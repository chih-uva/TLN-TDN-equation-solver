"""
Solve for U01 := h01 / r^2 using precomputed background grids and concatenated source.

Pipeline (per model):
  1) Build background arrays from TOV solution
  2) Integrate homogeneous/full U01 ODE on interior grid
  3) Form particular = full - homogeneous
  4) Match at stellar surface to exterior l=2 homogeneous basis
  5) Save U01 grids and matching diagnostics

Inputs:
  - config/h_eps_config.pkl
  - numpy_grid/h{h}_eps{eps}/RGrid_h{h}_eps{eps}.pkl (or Rgrid_...)
  - numpy_grid/h{h}_eps{eps}/PMSolGrid_h{h}_eps{eps}.pkl
  - numpy_grid/h{h}_eps{eps}/SourceGrid_concat_h{h}_eps{eps}.pkl

Outputs (under U01Grids/):
  - U01grid_hom_h{h}_eps{eps}.pkl
  - U01grid_full_h{h}_eps{eps}.pkl
  - U01grid_part_h{h}_eps{eps}.pkl
  - U01grid_matched_h{h}_eps{eps}.pkl
  - H01Match_h{h}_eps{eps}.pkl
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.special import hyp2f1
from tqdm import tqdm

from EOS_aux import PolytropeEOS_n1
from h00_aux import h00_approx, h00p_approx
from solver_tov import cgs_to_geom


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return pickle.load(f)


def resolve_paths(base_dir: Path, cfg: dict) -> dict:
    h_tok = str(cfg["h"])
    eps_tok = str(cfg["eps"])

    grid_root = base_dir / "numpy_grid"
    target_dir = grid_root / f"h{h_tok}_eps{eps_tok}"
    out_dir = target_dir / "U01Grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    r_path = target_dir / f"RGrid_h{h_tok}_eps{eps_tok}.pkl"
    if not r_path.exists():
        alt = target_dir / f"Rgrid_h{h_tok}_eps{eps_tok}.pkl"
        if alt.exists():
            r_path = alt

    pm_path = target_dir / f"PMSolGrid_h{h_tok}_eps{eps_tok}.pkl"
    src_path = target_dir / f"SourceGrid_concat_h{h_tok}_eps{eps_tok}.pkl"

    return {
        "target_dir": target_dir,
        "out_dir": out_dir,
        "r_path": r_path,
        "pm_path": pm_path,
        "src_path": src_path,
        "h_tok": h_tok,
        "eps_tok": eps_tok,
    }


def load_inputs(paths: dict):
    with open(paths["r_path"], "rb") as f:
        rgrid = pickle.load(f)
    with open(paths["pm_path"], "rb") as f:
        pmsol = pickle.load(f)
    with open(paths["src_path"], "rb") as f:
        source_concat = pickle.load(f)
    return rgrid, pmsol, source_concat


def build_source_profile(source_arr, source_sign: int = -1):
    if source_sign not in (-1, 1):
        raise ValueError("source_sign must be +1 or -1")
    return source_sign * np.asarray(source_arr, dtype=float)


def rk4_step(f, x, y, h, **kwargs):

    k1 = f(x, y, **kwargs)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1, **kwargs)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2, **kwargs)
    k4 = f(x + h, y + h * k3, **kwargs)
    
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def u01_rhs_indexed(r, Y, p_r, rho_r, lamb_r, drho, s_eff):
    u, up = Y

    if r <= 0.0:
        raise ZeroDivisionError(f"r must be positive, got r={r}")

    pi = np.pi
    e_l = np.exp(lamb_r)
    e2_l = e_l * e_l
    e3_l = e2_l * e_l

    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r6 = r3 * r3

    a_num = (
        -1.0
        - 3.0 * e_l
        + 3.0 * e2_l
        + e3_l
        + 96.0 * e2_l * (-5.0 + 2.0 * e_l) * (pi**2) * r4 * (p_r**2)
        + 512.0 * e3_l * (pi**3) * r6 * (p_r**3)
        - 20.0 * e_l * (-1.0 + e_l) * pi * r2 * rho_r
        + 4.0 * e_l * pi * r2 * p_r * (15.0 - 9.0 * e_l + 6.0 * e2_l - 40.0 * e_l * pi * r2 * rho_r)
        + 8.0 * e_l * pi * r3 * drho
    )

    d_den = r2 * (-1.0 + e_l + 8.0 * e_l * pi * r2 * p_r)
    if (not np.isfinite(d_den)) or abs(d_den) < 1e-30:
        raise ZeroDivisionError(f"Singular denominator at r={r}, D={d_den}")

    b_coeff = 1.0 + e_l + 4.0 * e_l * pi * r2 * p_r - 4.0 * e_l * pi * r2 * rho_r

    # Source transformation for u = h/r^2: S_h -> S_u = S_h / r^2
    source_u = s_eff / r2

    upp = (a_num / d_den) * u - ((4.0 + b_coeff) / r) * up - ((2.0 + 2.0 * b_coeff) / r2) * u + source_u
    
    return np.array([up, upp], dtype=float)


def integrate_on_grid_u01(
    y0,
    r_grid,
    p_arr,
    rho_arr,
    lamb_arr,
    drho_arr,
    source_arr,
    show_progress: bool = False,
    progress_desc: str = "",
):
    sol = np.zeros((len(r_grid), 2), dtype=float)
    sol[0] = y0

    y = y0.copy()
    step_iter = range(len(r_grid) - 1)
    pbar = None
    if show_progress:
        pbar = tqdm(
            step_iter,
            total=len(r_grid) - 1,
            desc=progress_desc,
            leave=False,
            unit="step",
        )
        step_iter = pbar

    for i in step_iter:
        r = r_grid[i]
        dr = r_grid[i + 1] - r_grid[i]
        if pbar is not None and (i % 2000 == 0 or i == len(r_grid) - 2):
            pbar.set_postfix_str(f"r={r:.6e}")
        params = {
            "p_r": p_arr[i],
            "rho_r": rho_arr[i],
            "lamb_r": lamb_arr[i],
            "drho": drho_arr[i],
            "s_eff": source_arr[i],
        }
        y = rk4_step(
            u01_rhs_indexed,
            r,
            y,
            dr,
            **params,
        )
        sol[i + 1] = y

    return sol


def u_to_h(u: float, up: float, r: float):
    h = (r * r) * u
    hp = (r * r) * up + 2.0 * r * u
    return h, hp


def eval_exterior_mh2r_l2_c4(r: float, m_tot: float):
    """
    Katagiri2409 Eq. (C4) closed-form l=2 implementation.
    """
    raise NotImplementedError("C4 closed form is not implemented yet. Use method='hypergeom'.")


def eval_exterior_mh2r_l2_hypergeom(r: float, m_tot: float):
    """
    Exterior basis for l=2 using Eq. (20):
      H_l^R = f * (M/r)^(l+1) * 2F1(l+1, l+3; 2l+2; 2M/r),
      f = 1 - 2M/r.

    Returns M*H_2^R and its analytic d/dr at radius r (outside star),
    using constant exterior mass M = m_tot.
    """
    if r <= 0.0:
        raise ValueError("r must be positive")

    l = 2
    z = 2.0 * m_tot / r
    x = m_tot / r
    f = 1.0 - z

    # H = f * x^(l+1) * 2F1(l+1,l+3;2l+2;z)
    F = hyp2f1(l + 1, l + 3, 2 * l + 2, z)
    h_ext = f * (x ** (l + 1)) * F

    # Analytic derivative
    # dF/dz = (ab/c) * 2F1(a+1,b+1;c+1;z)
    a = l + 1
    b = l + 3
    c = 2 * l + 2
    dF_dz = (a * b / c) * hyp2f1(a + 1, b + 1, c + 1, z)

    df_dr = 2.0 * m_tot / (r * r)
    dx3_dr = -(l + 1) * (x ** (l + 1)) / r
    dz_dr = -2.0 * m_tot / (r * r)
    dF_dr = dF_dz * dz_dr

    dh_dr = df_dr * (x ** (l + 1)) * F + f * dx3_dr * F + f * (x ** (l + 1)) * dF_dr

    # We match using M*H and derivative of M*H
    mh = m_tot * h_ext
    dmh_dr = m_tot * dh_dr
    return mh, dmh_dr


def eval_exterior_mh2r_l2(r: float, m_tot: float, method: str = "hypergeom", use_fd_derivative: bool = False):
    if method == "c4":
        mh, dmh_dr = eval_exterior_mh2r_l2_c4(r, m_tot)
    elif method == "hypergeom":
        mh, dmh_dr = eval_exterior_mh2r_l2_hypergeom(r, m_tot)
    else:
        raise ValueError(f"Unknown exterior method: {method}")

    if use_fd_derivative:
        # one-sided finite difference outside the surface with constant M_tot
        dr = max(1e-6 * r, 1e-9)
        mh_plus, _ = eval_exterior_mh2r_l2_hypergeom(r + dr, m_tot)
        dmh_dr = (mh_plus - mh) / dr

    return mh, dmh_dr


def match_surface_coeffs(r_surf, m_tot, u_hom_R, up_hom_R, u_part_R, up_part_R, exterior_method, use_fd_derivative):
    h_hom_R, hp_hom_R = u_to_h(u_hom_R, up_hom_R, r_surf)
    h_part_R, hp_part_R = u_to_h(u_part_R, up_part_R, r_surf)

    h_ext_R, hp_ext_R = eval_exterior_mh2r_l2(
        r_surf,
        m_tot,
        method=exterior_method,
        use_fd_derivative=use_fd_derivative,
    )

    denom = hp_hom_R * h_ext_R - h_hom_R * hp_ext_R
    if (not np.isfinite(denom)) or abs(denom) < 1e-30:
        return {
            "status": "singular_match",
            "A0": np.nan,
            "I2m": np.nan,
            "h_hom_R": h_hom_R,
            "hp_hom_R": hp_hom_R,
            "h_part_R": h_part_R,
            "hp_part_R": hp_part_R,
            "h_out_R": h_ext_R,
            "hp_out_R": hp_ext_R,
            "denom": denom,
        }

    # Same linear matching formulas used in old notebook
    A0 = (hp_ext_R * h_part_R - h_ext_R * hp_part_R) / denom
    I2m = (hp_hom_R * h_part_R - h_hom_R * hp_part_R) / denom

    h_in_matched_R = A0 * h_hom_R + h_part_R
    hp_in_matched_R = A0 * hp_hom_R + hp_part_R
    h_out_matched_R = I2m * h_ext_R
    hp_out_matched_R = I2m * hp_ext_R

    res_h = h_in_matched_R - h_out_matched_R
    res_hp = hp_in_matched_R - hp_out_matched_R

    rel_res_h = abs(res_h) / max(abs(h_out_matched_R), 1e-30)
    rel_res_hp = abs(res_hp) / max(abs(hp_out_matched_R), 1e-30)

    return {
        "status": "ok",
        "A0": A0,
        "I2m": I2m,
        "h_hom_R": h_hom_R,
        "hp_hom_R": hp_hom_R,
        "h_part_R": h_part_R,
        "hp_part_R": hp_part_R,
        "h_out_R": h_ext_R,
        "hp_out_R": hp_ext_R,
        "h_in_matched_R": h_in_matched_R,
        "hp_in_matched_R": hp_in_matched_R,
        "h_out_matched_R": h_out_matched_R,
        "hp_out_matched_R": hp_out_matched_R,
        "res_h": res_h,
        "res_hp": res_hp,
        "rel_res_h": rel_res_h,
        "rel_res_hp": rel_res_hp,
        "denom": denom,
    }


def main():
    parser = argparse.ArgumentParser(description="Solve U01=h01/r^2 and surface-match to exterior basis.")
    parser.add_argument("--source-sign", type=int, default=-1, choices=[-1, 1], help="Use +1 or -1 multiplier for input source grid.")
    parser.add_argument("--exterior-method", type=str, default="hypergeom", choices=["hypergeom", "c4"], help="Exterior basis implementation.")
    parser.add_argument("--fd-derivative", action="store_true", help="Use one-sided finite difference for exterior derivative.")
    parser.add_argument("--num-models", type=int, default=None, help="Optional limit on number of models to process.")
    parser.add_argument("--show-step-progress", action="store_true", help="Show per-integration RK4 step progress bars.")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "config" / "h_eps_config.pkl"

    cfg = load_config(config_path)
    paths = resolve_paths(base_dir, cfg)

    print("Loading:")
    print("  ", paths["r_path"])
    print("  ", paths["pm_path"])
    print("  ", paths["src_path"])

    rgrid_all, pmsol_all, source_concat_all = load_inputs(paths)

    d_num_cfg = int(cfg["d_num"])
    n_models = min(d_num_cfg, len(rgrid_all), len(pmsol_all), len(source_concat_all))
    if args.num_models is not None:
        n_models = min(n_models, int(args.num_models))

    k_cgs = float(cfg["K_cgs"])
    pc_arr_cgs = np.logspace(33.5, 36.5, d_num_cfg)
    k_geom, pc_arr = cgs_to_geom(pc_arr_cgs, 1, k_cgs)[:2]

    eos = PolytropeEOS_n1(K=k_geom)

    u01_hom_grid = []
    u01_full_grid = []
    u01_part_grid = []
    u01_matched_grid = []
    match_records = []

    print("\n===== U01 integration and matching starts =====\n")

    for nnn in tqdm(range(n_models), desc="U01 Solver", total=n_models):
        r_arr = np.asarray(rgrid_all[nnn], dtype=float)
        pm_arr = np.asarray(pmsol_all[nnn], dtype=float)
        source_arr_raw = np.asarray(source_concat_all[nnn], dtype=float)

        p_arr = pm_arr[:, 1]
        m_arr = pm_arr[:, 0]

        rho_arr = eos.energy_density(p_arr)
        lamb_arr = np.log(r_arr / (r_arr - 2.0 * m_arr))
        edge_order = 2 if len(r_arr) >= 3 else 1
        drho_arr = np.gradient(rho_arr, r_arr, edge_order=edge_order)

        arr_len = min(
            len(r_arr),
            len(p_arr),
            len(m_arr),
            len(rho_arr),
            len(lamb_arr),
            len(drho_arr),
            len(source_arr_raw),
        )
        surface_index = max(arr_len - 4, 1)

        # Interior region (same indexing convention as old notebook)
        r_use = r_arr[:surface_index]
        p_use = p_arr[:surface_index]
        m_use = m_arr[:surface_index]
        rho_use = rho_arr[:surface_index]
        lamb_use = lamb_arr[:surface_index]
        drho_use = drho_arr[:surface_index]
        source_use = build_source_profile(source_arr_raw[:surface_index], source_sign=args.source_sign)

        eps = float(r_use[0])
        pc = float(pc_arr[nnn])

        # Center initial condition from h00 homogeneous analytic expansion
        h_ini = h00_approx(r=eps, K=k_geom, pc=pc, a0=1.0, a8=0.0)
        hp_ini = h00p_approx(r=eps, K=k_geom, pc=pc, a0=1.0, a8=0.0)

        u_ini = h_ini / (eps * eps)
        up_ini = (eps * hp_ini - 2.0 * h_ini) / (eps**3)
        y0 = np.array([u_ini, up_ini], dtype=float)

        # homogeneous / full / particular
        source_zero = np.zeros_like(source_use)
        u_hom = integrate_on_grid_u01(
            y0,
            r_use,
            p_use,
            rho_use,
            lamb_use,
            drho_use,
            source_zero,
            show_progress=args.show_step_progress,
            progress_desc=f"Model {nnn + 1}/{n_models} hom",
        )
        u_full = integrate_on_grid_u01(
            y0,
            r_use,
            p_use,
            rho_use,
            lamb_use,
            drho_use,
            source_use,
            show_progress=args.show_step_progress,
            progress_desc=f"Model {nnn + 1}/{n_models} full",
        )
        u_part = u_full - u_hom

        r_surf = float(r_use[-1])
        m_tot = float(m_use[-1])

        match = match_surface_coeffs(
            r_surf=r_surf,
            m_tot=m_tot,
            u_hom_R=float(u_hom[-1, 0]),
            up_hom_R=float(u_hom[-1, 1]),
            u_part_R=float(u_part[-1, 0]),
            up_part_R=float(u_part[-1, 1]),
            exterior_method=args.exterior_method,
            use_fd_derivative=args.fd_derivative,
        )

        if match["status"] == "ok":
            a0 = float(match["A0"])
            u_matched = a0 * u_hom + u_part
        else:
            u_matched = np.full_like(u_hom, np.nan)

        u01_hom_grid.append(u_hom)
        u01_full_grid.append(u_full)
        u01_part_grid.append(u_part)
        u01_matched_grid.append(u_matched)

        rec = {
            "star_id": int(nnn),
            "status": match["status"],
            "surface_index": int(surface_index),
            "R_surf": r_surf,
            "M_tot": m_tot,
            "source_sign": int(args.source_sign),
            "exterior_method": str(args.exterior_method),
            "derivative_mode": "fd" if args.fd_derivative else "analytic",
        }
        rec.update(match)
        match_records.append(rec)

    h_tok = paths["h_tok"]
    eps_tok = paths["eps_tok"]
    out_dir = paths["out_dir"]

    out_hom = out_dir / f"U01grid_hom_h{h_tok}_eps{eps_tok}.pkl"
    out_full = out_dir / f"U01grid_full_h{h_tok}_eps{eps_tok}.pkl"
    out_part = out_dir / f"U01grid_part_h{h_tok}_eps{eps_tok}.pkl"
    out_match = out_dir / f"U01grid_matched_h{h_tok}_eps{eps_tok}.pkl"
    out_diag = out_dir / f"H01Match_h{h_tok}_eps{eps_tok}.pkl"

    print("\nWriting outputs...")
    with open(out_hom, "wb") as f:
        pickle.dump(u01_hom_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_full, "wb") as f:
        pickle.dump(u01_full_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_part, "wb") as f:
        pickle.dump(u01_part_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_match, "wb") as f:
        pickle.dump(u01_matched_grid, f, protocol=pickle.HIGHEST_PROTOCOL)

    diag_payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "h": h_tok,
            "eps": eps_tok,
            "d_num_config": d_num_cfg,
            "num_models_processed": n_models,
            "K_cgs": k_cgs,
            "K_geom": float(k_geom),
            "source_sign": int(args.source_sign),
            "exterior_method": str(args.exterior_method),
            "derivative_mode": "fd" if args.fd_derivative else "analytic",
            "inputs": {
                "RGrid": str(paths["r_path"]),
                "PMSolGrid": str(paths["pm_path"]),
                "SourceGrid_concat": str(paths["src_path"]),
            },
        },
        "records": match_records,
    }
    with open(out_diag, "wb") as f:
        pickle.dump(diag_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_ok = sum(1 for r in match_records if r.get("status") == "ok")
    print(f"Done. matched_ok={n_ok}/{len(match_records)}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
