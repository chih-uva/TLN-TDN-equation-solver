"""
Solve coupled first-order ODEs for (V_{eps omega 0}, W_{eps omega 0}) using
v_eqn.txt and w_eqn.txt (both set to zero) and precomputed background/U00 grids.

Equation source files:
  - wv_expressions/v_eqn.txt
  - wv_expressions/w_eqn.txt

Inputs:
  - config/h_eps_config.pkl
  - numpy_grid/h{h}_eps{eps}/RGrid_h{h}_eps{eps}.pkl (or Rgrid_...)
  - numpy_grid/h{h}_eps{eps}/PMSolGrid_h{h}_eps{eps}.pkl
  - numpy_grid/h{h}_eps{eps}/U00grid_h{h}_eps{eps}.pkl

Outputs:
  - numpy_grid/h{h}_eps{eps}/VWgrid_h{h}_eps{eps}.pkl
  - numpy_grid/h{h}_eps{eps}/VWmeta_h{h}_eps{eps}.pkl
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sympy import lambdify, symbols
from sympy.parsing.mathematica import parse_mathematica

from EOS_aux import PolytropeEOS_n1
from solver_tov import cgs_to_geom


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return pickle.load(f)


def resolve_paths(base_dir: Path, cfg: dict) -> dict:
    h_tok = str(cfg["h"])
    eps_tok = str(cfg["eps"])

    target_dir = base_dir / "numpy_grid" / f"h{h_tok}_eps{eps_tok}"

    r_path = target_dir / f"RGrid_h{h_tok}_eps{eps_tok}.pkl"
    if not r_path.exists():
        alt = target_dir / f"Rgrid_h{h_tok}_eps{eps_tok}.pkl"
        if alt.exists():
            r_path = alt

    return {
        "target_dir": target_dir,
        "r_path": r_path,
        "pm_path": target_dir / f"PMSolGrid_h{h_tok}_eps{eps_tok}.pkl",
        "u00_path": target_dir / f"U00grid_h{h_tok}_eps{eps_tok}.pkl",
        "eq_v_path": base_dir / "wv_expressions" / "v_eqn.txt",
        "eq_w_path": base_dir / "wv_expressions" / "w_eqn.txt",
        "ic_v_r6_path": base_dir / "wv_expressions" / "v0_r6",
        "ic_w_r6_path": base_dir / "wv_expressions" / "w0_r6.txt",
        "out_grid": target_dir / f"VWgrid_h{h_tok}_eps{eps_tok}.pkl",
        "out_meta": target_dir / f"VWmeta_h{h_tok}_eps{eps_tok}.pkl",
        "h_tok": h_tok,
        "eps_tok": eps_tok,
    }


def load_inputs(paths: dict):
    with open(paths["r_path"], "rb") as f:
        rgrid = pickle.load(f)
    with open(paths["pm_path"], "rb") as f:
        pmsol = pickle.load(f)
    with open(paths["u00_path"], "rb") as f:
        u00 = pickle.load(f)

    with open(paths["eq_v_path"], "r", encoding="utf-8") as f:
        v_eq_text = f.read()
    with open(paths["eq_w_path"], "r", encoding="utf-8") as f:
        w_eq_text = f.read()
    with open(paths["ic_v_r6_path"], "r", encoding="utf-8") as f:
        v0_r6_text = f.read()
    with open(paths["ic_w_r6_path"], "r", encoding="utf-8") as f:
        w0_r6_text = f.read()

    return rgrid, pmsol, u00, v_eq_text, w_eq_text, v0_r6_text, w0_r6_text


def _normalize_mma_expr(expr_text: str) -> str:
    return expr_text.replace("\\[Pi]", "Pi")


def compile_r6_ic_functions(v0_r6_text: str, w0_r6_text: str):
    """
    Compile Mathematica-format r^6 IC expressions into numeric callables.
    Expected symbols in expressions: r, K, pc, a0, v0.
    """
    expr_v = parse_mathematica(_normalize_mma_expr(v0_r6_text))
    expr_w = parse_mathematica(_normalize_mma_expr(w0_r6_text))

    r_sym, k_sym, pc_sym, a0_sym, v0_sym = symbols("r K pc a0 v0")
    v_fn = lambdify((r_sym, k_sym, pc_sym, a0_sym, v0_sym), expr_v, modules="numpy")
    w_fn = lambdify((r_sym, k_sym, pc_sym, a0_sym, v0_sym), expr_w, modules="numpy")
    return v_fn, w_fn


def compute_rhs_coeffs(r, p, rho, drho, lamb, h00, h00p):
    """
    Build linear coefficients from v_eqn=0 and w_eqn=0:
      V' = cV*V + cW*W + c0
      W' = dV*V + dW*W + d0

    The algebra is derived directly from the two provided Mathematica equations.
    """
    pi = np.pi
    e_l = np.exp(lamb)
    e_l_half = np.exp(0.5 * lamb)
    e_l_inv = np.exp(-lamb)

    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r6 = r3 * r3

    # Shared factor B = (-1 + e^lambda + 8 e^lambda pi r^2 p) used in w_eqn,
    # and (-3 + e^lambda + 8 e^lambda pi r^2 p) in v_eqn's V coefficient.
    b_w = -1.0 + e_l + 8.0 * e_l * pi * r2 * p
    b_v = -3.0 + e_l + 8.0 * e_l * pi * r2 * p

    # v_eqn structure pieces
    q_v = (
        3.0
        + 3.0 * e_l
        - 7.0 * (e_l**2)
        + (e_l**3)
        + 192.0 * (e_l**2) * (-2.0 + e_l) * (pi**2) * r4 * (p**2)
        + 512.0 * (e_l**3) * (pi**3) * r6 * (p**3)
        - 8.0 * e_l * (-3.0 + e_l) * pi * r2 * rho
        + 8.0
        * e_l
        * pi
        * r2
        * p
        * (8.0 - 13.0 * e_l + 3.0 * (e_l**2) - 8.0 * e_l * pi * r2 * rho)
    )

    hp_v = (
        12.0
        - 3.0 * e_l_inv
        - e_l
        + 32.0 * pi * r2 * p
        - 16.0 * e_l * pi * r2 * p
        - 64.0 * e_l * (pi**2) * r4 * (p**2)
    )

    c_v = b_v / r
    c_w = -e_l_half / r + (8.0 / 3.0) * e_l_half * pi * r * (p + rho)
    c_0 = ((e_l_inv * h00 * q_v) / r - h00p * hp_v) / 24.0

    # w_eqn structure pieces
    a_w = (
        512.0 * (e_l**3) * (pi**3) * r6 * (p**4)
        + (1.0 - 7.0 * e_l + 5.0 * (e_l**2) + (e_l**3)) * rho
        - 8.0 * e_l * (-1.0 + e_l) * pi * r2 * (rho**2)
        + 64.0
        * (e_l**2)
        * (pi**2)
        * r4
        * (p**3)
        * (-4.0 + 3.0 * e_l + 8.0 * e_l * pi * r2 * rho)
        + 8.0
        * e_l
        * pi
        * r2
        * (p**2)
        * (2.0 + e_l + 3.0 * (e_l**2) + 8.0 * e_l * (-5.0 + 3.0 * e_l) * pi * r2 * rho)
        + p
        * (
            1.0
            - 7.0 * e_l
            + 5.0 * (e_l**2)
            + (e_l**3)
            + 24.0 * e_l * (1.0 + (e_l**2)) * pi * r2 * rho
            - 64.0 * (e_l**2) * (pi**2) * r4 * (rho**2)
        )
        - 4.0 * e_l * r * drho
    )

    hp_w = h00p * r * (
        8.0 * e_l * pi * r2 * (p**2)
        - rho
        + e_l * rho
        + p * (-1.0 + e_l + 8.0 * e_l * pi * r2 * rho)
    )

    denom_w = b_w * 4.0 * e_l_half * r * (p + rho)

    d_v = -6.0 * e_l_half / r
    d_w = -(3.0 * (p + rho) + r * drho) / (r * (p + rho))
    d_0 = -(h00 * a_w + b_w * hp_w) / denom_w

    return c_v, c_w, c_0, d_v, d_w, d_0, b_w, denom_w


def rhs_vw(r, y, coeffs_row):
    v, w = y
    c_v, c_w, c_0, d_v, d_w, d_0 = coeffs_row
    vp = c_v * v + c_w * w + c_0
    wp = d_v * v + d_w * w + d_0
    return np.array([vp, wp], dtype=float)


def rk4_step(f, x, y, h, coeffs_i, coeffs_i1):
    # linear interpolation of coefficients at half-step
    coeffs_mid = 0.5 * (coeffs_i + coeffs_i1)

    k1 = f(x, y, coeffs_i)
    k2 = f(x + 0.5 * h, y + 0.5 * h * k1, coeffs_mid)
    k3 = f(x + 0.5 * h, y + 0.5 * h * k2, coeffs_mid)
    k4 = f(x + h, y + h * k3, coeffs_i1)

    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_vw_on_grid(y0, r_grid, coeffs):
    sol = np.zeros((len(r_grid), 2), dtype=float)
    sol[0] = y0

    y = y0.copy()
    for i in range(len(r_grid) - 1):
        dr = r_grid[i + 1] - r_grid[i]
        y = rk4_step(rhs_vw, r_grid[i], y, dr, coeffs[i], coeffs[i + 1])
        sol[i + 1] = y

    return sol


def main():
    parser = argparse.ArgumentParser(description="Solve coupled V/W system from v_eqn.txt and w_eqn.txt")
    parser.add_argument("--num-models", type=int, default=None, help="Optional limit on number of stellar models")
    parser.add_argument("--a0", type=float, default=1.0, help="a0 used by v0_r6/w0_r6 expressions")
    parser.add_argument("--v0", type=float, default=1.0, help="v0 used by v0_r6/w0_r6 expressions")
    parser.add_argument("--keep-full-length", action="store_true", help="Pad with NaN to original grid length")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cfg = load_config(base_dir / "config" / "h_eps_config.pkl")
    paths = resolve_paths(base_dir, cfg)

    print("Loading:")
    print("  ", paths["r_path"])
    print("  ", paths["pm_path"])
    print("  ", paths["u00_path"])
    print("  ", paths["eq_v_path"])
    print("  ", paths["eq_w_path"])
    print("  ", paths["ic_v_r6_path"])
    print("  ", paths["ic_w_r6_path"])

    rgrid_all, pmsol_all, u00_all, v_eq_text, w_eq_text, v0_r6_text, w0_r6_text = load_inputs(paths)
    v0_r6_fn, w0_r6_fn = compile_r6_ic_functions(v0_r6_text, w0_r6_text)

    d_num_cfg = int(cfg["d_num"])
    n_models = min(d_num_cfg, len(rgrid_all), len(pmsol_all), len(u00_all))
    if args.num_models is not None:
        n_models = min(n_models, int(args.num_models))

    k_cgs = float(cfg["K_cgs"])
    pc_arr_cgs = np.logspace(33.5, 36.5, d_num_cfg)
    k_geom, pc_arr = cgs_to_geom(pc_arr_cgs, 1, k_cgs)[:2]
    eos = PolytropeEOS_n1(K=k_geom)

    vw_grid = []
    records = []

    print("\n===== VW coupled solve starts =====\n")

    for nnn in tqdm(range(n_models), desc="VW Solver", total=n_models):
        r_arr = np.asarray(rgrid_all[nnn], dtype=float)
        pm_arr = np.asarray(pmsol_all[nnn], dtype=float)
        u00_arr = np.asarray(u00_all[nnn], dtype=float)

        p_arr = np.asarray(pm_arr[:, 1], dtype=float)
        m_arr = np.asarray(pm_arr[:, 0], dtype=float)

        rho_arr = eos.energy_density(p_arr)
        lamb_arr = np.log(r_arr / (r_arr - 2.0 * m_arr))
        edge_order = 2 if len(r_arr) >= 3 else 1
        drho_arr = np.gradient(rho_arr, r_arr, edge_order=edge_order)

        u_arr = np.asarray(u00_arr[:, 0], dtype=float)
        up_arr = np.asarray(u00_arr[:, 1], dtype=float)
        h00_arr = (r_arr * r_arr) * u_arr
        h00p_arr = (r_arr * r_arr) * up_arr + 2.0 * r_arr * u_arr

        arr_len = min(
            len(r_arr),
            len(p_arr),
            len(m_arr),
            len(rho_arr),
            len(lamb_arr),
            len(drho_arr),
            len(h00_arr),
            len(h00p_arr),
        )
        surface_index = max(arr_len - 4, 2)

        r_use = r_arr[:surface_index]
        p_use = p_arr[:surface_index]
        rho_use = rho_arr[:surface_index]
        drho_use = drho_arr[:surface_index]
        lamb_use = lamb_arr[:surface_index]
        h00_use = h00_arr[:surface_index]
        h00p_use = h00p_arr[:surface_index]

        coeffs = np.zeros((len(r_use), 6), dtype=float)
        singular_count = 0

        for i in range(len(r_use)):
            c_v, c_w, c_0, d_v, d_w, d_0, b_w, denom_w = compute_rhs_coeffs(
                r=r_use[i],
                p=p_use[i],
                rho=rho_use[i],
                drho=drho_use[i],
                lamb=lamb_use[i],
                h00=h00_use[i],
                h00p=h00p_use[i],
            )

            if (not np.isfinite(b_w)) or (not np.isfinite(denom_w)) or abs(denom_w) < 1e-30:
                singular_count += 1

            coeffs[i] = np.array([c_v, c_w, c_0, d_v, d_w, d_0], dtype=float)

        eps = float(r_use[0])
        pc = float(pc_arr[nnn])
        v_ic = float(v0_r6_fn(eps, k_geom, pc, args.a0, args.v0))
        w_ic = float(w0_r6_fn(eps, k_geom, pc, args.a0, args.v0))
        y0 = np.array([v_ic, w_ic], dtype=float)

        sol_interior = integrate_vw_on_grid(y0, r_use, coeffs)

        if args.keep_full_length:
            sol_full = np.full((len(r_arr), 2), np.nan, dtype=float)
            sol_full[:surface_index] = sol_interior
            vw_grid.append(sol_full)
        else:
            vw_grid.append(sol_interior)

        records.append(
            {
                "star_id": int(nnn),
                "status": "ok" if singular_count == 0 else "warn_singular_coeff",
                "arr_len": int(arr_len),
                "surface_index": int(surface_index),
                "R_surf": float(r_use[-1]),
                "M_surf": float(m_arr[surface_index - 1]),
                "a0": float(args.a0),
                "v0": float(args.v0),
                "V_ic": v_ic,
                "W_ic": w_ic,
                "singular_count": int(singular_count),
                "V_end": float(sol_interior[-1, 0]),
                "W_end": float(sol_interior[-1, 1]),
            }
        )

    print("\nWriting outputs...")
    with open(paths["out_grid"], "wb") as f:
        pickle.dump(vw_grid, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "h": paths["h_tok"],
            "eps": paths["eps_tok"],
            "d_num_config": d_num_cfg,
            "num_models_processed": n_models,
            "K_cgs": k_cgs,
            "K_geom": float(k_geom),
            "a0_input": float(args.a0),
            "v0_input": float(args.v0),
            "keep_full_length": bool(args.keep_full_length),
            "inputs": {
                "RGrid": str(paths["r_path"]),
                "PMSolGrid": str(paths["pm_path"]),
                "U00grid": str(paths["u00_path"]),
                "v_eqn": str(paths["eq_v_path"]),
                "w_eqn": str(paths["eq_w_path"]),
                "v0_r6": str(paths["ic_v_r6_path"]),
                "w0_r6": str(paths["ic_w_r6_path"]),
            },
            "equation_source_preview": {
                "v_eqn_head": v_eq_text[:220],
                "w_eqn_head": w_eq_text[:220],
                "v0_r6_head": v0_r6_text[:220],
                "w0_r6_head": w0_r6_text[:220],
            },
        },
        "records": records,
    }
    with open(paths["out_meta"], "wb") as f:
        pickle.dump(meta_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_warn = sum(1 for r in records if r["status"] != "ok")
    print(f"Done. models={len(records)}, warnings={n_warn}")
    print("Output:")
    print("  ", paths["out_grid"])
    print("  ", paths["out_meta"])


if __name__ == "__main__":
    main()
