"""
Solve V/W in three stages:
  1) full inhomogeneous system (existing V/W equations + full IC series)
  2) homogeneous system (homogeneous V/W equations + hom IC series)
  3) particular = full - homogeneous

Then enforce the surface condition (surface_condition_v0.txt) at the stellar
surface defined by the "shrink 4 indices" convention:
  surface_index = max(arr_len - 4, 2)
  r_surf = r_arr[surface_index - 1]

The matched solution is:
  V_matched = V_part + C * V_hom
  W_matched = W_part + C * W_hom
with C chosen so surface_condition_v0(r_surf) = 0.

Inputs:
  - config/h_eps_config.pkl
  - numpy_grid/h{h}_eps{eps}/RGrid_h{h}_eps{eps}.pkl (or Rgrid_...)
  - numpy_grid/h{h}_eps{eps}/PMSolGrid_h{h}_eps{eps}.pkl
  - numpy_grid/h{h}_eps{eps}/U00grid_h{h}_eps{eps}.pkl (or U00Grid_...)
  - wv_expressions/v_eqn.txt, w_eqn.txt
  - wv_expressions/v_eqn_hom.txt, w_eqn_hom.txt
  - wv_expressions/v0_r6(.txt), w0_r6(.txt)
  - wv_expressions/v0_r6_hom.txt, w0_r6_hom.txt
  - wv_expressions/surface_condition_v0.txt

Outputs (under numpy_grid/h{h}_eps{eps}/VWDecomp/):
  - VWfull_grid_h{h}_eps{eps}.pkl
  - VWhom_grid_h{h}_eps{eps}.pkl
  - VWpart_grid_h{h}_eps{eps}.pkl
  - VWmatched_grid_h{h}_eps{eps}.pkl
  - VWfull_hom_particular_meta_h{h}_eps{eps}.pkl
"""

from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sympy import lambdify, symbols
from sympy.parsing.mathematica import parse_mathematica
from tqdm import tqdm

from EOS_aux import PolytropeEOS_n1
from solver_tov import cgs_to_geom


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return pickle.load(f)


def _pick_existing(paths: list[Path], label: str) -> Path:
    for p in paths:
        if p.exists():
            return p
    tried = "\n  - ".join(str(p) for p in paths)
    raise FileNotFoundError(f"Missing {label}. Tried:\n  - {tried}")


def resolve_paths(base_dir: Path, cfg: dict) -> dict:
    h_tok = str(cfg["h"])
    eps_tok = str(cfg["eps"])

    target_dir = base_dir / "numpy_grid" / f"h{h_tok}_eps{eps_tok}"
    expr_dir = base_dir / "wv_expressions"
    out_dir = target_dir / "VWDecomp"
    out_dir.mkdir(parents=True, exist_ok=True)

    r_path = _pick_existing(
        [
            target_dir / f"RGrid_h{h_tok}_eps{eps_tok}.pkl",
            target_dir / f"Rgrid_h{h_tok}_eps{eps_tok}.pkl",
        ],
        "RGrid",
    )
    u00_path = _pick_existing(
        [
            target_dir / f"U00grid_h{h_tok}_eps{eps_tok}.pkl",
            target_dir / f"U00Grid_h{h_tok}_eps{eps_tok}.pkl",
        ],
        "U00grid",
    )

    return {
        "target_dir": target_dir,
        "out_dir": out_dir,
        "r_path": r_path,
        "pm_path": target_dir / f"PMSolGrid_h{h_tok}_eps{eps_tok}.pkl",
        "u00_path": u00_path,
        "eq_v_full": _pick_existing([expr_dir / "v_eqn.txt"], "v_eqn.txt"),
        "eq_w_full": _pick_existing([expr_dir / "w_eqn.txt"], "w_eqn.txt"),
        "eq_v_hom": _pick_existing([expr_dir / "v_eqn_hom.txt"], "v_eqn_hom.txt"),
        "eq_w_hom": _pick_existing([expr_dir / "w_eqn_hom.txt"], "w_eqn_hom.txt"),
        "ic_v_full": _pick_existing([expr_dir / "v0_r6.txt", expr_dir / "v0_r6"], "v0_r6"),
        "ic_w_full": _pick_existing([expr_dir / "w0_r6.txt", expr_dir / "w0_r6"], "w0_r6"),
        "ic_v_hom": _pick_existing([expr_dir / "v0_r6_hom.txt"], "v0_r6_hom.txt"),
        "ic_w_hom": _pick_existing([expr_dir / "w0_r6_hom.txt"], "w0_r6_hom.txt"),
        "surface_cond": _pick_existing([expr_dir / "surface_condition_v0.txt"], "surface_condition_v0.txt"),
        "out_full": out_dir / f"VWfull_grid_h{h_tok}_eps{eps_tok}.pkl",
        "out_hom": out_dir / f"VWhom_grid_h{h_tok}_eps{eps_tok}.pkl",
        "out_part": out_dir / f"VWpart_grid_h{h_tok}_eps{eps_tok}.pkl",
        "out_matched": out_dir / f"VWmatched_grid_h{h_tok}_eps{eps_tok}.pkl",
        "out_meta": out_dir / f"VWfull_hom_particular_meta_h{h_tok}_eps{eps_tok}.pkl",
        "h_tok": h_tok,
        "eps_tok": eps_tok,
    }


def extract_u00_arrays(u00_model):
    if isinstance(u00_model, dict):
        if "u00_grid" in u00_model and "u00p_grid" in u00_model:
            return np.asarray(u00_model["u00_grid"], dtype=float), np.asarray(u00_model["u00p_grid"], dtype=float)
        if "u00" in u00_model and "u00p" in u00_model:
            return np.asarray(u00_model["u00"], dtype=float), np.asarray(u00_model["u00p"], dtype=float)

    arr = np.asarray(u00_model)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)

    raise ValueError("Unsupported U00 model format; expected Nx2 array or dict with u00/u00p arrays")


def _normalize_mma_expr(expr_text: str) -> str:
    return expr_text.replace("\\[Pi]", "Pi")


def compile_r6_ic_functions(v0_r6_text: str, w0_r6_text: str):
    expr_v = parse_mathematica(_normalize_mma_expr(v0_r6_text))
    expr_w = parse_mathematica(_normalize_mma_expr(w0_r6_text))
    r_sym, k_sym, pc_sym, a0_sym, v0_sym = symbols("r K pc a0 v0")
    v_fn = lambdify((r_sym, k_sym, pc_sym, a0_sym, v0_sym), expr_v, modules="numpy")
    w_fn = lambdify((r_sym, k_sym, pc_sym, a0_sym, v0_sym), expr_w, modules="numpy")
    return v_fn, w_fn


def compute_full_coeffs(r, p, rho, drho, lamb, h00, h00p):
    """
    Full system:
      V' = cV*V + cW*W + c0
      W' = dV*V + dW*W + d0
    """
    pi = np.pi
    e_l = np.exp(lamb)
    e_l_half = np.exp(0.5 * lamb)
    e_l_inv = np.exp(-lamb)

    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r6 = r3 * r3

    b_w = -1.0 + e_l + 8.0 * e_l * pi * r2 * p
    b_v = -3.0 + e_l + 8.0 * e_l * pi * r2 * p

    q_v = (
        3.0
        + 3.0 * e_l
        - 7.0 * (e_l**2)
        + (e_l**3)
        + 192.0 * (e_l**2) * (-2.0 + e_l) * (pi**2) * r4 * (p**2)
        + 512.0 * (e_l**3) * (pi**3) * r6 * (p**3)
        - 8.0 * e_l * (-3.0 + e_l) * pi * r2 * rho
        + 8.0 * e_l * pi * r2 * p * (8.0 - 13.0 * e_l + 3.0 * (e_l**2) - 8.0 * e_l * pi * r2 * rho)
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

    a_w = (
        512.0 * (e_l**3) * (pi**3) * r6 * (p**4)
        + (1.0 - 7.0 * e_l + 5.0 * (e_l**2) + (e_l**3)) * rho
        - 8.0 * e_l * (-1.0 + e_l) * pi * r2 * (rho**2)
        + 64.0 * (e_l**2) * (pi**2) * r4 * (p**3) * (-4.0 + 3.0 * e_l + 8.0 * e_l * pi * r2 * rho)
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
        8.0 * e_l * pi * r2 * (p**2) - rho + e_l * rho + p * (-1.0 + e_l + 8.0 * e_l * pi * r2 * rho)
    )

    denom_w = b_w * 4.0 * e_l_half * r * (p + rho)

    d_v = -6.0 * e_l_half / r
    d_w = -(3.0 * (p + rho) + r * drho) / (r * (p + rho))
    d_0 = -(h00 * a_w + b_w * hp_w) / denom_w

    return np.array([c_v, c_w, c_0, d_v, d_w, d_0], dtype=float), b_w, denom_w


def compute_hom_coeffs(r, p, rho, drho, lamb):
    """
    Homogeneous system from v_eqn_hom.txt and w_eqn_hom.txt:
      V' = cV*V + cW*W
      W' = dV*V + dW*W
    """
    pi = np.pi
    e_l = np.exp(lamb)
    e_l_half = np.exp(0.5 * lamb)

    r2 = r * r
    b_v = -3.0 + e_l + 8.0 * e_l * pi * r2 * p

    c_v = b_v / r
    c_w = -e_l_half / r + (8.0 / 3.0) * e_l_half * pi * r * (p + rho)
    c_0 = 0.0

    d_v = -6.0 * e_l_half / r
    d_w = -(3.0 * (p + rho) + r * drho) / (r * (p + rho))
    d_0 = 0.0

    return np.array([c_v, c_w, c_0, d_v, d_w, d_0], dtype=float)


def rhs_vw(_, y, coeffs_row):
    v, w = y
    c_v, c_w, c_0, d_v, d_w, d_0 = coeffs_row
    vp = c_v * v + c_w * w + c_0
    wp = d_v * v + d_w * w + d_0
    return np.array([vp, wp], dtype=float)


def rk4_step(f, x, y, h, coeffs_i, coeffs_i1):
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


def eval_surface_condition_v0(r, p, rho, lamb, h00, h00p, v, w, wp):
    """
    surface_condition_v0.txt:
      1/4 * (
        24 V + 12 exp(-lambda/2) W
        + H * (...)
        + r H' - exp(-lambda) r H' + 8 pi r^3 p H'
        + 4 exp(-lambda/2) r W'
      )
    """
    pi = np.pi
    e_l = np.exp(lamb)
    e_mhalf = np.exp(-0.5 * lamb)
    e_minv = np.exp(-lamb)
    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2

    h_coeff = (
        6.0
        - e_minv
        + e_l
        + 8.0 * (-3.0 + 2.0 * e_l) * pi * r2 * p
        + 64.0 * e_l * (pi**2) * r4 * (p**2)
        - 8.0 * pi * r2 * rho
    )

    return 0.25 * (
        24.0 * v
        + 12.0 * e_mhalf * w
        + h00 * h_coeff
        + r * h00p
        - e_minv * r * h00p
        + 8.0 * pi * r3 * p * h00p
        + 4.0 * e_mhalf * r * wp
    )


def maybe_pad(sol, full_len: int, used_len: int, keep_full_length: bool):
    if not keep_full_length:
        return sol
    out = np.full((full_len, 2), np.nan, dtype=float)
    out[:used_len] = sol
    return out


def main():
    parser = argparse.ArgumentParser(description="Solve full/homogeneous V/W, build particular, and fit C from surface_condition_v0=0.")
    parser.add_argument("--num-models", type=int, default=None, help="Optional limit on number of stellar models")
    parser.add_argument("--a0-full", type=float, default=1.0, help="a0 for full IC series")
    parser.add_argument("--v0-full", type=float, default=1.0, help="v0 for full IC series")
    parser.add_argument("--a0-hom", type=float, default=1.0, help="a0 for homogeneous IC series")
    parser.add_argument("--v0-hom", type=float, default=1.0, help="v0 for homogeneous IC series")
    parser.add_argument("--keep-full-length", action="store_true", help="Pad outputs with NaN to full original grid length")
    parser.add_argument("--denom-threshold", type=float, default=1e-20, help="Small-denominator threshold for C solve")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cfg = load_config(base_dir / "config" / "h_eps_config.pkl")
    paths = resolve_paths(base_dir, cfg)

    print("Loading:")
    print("  ", paths["r_path"])
    print("  ", paths["pm_path"])
    print("  ", paths["u00_path"])
    print("  ", paths["ic_v_full"])
    print("  ", paths["ic_w_full"])
    print("  ", paths["ic_v_hom"])
    print("  ", paths["ic_w_hom"])
    print("  ", paths["surface_cond"])

    with open(paths["r_path"], "rb") as f:
        rgrid_all = pickle.load(f)
    with open(paths["pm_path"], "rb") as f:
        pmsol_all = pickle.load(f)
    with open(paths["u00_path"], "rb") as f:
        u00_all = pickle.load(f)

    with open(paths["eq_v_full"], "r", encoding="utf-8") as f:
        v_eq_full_text = f.read()
    with open(paths["eq_w_full"], "r", encoding="utf-8") as f:
        w_eq_full_text = f.read()
    with open(paths["eq_v_hom"], "r", encoding="utf-8") as f:
        v_eq_hom_text = f.read()
    with open(paths["eq_w_hom"], "r", encoding="utf-8") as f:
        w_eq_hom_text = f.read()
    with open(paths["surface_cond"], "r", encoding="utf-8") as f:
        surface_cond_text = f.read()

    with open(paths["ic_v_full"], "r", encoding="utf-8") as f:
        v0_full_text = f.read()
    with open(paths["ic_w_full"], "r", encoding="utf-8") as f:
        w0_full_text = f.read()
    with open(paths["ic_v_hom"], "r", encoding="utf-8") as f:
        v0_hom_text = f.read()
    with open(paths["ic_w_hom"], "r", encoding="utf-8") as f:
        w0_hom_text = f.read()

    v0_full_fn, w0_full_fn = compile_r6_ic_functions(v0_full_text, w0_full_text)
    v0_hom_fn, w0_hom_fn = compile_r6_ic_functions(v0_hom_text, w0_hom_text)

    d_num_cfg = int(cfg["d_num"])
    n_models = min(d_num_cfg, len(rgrid_all), len(pmsol_all), len(u00_all))
    if args.num_models is not None:
        n_models = min(n_models, int(args.num_models))

    k_cgs = float(cfg["K_cgs"])
    pc_arr_cgs = np.logspace(33.5, 36.5, d_num_cfg)
    k_geom, pc_arr = cgs_to_geom(pc_arr_cgs, 1, k_cgs)[:2]
    eos = PolytropeEOS_n1(K=k_geom)

    full_grid = []
    hom_grid = []
    part_grid = []
    matched_grid = []
    records = []

    print("\n===== VW full/hom/particular solve starts =====\n")
    for nnn in tqdm(range(n_models), desc="VW full+hom", total=n_models):
        r_arr = np.asarray(rgrid_all[nnn], dtype=float)
        pm_arr = np.asarray(pmsol_all[nnn], dtype=float)
        p_arr = np.asarray(pm_arr[:, 1], dtype=float)
        m_arr = np.asarray(pm_arr[:, 0], dtype=float)

        rho_arr = eos.energy_density(p_arr)
        lamb_arr = np.log(r_arr / (r_arr - 2.0 * m_arr))
        edge_order = 2 if len(r_arr) >= 3 else 1
        drho_arr = np.gradient(rho_arr, r_arr, edge_order=edge_order)

        u_arr, up_arr = extract_u00_arrays(u00_all[nnn])
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

        coeffs_full = np.zeros((len(r_use), 6), dtype=float)
        coeffs_hom = np.zeros((len(r_use), 6), dtype=float)
        singular_count_full = 0

        for i in range(len(r_use)):
            row_full, b_w, denom_w = compute_full_coeffs(
                r=r_use[i],
                p=p_use[i],
                rho=rho_use[i],
                drho=drho_use[i],
                lamb=lamb_use[i],
                h00=h00_use[i],
                h00p=h00p_use[i],
            )
            if (not np.isfinite(b_w)) or (not np.isfinite(denom_w)) or abs(denom_w) < 1e-30:
                singular_count_full += 1
            coeffs_full[i] = row_full
            coeffs_hom[i] = compute_hom_coeffs(
                r=r_use[i],
                p=p_use[i],
                rho=rho_use[i],
                drho=drho_use[i],
                lamb=lamb_use[i],
            )

        eps0 = float(r_use[0])
        pc = float(pc_arr[nnn])

        y0_full = np.array(
            [
                float(v0_full_fn(eps0, k_geom, pc, args.a0_full, args.v0_full)),
                float(w0_full_fn(eps0, k_geom, pc, args.a0_full, args.v0_full)),
            ],
            dtype=float,
        )
        y0_hom = np.array(
            [
                float(v0_hom_fn(eps0, k_geom, pc, args.a0_hom, args.v0_hom)),
                float(w0_hom_fn(eps0, k_geom, pc, args.a0_hom, args.v0_hom)),
            ],
            dtype=float,
        )

        sol_full = integrate_vw_on_grid(y0_full, r_use, coeffs_full)
        sol_hom = integrate_vw_on_grid(y0_hom, r_use, coeffs_hom)
        sol_part = sol_full - sol_hom

        # Surface values (surface_index uses "shrink 4"; last interior point is [-1] of r_use)
        i_surf = len(r_use) - 1
        r_surf = float(r_use[i_surf])
        p_surf = float(p_use[i_surf])
        rho_surf = float(rho_use[i_surf])
        lamb_surf = float(lamb_use[i_surf])
        h_surf = float(h00_use[i_surf])
        hp_surf = float(h00p_use[i_surf])
        m_surf = float(m_arr[surface_index - 1])

        v_h = float(sol_hom[i_surf, 0])
        w_h = float(sol_hom[i_surf, 1])
        v_p = float(sol_part[i_surf, 0])
        w_p = float(sol_part[i_surf, 1])
        v_f = float(sol_full[i_surf, 0])
        w_f = float(sol_full[i_surf, 1])

        d_v_f, d_w_f, d0_f = float(coeffs_full[i_surf, 3]), float(coeffs_full[i_surf, 4]), float(coeffs_full[i_surf, 5])
        d_v_h, d_w_h, d0_h = float(coeffs_hom[i_surf, 3]), float(coeffs_hom[i_surf, 4]), float(coeffs_hom[i_surf, 5])

        wp_full = d_v_f * v_f + d_w_f * w_f + d0_f
        wp_hom = d_v_h * v_h + d_w_h * w_h + d0_h
        wp_part = wp_full - wp_hom

        cond_const = float(
            eval_surface_condition_v0(r_surf, p_surf, rho_surf, lamb_surf, h_surf, hp_surf, 0.0, 0.0, 0.0)
        )
        cond_part = float(
            eval_surface_condition_v0(r_surf, p_surf, rho_surf, lamb_surf, h_surf, hp_surf, v_p, w_p, wp_part)
        )
        cond_hom = float(
            eval_surface_condition_v0(r_surf, p_surf, rho_surf, lamb_surf, h_surf, hp_surf, v_h, w_h, wp_hom)
        )
        cond_full = float(
            eval_surface_condition_v0(r_surf, p_surf, rho_surf, lamb_surf, h_surf, hp_surf, v_f, w_f, wp_full)
        )

        lin_hom = cond_hom - cond_const
        if (not np.isfinite(lin_hom)) or abs(lin_hom) < float(args.denom_threshold):
            c_scale = np.nan
            status = "singular_c_denom"
            cond_matched = np.nan
            sol_matched = np.full_like(sol_part, np.nan)
        else:
            c_scale = -cond_part / lin_hom
            status = "ok"
            sol_matched = sol_part + c_scale * sol_hom
            v_m = float(sol_matched[i_surf, 0])
            w_m = float(sol_matched[i_surf, 1])
            wp_matched = wp_part + c_scale * wp_hom
            cond_matched = float(
                eval_surface_condition_v0(r_surf, p_surf, rho_surf, lamb_surf, h_surf, hp_surf, v_m, w_m, wp_matched)
            )

        full_grid.append(maybe_pad(sol_full, len(r_arr), surface_index, args.keep_full_length))
        hom_grid.append(maybe_pad(sol_hom, len(r_arr), surface_index, args.keep_full_length))
        part_grid.append(maybe_pad(sol_part, len(r_arr), surface_index, args.keep_full_length))
        matched_grid.append(maybe_pad(sol_matched, len(r_arr), surface_index, args.keep_full_length))

        records.append(
            {
                "star_id": int(nnn),
                "status": status,
                "arr_len": int(arr_len),
                "surface_index": int(surface_index),
                "R_surf": r_surf,
                "M_surf": m_surf,
                "V_full_ic": float(y0_full[0]),
                "W_full_ic": float(y0_full[1]),
                "V_hom_ic": float(y0_hom[0]),
                "W_hom_ic": float(y0_hom[1]),
                "singular_count_full_coeff": int(singular_count_full),
                "C": float(c_scale),
                "lin_hom": float(lin_hom),
                "surface_condition_const": cond_const,
                "surface_condition_full": cond_full,
                "surface_condition_hom": cond_hom,
                "surface_condition_part": cond_part,
                "surface_condition_matched": float(cond_matched),
                "V_surf_full": v_f,
                "W_surf_full": w_f,
                "Wp_surf_full": float(wp_full),
                "V_surf_hom": v_h,
                "W_surf_hom": w_h,
                "Wp_surf_hom": float(wp_hom),
                "V_surf_part": v_p,
                "W_surf_part": w_p,
                "Wp_surf_part": float(wp_part),
            }
        )

    print("\nWriting outputs...")
    with open(paths["out_full"], "wb") as f:
        pickle.dump(full_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["out_hom"], "wb") as f:
        pickle.dump(hom_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["out_part"], "wb") as f:
        pickle.dump(part_grid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["out_matched"], "wb") as f:
        pickle.dump(matched_grid, f, protocol=pickle.HIGHEST_PROTOCOL)

    meta_payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "h": paths["h_tok"],
            "eps": paths["eps_tok"],
            "d_num_config": d_num_cfg,
            "num_models_processed": n_models,
            "K_cgs": float(k_cgs),
            "K_geom": float(k_geom),
            "a0_full": float(args.a0_full),
            "v0_full": float(args.v0_full),
            "a0_hom": float(args.a0_hom),
            "v0_hom": float(args.v0_hom),
            "keep_full_length": bool(args.keep_full_length),
            "surface_index_logic": "surface_index=max(min(len(r),len(p),len(m),len(rho),len(lambda),len(drho),len(h00),len(h00p))-4,2)",
            "inputs": {
                "RGrid": str(paths["r_path"]),
                "PMSolGrid": str(paths["pm_path"]),
                "U00grid": str(paths["u00_path"]),
                "v_eqn_full": str(paths["eq_v_full"]),
                "w_eqn_full": str(paths["eq_w_full"]),
                "v_eqn_hom": str(paths["eq_v_hom"]),
                "w_eqn_hom": str(paths["eq_w_hom"]),
                "v0_r6_full": str(paths["ic_v_full"]),
                "w0_r6_full": str(paths["ic_w_full"]),
                "v0_r6_hom": str(paths["ic_v_hom"]),
                "w0_r6_hom": str(paths["ic_w_hom"]),
                "surface_condition_v0": str(paths["surface_cond"]),
            },
            "expression_previews": {
                "v_eqn_full_head": v_eq_full_text[:220],
                "w_eqn_full_head": w_eq_full_text[:220],
                "v_eqn_hom_head": v_eq_hom_text[:220],
                "w_eqn_hom_head": w_eq_hom_text[:220],
                "surface_condition_v0_head": surface_cond_text[:220],
            },
            "notes": {
                "particular_definition": "VWpart = VWfull - VWhom",
                "matched_definition": "VWmatched = VWpart + C*VWhom, with C from surface_condition_v0(R_surf)=0",
                "surface_condition_handling": "C = -cond_part/(cond_hom-cond_const), cond_const from (V=W=Wp=0)",
            },
        },
        "records": records,
    }
    with open(paths["out_meta"], "wb") as f:
        pickle.dump(meta_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_sing = len(records) - n_ok
    print(f"Done. models={len(records)}, ok={n_ok}, singular_c_denom={n_sing}")
    print("Output:")
    print("  ", paths["out_full"])
    print("  ", paths["out_hom"])
    print("  ", paths["out_part"])
    print("  ", paths["out_matched"])
    print("  ", paths["out_meta"])


if __name__ == "__main__":
    main()
