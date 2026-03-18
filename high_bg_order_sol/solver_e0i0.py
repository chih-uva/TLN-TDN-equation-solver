"""
Compute static exterior coefficients (E0, I0) = (E^{(0)}_{2m}, I^{+(0)}_{2m}) for l=2
from saved U00 grid outputs.

Surface index logic is intentionally kept identical to solver_h01.py:
  arr_len = min(len(r), len(p), len(m), len(rho), len(lambda), len(drho), len(source))
  surface_index = max(arr_len - 4, 1)

Inputs:
  - config/h_eps_config.pkl
  - numpy_grid/h{h}_eps{eps}/U00grid_h{h}_eps{eps}.pkl
  - numpy_grid/h{h}_eps{eps}/PMSolGrid_h{h}_eps{eps}.pkl
  - numpy_grid/h{h}_eps{eps}/RGrid_h{h}_eps{eps}.pkl (or Rgrid_...)
  - numpy_grid/h{h}_eps{eps}/SourceGrid_concat_h{h}_eps{eps}.pkl

Output:
  - numpy_grid/h{h}_eps{eps}/E0I0_from_u00_h{h}_eps{eps}.pkl
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
        "src_path": target_dir / f"SourceGrid_concat_h{h_tok}_eps{eps_tok}.pkl",
        "out_path": target_dir / f"E0I0_from_u00_h{h_tok}_eps{eps_tok}.pkl",
        "h_tok": h_tok,
        "eps_tok": eps_tok,
    }


def extract_u00_arrays(u00_model):
    """Return u00, u00p arrays from one model entry."""
    if isinstance(u00_model, dict):
        if "u00_grid" in u00_model and "u00p_grid" in u00_model:
            return np.asarray(u00_model["u00_grid"], dtype=float), np.asarray(u00_model["u00p_grid"], dtype=float)
        if "u00" in u00_model and "u00p" in u00_model:
            return np.asarray(u00_model["u00"], dtype=float), np.asarray(u00_model["u00p"], dtype=float)

    arr = np.asarray(u00_model)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return np.asarray(arr[:, 0], dtype=float), np.asarray(arr[:, 1], dtype=float)

    raise ValueError("Unsupported U00 model format; expected Nx2 array or dict with u00/u00p arrays")


def u_to_h_at_r(u, up, r):
    h = (r * r) * u
    hp = (r * r) * up + 2.0 * r * u
    return h, hp


def compute_surface_index_h01_style(r_arr, p_arr, m_arr, source_arr, eos):
    """Exact same index logic used by solver_h01.py."""
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
        len(source_arr),
    )
    surface_index = max(arr_len - 4, 1)
    return surface_index, arr_len

# Polynomial definition
def H2R_l2(r, m_tot):
    z = 2.0 * m_tot / r
    f = 1.0 - z
    return f * (m_tot / r) ** 3 * hyp2f1(3, 5, 6, z)

# C4 definition: H2T = f * (r^2 / M^2)
def H2T_l2(r, m_tot):
    f = 1.0 - 2.0 * m_tot / r
    return f * (r / m_tot) ** 2


def relative_error(a, b, floor=1e-30):
    return abs(a - b) / max(abs(b), floor)


def main():
    parser = argparse.ArgumentParser(description="Compute E0/I0 from U00 grids with h01-style surface index.")
    parser.add_argument("--num-models", type=int, default=None, help="Optional limit on number of stars")
    parser.add_argument("--det-threshold", type=float, default=1e-14, help="Warn when |det| below this value")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    cfg = load_config(base_dir / "config" / "h_eps_config.pkl")
    paths = resolve_paths(base_dir, cfg)

    print("Loading:")
    print("  ", paths["r_path"])
    print("  ", paths["pm_path"])
    print("  ", paths["u00_path"])
    print("  ", paths["src_path"])

    with open(paths["r_path"], "rb") as f:
        rgrid_all = pickle.load(f)
    with open(paths["pm_path"], "rb") as f:
        pmsol_all = pickle.load(f)
    with open(paths["u00_path"], "rb") as f:
        u00_all = pickle.load(f)
    with open(paths["src_path"], "rb") as f:
        source_all = pickle.load(f)

    d_num_cfg = int(cfg["d_num"])
    n_models = min(d_num_cfg, len(rgrid_all), len(pmsol_all), len(u00_all), len(source_all))
    if args.num_models is not None:
        n_models = min(n_models, int(args.num_models))

    k_cgs = float(cfg["K_cgs"])
    k_geom = cgs_to_geom(np.array([1e35]), 1, k_cgs)[0]
    eos = PolytropeEOS_n1(K=k_geom)

    records = []
    tiny_det_count = 0

    print("\n===== E0/I0 matching from U00 starts =====\n")

    for i in tqdm(range(n_models), desc="E0I0", total=n_models):
        r_arr = np.asarray(rgrid_all[i], dtype=float)
        pm_arr = np.asarray(pmsol_all[i], dtype=float)
        p_arr = np.asarray(pm_arr[:, 1], dtype=float)
        m_arr = np.asarray(pm_arr[:, 0], dtype=float)

        u00_arr, u00p_arr = extract_u00_arrays(u00_all[i])
        source_arr = np.asarray(source_all[i], dtype=float)

        sidx, arr_len = compute_surface_index_h01_style(r_arr, p_arr, m_arr, source_arr, eos)

        R = float(r_arr[sidx])
        M_tot = float(m_arr[sidx])
        uR = float(u00_arr[sidx])
        uRp = float(u00p_arr[sidx])
        hR, hRp = u_to_h_at_r(uR, uRp, R)

        # One-sided finite difference just outside the star, with constant M_tot
        h_step_near_surface = float(abs(r_arr[sidx] - r_arr[sidx - 1])) if sidx > 0 else float(abs(r_arr[1] - r_arr[0]))
        delta = max(5.0 * h_step_near_surface, 1e-6 * R)

        HT = float(H2T_l2(R, M_tot))
        HR = float(H2R_l2(R, M_tot))
        HTp = float((H2T_l2(R + delta, M_tot) - HT) / delta)
        HRp = float((H2R_l2(R + delta, M_tot) - HR) / delta)

        A = np.array([[HT, HR], [HTp, HRp]], dtype=float)
        b = np.array([hR, hRp], dtype=float)

        det = float(HT * HRp - HTp * HR)
        tiny_det = abs(det) < args.det_threshold
        if tiny_det:
            tiny_det_count += 1

        try:
            E0, I0 = np.linalg.solve(A, b)
            solve_status = "ok"
        except np.linalg.LinAlgError:
            E0, I0 = np.nan, np.nan
            solve_status = "singular"

        hR_rec = float(E0 * HT + I0 * HR) if np.isfinite(E0) and np.isfinite(I0) else np.nan
        hRp_rec = float(E0 * HTp + I0 * HRp) if np.isfinite(E0) and np.isfinite(I0) else np.nan

        rel_err_hR = relative_error(hR_rec, hR) if np.isfinite(hR_rec) else np.nan
        rel_err_hRp = relative_error(hRp_rec, hRp) if np.isfinite(hRp_rec) else np.nan

        r_test = 10.0 * R
        HT_test = float(H2T_l2(r_test, M_tot))
        HR_test = float(H2R_l2(r_test, M_tot))

        # Asymptotic scaling indicators
        tidal_scaled = HT_test / (r_test * r_test)
        response_scaled = HR_test * (r_test**3)

        records.append(
            {
                "star_id": int(i),
                "status": solve_status,
                "tiny_det": bool(tiny_det),
                "det": det,
                "arr_len": int(arr_len),
                "surface_index": int(sidx),
                "R": R,
                "M_tot": M_tot,
                "delta": float(delta),
                "h_step_near_surface": float(h_step_near_surface),
                "hR": hR,
                "hRp": hRp,
                "HT": HT,
                "HR": HR,
                "HTp": HTp,
                "HRp": HRp,
                "E0": float(E0),
                "I0": float(I0),
                "hR_rec": hR_rec,
                "hRp_rec": hRp_rec,
                "rel_err_hR": float(rel_err_hR),
                "rel_err_hRp": float(rel_err_hRp),
                "r_test": float(r_test),
                "H2T_r_test": HT_test,
                "H2R_r_test": HR_test,
                "tidal_scaled_H2T_over_r2": float(tidal_scaled),
                "response_scaled_H2R_times_r3": float(response_scaled),
            }
        )

    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "h": paths["h_tok"],
            "eps": paths["eps_tok"],
            "d_num_config": d_num_cfg,
            "num_models_processed": n_models,
            "det_threshold": float(args.det_threshold),
            "surface_index_logic": "same as solver_h01: surface_index=max(min(len(r),len(p),len(m),len(rho),len(lambda),len(drho),len(source))-4,1)",
            "inputs": {
                "RGrid": str(paths["r_path"]),
                "PMSolGrid": str(paths["pm_path"]),
                "U00grid": str(paths["u00_path"]),
                "SourceGrid_concat": str(paths["src_path"]),
            },
            "basis_definitions": {
                "H2R": "f*(M/r)^3*hyp2f1(3,5,6,2M/r), f=1-2M/r",
                "H2T": "C4 definition: f*(r/M)^2 (effective hypergeometric factor set to 1)",
                "derivatives": "one-sided finite diff outside star with constant M_tot",
            },
            "tiny_det_count": tiny_det_count,
        },
        "records": records,
    }

    with open(paths["out_path"], "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"Done. solved={n_ok}/{len(records)}, tiny_det={tiny_det_count}")
    print("Output:", paths["out_path"])


if __name__ == "__main__":
    main()
