"""
Concatenate analytical (r^7) and numerical source terms using an error-based
transition window.

Rules:
  - r_min: first radius where error reaches 1e-3 outside the noisy region
  - r_max: first radius where error reaches 1e-1 outside the noisy region
  - r < r_min: analytical
  - r > r_max: numerical
  - r_min..r_max: smooth C2 transition (quintic smoothstep)

Input:
  - SourceGrid_full_h{h}_eps{eps}.pkl (numerical source)
  - RGrid_h{h}_eps{eps}.pkl (radius grid)
  - NuGrid_h{h}_eps{eps}.pkl (nu grid for nu_c)

Output:
  - SourceGrid_concat_h{h}_eps{eps}.pkl
"""

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from solver_tov import cgs_to_geom
from source_aux import source_r7


def quintic_smooth_step(t: np.ndarray) -> np.ndarray:
    """C2 smoothstep with zero 1st/2nd derivatives at boundaries."""
    return (6.0 * t**5) - (15.0 * t**4) + (10.0 * t**3)


def find_transition_indices(err: np.ndarray, start_idx: int, thr_min: float, thr_max: float):
    """Find r_min and r_max indices based on error thresholds."""
    n = len(err)
    r_min = start_idx
    r_max = n - 1

    for i in range(start_idx, n):
        if err[i] >= thr_min:
            r_min = i
            break

    for i in range(r_min, n):
        if err[i] >= thr_max:
            r_max = i
            break

    return r_min, r_max


def main():
    config_path = Path(
        "/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/config/h_eps_config.pkl"
    )
    with open(config_path, "rb") as f:
        h_eps_dic = pickle.load(f)

    h = float(h_eps_dic["h"])
    eps = float(h_eps_dic["eps"])
    d_num = int(h_eps_dic["d_num"])
    K_cgs = float(h_eps_dic["K_cgs"])

    print(f"\n===== config loaded: h = {h}, eps = {eps}, d_num = {d_num}, K_cgs = {K_cgs} =====\n")

    grid_path = Path(
        "/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/numpy_grid"
    )
    target_dir = grid_path / f"h{h_eps_dic['h']}_eps{h_eps_dic['eps']}"

    r_path = target_dir / f"RGrid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    nu_path = target_dir / f"NuGrid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    src_path = target_dir / f"SourceGrid_full_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"

    if not r_path.exists():
        alt = target_dir / f"Rgrid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
        if alt.exists():
            r_path = alt

    print("Loading:")
    print("  ", r_path)
    print("  ", nu_path)
    print("  ", src_path)

    with open(r_path, "rb") as f:
        Rgrid = pickle.load(f)
    with open(nu_path, "rb") as f:
        NuGrid = pickle.load(f)
    with open(src_path, "rb") as f:
        SourceFull = pickle.load(f)

    PcArr_cgs = np.logspace(33.5, 36.5, d_num)
    K_geom, PcArr = cgs_to_geom(PcArr_cgs, 1, K_cgs)[:2]

    SourceConcat = []

    print("\n===== Source concatenation starts =====\n")

    for nnn in tqdm(range(d_num), desc="Source concat", total=d_num):
        RArr = np.array(Rgrid[nnn])
        nu = np.array(NuGrid[nnn])
        src_num = np.array(SourceFull[nnn])

        if np.iscomplexobj(src_num):
            src_num = src_num.imag

        # Align lengths: keep as long as possible, but consistent
        n = min(len(src_num), len(RArr) - 1, len(nu) - 1)
        r = RArr[:n]
        src_num = src_num[:n]
        nu = nu[:n]

        pc = PcArr[nnn]
        T = 6e-6
        src_r7 = source_r7(r, K_geom, pc, a0=1.0, nu_c=nu[0], T=T)

        # Relative error with softening to avoid division by tiny values
        soft_slice = src_num[int(0.1 * n) : int(0.2 * n)]
        softening = np.max(np.abs(soft_slice)) * 0.1 if len(soft_slice) else 0.0
        if not np.isfinite(softening) or softening == 0.0:
            softening = 1e-30

        err = np.abs(src_num - src_r7) / (np.abs(src_num) + softening)

        noisy_end = int(0.03 * n)
        r_min_idx, r_max_idx = find_transition_indices(err, noisy_end, 10**(-3.5), 10**(-2.5))

        # Guard against invalid ordering
        if r_max_idx <= r_min_idx:
            r_max_idx = min(r_min_idx + 1, n - 1)

        # Build smooth transition weights
        t = (r - r[r_min_idx]) / (r[r_max_idx] - r[r_min_idx])
        t = np.clip(t, 0.0, 1.0)
        t_smooth = quintic_smooth_step(t)

        src_concat = src_r7 * (1.0 - t_smooth) + src_num * t_smooth
        SourceConcat.append(src_concat)

    out_concat = target_dir / f"SourceGrid_concat_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"

    print("\nWriting concatenated source grid...")
    with open(out_concat, "wb") as f:
        pickle.dump(SourceConcat, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    main()
