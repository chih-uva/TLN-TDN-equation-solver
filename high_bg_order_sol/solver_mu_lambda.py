"""
Compute NuGrid and LambGrid from precomputed TOV backgrounds.

Nu is matched at the stellar surface:
  nu_R = log(1 - 2 M(R)/R), then shift the integrated nu(r) so nu(R) = nu_R.

Outputs:
  - NuGrid_h{h}_eps{eps}.pkl
  - LambGrid_h{h}_eps{eps}.pkl
"""

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from solver_tov import nu_rhs, rk4_step_nu


def compute_nu_matched(RArr, PArr, MArr):
    """
    Integrate nu' and match the surface value as in low_bg_order_sol/solver-h00_old.ipynb.
    """
    nu_raw = np.zeros(len(RArr), dtype=float)
    for i, r in enumerate(RArr[:-1]):
        dr = RArr[i + 1] - RArr[i]
        nu_raw[i + 1] = rk4_step_nu(nu_rhs, r, nu_raw[i], dr, PArr[i], MArr[i])

    # Surface matching using the -2 index (consistent with notebook)
    nu_R = np.log(1.0 - 2.0 * MArr[-2] / RArr[-2])
    nu = nu_raw + (nu_R - nu_raw[-2])
    return nu


def main():
    config_path = Path("/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/config/h_eps_config.pkl")
    with open(config_path, "rb") as f:
        h_eps_dic = pickle.load(f)

    h = str(h_eps_dic["h"])
    eps = str(h_eps_dic["eps"])

    grid_path = Path("/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/numpy_grid")
    target_dir = grid_path / f"h{h}_eps{eps}"

    r_path = target_dir / f"RGrid_h{h}_eps{eps}.pkl"
    pm_path = target_dir / f"PMSolGrid_h{h}_eps{eps}.pkl"

    if not r_path.exists():
        alt = target_dir / f"Rgrid_h{h}_eps{eps}.pkl"
        if alt.exists():
            r_path = alt

    print("Loading:")
    print("  ", r_path)
    print("  ", pm_path)

    with open(r_path, "rb") as f:
        Rgrid = pickle.load(f)
    with open(pm_path, "rb") as f:
        PMSol = pickle.load(f)

    NuGrid = []
    LambGrid = []

    print("\n===== Computing Nu/Lambda grids =====\n")
    for nnn in tqdm(range(len(Rgrid)), desc="Nu/Lambda", total=len(Rgrid)):
        PMSolArr = np.array(PMSol[nnn])
        RArr = np.array(Rgrid[nnn])
        PArr = PMSolArr[:, 1]
        MArr = PMSolArr[:, 0]

        LambArr = np.log(RArr / (RArr - 2.0 * MArr))
        NuArr = compute_nu_matched(RArr, PArr, MArr)

        LambGrid.append(LambArr)
        NuGrid.append(NuArr)

    out_nu = target_dir / f"NuGrid_h{h}_eps{eps}.pkl"
    out_lamb = target_dir / f"LambGrid_h{h}_eps{eps}.pkl"

    print("\nWriting outputs...")
    with open(out_nu, "wb") as f:
        pickle.dump(NuGrid, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out_lamb, "wb") as f:
        pickle.dump(LambGrid, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    main()
