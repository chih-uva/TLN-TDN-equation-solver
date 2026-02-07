"""
Compute the L=2 source term using precomputed background and H00 grids.

Inputs:
  - RGrid_h{h}_eps{eps}.pkl
  - PMSolGrid_h{h}_eps{eps}.pkl
  - UGrid_h{h}_eps{eps}.pkl

Outputs:
  - SourceGrid_full_h{h}_eps{eps}.pkl
  - SourceGrid_r3_h{h}_eps{eps}.pkl
  - SourceGrid_r5_h{h}_eps{eps}.pkl
  - SourceGrid_r7_h{h}_eps{eps}.pkl
"""

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from EOS_aux import PolytropeEOS_n1
from solver_tov import cgs_to_geom
from source_aux import source_full, source_r3, source_r5, source_r7


def main():
    config_path = Path("/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/config/h_eps_config.pkl")
    with open(config_path, "rb") as f:
        h_eps_dic = pickle.load(f)

    h = float(h_eps_dic["h"])
    eps = float(h_eps_dic["eps"])
    d_num = int(h_eps_dic["d_num"])
    K_cgs = float(h_eps_dic["K_cgs"])

    print(f"\n===== config loaded: h = {h}, eps = {eps}, d_num = {d_num}, K_cgs = {K_cgs} =====\n")

    grid_path = Path("/Users/sztk.ch/Work/grad-school/Research/TLN-TDN Project/Programs/Equation Solver/high_bg_order_sol/numpy_grid")
    target_dir = grid_path / f"h{h_eps_dic['h']}_eps{h_eps_dic['eps']}"

    r_path = target_dir / f"RGrid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    pm_path = target_dir / f"PMSolGrid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    u_path = target_dir / f"U00Grid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    nu_path = target_dir / f"NuGrid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"

    if not r_path.exists():
        alt = target_dir / f"Rgrid_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
        if alt.exists():
            r_path = alt

    print("Loading:")
    print("  ", r_path)
    print("  ", pm_path)
    print("  ", u_path)
    print("  ", nu_path)

    with open(r_path, "rb") as f:
        Rgrid = pickle.load(f)
    with open(pm_path, "rb") as f:
        PMSol = pickle.load(f)
    with open(u_path, "rb") as f:
        UGrid = pickle.load(f)
    with open(nu_path, "rb") as f:
        NuGrid = pickle.load(f)

    PcArr_cgs = np.logspace(33.5, 36.5, d_num)
    K_geom, PcArr = cgs_to_geom(PcArr_cgs, 1, K_cgs)[:2]

    n1polytrope = PolytropeEOS_n1(K_geom)

    SourceFull = []
    # Analytic series sources are not saved in this solver (computed on-demand in notebooks)
    # SourceR3 = []
    # SourceR5 = []
    # SourceR7 = []

    print("\n===== Source computation starts =====\n")

    for nnn in tqdm(range(d_num), desc="Source solver", total=d_num):
        PMSolArr = np.array(PMSol[nnn])
        RArr = np.array(Rgrid[nnn])
        PArr = PMSolArr[:, 1]
        MArr = PMSolArr[:, 0]

        # align with U grid (length len(RArr)-1)
        r = RArr[:-1]
        P = PArr[:-1]
        M = MArr[:-1]

        u_num = np.array(UGrid[nnn])[:, 0]
        du_num = np.array(UGrid[nnn])[:, 1]

        # Align lengths defensively (UGrid length may differ by 1)
        n = min(len(r), len(u_num))
        r = r[:n]
        P = P[:n]
        M = M[:n]
        u_num = u_num[:n]
        du_num = du_num[:n]

        h00 = u_num * (r**2)
        dh00 = du_num * (r**2) + 2.0 * u_num * r

        rho = n1polytrope.energy_density(P)
        drho = np.gradient(rho, r, edge_order=2)
        d2rho = np.gradient(drho, r, edge_order=2)

        lamb = np.log(r / (r - 2.0 * M))
        nu = np.array(NuGrid[nnn])[:-1]

        # Match solver-source_old.ipynb: enforce a common length and trim by 4
        arr_len = min(
            len(r),
            len(P),
            len(M),
            len(rho),
            len(lamb),
            len(nu),
            len(drho),
            len(d2rho),
            len(h00),
            len(dh00),
        )
        arr_len = max(arr_len - 4, 0)

        r = r[:arr_len]
        P = P[:arr_len]
        M = M[:arr_len]
        rho = rho[:arr_len]
        drho = drho[:arr_len]
        d2rho = d2rho[:arr_len]
        h00 = h00[:arr_len]
        dh00 = dh00[:arr_len]
        lamb = lamb[:arr_len]
        nu = nu[:arr_len]

        # eta, deta, d2eta follow solver-source_old.ipynb convention
        T = 6e-6
        eta = T * rho**2
        deta = 2.0 * T * rho
        d2eta = 2.0 * T

        src_full = source_full(r, P, rho, drho, d2rho, h00, dh00, lamb, nu, eta, deta, d2eta)
        SourceFull.append(src_full)

        # Analytic series sources are intentionally skipped here.
        # pc = PcArr[nnn]
        # src_r3 = source_r3(r, K_geom, pc, a0=1.0, nu_c=0.0, T=1.0)
        # src_r5 = source_r5(r, K_geom, pc, a0=1.0, nu_c=0.0, T=1.0)
        # src_r7 = source_r7(r, K_geom, pc, a0=1.0, nu_c=0.0, T=1.0)
        # SourceR3.append(src_r3)
        # SourceR5.append(src_r5)
        # SourceR7.append(src_r7)

    # Save
    out_full = target_dir / f"SourceGrid_full_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    # out_r3 = target_dir / f"SourceGrid_r3_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    # out_r5 = target_dir / f"SourceGrid_r5_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"
    # out_r7 = target_dir / f"SourceGrid_r7_h{h_eps_dic['h']}_eps{h_eps_dic['eps']}.pkl"

    print("\nWriting source grids...")
    with open(out_full, "wb") as f:
        pickle.dump(SourceFull, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(out_r3, "wb") as f:
    #     pickle.dump(SourceR3, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(out_r5, "wb") as f:
    #     pickle.dump(SourceR5, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(out_r7, "wb") as f:
    #     pickle.dump(SourceR7, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


if __name__ == "__main__":
    main()
