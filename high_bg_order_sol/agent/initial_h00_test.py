import math
import sys
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt

# Ensure project modules are importable when running from this file's directory.
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_DIR = _THIS_DIR.parent
if str(_PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJ_DIR))

from tov_series_aux import p_approx, m_approx
from h00_aux import h00_approx, h00p_approx


# -----------------------------
# Physical constants (cgs)
# -----------------------------
G_CGS = 6.67430e-8     # cm^3 g^-1 s^-2
C_CGS = 2.99792458e10  # cm s^-1
CM2KM = 1.0e5
TINY = 1.0e-30


# -----------------------------
# EOS and unit conversion
# -----------------------------

def energy_eos(p, K):
    # rho(p) for n=1 polytrope: rho = ((1+sqrt(4Kp))^2 - 1)/(4K)
    p = np.maximum(p, 0.0)
    s = np.sqrt(4.0 * K * p)
    return ((1.0 + s) ** 2 - 1.0) / (4.0 * K)


def drho_dp(p, K):
    # derivative of rho wrt p for the same EOS
    p = np.maximum(p, 0.0)
    s = np.sqrt(4.0 * K * p)
    s = np.maximum(s, 1.0e-30)
    return 1.0 + 1.0 / s


def cgs_to_geom(p_cgs, n, K_cgs):
    # p_geom in km^-2
    p_geom = p_cgs * G_CGS / (C_CGS**4) * (CM2KM**2)
    # K_geom in km^(2/n)
    K_geom = K_cgs / (G_CGS ** (1.0 / n)) / (C_CGS ** (2.0 - 2.0 / n)) / (CM2KM ** (2.0 / n))
    # rho_geom in km^-2
    rho_geom = energy_eos(p_geom, K_geom)
    return K_geom, p_geom, rho_geom


# -----------------------------
# ODE system: TOV + h00
# -----------------------------

def tov_rhs(r, m, p, K):
    rho = energy_eos(p, K)
    dm_dr = 4.0 * math.pi * r**2 * rho
    dp_dr = - (rho + p) * (m + 4.0 * math.pi * r**3 * p) / (r * (r - 2.0 * m))
    return dm_dr, dp_dr


def h00_rhs(r, h00, h00p, p, m, K):
    if p <= 0.0:
        return 0.0, 0.0
    rho = energy_eos(p, K)
    dm_dr, dp_dr = tov_rhs(r, m, p, K)
    drho = drho_dp(p, K) * dp_dr

    lamb = math.log(r / (r - 2.0 * m))
    eL = math.exp(lamb)
    e2L = eL * eL
    e3L = e2L * eL

    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r6 = r3 * r3

    A = (
        -1.0 - 3.0 * eL + 3.0 * e2L + e3L
        + 96.0 * e2L * (-5.0 + 2.0 * eL) * (math.pi**2) * r4 * (p**2)
        + 512.0 * e3L * (math.pi**3) * r6 * (p**3)
        - 20.0 * eL * (-1.0 + eL) * math.pi * r2 * rho
        + 4.0 * eL * math.pi * r2 * p * (15.0 - 9.0 * eL + 6.0 * e2L - 40.0 * eL * math.pi * r2 * rho)
        + 8.0 * eL * math.pi * r3 * drho
    )

    D = r2 * (-1.0 + eL + 8.0 * eL * math.pi * r2 * p)
    if D == 0.0 or not math.isfinite(D):
        raise ZeroDivisionError(f"Singular denominator at r={r}")

    hp_coeff = (1.0 + eL + 4.0 * eL * math.pi * r2 * p - 4.0 * eL * math.pi * r2 * rho) / r

    h00pp = (A / D) * h00 - hp_coeff * h00p
    return h00p, h00pp


def rk4_step(r, y, h, K):
    # y = [m, p, h00, h00p]
    def f(r_local, y_local):
        m, p, h00, h00p = y_local
        dm, dp = tov_rhs(r_local, m, p, K)
        dh00, dh00p = h00_rhs(r_local, h00, h00p, p, m, K)
        return np.array([dm, dp, dh00, dh00p], dtype=float)

    k1 = f(r, y)
    k2 = f(r + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(r + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(r + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# -----------------------------
# Experiment settings
# -----------------------------

K_CGS = 1.6e5
N_POLY = 1

# central-pressure list (existing grid)
PC_ALL = np.logspace(33.5, 36.5, 25)
PC_IDX = np.linspace(0, len(PC_ALL) - 1, 5, dtype=int)
PC_LIST = PC_ALL[PC_IDX]

# sweep parameters
H_LIST = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
R0_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


def robust_std(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.nan
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return 1.4826 * mad


def integrate_star(pc_cgs, h, r0):
    K_geom, pc_geom, _ = cgs_to_geom(pc_cgs, N_POLY, K_CGS)

    p0 = p_approx(r0, K_geom, pc_geom)
    m0 = m_approx(r0, K_geom, pc_geom)

    h00_0 = h00_approx(r0, K_geom, pc_geom, a0=1.0, a8=0.0)
    h00p_0 = h00p_approx(r0, K_geom, pc_geom, a0=1.0, a8=0.0)

    y = np.array([m0, p0, h00_0, h00p_0], dtype=float)

    # integrate to surface (p -> 0) with a safety cap
    r_max_cap = 1
    n_steps = max(3, int(math.ceil((r_max_cap - r0) / h)))

    r_arr = np.zeros(n_steps + 1)
    h00_arr = np.zeros(n_steps + 1)
    ratio_arr = np.zeros(n_steps + 1)

    r = r0
    r_arr[0] = r
    h00_arr[0] = y[2]
    ratio_arr[0] = y[2] / (r * r)

    for i in range(1, n_steps + 1):
        if y[1] <= 0.0 or not np.isfinite(y[1]) or not np.isfinite(y[0]):
            r_arr = r_arr[:i]
            h00_arr = h00_arr[:i]
            ratio_arr = ratio_arr[:i]
            break
        if r - 2.0 * y[0] <= 0.0:
            r_arr = r_arr[:i]
            h00_arr = h00_arr[:i]
            ratio_arr = ratio_arr[:i]
            break
        y = rk4_step(r, y, h, K_geom)
        r = r + h
        r_arr[i] = r
        h00_arr[i] = y[2]
        ratio_arr[i] = y[2] / (r * r)

        if y[1] <= 0.0 or not np.isfinite(y[2]) or not np.isfinite(y[3]):
            r_arr = r_arr[: i + 1]
            h00_arr = h00_arr[: i + 1]
            ratio_arr = ratio_arr[: i + 1]
            break

    return r_arr, h00_arr, ratio_arr, (K_geom, pc_geom)


def analytic_y(r, K_geom, pc_geom, a0=1.0, a8=0.0):
    h = h00_approx(r, K_geom, pc_geom, a0=a0, a8=a8)
    return h / (r * r)


def plot_for_pc(pc, rows, raw, output_dir):
    pc_rows = [r for r in rows if r["Pc_cgs"] == pc]
    if not pc_rows:
        return

    # Figure 1: small-r overlay in 2x3 grid (one subplot per r0)
    fig1, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160, sharey=True)
    axes = axes.reshape(-1)
    for i, r0 in enumerate(R0_LIST):
        ax = axes[i]
        for h in H_LIST:
            key = f"Pc{pc:.3e}_h{h:.0e}_r0{r0:.0e}"
            r = raw[f"{key}_r"]
            y = raw[f"{key}_y"]
            ax.plot(r, y, label=f"h={h:.0e}")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(f"r0={r0:.0e}")
        ax.grid(alpha=0.2)
        if i % 3 == 0:
            ax.set_ylabel("y=h00/r^2")
        ax.set_xlabel("r (km)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    fig1.suptitle(f"Pc={pc:.3e} small-r y(r)")
    fig1.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig1.savefig(output_dir / f"Pc_{pc:.3e}_y_smallr_grid.png")
    plt.close(fig1)

    # Figure 2: small-r relative deviation
    fig2, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160, sharey=True)
    axes = axes.reshape(-1)
    for i, r0 in enumerate(R0_LIST):
        ax = axes[i]
        for h in H_LIST:
            key = f"Pc{pc:.3e}_h{h:.0e}_r0{r0:.0e}"
            r = raw[f"{key}_r"]
            y = raw[f"{key}_y"]
            y_ana = raw[f"{key}_y_ana"]
            delta = np.abs(y - y_ana) / np.maximum(np.abs(y_ana), TINY)
            ax.plot(r, delta, label=f"h={h:.0e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"r0={r0:.0e}")
        ax.grid(alpha=0.2)
        if i % 3 == 0:
            ax.set_ylabel("rel err")
        ax.set_xlabel("r (km)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    fig2.suptitle(f"Pc={pc:.3e} small-r rel err")
    fig2.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig2.savefig(output_dir / f"Pc_{pc:.3e}_relerr_smallr_grid.png")
    plt.close(fig2)

    # Figure 3: step-size convergence at fixed r0 (ref = h_min)
    fig3, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160, sharey=True)
    axes = axes.reshape(-1)
    h_ref = min(H_LIST)
    for i, r0 in enumerate(R0_LIST):
        ax = axes[i]
        key_ref = f"Pc{pc:.3e}_h{h_ref:.0e}_r0{r0:.0e}"
        r_ref = raw[f"{key_ref}_r"]
        y_ref = raw[f"{key_ref}_y"]

        r_min = max(r_ref[0], r0)
        r_max = min(r_ref[-1], 1.0e-1)
        if r_max <= r_min:
            continue
        r_grid = np.logspace(np.log10(r_min), np.log10(r_max), 200)
        y_ref_i = np.interp(r_grid, r_ref, y_ref)

        for h in H_LIST:
            if h == h_ref:
                continue
            key = f"Pc{pc:.3e}_h{h:.0e}_r0{r0:.0e}"
            r = raw[f"{key}_r"]
            y = raw[f"{key}_y"]
            y_i = np.interp(r_grid, r, y)
            diff = np.abs(y_i - y_ref_i) / np.maximum(np.abs(y_ref_i), TINY)
            ax.plot(r_grid, diff, label=f"h={h:.0e}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"r0={r0:.0e}")
        ax.grid(alpha=0.2)
        if i % 3 == 0:
            ax.set_ylabel("rel diff vs h_min")
        ax.set_xlabel("r (km)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
    fig3.suptitle(f"Pc={pc:.3e} convergence vs h_min={h_ref:.0e}")
    fig3.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig3.savefig(output_dir / f"Pc_{pc:.3e}_convergence_by_h.png")
    plt.close(fig3)

    # Figure 4: heatmap over (h, r0) for rel_err_a0 and spike_amp
    for metric in ("rel_err_a0", "spike_amp"):
        z = np.zeros((len(R0_LIST), len(H_LIST)))
        for i, r0 in enumerate(R0_LIST):
            for j, h in enumerate(H_LIST):
                for row in pc_rows:
                    if row["r0"] == r0 and row["h"] == h:
                        z[i, j] = row[metric]
                        break
        fig4, ax = plt.subplots(figsize=(6, 4.5), dpi=160)
        im = ax.imshow(z, origin="lower", aspect="auto")
        ax.set_xticks(range(len(H_LIST)))
        ax.set_yticks(range(len(R0_LIST)))
        ax.set_xticklabels([f"{h:.0e}" for h in H_LIST])
        ax.set_yticklabels([f"{r0:.0e}" for r0 in R0_LIST])
        ax.set_xlabel("h")
        ax.set_ylabel("r0")
        ax.set_title(f"Pc={pc:.3e} {metric}")
        fig4.colorbar(im, ax=ax, shrink=0.85)
        fig4.tight_layout()
        fig4.savefig(output_dir / f"Pc_{pc:.3e}_heatmap_{metric}.png")
        plt.close(fig4)


def compute_convergence(rows, raw):
    for pc in PC_LIST:
        for r0 in R0_LIST:
            h_sorted = sorted(H_LIST, reverse=True)
            for i in range(len(h_sorted) - 1):
                h_i = h_sorted[i]
                h_j = h_sorted[i + 1]
                key_i = f"Pc{pc:.3e}_h{h_i:.0e}_r0{r0:.0e}"
                key_j = f"Pc{pc:.3e}_h{h_j:.0e}_r0{r0:.0e}"
                r_i = raw[f"{key_i}_r"]
                y_i = raw[f"{key_i}_y"]
                r_j = raw[f"{key_j}_r"]
                y_j = raw[f"{key_j}_y"]
                r_min = max(r_i[0], r_j[0], r0)
                r_max = min(r_i[-1], r_j[-1], 1.0e-1)
                if r_max <= r_min:
                    diff = np.nan
                else:
                    r_grid = np.logspace(np.log10(r_min), np.log10(r_max), 200)
                    y_i_i = np.interp(r_grid, r_i, y_i)
                    y_j_i = np.interp(r_grid, r_j, y_j)
                    diff = float(np.max(np.abs(y_i_i - y_j_i) / np.maximum(np.abs(y_j_i), TINY)))

                for row in rows:
                    if row["Pc_cgs"] == pc and row["r0"] == r0 and row["h"] == h_i:
                        row[f"diff_to_{h_j:.0e}"] = diff
    return rows


def report_summary(rows):
    print("Summary ranking by Pc (min rel_err_a0, min spike_amp):")
    for pc in PC_LIST:
        pc_rows = [r for r in rows if r["Pc_cgs"] == pc]
        if not pc_rows:
            continue
        best_rel = min(pc_rows, key=lambda r: r["rel_err_a0"])
        best_spk = min(pc_rows, key=lambda r: r["spike_amp"])
        print(f"Pc={pc:.3e}:")
        print(f"  best rel_err_a0: h={best_rel['h']:.0e}, r0={best_rel['r0']:.0e}, rel_err={best_rel['rel_err_a0']:.3e}")
        print(f"  best spike_amp: h={best_spk['h']:.0e}, r0={best_spk['r0']:.0e}, spike={best_spk['spike_amp']:.3e}")

        spikes = {}
        for r0 in R0_LIST:
            subset = [r for r in pc_rows if r["r0"] == r0]
            if subset:
                spikes[r0] = min(subset, key=lambda r: r["spike_amp"])["spike_amp"]
        if spikes:
            r0_sorted = sorted(spikes.keys())
            worsening = all(spikes[r0_sorted[i]] <= spikes[r0_sorted[i + 1]] for i in range(len(r0_sorted) - 1))
            if worsening:
                print("  note: decreasing r0 appears to worsen spike_amp (possible contamination).")


def run_sweep(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    npz_path = output_dir / "raw_arrays.npz"

    rows = []
    raw = {}

    total_runs = len(PC_LIST) * len(R0_LIST) * len(H_LIST)
    run_idx = 0
    t_start = time.perf_counter()

    for pc in PC_LIST:
        for r0 in R0_LIST:
            for h in H_LIST:
                run_idx += 1
                if run_idx == 1 or run_idx % 10 == 0 or run_idx == total_runs:
                    elapsed = time.perf_counter() - t_start
                    rate = run_idx / max(elapsed, 1e-9)
                    eta = (total_runs - run_idx) / max(rate, 1e-9)
                    print(
                        f"[{run_idx:4d}/{total_runs}] "
                        f"Pc={pc:.2e} h={h:.0e} r0={r0:.0e} "
                        f"elapsed={elapsed:6.1f}s ETA={eta:6.1f}s"
                    )

                r_arr, h00_arr, y_arr, (K_geom, pc_geom) = integrate_star(pc, h, r0)

                r_win = min(50.0 * r0, 1.0e-2)
                win_mask = (r_arr >= r0) & (r_arr <= r_win)
                y_win = y_arr[win_mask]

                a0_est = np.median(y_win) if y_win.size else np.nan
                a0_std = robust_std(y_win)

                a0_ana = 1.0
                rel_err_a0 = abs(a0_est - a0_ana) / max(abs(a0_ana), TINY)

                if y_win.size:
                    spike_amp = float(np.max(y_win) - np.min(y_win))
                    dev = np.abs(y_win - a0_est)
                    spike_loc = float(r_arr[win_mask][np.argmax(dev)])
                else:
                    spike_amp = np.nan
                    spike_loc = np.nan

                key = f"Pc{pc:.3e}_h{h:.0e}_r0{r0:.0e}"
                raw[f"{key}_r"] = r_arr
                raw[f"{key}_h00"] = h00_arr
                raw[f"{key}_y"] = y_arr
                raw[f"{key}_y_ana"] = analytic_y(r_arr, K_geom, pc_geom, a0=1.0, a8=0.0)

                rows.append(
                    dict(
                        Pc_cgs=pc,
                        h=h,
                        r0=r0,
                        r_win=r_win,
                        a0_est=a0_est,
                        a0_std=a0_std,
                        a0_ana=a0_ana,
                        rel_err_a0=rel_err_a0,
                        spike_amp=spike_amp,
                        spike_loc=spike_loc,
                        n_points=len(r_arr),
                    )
                )

        plot_for_pc(pc, rows, raw, output_dir)

    rows = compute_convergence(rows, raw)

    header = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in header) + "\n")

    np.savez_compressed(npz_path, **raw)

    report_summary(rows)
    total_time = time.perf_counter() - t_start
    print(f"Total runtime: {total_time:.1f} s")

# def run_sweep_c(output_dir):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     csv_path = output_dir / "summary.csv"
#     npz_path = output_dir / "raw_arrays.npz"

#     rows = []
#     raw = {}
    
#     total_runs = len(PC_LIST) * len(R0_LIST) * len(H_LIST)
#     run_idx = 0
#     t_start = time.perf_counter()

#     for pc in PC_LIST:
#         for r0 in R0_LIST:
#             for h in H_LIST:
#                 run_idx += 1
#                 if run_idx == 1 or run_idx % 10 == 0 or run_idx == total_runs:
#                     elapsed = time.perf_counter() - t_start
#                     rate = run_idx / max(elapsed, 1e-9)
#                     eta = (total_runs - run_idx) / max(rate, 1e-9)
#                     print(
#                         f"[{run_idx:4d}/{total_runs}] "
#                         f"Pc={pc:.2e} h={h:.0e} r0={r0:.0e} "
#                         f"elapsed={elapsed:6.1f}s ETA={eta:6.1f}s"
#                     )

#                 r_arr, h00_arr, y_arr, (K_geom, pc_geom) = integrate_star(pc, h, r0)

#                 r_win = min(50.0 * r0, 1.0e-2)
#                 win_mask = (r_arr >= r0) & (r_arr <= r_win)
#                 y_win = y_arr[win_mask]

#                 a0_est = np.median(y_win) if y_win.size else np.nan
#                 a0_std = robust_std(y_win)

#                 a0_ana = 1.0
#                 rel_err_a0 = abs(a0_est - a0_ana) / max(abs(a0_ana), TINY)

#                 if y_win.size:
#                     spike_amp = float(np.max(y_win) - np.min(y_win))
#                     dev = np.abs(y_win - a0_est)
#                     spike_loc = float(r_arr[win_mask][np.argmax(dev)])
#                 else:
#                     spike_amp = np.nan
#                     spike_loc = np.nan

#                 key = f"Pc{pc:.3e}_h{h:.0e}_r0{r0:.0e}"
#                 raw[f"{key}_r"] = r_arr
#                 raw[f"{key}_h00"] = h00_arr
#                 raw[f"{key}_y"] = y_arr
#                 raw[f"{key}_y_ana"] = analytic_y(r_arr, K_geom, pc_geom, a0=1.0, a8=0.0)

#                 rows.append(
#                     dict(
#                         Pc_cgs=pc,
#                         h=h,
#                         r0=r0,
#                         r_win=r_win,
#                         a0_est=a0_est,
#                         a0_std=a0_std,
#                         a0_ana=a0_ana,
#                         rel_err_a0=rel_err_a0,
#                         spike_amp=spike_amp,
#                         spike_loc=spike_loc,
#                         n_points=len(r_arr),
#                     )
#                 )

#         plot_for_pc(pc, rows, raw, output_dir)


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _THIS_DIR / "initial_h00_test_outputs" / timestamp

    run_sweep(out_dir)
