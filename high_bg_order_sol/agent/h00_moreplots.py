#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

TINY = 1e-30


def load_summary(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # cast
            row = dict(r)
            for k in ["Pc_cgs", "h", "r0", "r_win", "a0_est", "a0_std", "a0_ana", "rel_err_a0",
                      "spike_amp", "spike_loc"]:
                if k in row and row[k] != "":
                    row[k] = float(row[k])
                elif k in row:
                    row[k] = float("nan")
            rows.append(row)
    return rows


def key(pc, h, r0):
    return f"Pc{pc:.3e}_h{h:.0e}_r0{r0:.0e}"


def win_mask(r, r0, r_win):
    return (r >= r0) & (r <= r_win)


def safe_logspace(rmin, rmax, n=300):
    rmin = float(rmin)
    rmax = float(rmax)
    if not (rmin > 0 and rmax > rmin):
        return None
    return np.logspace(np.log10(rmin), np.log10(rmax), n)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main(output_dir: str):
    out = Path(output_dir).resolve()
    csv_path = out / "summary.csv"
    npz_path = out / "raw_arrays.npz"
    if not csv_path.exists() or not npz_path.exists():
        raise FileNotFoundError("Need summary.csv and raw_arrays.npz in the output directory.")

    rows = load_summary(csv_path)
    raw = np.load(npz_path, allow_pickle=True)

    pcs = sorted({r["Pc_cgs"] for r in rows if math.isfinite(r["Pc_cgs"])})
    hs = sorted({r["h"] for r in rows if math.isfinite(r["h"])})
    r0s = sorted({r["r0"] for r in rows if math.isfinite(r["r0"])}, reverse=False)

    fig_dir = out / "more_plots"
    ensure_dir(fig_dir)

    # ------------------------------------------------------------
    # A) For each h: 2x3 grid over r0, overlay different Pc curves
    #     Plot residual Δy = y - y_ana
    #     Produce both log-x and linear-x versions
    # ------------------------------------------------------------
    for h in hs:
        for xmode in ["log", "linear"]:
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160, sharey=True)
            axes = axes.reshape(-1)
            used_any = False

            for i, r0 in enumerate(r0s[:6]):  # expect 6
                ax = axes[i]
                # compute r_win from any row matching (h,r0)
                rwin_list = [r["r_win"] for r in rows if r["h"] == h and r["r0"] == r0 and math.isfinite(r["r_win"])]
                if not rwin_list:
                    ax.set_title(f"r0={r0:.0e} (no data)")
                    continue
                r_win = float(np.nanmedian(rwin_list))
                if not (r_win > r0):
                    ax.set_title(f"r0={r0:.0e} (empty window)")
                    continue

                for pc in pcs:
                    k = key(pc, h, r0)
                    rk = f"{k}_r"
                    yk = f"{k}_y"
                    yak = f"{k}_y_ana"
                    if rk not in raw or yk not in raw or yak not in raw:
                        continue
                    r = raw[rk]
                    y = raw[yk]
                    ya = raw[yak]

                    m = win_mask(r, r0, r_win)
                    if not np.any(m):
                        continue
                    rr = r[m]
                    dy = (y[m] - ya[m])

                    ax.plot(rr, dy, label=f"Pc={pc:.2e}")
                    used_any = True

                if xmode == "log":
                    ax.set_xscale("log")
                ax.grid(alpha=0.2)
                ax.set_title(f"r0={r0:.0e}")
                ax.set_xlabel("r (km)")
                if i % 3 == 0:
                    ax.set_ylabel("Δy = (h00/r^2) - (h00_ana/r^2)")

                # For linear mode, zoom the x-range to the window so spikes are visible
                if xmode == "linear":
                    ax.set_xlim(r0, r_win)

            if used_any:
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=7)
                fig.suptitle(f"Residual Δy overlays across Pc | h={h:.0e} | x={xmode}")
                fig.tight_layout(rect=[0, 0, 0.88, 0.95])
                fig.savefig(fig_dir / f"overlayPc_residual_dy_grid_h{h:.0e}_x{xmode}.png")
            plt.close(fig)

    # ------------------------------------------------------------
    # B) Peak metrics vs Pc at fixed (h, r0)
    #     peak_abs, peak_rel, r_peak using analytic-aware residuals
    #     One figure per metric: panels=r0, curves=h
    # ------------------------------------------------------------
    metrics = ["peak_abs", "peak_rel", "r_peak"]
    for metric in metrics:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160, sharex=True)
        axes = axes.reshape(-1)

        for i, r0 in enumerate(r0s[:6]):
            ax = axes[i]
            for h in hs:
                pc_vals = []
                met_vals = []
                for pc in pcs:
                    k = key(pc, h, r0)
                    rk = f"{k}_r"
                    yk = f"{k}_y"
                    yak = f"{k}_y_ana"
                    if rk not in raw or yk not in raw or yak not in raw:
                        continue

                    # get r_win from summary if possible
                    rwin_list = [r["r_win"] for r in rows if r["Pc_cgs"] == pc and r["h"] == h and r["r0"] == r0]
                    if not rwin_list or not math.isfinite(rwin_list[0]):
                        continue
                    r_win = float(rwin_list[0])
                    if not (r_win > r0):
                        continue

                    r = raw[rk]
                    y = raw[yk]
                    ya = raw[yak]
                    m = win_mask(r, r0, r_win)
                    if not np.any(m):
                        continue

                    rr = r[m]
                    dy = y[m] - ya[m]
                    denom = np.maximum(np.abs(ya[m]), TINY)
                    rel = np.abs(dy) / denom

                    peak_abs = float(np.max(np.abs(dy)))
                    peak_rel = float(np.max(rel))
                    r_peak = float(rr[int(np.argmax(rel))])

                    pc_vals.append(pc)
                    if metric == "peak_abs":
                        met_vals.append(peak_abs)
                    elif metric == "peak_rel":
                        met_vals.append(peak_rel)
                    else:
                        met_vals.append(r_peak)

                if pc_vals:
                    # x-axis in log Pc is usually more readable
                    pc_vals = np.array(pc_vals)
                    met_vals = np.array(met_vals)
                    order = np.argsort(pc_vals)
                    ax.plot(pc_vals[order], met_vals[order], marker="o", ms=3, lw=1.0, label=f"h={h:.0e}")

            ax.set_xscale("log")
            ax.grid(alpha=0.2)
            ax.set_title(f"r0={r0:.0e}")
            ax.set_xlabel("Pc (cgs)")
            if i % 3 == 0:
                ax.set_ylabel(metric)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=8)
        fig.suptitle(f"{metric} vs Pc at fixed (h,r0) | analytic-aware")
        fig.tight_layout(rect=[0, 0, 0.88, 0.95])
        fig.savefig(fig_dir / f"peak_metrics_{metric}_vs_Pc_panels_r0.png")
        plt.close(fig)

    # ------------------------------------------------------------
    # C) Shape-collapse test:
    #     For each (h,r0): plot normalized residual vs x=r/r0
    #     Combine all Pc curves; make both log-x and linear-x
    #     To keep output manageable: make a figure for each h (panels=r0)
    # ------------------------------------------------------------
    for h in hs:
        for xmode in ["log", "linear"]:
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=160, sharey=True)
            axes = axes.reshape(-1)

            for i, r0 in enumerate(r0s[:6]):
                ax = axes[i]
                for pc in pcs:
                    k = key(pc, h, r0)
                    rk = f"{k}_r"
                    yk = f"{k}_y"
                    yak = f"{k}_y_ana"
                    if rk not in raw or yk not in raw or yak not in raw:
                        continue

                    rwin_list = [r["r_win"] for r in rows if r["Pc_cgs"] == pc and r["h"] == h and r["r0"] == r0]
                    if not rwin_list or not math.isfinite(rwin_list[0]):
                        continue
                    r_win = float(rwin_list[0])
                    if not (r_win > r0):
                        continue

                    r = raw[rk]
                    y = raw[yk]
                    ya = raw[yak]
                    m = win_mask(r, r0, r_win)
                    if not np.any(m):
                        continue

                    rr = r[m]
                    dy = (y[m] - ya[m])
                    peak = np.max(np.abs(dy))
                    if not (peak > 0):
                        continue
                    dy_norm = dy / peak
                    x = rr / r0

                    ax.plot(x, dy_norm, label=f"Pc={pc:.2e}")

                if xmode == "log":
                    ax.set_xscale("log")
                ax.grid(alpha=0.2)
                ax.set_title(f"r0={r0:.0e}")
                ax.set_xlabel("x = r/r0")
                if i % 3 == 0:
                    ax.set_ylabel("Δy normalized by peak")

                if xmode == "linear":
                    ax.set_xlim(1.0, 50.0)  # focus on the window scale, adjust if you want

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=7)
            fig.suptitle(f"Shape collapse Δy/peak vs r/r0 | h={h:.0e} | x={xmode}")
            fig.tight_layout(rect=[0, 0, 0.88, 0.95])
            fig.savefig(fig_dir / f"shape_collapse_h{h:.0e}_x{xmode}.png")
            plt.close(fig)

    print(f"Done. Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python analyze_initial_h00_raw_more_plots.py <output_dir_with_summary_and_npz>")
    main(sys.argv[1])
