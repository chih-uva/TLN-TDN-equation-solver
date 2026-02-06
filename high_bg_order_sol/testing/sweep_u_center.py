#!/usr/bin/env python
"""
Parameter sweep for central behavior of u(r)=h00(r)/r^2 under varying step size h and start radius eps.

Central-only is default and fast. Full runs are optional with --full.
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent folder to sys.path to import helper functions
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tov_series_aux import p_approx, m_approx
from h00_aux import h00_approx, h00p_approx


# -----------------------------
# Constants and EOS definitions
# -----------------------------
K_cgs = 1.6e5
n_poly = 1

def PolyEOS(p, n, K):
    rr = (p / K) ** (n / (n + 1))
    return rr

def EnergyEOS(p, n, K):
    rr = ((1 + np.sqrt(4 * K * p)) ** 2 - 1) / (4 * K)
    return rr

G_cgs = 6.67430e-8
c_cgs = 2.99792458e10
M_sun_cgs = 1.989e33
cm2km = 1e5

def cgs_to_geom(p_cgs, n, K_cgs, eos=PolyEOS):
    p_geom = p_cgs * G_cgs / c_cgs**4 * cm2km**2
    K_geom = K_cgs / G_cgs**(1 / n) / c_cgs**(2 - 2 / n) / cm2km**(2 / n)
    rho_geom = eos(p_geom, n, K_geom)
    return K_geom, p_geom, rho_geom


def tov_rhs(r, y, eos, n, K):
    m, p = y[0], y[1]
    rhoEnergy = eos(p, n, K)
    dm_dr = 4 * np.pi * r**2 * rhoEnergy
    dp_dr = - (rhoEnergy + p) * (m + 4*np.pi*(r**3)*p) / (r * (r - 2*m))
    return [dm_dr, dp_dr]


def rk4_step(func, r, y, h, eos=EnergyEOS, n=n_poly, K=K_cgs):
    k1 = np.array(func(r, y, eos, n, K))
    k2 = np.array(func(r + h/2, y + h*k1/2, eos, n, K))
    k3 = np.array(func(r + h/2, y + h*k2/2, eos, n, K))
    k4 = np.array(func(r + h, y + h*k3, eos, n, K))
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)


def rk4_step_generic(f, x, y, h, **kwargs):
    k1 = f(x, y, **kwargs)
    k2 = f(x + 0.5*h, y + 0.5*h*k1, **kwargs)
    k3 = f(x + 0.5*h, y + 0.5*h*k2, **kwargs)
    k4 = f(x + h, y + h*k3, **kwargs)
    return y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)


def u_pp_rhs(r, Y, p_r, rho_r, lamb_r, drho):
    u, up = Y
    pi = np.pi

    eL = np.exp(lamb_r)
    e2L = eL * eL
    e3L = e2L * eL

    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r6 = r3 * r3

    A = (
        -1.0 - 3.0*eL + 3.0*e2L + e3L
        + 96.0*e2L*(-5.0 + 2.0*eL)*(pi**2)*r4*(p_r**2)
        + 512.0*e3L*(pi**3)*r6*(p_r**3)
        - 20.0*eL*(-1.0 + eL)*pi*r2*rho_r
        + 4.0*eL*pi*r2*p_r*(15.0 - 9.0*eL + 6.0*e2L - 40.0*eL*pi*r2*rho_r)
        + 8.0*eL*pi*r3*drho
    )

    D = r2 * (-1.0 + eL + 8.0*eL*pi*r2*p_r)
    if D == 0.0 or not np.isfinite(D):
        raise ZeroDivisionError(f"Denominator is singular at r={r}")

    B = 1.0 + eL + 4.0*eL*pi*r2*p_r - 4.0*eL*pi*r2*rho_r

    upp = (A/D)*u - ((4.0 + B)/r)*up - ((2.0 + 2.0*B)/r2)*u

    return np.array([up, upp], dtype=float)


# -----------------------------
# Helpers
# -----------------------------

def _parse_token(tok: str) -> float:
    try:
        return float(tok)
    except ValueError:
        # Support tokens like 1e-3.5 or 2E2.25
        s = tok.strip()
        if 'e' in s.lower():
            base_str, exp_str = s.lower().split('e', 1)
            return float(base_str) * (10.0 ** float(exp_str))
        raise


def parse_list(arg: str) -> List[float]:
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    return [_parse_token(p) for p in parts]


def fmt_float(x: float) -> str:
    s = f"{x:.0e}"
    s = s.replace('+', '')
    s = s.replace('-', 'm')
    return s


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class ComboResult:
    pc: float
    eps: float
    h: float
    n_center: int
    r_end_center: float
    max_abs_err: float
    max_rel_err: float
    spike_amp: float
    max_du: float
    max_abs_err_over_h: float
    max_du_over_pc: float
    surface_r: float
    n_points: int
    runtime_s: float


# -----------------------------
# Core computation
# -----------------------------

def integrate_tov(pc_geom: float, K_geom: float, eps: float, h: float, n_center: int, central_only: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Initial conditions from series
    p0 = p_approx(eps, K_geom, pc_geom)
    m0 = m_approx(eps, K_geom, pc_geom)

    Sol = [np.array([m0, p0], dtype=float)]
    R_l = [eps]

    # stop criterion from notebook
    def keep_going(p_start, p_now):
        if p_now <= 0 or p_start <= 0:
            return False
        return (np.log10(p_start / p_now) <= 10.1)

    max_steps = n_center if central_only else None

    step_count = 0
    while True:
        if max_steps is not None and step_count >= max_steps:
            break
        if not keep_going(Sol[0][1], Sol[-1][1]):
            break

        newStep = rk4_step(tov_rhs, R_l[-1], Sol[-1], h, EnergyEOS, n_poly, K_geom)
        Sol.append(newStep)
        R_l.append(R_l[-1] + h)
        step_count += 1

    SolArr = np.array(Sol)
    RArr = np.array(R_l)
    MArr = SolArr[:, 0]
    PArr = SolArr[:, 1]
    return RArr, MArr, PArr


def solve_u(RArr: np.ndarray, PArr: np.ndarray, MArr: np.ndarray, K_geom: float, pc_geom: float) -> np.ndarray:
    RhoArr = EnergyEOS(PArr, n_poly, K_geom)
    LambArr = np.log(RArr / (RArr - 2.0*MArr))
    edge_order = 2 if len(RArr) >= 3 else 1
    DrhoArr = np.gradient(RhoArr, RArr, edge_order=edge_order)

    r0 = RArr[0]
    h0 = h00_approx(r=r0, K=K_geom, pc=pc_geom, a0=1, a8=0)
    hp0 = h00p_approx(r=r0, K=K_geom, pc=pc_geom, a0=1, a8=0)

    u0 = h0 / (r0**2)
    up0 = (r0*hp0 - 2.0*h0) / (r0**3)

    USol = np.zeros((RArr.shape[0], 2), dtype=float)
    USol[0] = np.array([u0, up0], dtype=float)

    for i, r in enumerate(RArr[:-1]):
        dr = RArr[i+1] - RArr[i]
        params = {
            "p_r": PArr[i],
            "rho_r": RhoArr[i],
            "lamb_r": LambArr[i],
            "drho": DrhoArr[i],
        }
        USol[i+1] = rk4_step_generic(u_pp_rhs, r, USol[i], dr, **params)

    return USol


def compute_metrics(u_num: np.ndarray, u_ana: np.ndarray, n_center: int) -> Tuple[float, float, float]:
    n_use = min(n_center + 1, len(u_num))
    if n_use < 3:
        n_use = len(u_num)
    u_num_c = u_num[:n_use]
    u_ana_c = u_ana[:n_use]

    abs_err = np.abs(u_num_c - u_ana_c)
    max_abs_err = float(np.max(abs_err)) if abs_err.size else float('nan')

    denom = max(np.max(np.abs(u_ana_c)), 1e-30) if u_ana_c.size else 1e-30
    # denom = np.abs(u_ana_c)
    max_rel_err = max_abs_err / denom

    # spike metric
    last_n = max(1, int(0.2 * n_use))
    baseline = np.median(u_num_c[-last_n:])
    spike_amp = float(np.max(np.abs(u_num_c - baseline))) if u_num_c.size else float('nan')

    return max_abs_err, max_rel_err, spike_amp


def compute_max_du(u_num: np.ndarray, r_arr: np.ndarray) -> float:
    if len(u_num) < 2:
        return float('nan')
    du = np.diff(u_num)
    dr = np.diff(r_arr)
    with np.errstate(divide='ignore', invalid='ignore'):
        du_dr = du / dr
    return float(np.max(np.abs(du_dr)))


# -----------------------------
# Plotting
# -----------------------------

def plot_central_grid(
    out_path: Path,
    tag: str,
    r_list: List[np.ndarray],
    u_list: List[np.ndarray],
    uana_list: List[np.ndarray],
    logx: bool,
    title: str = None,
    pc_labels: List[Tuple[int, float]] = None,
    metrics: Dict[str, List[float]] = None,
) -> None:
    n = len(r_list)
    if n == 25:
        rows, cols = 5, 5
    else:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), dpi=200)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r = r_list[i] if i < n else None
        ax = axes[i // cols][i % cols]
        if r is None:
            ax.axis('off')
            continue
        ax.plot(r, u_list[i], 'b-', label="Numerical", linewidth=2)
        ax.plot(r, uana_list[i], 'r--', label="Analytic", linewidth=2)
        if logx and np.all(r > 0):
            ax.set_xscale("log")
        # Annotate pc index/value and per-pc metrics if provided
        if pc_labels is not None and metrics is not None and i < len(pc_labels):
            idx, pcv = pc_labels[i]
            abs_e = metrics["abs"][i]
            rel_e = metrics["rel"][i]
            spk_e = metrics["spk"][i]
            du_e = metrics.get("du", [float('nan')]*len(pc_labels))[i]
            du_pc_e = metrics.get("du_pc", [float('nan')]*len(pc_labels))[i]
            ax.text(
                0.02, 0.98,
                f"pc[{idx}]={pcv:.2e}\nabs={abs_e:.1e} rel={rel_e:.1e}\nspk={spk_e:.1e}\nmax|du/dr|={du_e:.1e}\nmax|du/dr|/pc={du_pc_e:.1e}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_overlay_eps(
    out_path: Path,
    r_by_eps: Dict[float, List[np.ndarray]],
    u_by_eps: Dict[float, List[np.ndarray]],
    uana_by_eps: Dict[float, List[np.ndarray]],
    eps_list: List[float],
    title: str = None,
    mode: str = "raw",
) -> None:
    # mode: raw | ratio | relerr
    # all eps share the same number of models
    first_eps = eps_list[0]
    n = len(uana_by_eps[first_eps])
    if n == 25:
        rows, cols = 5, 5
    else:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), dpi=200)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    colors = plt.cm.viridis(np.linspace(0, 1, len(eps_list)))

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i >= n:
            ax.axis('off')
            continue
        for c, eps in zip(colors, eps_list):
            r = r_by_eps[eps][i]
            u = u_by_eps[eps][i]
            ua = uana_by_eps[eps][i]
            if mode == "ratio":
                y_num = u / (ua + 1e-30)
                y_ana = np.ones_like(ua)
            elif mode == "relerr":
                y_num = np.abs(u - ua) / np.maximum(np.abs(ua), 1e-30)
                y_ana = np.zeros_like(ua)
            else:
                y_num = u
                y_ana = ua
            ax.plot(r, y_num, color=c, linewidth=1.3, label=f"eps={eps:.1e}")
            ax.plot(r, y_ana, color=c, linestyle='--', linewidth=1.0, alpha=0.6)
        if i == 0:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_du_grid(
    out_path: Path,
    r_list: List[np.ndarray],
    du_list: List[np.ndarray],
    logx: bool,
    title: str = None,
    pc_labels: List[Tuple[int, float]] = None,
    du_metrics: List[float] = None,
) -> None:
    n = len(r_list)
    if n == 25:
        rows, cols = 5, 5
    else:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(18, 18), dpi=200)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for i in range(rows * cols):
        r = r_list[i] if i < n else None
        ax = axes[i // cols][i % cols]
        if r is None:
            ax.axis('off')
            continue
        ax.plot(r, du_list[i], 'k-', linewidth=2)
        if logx and np.all(r > 0):
            ax.set_xscale("log")
        if pc_labels is not None and du_metrics is not None and i < len(pc_labels):
            idx, pcv = pc_labels[i]
            ax.text(
                0.02, 0.98,
                f"pc[{idx}]={pcv:.2e}\nmax|du/dr|={du_metrics[i]:.1e}",
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def plot_metrics_line(out_path: Path, eps_list: List[float], metric_by_h: Dict[float, List[float]], ylabel: str) -> None:
    plt.figure(figsize=(8, 6), dpi=160)
    for h, vals in metric_by_h.items():
        plt.plot(eps_list, vals, '-o', label=f"h={h:.1e}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("eps")
    plt.ylabel(ylabel)
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_heatmap(
    out_path: Path,
    eps_list: List[float],
    h_list: List[float],
    grid: np.ndarray,
    title: str,
    best_ij: Tuple[int, int] = None,
) -> None:
    plt.figure(figsize=(8, 6), dpi=160)
    im = plt.imshow(grid, aspect='auto', origin='lower')
    plt.colorbar(im, label=title)
    plt.xticks(range(len(eps_list)), [f"{e:.1e}" for e in eps_list], rotation=45)
    plt.yticks(range(len(h_list)), [f"{h:.1e}" for h in h_list])
    plt.xlabel("eps")
    plt.ylabel("h")
    plt.title(title)
    if best_ij is not None:
        i, j = best_ij
        plt.scatter([j], [i], s=120, facecolors='none', edgecolors='w', linewidths=2)
        plt.scatter([j], [i], s=60, facecolors='none', edgecolors='k', linewidths=1)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Sweep eps and h for central u(r) behavior.")
    parser.add_argument("--h-list", type=str, default="1e-4,5e-5,2e-5,1e-5")
    parser.add_argument("--eps-list", type=str, default="1e-3,5e-4,2e-4,1e-4,5e-5")
    parser.add_argument("--central-only", action="store_true", default=False, help="Explicitly enable central-only (default).")
    parser.add_argument("--full", action="store_true", default=False, help="Run full integration for top-k pairs after central sweep.")
    parser.add_argument("--topk", type=int, default=3, help="Top-k (eps,h) pairs for full runs.")
    default_out_root = str(Path(__file__).resolve().parents[1] / "testing" / "output")
    parser.add_argument("--out-root", type=str, default=default_out_root, help="Output root directory (absolute).")
    parser.add_argument("--r-center-target", type=float, default=0.05, help="Target radius for central window.")
    args = parser.parse_args()

    h_list = parse_list(args.h_list)
    eps_list = parse_list(args.eps_list)

    # central-only is default
    central_only = True

    # pc list from notebook
    d_num = 12
    PcArr_cgs = np.logspace(33.5, 36.5, d_num)
    K_geom, PcArr_geom = cgs_to_geom(PcArr_cgs, n_poly, K_cgs, eos=EnergyEOS)[:2]

    # output folder
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"u_center_sweep_{ts}"
    fig_dir = out_dir / "figures"
    npz_dir = out_dir / "npz"
    full_dir = out_dir / "full"
    ensure_dir(fig_dir)
    ensure_dir(npz_dir)
    if args.full:
        ensure_dir(full_dir)

    # metadata
    metadata = {
        "pc_list_cgs": PcArr_cgs.tolist(),
        "pc_list_geom": PcArr_geom.tolist(),
        "h_list": h_list,
        "eps_list": eps_list,
        "r_center_target": args.r_center_target,
        "n_center_rule": "n_center=round(r_center_target/h)",
        "central_only_default": True,
        "full_enabled": args.full,
        "topk": args.topk,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    summary_rows: List[ComboResult] = []

    # store per (eps,h) for plotting
    central_curves: Dict[Tuple[float, float], Dict[str, List[np.ndarray]]] = {}

    for h in tqdm(h_list, desc="h sweep"):
        n_center = int(round(args.r_center_target / h))
        r_end_center = None
        for eps in tqdm(eps_list, desc=f"eps sweep (h={h:.1e})", leave=False):
            r_end_center = eps + n_center * h
            r_list = []
            u_list = []
            uana_list = []
            rdu_list = []
            du_list = []

            for pc in tqdm(PcArr_geom, desc=f"pc sweep (h={h:.1e}, eps={eps:.1e})", leave=False):
                t0 = datetime.now().timestamp()
                RArr, MArr, PArr = integrate_tov(pc, K_geom, eps, h, n_center, central_only=central_only)

                # keep only central window (already capped in central_only)
                USol = solve_u(RArr, PArr, MArr, K_geom, pc)
                u_num = USol[:, 0]
                u_ana = h00_approx(r=RArr, K=K_geom, pc=pc, a0=1, a8=0) / (RArr**2)

                max_abs_err, max_rel_err, spike_amp = compute_metrics(u_num, u_ana, n_center)
                max_du = compute_max_du(u_num, RArr)
                max_abs_err_over_h = max_abs_err / h if h > 0 else float('nan')
                max_du_over_pc = max_du / pc if pc > 0 else float('nan')

                t1 = datetime.now().timestamp()

                r_list.append(RArr)
                u_list.append(u_num)
                uana_list.append(u_ana)
                r_du = 0.5 * (RArr[:-1] + RArr[1:]) if len(RArr) > 1 else RArr
                du_dr = np.diff(u_num) / np.diff(RArr) if len(RArr) > 1 else np.array([np.nan])
                rdu_list.append(r_du)
                du_list.append(du_dr)

                npz_name = npz_dir / f"pc{fmt_float(pc)}__eps{fmt_float(eps)}__h{fmt_float(h)}.npz"
                np.savez(npz_name, r=RArr, u_num=u_num, u_analytic=u_ana, p=PArr, m=MArr)

                summary_rows.append(
                    ComboResult(
                        pc=pc,
                        eps=eps,
                        h=h,
                        n_center=n_center,
                        r_end_center=r_end_center,
                        max_abs_err=max_abs_err,
                        max_rel_err=max_rel_err,
                        spike_amp=spike_amp,
                        max_du=max_du,
                        max_abs_err_over_h=max_abs_err_over_h,
                        max_du_over_pc=max_du_over_pc,
                        surface_r=float(RArr[-1]),
                        n_points=len(RArr),
                        runtime_s=float(t1 - t0),
                    )
                )

            # store for plotting
            central_curves[(eps, h)] = {
                "r": r_list,
                "u": u_list,
                "uana": uana_list,
            }

            # central plots per (eps,h)
            pc_labels = list(enumerate(PcArr_geom))
            # reuse per-pc metrics computed in this (eps,h) loop
            metrics = {
                "abs": [r.max_abs_err for r in summary_rows if r.eps == eps and r.h == h][-len(PcArr_geom):],
                "rel": [r.max_rel_err for r in summary_rows if r.eps == eps and r.h == h][-len(PcArr_geom):],
                "spk": [r.spike_amp for r in summary_rows if r.eps == eps and r.h == h][-len(PcArr_geom):],
                "du": [r.max_du for r in summary_rows if r.eps == eps and r.h == h][-len(PcArr_geom):],
                "du_pc": [r.max_du_over_pc for r in summary_rows if r.eps == eps and r.h == h][-len(PcArr_geom):],
            }
            plot_central_grid(
                fig_dir / f"central_u_linear__eps{fmt_float(eps)}__h{fmt_float(h)}.png",
                tag="linear",
                r_list=r_list,
                u_list=u_list,
                uana_list=uana_list,
                logx=False,
                title=f"central u(r): eps={eps:.2e}, h={h:.2e} (linear x)",
                pc_labels=pc_labels,
                metrics=metrics,
            )
            plot_du_grid(
                fig_dir / f"central_du_linear__eps{fmt_float(eps)}__h{fmt_float(h)}.png",
                r_list=rdu_list,
                du_list=du_list,
                logx=False,
                title=f"central du/dr: eps={eps:.2e}, h={h:.2e} (linear x)",
                pc_labels=pc_labels,
                du_metrics=metrics["du"],
            )
            plot_central_grid(
                fig_dir / f"central_u_logx__eps{fmt_float(eps)}__h{fmt_float(h)}.png",
                tag="logx",
                r_list=r_list,
                u_list=u_list,
                uana_list=uana_list,
                logx=True,
                title=f"central u(r): eps={eps:.2e}, h={h:.2e} (log x)",
                pc_labels=pc_labels,
                metrics=metrics,
            )
            plot_du_grid(
                fig_dir / f"central_du_logx__eps{fmt_float(eps)}__h{fmt_float(h)}.png",
                r_list=rdu_list,
                du_list=du_list,
                logx=True,
                title=f"central du/dr: eps={eps:.2e}, h={h:.2e} (log x)",
                pc_labels=pc_labels,
                du_metrics=metrics["du"],
            )

        # compute best eps for this h based on worst max_abs_err (local)
        best_eps = None
        best_val = None
        for eps in eps_list:
            vals = [r.max_abs_err for r in summary_rows if r.eps == eps and r.h == h]
            if not vals:
                continue
            worst = max(vals)
            if best_val is None or worst < best_val:
                best_val = worst
                best_eps = eps
        best_tag = f"best eps={best_eps:.1e}" if best_eps is not None else "best eps=NA"

        # overlay eps for each h
        r_by_eps = {eps: central_curves[(eps, h)]["r"] for eps in eps_list}
        u_by_eps = {eps: central_curves[(eps, h)]["u"] for eps in eps_list}
        uana_by_eps = {eps: central_curves[(eps, h)]["uana"] for eps in eps_list}
        plot_overlay_eps(
            fig_dir / f"overlay_eps__h{fmt_float(h)}.png",
            r_by_eps=r_by_eps,
            u_by_eps=u_by_eps,
            uana_by_eps=uana_by_eps,
            eps_list=eps_list,
            title=f"overlay eps at fixed h={h:.2e} (central window), {best_tag}",
            mode="raw",
        )
        # ratio overlay: u_num/u_analytic (baseline at 1)
        plot_overlay_eps(
            fig_dir / f"overlay_eps_ratio__h{fmt_float(h)}.png",
            r_by_eps=r_by_eps,
            u_by_eps=u_by_eps,
            uana_by_eps=uana_by_eps,
            eps_list=eps_list,
            title=f"overlay eps (u/u_analytic) at fixed h={h:.2e}, {best_tag}",
            mode="ratio",
        )
        # relative error overlay: |u_num-u_analytic|/|u_analytic|
        plot_overlay_eps(
            fig_dir / f"overlay_eps_relerr__h{fmt_float(h)}.png",
            r_by_eps=r_by_eps,
            u_by_eps=u_by_eps,
            uana_by_eps=uana_by_eps,
            eps_list=eps_list,
            title=f"overlay eps (rel. error) at fixed h={h:.2e}, {best_tag}",
            mode="relerr",
        )

    # summary.csv
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pc_geom", "eps", "h", "n_center", "r_end_center",
            "max_abs_err", "max_rel_err", "spike_amp", "max_du", "max_abs_err_over_h", "max_du_over_pc",
            "surface_r", "n_points", "runtime_s",
        ])
        for row in summary_rows:
            writer.writerow([
                row.pc, row.eps, row.h, row.n_center, row.r_end_center,
                row.max_abs_err, row.max_rel_err, row.spike_amp, row.max_du, row.max_abs_err_over_h, row.max_du_over_pc,
                row.surface_r, row.n_points, row.runtime_s,
            ])

    # aggregate ranking
    ranking = {}
    for row in summary_rows:
        key = (row.eps, row.h)
        if key not in ranking:
            ranking[key] = {
                "max_abs_err": row.max_abs_err,
                "spike_amp": row.spike_amp,
                "max_rel_err": row.max_rel_err,
                "max_du": row.max_du,
                "max_abs_err_over_h": row.max_abs_err_over_h,
                "max_du_over_pc": row.max_du_over_pc,
            }
        else:
            ranking[key]["max_abs_err"] = max(ranking[key]["max_abs_err"], row.max_abs_err)
            ranking[key]["spike_amp"] = max(ranking[key]["spike_amp"], row.spike_amp)
            ranking[key]["max_rel_err"] = max(ranking[key]["max_rel_err"], row.max_rel_err)
            ranking[key]["max_du"] = max(ranking[key]["max_du"], row.max_du)
            ranking[key]["max_abs_err_over_h"] = max(ranking[key]["max_abs_err_over_h"], row.max_abs_err_over_h)
            ranking[key]["max_du_over_pc"] = max(ranking[key]["max_du_over_pc"], row.max_du_over_pc)

    # eps* per h (minimize worst max_abs_err)
    eps_star_by_h = {}
    for h in h_list:
        candidates = [(eps, ranking[(eps, h)]["max_abs_err"]) for eps in eps_list]
        eps_star_by_h[h] = min(candidates, key=lambda x: x[1])[0]

    with open(out_dir / "ranking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["eps", "h", "worst_max_abs_err", "worst_max_rel_err", "worst_spike_amp", "worst_max_du", "worst_max_abs_err_over_h", "worst_max_du_over_pc", "eps_star_for_h"])
        for h in h_list:
            for eps in eps_list:
                writer.writerow([
                    eps, h,
                    ranking[(eps, h)]["max_abs_err"],
                    ranking[(eps, h)]["max_rel_err"],
                    ranking[(eps, h)]["spike_amp"],
                    ranking[(eps, h)]["max_du"],
                    ranking[(eps, h)]["max_abs_err_over_h"],
                    ranking[(eps, h)]["max_du_over_pc"],
                    eps_star_by_h[h] if eps == eps_star_by_h[h] else "",
                ])

    # best overall (by worst max_abs_err)
    best_pair = min(ranking.items(), key=lambda kv: kv[1]["max_abs_err"])
    best_eps, best_h = best_pair[0]
    best_metrics = best_pair[1]

    print("Best (eps, h) by worst max_abs_err:")
    print(f"  eps = {best_eps:.3e}, h = {best_h:.3e}")
    print(f"  worst max_abs_err = {best_metrics['max_abs_err']:.3e}")
    print(f"  worst max_rel_err = {best_metrics['max_rel_err']:.3e}")
    print(f"  worst spike_amp  = {best_metrics['spike_amp']:.3e}")
    print(f"  worst max_du     = {best_metrics['max_du']:.3e}")
    print(f"  worst max_abs_err_over_h = {best_metrics['max_abs_err_over_h']:.3e}")
    print(f"  worst max_du_over_pc = {best_metrics['max_du_over_pc']:.3e}")

    # metric plots
    # metric visualizations
    metric_abs_by_h = {h: [ranking[(eps, h)]["max_abs_err"] for eps in eps_list] for h in h_list}
    metric_rel_by_h = {h: [ranking[(eps, h)]["max_rel_err"] for eps in eps_list] for h in h_list}
    metric_spk_by_h = {h: [ranking[(eps, h)]["spike_amp"] for eps in eps_list] for h in h_list}
    metric_du_by_h = {h: [ranking[(eps, h)]["max_du"] for eps in eps_list] for h in h_list}
    metric_abs_over_h_by_h = {h: [ranking[(eps, h)]["max_abs_err_over_h"] for eps in eps_list] for h in h_list}
    metric_du_over_pc_by_h = {h: [ranking[(eps, h)]["max_du_over_pc"] for eps in eps_list] for h in h_list}

    plot_metrics_line(fig_dir / "metric_vs_eps__by_h.png", eps_list, metric_abs_by_h, ylabel="worst max_abs_err")
    plot_metrics_line(fig_dir / "metric_rel_vs_eps__by_h.png", eps_list, metric_rel_by_h, ylabel="worst max_rel_err")
    plot_metrics_line(fig_dir / "metric_spike_vs_eps__by_h.png", eps_list, metric_spk_by_h, ylabel="worst spike_amp")
    plot_metrics_line(fig_dir / "metric_du_vs_eps__by_h.png", eps_list, metric_du_by_h, ylabel="worst max|du/dr|")
    plot_metrics_line(fig_dir / "metric_abs_over_h_vs_eps__by_h.png", eps_list, metric_abs_over_h_by_h, ylabel="worst max_abs_err/h")
    plot_metrics_line(fig_dir / "metric_du_over_pc_vs_eps__by_h.png", eps_list, metric_du_over_pc_by_h, ylabel="worst max|du/dr|/pc")

    # heatmap
    grid_abs = np.zeros((len(h_list), len(eps_list)), dtype=float)
    grid_rel = np.zeros((len(h_list), len(eps_list)), dtype=float)
    grid_spk = np.zeros((len(h_list), len(eps_list)), dtype=float)
    grid_du = np.zeros((len(h_list), len(eps_list)), dtype=float)
    grid_abs_over_h = np.zeros((len(h_list), len(eps_list)), dtype=float)
    grid_du_over_pc = np.zeros((len(h_list), len(eps_list)), dtype=float)
    for i, h in enumerate(h_list):
        for j, eps in enumerate(eps_list):
            grid_abs[i, j] = ranking[(eps, h)]["max_abs_err"]
            grid_rel[i, j] = ranking[(eps, h)]["max_rel_err"]
            grid_spk[i, j] = ranking[(eps, h)]["spike_amp"]
            grid_du[i, j] = ranking[(eps, h)]["max_du"]
            grid_abs_over_h[i, j] = ranking[(eps, h)]["max_abs_err_over_h"]
            grid_du_over_pc[i, j] = ranking[(eps, h)]["max_du_over_pc"]

    best_abs = np.unravel_index(np.argmin(grid_abs), grid_abs.shape)
    best_rel = np.unravel_index(np.argmin(grid_rel), grid_rel.shape)
    best_spk = np.unravel_index(np.argmin(grid_spk), grid_spk.shape)
    best_du = np.unravel_index(np.argmin(grid_du), grid_du.shape)
    best_abs_over_h = np.unravel_index(np.argmin(grid_abs_over_h), grid_abs_over_h.shape)
    best_du_over_pc = np.unravel_index(np.argmin(grid_du_over_pc), grid_du_over_pc.shape)

    plot_heatmap(fig_dir / "heatmap_metric_eps_h.png", eps_list, h_list, grid_abs, title="worst max_abs_err", best_ij=best_abs)
    plot_heatmap(fig_dir / "heatmap_metric_rel_eps_h.png", eps_list, h_list, grid_rel, title="worst max_rel_err", best_ij=best_rel)
    plot_heatmap(fig_dir / "heatmap_metric_spike_eps_h.png", eps_list, h_list, grid_spk, title="worst spike_amp", best_ij=best_spk)
    plot_heatmap(fig_dir / "heatmap_metric_du_eps_h.png", eps_list, h_list, grid_du, title="worst max|du/dr|", best_ij=best_du)
    plot_heatmap(fig_dir / "heatmap_metric_abs_over_h_eps_h.png", eps_list, h_list, grid_abs_over_h, title="worst max_abs_err/h", best_ij=best_abs_over_h)
    plot_heatmap(fig_dir / "heatmap_metric_du_over_pc_eps_h.png", eps_list, h_list, grid_du_over_pc, title="worst max|du/dr|/pc", best_ij=best_du_over_pc)

    # full runs (topk)
    if args.full:
        sorted_pairs = sorted(ranking.items(), key=lambda kv: kv[1]["max_abs_err"])
        top_pairs = [kv[0] for kv in sorted_pairs[:args.topk]]

        for eps, h in tqdm(top_pairs, desc="full top-k"):
            r_list = []
            u_list = []
            uana_list = []

            for pc in tqdm(PcArr_geom, desc=f"full pc sweep (h={h:.1e}, eps={eps:.1e})", leave=False):
                RArr, MArr, PArr = integrate_tov(pc, K_geom, eps, h, n_center=int(round(args.r_center_target / h)), central_only=False)
                USol = solve_u(RArr, PArr, MArr, K_geom, pc)
                u_num = USol[:, 0]
                u_ana = h00_approx(r=RArr, K=K_geom, pc=pc, a0=1, a8=0) / (RArr**2)

                r_list.append(RArr)
                u_list.append(u_num)
                uana_list.append(u_ana)

                npz_name = full_dir / f"full_pc{fmt_float(pc)}__eps{fmt_float(eps)}__h{fmt_float(h)}.npz"
                np.savez(npz_name, r=RArr, u_num=u_num, u_analytic=u_ana, p=PArr, m=MArr)

            plot_central_grid(
                full_dir / f"full_u__eps{fmt_float(eps)}__h{fmt_float(h)}.png",
                tag="full",
                r_list=r_list,
                u_list=u_list,
                uana_list=uana_list,
                logx=False,
                title=f"full u(r): eps={eps:.2e}, h={h:.2e}",
            )


if __name__ == "__main__":
    main()
