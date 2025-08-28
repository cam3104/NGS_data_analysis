#!/usr/bin/env python3
# 250420_correlation_plus.py
"""
Pairwise association analysis with multiple measures and visualizations.

Supported association measures:
- pearson, spearman
- dcor (distance correlation; pure NumPy implementation)
- mi (mutual information; sklearn)
- mic (maximal information coefficient; minepy if available)
- hsic (Gaussian-kernel HSIC; pure NumPy implementation, optional permutation test)

Visualizations:
- scatter (with linear regression line)
- lowess (scatter + LOWESS curve; requires statsmodels, optional)
- hexbin (2D hexbins)
- kde2d (2D KDE contours; requires scipy)

Usage examples:
  python 250823_correlation_plus.py data.tsv --col1 meth_diff --col2 log1p_fc \
    --methods pearson spearman dcor mi hsic --visuals scatter hexbin kde2d --prefix out/foo

  # MIC
  python 250823_correlation_plus.py data.tsv --col1 meth_diff --col2 log1p_fc \
    --methods mic --prefix out/mic_test
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# optional deps
_MI_OK = False
try:
    from sklearn.feature_selection import mutual_info_regression
    _MI_OK = True
except Exception:
    pass

_MIC_OK = False
try:
    from minepy import MINE
    _MIC_OK = True
except Exception:
    pass

_STATS_OK = False
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    _STSATS_OK = True
except Exception:
    _STSATS_OK = False

_SCIPY_KDE_OK = False
try:
    from scipy.stats import gaussian_kde
    _SCIPY_KDE_OK = True
except Exception:
    pass

import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _ensure_1d(x):
    x = np.asarray(x).reshape(-1)
    return x


def _standardize(a):
    a = _ensure_1d(a)
    mu = np.nanmean(a)
    sd = np.nanstd(a)
    if sd == 0 or np.isnan(sd):
        return a * 0.0
    return (a - mu) / sd


# ----------------------------
# Association measures
# ----------------------------
def assoc_pearson(x, y):
    r, p = pearsonr(x, y)
    return {"stat": float(r), "pvalue": float(p)}

def assoc_spearman(x, y):
    r, p = spearmanr(x, y)
    return {"stat": float(r), "pvalue": float(p)}

def assoc_dcor(x, y):
    """
    Distance correlation (Szekely & Rizzo) – biased estimator.
    Returns value in [0,1]. 0 iff independent (for population).
    """
    x = _ensure_1d(x)
    y = _ensure_1d(y)
    # distance matrices
    ax = np.abs(x[:, None] - x[None, :])
    ay = np.abs(y[:, None] - y[None, :])
    # double-centering
    Ax = ax - ax.mean(0)[None, :] - ax.mean(1)[:, None] + ax.mean()
    Ay = ay - ay.mean(0)[None, :] - ay.mean(1)[:, None] + ay.mean()
    # distance covariance / variance
    dCov2 = (Ax * Ay).mean()
    dVarX = (Ax * Ax).mean()
    dVarY = (Ay * Ay).mean()
    denom = np.sqrt(max(dVarX, 0.0) * max(dVarY, 0.0))
    if denom <= 0:
        return {"stat": 0.0, "pvalue": np.nan}
    dCor = np.sqrt(max(dCov2, 0.0)) / np.sqrt(denom)
    return {"stat": float(dCor), "pvalue": np.nan}  # no analytic p-value

def assoc_mi(x, y, n_neighbors=3, n_repeats=5, random_state=0):
    if not _MI_OK:
        return {"stat": np.nan, "pvalue": np.nan, "note": "sklearn not available"}
    x = _ensure_1d(x)
    y = _ensure_1d(y)
    # sklearn MI expects 2D X
    X = x.reshape(-1, 1)
    mi_vals = []
    for seed in range(n_repeats):
        mi = mutual_info_regression(
            X, y, n_neighbors=n_neighbors, random_state=(random_state + seed)
        )[0]
        mi_vals.append(mi)
    return {"stat": float(np.mean(mi_vals)), "pvalue": np.nan}

def assoc_mic(x, y, est="mic_approx"):
    if not _MIC_OK:
        return {"stat": np.nan, "pvalue": np.nan, "note": "minepy not available"}
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(_ensure_1d(x), _ensure_1d(y))
    if est == "mic_approx":
        val = mine.mic()
    else:
        val = mine.mic()
    return {"stat": float(val), "pvalue": np.nan}

def gauss_kernel(a, sigma=None):
    a = _ensure_1d(a)
    if sigma is None:
        # median heuristic on pairwise distances
        d = np.abs(a[:, None] - a[None, :])
        med = np.median(d)
        sigma = med if med > 0 else np.std(a) + 1e-8
    K = np.exp(-((a[:, None] - a[None, :]) ** 2) / (2 * sigma ** 2))
    return K

def assoc_hsic(x, y, sigma_x=None, sigma_y=None, n_perm=0, rng=0):
    """
    Gaussian-kernel HSIC (biased). Optionally permutation p-value.
    """
    x = _ensure_1d(x)
    y = _ensure_1d(y)
    n = len(x)
    H = np.eye(n) - np.ones((n, n)) / n
    Kx = gauss_kernel(x, sigma_x)
    Ky = gauss_kernel(y, sigma_y)
    HKxH = H @ Kx @ H
    HKyH = H @ Ky @ H
    hsic = (HKxH * HKyH).sum() / ((n - 1) ** 2)

    pval = np.nan
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(rng)
        perm_vals = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            Ky_perm = gauss_kernel(y_perm, sigma_y)
            HKyH_perm = H @ Ky_perm @ H
            perm_vals.append((HKxH * HKyH_perm).sum() / ((n - 1) ** 2))
        perm_vals = np.asarray(perm_vals)
        pval = (np.sum(perm_vals >= hsic) + 1) / (n_perm + 1)

    return {"stat": float(hsic), "pvalue": float(pval) if not np.isnan(pval) else np.nan}


# ----------------------------
# Plots
# ----------------------------
def plot_scatter(df, xcol, ycol, outpng):
    plt.figure(figsize=(6, 6))
    x = df[xcol].values
    y = df[ycol].values
    plt.scatter(x, y, s=8, alpha=0.7)
    # linear fit
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() > 2:
        b = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
        yy = b[0] * xx + b[1]
        plt.plot(xx, yy)
    plt.axhline(0, lw=0.7)
    plt.axvline(0, lw=0.7)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(f"Scatter: {xcol} vs {ycol}")
    plt.tight_layout(); plt.savefig(outpng, dpi=300); plt.close()

def plot_lowess(df, xcol, ycol, outpng):
    plt.figure(figsize=(6, 6))
    x = df[xcol].values
    y = df[ycol].values
    plt.scatter(x, y, s=8, alpha=0.6)
    if _STSATS_OK:
        ok = np.isfinite(x) & np.isfinite(y)
        smoothed = lowess(y[ok], x[ok], frac=0.3, return_sorted=True)
        plt.plot(smoothed[:, 0], smoothed[:, 1])
    else:
        plt.text(0.02, 0.98, "statsmodels not available", transform=plt.gca().transAxes,
                 va="top")
    plt.axhline(0, lw=0.7); plt.axvline(0, lw=0.7)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(f"LOWESS: {xcol} vs {ycol}")
    plt.tight_layout(); plt.savefig(outpng, dpi=300); plt.close()

def plot_hexbin(df, xcol, ycol, outpng, gridsize=40):
    plt.figure(figsize=(6, 6))
    plt.hexbin(df[xcol], df[ycol], gridsize=gridsize, mincnt=1)
    plt.axhline(0, lw=0.7); plt.axvline(0, lw=0.7)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(f"Hexbin: {xcol} vs {ycol}")
    cb = plt.colorbar()
    cb.set_label("count")
    plt.tight_layout(); plt.savefig(outpng, dpi=300); plt.close()

def plot_kde2d(df, xcol, ycol, outpng, levels=6):
    plt.figure(figsize=(6, 6))
    x = df[xcol].values; y = df[ycol].values
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    plt.scatter(x, y, s=4, alpha=0.4)
    if _SCIPY_KDE_OK and len(x) > 10:
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                             np.linspace(ymin, ymax, 200))
        xy = np.vstack([xx.ravel(), yy.ravel()])
        kde = gaussian_kde(np.vstack([x, y]))
        zz = kde(xy).reshape(xx.shape)
        cs = plt.contour(xx, yy, zz, levels=levels)
        plt.clabel(cs, inline=1, fontsize=8)
    else:
        plt.text(0.02, 0.98, "scipy KDE unavailable", transform=plt.gca().transAxes,
                 va="top")
    plt.axhline(0, lw=0.7); plt.axvline(0, lw=0.7)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(f"2D KDE: {xcol} vs {ycol}")
    plt.tight_layout(); plt.savefig(outpng, dpi=300); plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Flexible correlation/association analysis with visuals.")
    ap.add_argument("input_file", help="Input table (TSV by default)")
    ap.add_argument("--col1", required=True, help="First column")
    ap.add_argument("--col2", required=True, help="Second column")
    ap.add_argument("--sep", default="\t", help="Field separator (default: tab)")
    ap.add_argument("--prefix", default="corr_out/out", help="Output prefix (dir/fileprefix)")
    ap.add_argument("--methods", nargs="+",
                    default=["pearson", "spearman"],
                    choices=["pearson", "spearman", "dcor", "mi", "mic", "hsic"],
                    help="Association measures to compute")
    ap.add_argument("--visuals", nargs="+",
                    default=["scatter"],
                    choices=["scatter", "lowess", "hexbin", "kde2d"],
                    help="Visual aids to save")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with NA in either column")
    ap.add_argument("--standardize", action="store_true",
                    help="Z-score standardize both columns before analysis")
    ap.add_argument("--hsic-perm", type=int, default=0,
                    help="HSIC permutation count for p-value (0 = skip)")
    ap.add_argument("--mi-neighbors", type=int, default=3, help="n_neighbors for MI")
    args = ap.parse_args()

    df = pd.read_csv(args.input_file, sep=args.sep)
    if args.dropna:
        df = df[[args.col1, args.col2]].dropna().copy()

    x = df[args.col1].values
    y = df[args.col2].values
    if args.standardize:
        x = _standardize(x)
        y = _standardize(y)

    # ensure output dir
    out_prefix = Path(args.prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for m in args.methods:
        if m == "pearson":
            results["pearson"] = assoc_pearson(x, y)
        elif m == "spearman":
            results["spearman"] = assoc_spearman(x, y)
        elif m == "dcor":
            results["dcor"] = assoc_dcor(x, y)
        elif m == "mi":
            results["mi"] = assoc_mi(x, y, n_neighbors=args.mi_neighbors)
        elif m == "mic":
            results["mic"] = assoc_mic(x, y)
        elif m == "hsic":
            results["hsic"] = assoc_hsic(x, y, n_perm=args.hsic_perm)
        else:
            continue

    # write out results
    txt_path = f"{out_prefix}_assoc.txt"
    json_path = f"{out_prefix}_assoc.json"
    with open(txt_path, "w") as f:
        f.write(f"# columns: {args.col1} vs {args.col2}\n")
        for k, v in results.items():
            note = f"  note={v.get('note')}" if "note" in v else ""
            f.write(f"{k}\tstat={v['stat']:.6g}\tpvalue={v['pvalue']}{note}\n")
    with open(json_path, "w") as f:
        json.dump({"col1": args.col1, "col2": args.col2, "results": results}, f, indent=2)

    # visuals
    for vis in args.visuals:
        if vis == "scatter":
            plot_scatter(df, args.col1, args.col2, f"{out_prefix}_scatter.png")
        elif vis == "lowess":
            plot_lowess(df, args.col1, args.col2, f"{out_prefix}_lowess.png")
        elif vis == "hexbin":
            plot_hexbin(df, args.col1, args.col2, f"{out_prefix}_hexbin.png")
        elif vis == "kde2d":
            plot_kde2d(df, args.col1, args.col2, f"{out_prefix}_kde2d.png")

    print(f"[done] results -> {txt_path}, {json_path}")
    print(f"[done] figures -> {out_prefix}_*.png")


if __name__ == "__main__":
    main()
