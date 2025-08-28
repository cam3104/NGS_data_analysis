#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA plotting utility
- Choose whether samples are in rows or columns
- Read a meta file to color points by condition
- Save PCA scores, loadings, and a PNG plot

Example usages
1) Samples are rows, first column is sample ID, meta without header
   python 250818_draw_PCA_refactored.py \
     --input matrix.tsv --samples-axis rows --index-col 0 \
     --meta-file meta.tsv --output-dir out --prefix test

2) Samples are columns, first row has sample IDs, meta has header 'sample' and 'condition'
   python 250818_draw_PCA_refactored.py \
     --input matrix.tsv --samples-axis columns --index-col 0 \
     --meta-file meta_header.tsv --meta-has-header \
     --meta-sample-col sample --meta-condition-col condition \
     --output-dir out --prefix test
"""

import argparse
import os
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ------------------------------- IO helpers ------------------------------- #

def _parse_index_col(idx: Optional[str]) -> Optional[Union[int, str]]:
    if idx is None:
        return 0  # default to first column as ID
    # allow integer-like strings
    try:
        return int(idx)
    except (TypeError, ValueError):
        return idx


def load_matrix(path: str, samples_axis: str, index_col: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=0)

    idx = _parse_index_col(index_col)
    if idx is not None:
        # If name provided and exists, set it; if int within bounds, set by position
        if isinstance(idx, str) and idx in df.columns:
            df = df.set_index(idx)
        elif isinstance(idx, int) and 0 <= idx < df.shape[1]:
            df = df.set_index(df.columns[idx])
        # else: leave as-is

    # If samples are in columns, transpose to make rows = samples
    if samples_axis == "columns":
        df = df.T

    # After here: rows are samples, columns are features
    # Coerce to numeric, allowing some non-numeric columns to be dropped
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    # Drop columns that are entirely NaN after coercion
    df_numeric = df_numeric.dropna(axis=1, how="all")
    # Fill any remaining NaNs with 0 by default (controlled by CLI later if desired)
    return df_numeric


def load_meta(
    path: str,
    has_header: bool,
    sep: str,
    sample_col: str,
    condition_col: str,
) -> pd.DataFrame:
    if has_header:
        meta = pd.read_csv(path, sep=sep)
    else:
        meta = pd.read_csv(path, sep=sep, header=None, names=[sample_col, condition_col])
    # ensure string dtype for safety
    meta[sample_col] = meta[sample_col].astype(str)
    meta[condition_col] = meta[condition_col].astype(str)
    meta = meta.dropna(subset=[sample_col, condition_col])
    meta = meta.drop_duplicates(subset=[sample_col])
    return meta


# ------------------------------- PCA core ------------------------------- #

def run_pca(
    X: pd.DataFrame,
    n_components: int = 2,
    scale: bool = True,
    var_threshold: Optional[float] = None,
    topk_features: Optional[int] = None,
    random_state: Optional[int] = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Parameters
    X              rows are samples, columns are features
    var_threshold   remove features with variance < threshold (computed on unscaled data)
    topk_features   keep only top-k most variable features (on unscaled data)
    Returns
    scores_df       dataframe indexed by sample with PCA components
    loadings_df     dataframe indexed by feature with component loadings
    explained_var   array of explained variance ratio
    """
    # Feature filtering on unscaled data
    X0 = X.copy()
    if var_threshold is not None:
        variances = X0.var(axis=0, ddof=1)
        keep = variances[variances >= var_threshold].index
        X0 = X0[keep]
    if topk_features is not None and topk_features > 0 and X0.shape[1] > topk_features:
        variances = X0.var(axis=0, ddof=1)
        topk_idx = variances.sort_values(ascending=False).head(topk_features).index
        X0 = X0[topk_idx]

    # Fill remaining NaNs after filtering
    X0 = X0.fillna(0.0)

    # Scale if requested
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = scaler.fit_transform(X0)
    else:
        # center only to make PCA equivalent to correlation if needed
        X_scaled = X0.values - X0.values.mean(axis=0, keepdims=True)

    # PCA
    n_components = min(n_components, min(X_scaled.shape))
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=random_state)
    scores = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    # Prepare outputs
    comp_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, index=X0.index, columns=comp_cols)

    loadings = pca.components_.T  # features x components
    loadings_df = pd.DataFrame(loadings, index=X0.columns, columns=comp_cols)

    return scores_df, loadings_df, explained


# ------------------------------- Plotting ------------------------------- #

def plot_pca(
    scores_df: pd.DataFrame,
    meta: pd.DataFrame,
    sample_col: str,
    condition_col: str,
    prefix: str,
    output_dir: str,
    annotate: bool = False,
    style_col: Optional[str] = None,
    palette: Optional[str] = "tab10",
    explained: Optional[np.ndarray] = None,
) -> str:
    # Merge scores with meta
    scores_df = scores_df.copy()
    scores_df[sample_col] = scores_df.index.astype(str)
    plot_df = scores_df.merge(meta[[sample_col, condition_col] + ([style_col] if style_col and style_col in meta.columns else [])],
                              on=sample_col, how="inner")

    if plot_df.empty:
        raise ValueError("No overlapping samples between matrix and meta. Check IDs and axis settings.")

    # palette based on number of conditions
    n_cond = plot_df[condition_col].nunique()
    colors = sns.color_palette(palette, n_colors=n_cond)
    cond_order = sorted(plot_df[condition_col].unique())
    color_map = dict(zip(cond_order, colors))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue=condition_col,
        style=(style_col if style_col and style_col in plot_df.columns else None),
        palette=color_map,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        s=60,
    )

    if annotate:
        for _, r in plot_df.iterrows():
            plt.text(r["PC1"], r["PC2"], str(r[sample_col]), fontsize=8, ha="left", va="bottom")

    if explained is not None and len(explained) >= 2:
        xlab = f"PC1 ({explained[0]*100:.1f}%)"
        ylab = f"PC2 ({explained[1]*100:.1f}%)"
    else:
        xlab, ylab = "PC1", "PC2"

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(f"PCA — {prefix}")
    plt.legend(title=condition_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{prefix}_PCA.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    return out_png


# ------------------------------- Main ------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Draw PCA with row/column orientation and meta-based coloring")

    # Inputs
    ap.add_argument("--input", required=True, help="Input matrix TSV. If samples are columns, use --samples-axis columns")
    ap.add_argument("--meta-file", required=True, help="Meta file: two columns [sample, condition] by default")

    # Orientation
    ap.add_argument("--samples-axis", choices=["rows", "columns"], default="rows",
                    help="Where the samples live in the input matrix")
    ap.add_argument("--index-col", default="0",
                    help="Column index or name to use as sample ID before orientation fix. Default 0 (first column)")

    # Meta options
    ap.add_argument("--meta-has-header", action="store_true", help="Set if the meta file has a header row")
    ap.add_argument("--meta-sep", default="\t", help="Separator for meta file. Default: tab")
    ap.add_argument("--meta-sample-col", default="sample", help="Sample ID column in meta")
    ap.add_argument("--meta-condition-col", default="condition", help="Condition column in meta")
    ap.add_argument("--meta-style-col", default=None, help="Optional column in meta to map to point style")

    # PCA params
    ap.add_argument("--n-components", type=int, default=2, help="Number of PCA components to compute")
    ap.add_argument("--no-scale", action="store_true", help="Disable standard scaling before PCA")
    ap.add_argument("--var-threshold", type=float, default=None, help="Drop features with variance < threshold")
    ap.add_argument("--topk-features", type=int, default=None, help="Keep only top-k most variable features")

    # Plot/output
    ap.add_argument("--annotate", action="store_true", help="Annotate points with sample IDs")
    ap.add_argument("--palette", default="tab10", help="Seaborn palette name")
    ap.add_argument("--output-dir", default=".", help="Output directory")
    ap.add_argument("--prefix", default="PCA", help="Output prefix")

    args = ap.parse_args()

    # Load and orient matrix (rows=samples after this step)
    X = load_matrix(args.input, args.samples_axis, args.index_col)
    # Ensure sample IDs are strings for merging
    X.index = X.index.astype(str)

    # Load meta
    meta = load_meta(
        args.meta_file,
        has_header=args.meta_has_header,
        sep=args.meta_sep,
        sample_col=args.meta_sample_col,
        condition_col=args.meta_condition_col,
    )

    # Intersect by sample
    common = sorted(set(X.index).intersection(set(meta[args.meta_sample_col].astype(str))))
    if len(common) == 0:
        raise SystemExit("No overlapping samples between matrix and meta. Check --samples-axis, --index-col, and meta IDs.")

    X = X.loc[common]

    # PCA
    scores_df, loadings_df, explained = run_pca(
        X,
        n_components=args.n_components,
        scale=not args.no_scale,
        var_threshold=args.var_threshold,
        topk_features=args.topk_features,
        random_state=0,
    )

    # Plot
    out_png = plot_pca(
        scores_df,
        meta,
        sample_col=args.meta_sample_col,
        condition_col=args.meta_condition_col,
        prefix=args.prefix,
        output_dir=args.output_dir,
        annotate=args.annotate,
        style_col=args.meta_style_col,
        palette=args.palette,
        explained=explained,
    )

    # Save tables
    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, f"{args.prefix}_PCA_scores.tsv")
    loadings_path = os.path.join(args.output_dir, f"{args.prefix}_PCA_loadings.tsv")
    evr_path = os.path.join(args.output_dir, f"{args.prefix}_PCA_explained_variance.tsv")

    scores_df.to_csv(scores_path, sep='\t')
    loadings_df.to_csv(loadings_path, sep='\t')
    pd.Series(explained, index=[f"PC{i+1}" for i in range(len(explained))]).to_csv(evr_path, sep='\t', header=False)

    print(f"Saved plot: {out_png}")
    print(f"Saved scores: {scores_path}")
    print(f"Saved loadings: {loadings_path}")
    print(f"Saved explained variance: {evr_path}")


if __name__ == "__main__":
    main()
