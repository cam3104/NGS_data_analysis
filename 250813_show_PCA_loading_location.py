#!/usr/bin/env python3
# usage: python3 250813_show_PCA_loading_location.py -i <input_file> -o <output_image> 
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_loading(df: pd.DataFrame,
                 title: str,
                 out_path: Path,
                 feature_col: str = "Unnamed: 0",
                 x_col: str = "PC1_loading",
                 y_col: str = "PC2_loading",
                 text_dx: float = 0.02,
                 figsize=(6, 6),
                 legend: bool = True,
                 dpi: int = 300):
    # feature & coordinates
    features = df[feature_col].astype(str).tolist()
    pc1 = df[x_col].astype(float).tolist()
    pc2 = df[y_col].astype(float).tolist()

    
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colors = {feat: palette[i % len(palette)] for i, feat in enumerate(features)}

    fig, ax = plt.subplots(figsize=figsize)

    
    for feat, x, y in zip(features, pc1, pc2):
        ax.scatter(x, y, s=100, color=colors[feat])
        ax.text(x + text_dx, y, feat, fontsize=10)

    
    ax.axhline(0, color="gray", linewidth=1, linestyle="--", zorder=0)
    ax.axvline(0, color="gray", linewidth=1, linestyle="--", zorder=0)

    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    if legend:
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              label=f, markerfacecolor=colors[f], markersize=10)
                   for f in features]
        ax.legend(handles=handles, title="Feature", loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot PCA loading (PC1 vs PC2) from a single file.")
    parser.add_argument("-i", "--input", required=True, help="Input PCA loading TSV/CSV file")
    parser.add_argument("-o", "--output", required=True, help="Output image file path")
    parser.add_argument("--feature_col", default="Unnamed: 0", help="Feature column name")
    parser.add_argument("--x_col", default="PC1_loading", help="X-axis column name")
    parser.add_argument("--y_col", default="PC2_loading", help="Y-axis column name")
    parser.add_argument("--text_dx", type=float, default=0.02, help="Text label X-offset (default: 0.02)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6, 6), help="Figure size in inches")
    parser.add_argument("--dpi", type=int, default=300, help="Image DPI")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output format")
    parser.add_argument("--sep", default="\t", help="File separator (default: tab)")
    parser.add_argument("--no_legend", action="store_true", help="Do not show legend")
    args = parser.parse_args()

    
    df = pd.read_csv(args.input, sep=args.sep)

    # title and output path
    title = Path(args.input).stem
    out_path = Path(args.output)


    plot_loading(
        df=df,
        title=title,
        out_path=out_path,
        feature_col=args.feature_col,
        x_col=args.x_col,
        y_col=args.y_col,
        text_dx=args.text_dx,
        figsize=tuple(args.figsize),
        legend=not args.no_legend,
        dpi=args.dpi
    )

    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
