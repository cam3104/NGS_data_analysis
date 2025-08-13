#usage: python3 250120_draw_GO_plot.py input_file  output_path -t 2000 -p prefix
#######################################
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
from datetime import datetime
import matplotlib as mpl
import matplotlib.ticker as ticker

def process_and_plot_go_data(input_go_file, output_directory,term_size_threshold, prefix):
    # Attempt to read the input file with various separators
    sep_options = [",", "\t", ";"]
    for sep in sep_options:
        try:
            go_data = pd.read_csv(input_go_file, sep=sep)
            break  # Successfully read the file
        except pd.errors.ParserError:
            continue
    else:
        raise ValueError(
            "Unable to parse the input file with the provided separators. Please check the file format."
        )

    # Validate required columns
    required_columns = ["source","term_name", "negative_log10_of_adjusted_p_value","term_size" ,"intersection_size","query_size"]
    if not all(col in go_data.columns for col in required_columns):
        raise ValueError(
            f"Input file must contain the following columns: {required_columns}"
        )
    # filter with GO:BP & given term_size thereshold
    go_data = go_data[go_data["source"].str.contains("GO:BP")]
    go_data = go_data[go_data["term_size"] < term_size_threshold]
    go_data["gene_ratio"] =round(go_data["intersection_size"] / go_data["query_size"],2) 

    # Sort data by gene_ratio in descending order
    go_data = go_data.sort_values(
        by="negative_log10_of_adjusted_p_value", ascending=False
    )

    # Prepare output file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(
        output_directory, f"{prefix}_{timestamp}_GO_result.png"
    )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))  # Increase figure size for better spacing
    ax.scatter(
        go_data["negative_log10_of_adjusted_p_value"],
        go_data["term_name"],
        c=go_data["gene_ratio"],  # Color by adjusted p-value
        s=go_data["intersection_size"]*3,  # Scale dot size by intersection size
        cmap="viridis",
        alpha=0.8,
        edgecolors="black"
    )

    # Create size legend using proxy artists
    size_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='5', markersize=5, markerfacecolor='gray'),
        Line2D([0], [0], marker='o', color='w', label='20', markersize=10, markerfacecolor='gray'),
        Line2D([0], [0], marker='o', color='w', label='50', markersize=15, markerfacecolor='gray'),
        Line2D([0], [0], marker='o', color='w', label='100', markersize=20, markerfacecolor='gray'),
    ]

    # Add a legend for size
    ax.legend(
        handles=size_legend_elements,
        title="Intersection Size",
        loc="upper left",  # Move the legend to the upper left
        bbox_to_anchor=(1, 1),  # Shift further to the right of the plot
        labelspacing=0.8,  # Spacing between labels
        borderpad=1.2,  # Padding between the border and the content
        frameon=True,
        fontsize=12,
        title_fontsize=12,
    )

    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.25])  # [left, bottom, width, height]
    cbar = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap="viridis",
        orientation="vertical",  # Set to vertical
        norm=mpl.colors.Normalize(vmin=go_data["gene_ratio"].min(),
                              vmax=go_data["gene_ratio"].max())
    )
    cbar.set_label("Gene ratio", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Add axis labels and title
    ax.set_xlabel("-log(adjusted p-value)", fontsize=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.set_ylabel("GO term", fontsize=14)
    ax.set_title("Gene Ontology Analysis", fontsize=14)
    ax.invert_yaxis()  # Reverse order for better readability
    ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent overlap
    fig.subplots_adjust(bottom=0.2, top=0.85, left=0.2, right=0.9)  # Adjust margins 
    # 시각적 이해
    #Figure 좌표 (0, 0) ~ (1, 1):

    #(0, 0): Figure의 왼쪽 아래 모서리.
    #(1, 1): Figure의 오른쪽 위 모서리.
    #Axes 위치:

    #left=0.92: Figure 오른쪽으로 92% 지점에서 시작.
    #bottom=0.3: 아래에서 30% 지점에서 시작.
    #width=0.02: 너비는 figure 전체의 2%를 차지.
    #height=0.4: 높이는 figure 전체의 40%를 차지.
    
    plt.savefig(output_file_path, bbox_inches="tight")  # Save figure with adjusted bounding box
    print(f"GO result saved to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Process and visualize Gene Ontology (GO) data. "
            "Ensure the input is a valid CSV file with columns like 'term_name' and 'negative_log10_of_adjusted_p_value'."
        )
    )
    parser.add_argument(
        "input_go_file", type=str, help="Path to the input file containing GO data."
    )
    parser.add_argument(
        "output_directory", type=str, help="Directory to store the output results."
    )
    parser.add_argument(
        "-t","--term_size_threshold",
        type=int,
        default=5000,
        help="Threshold for term size to filter GO terms.",
    )
    parser.add_argument(
        "-p","--prefix", type=str, default="output", help="Prefix for the output files."
    )
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_go_file):
        raise FileNotFoundError(f"Input file not found: {args.input_go_file}")

    # Create output directory if it does not exist
    try:
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
    except OSError as e:
        raise RuntimeError(f"Failed to create the output directory: {e}")

    # Process and visualize GO data
    process_and_plot_go_data(
        args.input_go_file, args.output_directory,args.term_size_threshold, args.prefix
    )

if __name__ == "__main__":
    main()
