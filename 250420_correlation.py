    # Example usage:
    # python correlation_analysis.py input_file.txt col1 col2
    # where input_file.txt is the input file and col1 and col2 are the column names

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def calculate_correlation(input_file, col1, col2,prefix):
    # Load the data
    df = pd.read_csv(input_file, sep="\t")

    # Calculate Pearson and Spearman correlation between MDV_sum and PC2
    pearson_corr, p_pearson = pearsonr(df[col1], df[col2])
    spearman_corr, p_spearman = spearmanr(df[col1], df[col2])

    # Print the results
    print(f'Pearson correlation between {col1} and {col2}: {pearson_corr:.4f}, p-value: {p_pearson:.4e}')
    print(f'Spearman correlation between {col1} and {col2}: {spearman_corr:.4f}, p-value: {p_spearman:.4e}')
    #save the results to a file
    with open(f'{prefix}_correlation_results.txt', 'w') as f:
        f.write(f'Pearson correlation between {col1} and {col2}: {pearson_corr:.4f}, p-value: {p_pearson:.4e}\n')
        f.write(f'Spearman correlation between {col1} and {col2}: {spearman_corr:.4f}, p-value: {p_spearman:.4e}\n')
    

def plot_correlation(df, col1, col2,prefix):
    # Plotting the scatter plot with regression line
    plt.figure(figsize=(8, 6))
    sns.regplot(x=col1, y=col2, data=df, scatter_kws={'s': 10}, line_kws={'color': 'red'})
    # Set the title and labels
    plt.title(f'correlation plot of {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.grid(True)
    #save the plot
    plt.savefig(f'{prefix}_correlation_plot.png', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate and plot correlation between two columns in a file.')
    parser.add_argument('input_file', type=str, help='Input file path')
    parser.add_argument('--col1', type=str, help='First column name')
    parser.add_argument('--col2', type=str, help='Second column name')
    parser.add_argument('--prefix', type=str, help='Prefix for output files')
    args = parser.parse_args()

    # Calculate correlation
    calculate_correlation(args.input_file, args.col1, args.col2,args.prefix)

    # Load the data again for plotting
    df = pd.read_csv(args.input_file, sep="\t")
    
    # Plot correlation
    plot_correlation(df, args.col1, args.col2,args.prefix)

if __name__ == "__main__":
    main()

    