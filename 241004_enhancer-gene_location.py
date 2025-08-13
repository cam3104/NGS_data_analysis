import pandas as pd
import multiprocessing as mp
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

def calculate_location(start, end, tad_start, tad_end):
    tad_size = tad_end - tad_start
    tad_mid = (tad_start + tad_end) / 2
    location = ((start + end) / 2 - tad_mid) / tad_size * 200
    return location

def process_region_numpy(TAD_chunk, enhancer_region):
    enhancer_results = []
    gene_results = []

    # NumPy 배열로 변환하여 처리
    tad_chrom = TAD_chunk[:, 0]
    tad_start = TAD_chunk[:, 1]
    tad_end = TAD_chunk[:, 2]

    enh_chrom = enhancer_region[:, 0]
    enh_start = enhancer_region[:, 1]
    enh_end = enhancer_region[:, 2]

    gene_chrom = enhancer_region[:, 5]
    gene_start = enhancer_region[:, 6]
    gene_end = enhancer_region[:, 7]
    
    for f in range(len(tad_chrom)):
        for i in range(len(enh_chrom)):
            # Enhancer 위치 계산
            if (tad_chrom[f] == enh_chrom[i]) and (tad_start[f] <= enh_start[i]) and (tad_end[f] >= enh_end[i]):
                enh_location = calculate_location(enh_start[i], enh_end[i], tad_start[f], tad_end[f])
                enhancer_results.append({'chrom': enh_chrom[i], 'location': enh_location, 'enhancer': enhancer_region[i, 6]})

            # Gene 위치 계산
            if (tad_chrom[f] == gene_chrom[i]) and (tad_start[f] <= gene_start[i]) and (tad_end[f] >= gene_end[i]):
                gene_location = calculate_location(gene_start[i], gene_end[i], tad_start[f], tad_end[f])
                gene_results.append({'chrom': gene_chrom[i], 'location': gene_location, 'gene': enhancer_region[i, 7]})

    return enhancer_results, gene_results

def find_enhancer_gene_in_TAD_parallel(TAD_table, enhancer_table, num_workers=4, chunksize=1000):
    # enhancer_table의 데이터를 NumPy 배열로 변환
    enhancer_region = pd.read_csv(enhancer_table, sep="\t", header=None).to_numpy()

    enhancer_results = []
    gene_results = []

    # multiprocessing을 사용하여 각 청크를 병렬 처리
    with mp.Pool(processes=num_workers) as pool:
        # TAD 데이터를 청크 단위로 읽고 NumPy 배열로 변환
        for TAD_chunk in pd.read_csv(TAD_table, sep="\t", header=None, chunksize=chunksize):
            TAD_chunk_np = TAD_chunk.to_numpy()
            # 각 청크를 병렬 처리
            results = pool.starmap(process_region_numpy, [(TAD_chunk_np, enhancer_region)])
            
            # 결과 합치기
            for enh_res, gene_res in results:
                enhancer_results.extend(enh_res)
                gene_results.extend(gene_res)

    return enhancer_results, gene_results

def plot_and_save_locations(enhancer_results, gene_results, output_directory, prefix):
    enhancer_df = pd.DataFrame(enhancer_results)
    gene_df = pd.DataFrame(gene_results)

    # 빈도수를 계산할 위치 데이터
    enhancer_locations = enhancer_df['location'] if not enhancer_df.empty else []
    gene_locations = gene_df['location'] if not gene_df.empty else []

    plt.figure(figsize=(10, 6))

    # Enhancer 위치에 대한 히스토그램 계산 후, scatter로 표시
    if len(enhancer_locations) > 0:
        enhancer_hist, enhancer_bins = np.histogram(enhancer_locations, bins=1000)
        enhancer_bin_centers = (enhancer_bins[:-1] + enhancer_bins[1:]) / 2
        plt.scatter(enhancer_bin_centers, enhancer_hist, color='blue', label='Enhancers', alpha=0.6)

    # Gene 위치에 대한 히스토그램 계산 후, scatter로 표시
    if len(gene_locations) > 0:
        gene_hist, gene_bins = np.histogram(gene_locations, bins=1000)
        gene_bin_centers = (gene_bins[:-1] + gene_bins[1:]) / 2
        plt.scatter(gene_bin_centers, gene_hist, color='red', label='Genes', alpha=0.6)

    plt.xlabel('Location within TAD(normalized)')
    plt.ylabel('Frequency')
    plt.xlim(-100, 100)
    plt.title('Enhancer and Gene Location Frequency (Scatter Histogram)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(output_directory, f"{prefix}_scatter_histogram_plot.png")
    plt.savefig(plot_path)
    plt.show()

    # Save the DataFrames as CSV files
    enhancer_csv_path = os.path.join(output_directory, f"{prefix}_enhancer_locations.csv")
    gene_csv_path = os.path.join(output_directory, f"{prefix}_gene_locations.csv")

    enhancer_df.to_csv(enhancer_csv_path, index=False)
    gene_df.to_csv(gene_csv_path, index=False)

    print(f"Plot saved to: {plot_path}")
    print(f"Enhancer data saved to: {enhancer_csv_path}")
    print(f"Gene data saved to: {gene_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Process and analyze gene data.")
    parser.add_argument("enhancer_table", type=str, help="Path to the input file containing enhancer data.")
    parser.add_argument("TAD_table", type=str, help="Path to the input file containing TAD data.")
    parser.add_argument("output_directory", type=str, help="Directory to store the output results.")
    parser.add_argument("--prefix", type=str, default="output", help="Prefix for the output files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--chunksize", type=int, default=1000, help="Number of rows to process per chunk.")
    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    enhancer_results, gene_results = find_enhancer_gene_in_TAD_parallel(args.TAD_table, args.enhancer_table, args.workers, args.chunksize)
    
    plot_and_save_locations(enhancer_results, gene_results, args.output_directory, args.prefix)

if __name__ == "__main__":
    main()
