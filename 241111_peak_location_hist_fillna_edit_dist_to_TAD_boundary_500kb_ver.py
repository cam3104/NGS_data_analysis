
# usage : python 241111_peak_location_hist_fillna_edit_dist_to_TAD_boundary_500kb_ver.py [peak_table] [TAD_table] [output_directory] --prefix [prefix] --workers [workers] --chunksize [chunksize]
# y축 값은 Frequency / Total Data Count로 설정 , 직접 설정해야함
# x축 눈금 설정 (사용자 정의 위치와 레이블) : xticks = [-500000, -250000, 0,  250000, 500000]  # 원하는 눈금 위치, xlabels = ['0', '250', '500', '250', '0']  
################################################

import pandas as pd
import multiprocessing as mp
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_location(start, end, tad_start, tad_end):
    tad_mid = (tad_start + tad_end) / 2
    location = 0  # Default value, in case none of the conditions match
    if (start >= tad_start) and (end <= tad_mid):
        location = tad_start - start  
    elif (start >= tad_mid) and (end <= tad_end):
        location = tad_end - end    
    return location

def process_region_numpy(TAD_chunk, peak_region):
    
    peak_results = []

    tad_chrom = TAD_chunk[:, 0]
    tad_start = TAD_chunk[:, 1]
    tad_end = TAD_chunk[:, 2]

    peak_chrom = peak_region[:, 0]
    peak_start = peak_region[:, 1]
    peak_end = peak_region[:, 2]

    for f in range(len(tad_chrom)):
        for i in range(len(peak_chrom)):
            if (tad_chrom[f] == peak_chrom[i]) and (tad_start[f] <= peak_start[i]) and (tad_end[f] >= peak_end[i]):
                peak_location = calculate_location(peak_start[i], peak_end[i], tad_start[f], tad_end[f])
                if peak_location <= 500000 and peak_location >= 0 :
                    peak_results.append({
                        'chrom': peak_chrom[i], 
                        'location': 500000 - peak_location, 
                        'peak_name': peak_region[i, 3], 
                        'tad_start': tad_start[f], 
                        'tad_end': tad_end[f]
                    })
                elif peak_location >= -500000 and peak_location <= 0:
                    peak_results.append({
                        'chrom': peak_chrom[i], 
                        'location': -500000 - peak_location, 
                        'peak_name': peak_region[i, 3], 
                        'tad_start': tad_start[f], 
                        'tad_end': tad_end[f]
                    })

    return peak_results

def find_enhancer_gene_in_TAD_parallel(TAD_table, enhancer_table, num_workers=4, chunksize=1000):
    peak_region = pd.read_csv(enhancer_table, sep="\t", header=None).to_numpy()

    peak_results = []


    with mp.Pool(processes=num_workers) as pool:
        for TAD_chunk in pd.read_csv(TAD_table, sep="\t", header=None, chunksize=chunksize):
            TAD_chunk_np = TAD_chunk.to_numpy()
            results = pool.starmap(process_region_numpy, [(TAD_chunk_np, peak_region)])
            
            for enh_res in results:
                peak_results.extend(enh_res)
                
    return peak_results

def plot_and_save_locations(peak_results, output_directory, prefix):
    peak_df = pd.DataFrame(peak_results)
    peak_locations = peak_df['location'] if not peak_df.empty else []
    plt.figure(figsize=(14, 10))

    # Enhancer locations histogram
    if len(peak_locations) > 0:
        enhancer_hist, peak_bins = np.histogram(peak_locations, bins=100)
        enhancer_hist_series = pd.Series(enhancer_hist)
        peak_total = len(peak_locations)

        peak_moving_avg = enhancer_hist_series.rolling(window=25, center=True, min_periods=1).mean()

        # Adjust moving average by overlap correction
        peak_adjusted = peak_moving_avg / peak_total 

        plt.plot(peak_bins[:-1], peak_adjusted, color='red', label='peak moving avg')
    
    # x축 눈금 설정 (사용자 정의 위치와 레이블)
    xticks = [-500000, -250000, 0,  250000, 500000]  # 원하는 눈금 위치
    xlabels = ['0', '250', '500', '250', '0']  # 원하는 레이블

    plt.xticks(xticks, labels=xlabels, fontsize=12)  # 눈금과 레이블 설정
    plt.ylim(0,0.1)
    plt.xlabel('Distance from TAD boundary(kb)')
    plt.ylabel('Frequency / Total Data Count')
    plt.title('Peak Location Frequency within TADs')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_directory, f"{prefix}_location_histogram_rolling_avg_overlap_500kb_distance_to_TAD_boundary.png")
    plt.savefig(plot_path)
    plt.show()

    peak_csv_path = os.path.join(output_directory, f"{prefix}_peak_locations.csv")


    peak_df.to_csv(peak_csv_path, index=False)
   
    print(f"Plot saved to: {plot_path}")
    print(f"Enhancer data saved to: {peak_csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Process and analyze gene data.")
    parser.add_argument("peak_table", type=str, help="Path to the input file containing peak data.")
    parser.add_argument("TAD_table", type=str, help="Path to the input file containing TAD data.")
    parser.add_argument("output_directory", type=str, help="Directory to store the output results.")
    parser.add_argument("--prefix", type=str, default="output", help="Prefix for the output files.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--chunksize", type=int, default=1000, help="Number of rows to process per chunk.")
    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    peak_results = find_enhancer_gene_in_TAD_parallel(args.TAD_table, args.peak_table, args.workers, args.chunksize)
    
    plot_and_save_locations(peak_results,args.output_directory, args.prefix)

if __name__ == "__main__":
    main()
