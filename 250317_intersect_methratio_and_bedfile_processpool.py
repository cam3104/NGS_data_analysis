
# ✅ 글로벌 변수 설정 함수
#ProcessPoolExecutor를 사용할 때, 각 프로세스는 독립적인 메모리 공간을 가지므로 reference_df를 여러 번 읽어야 하는 문제가 발생합니다.

#만약 각 프로세스가 매번 reference_df를 읽어들인다면:
#메모리 낭비: reference_df가 크다면, 각 프로세스마다 중복 저장되므로 메모리가 크게 소비됨.
#속도 저하: 각 프로세스가 같은 파일을 여러 번 읽고 변환하면 불필요한 시간이 소요됨.
#이를 방지하기 위해, 한 번만 읽어들인 reference_df를 전역 변수로 설정하여 각 프로세스에서 공유하도록 합니다.
import pandas as pd
import numpy as np
import os
from datetime import datetime
import argparse
import re
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from bisect import bisect_left
from collections import defaultdict

# ✅ reference를 chrom별로 나누고 정렬된 dict로 저장
reference_dict = {}
reference_starts = {}

def init_worker(ref_df):
    global reference_dict, reference_starts
    reference_dict = defaultdict(pd.DataFrame)
    reference_starts = {}

    for chrom in ref_df['chrom'].unique():
        chrom_df = ref_df[ref_df['chrom'] == chrom].copy()
        chrom_df.sort_values("start", inplace=True)
        reference_dict[chrom] = chrom_df.reset_index(drop=True)
        reference_starts[chrom] = chrom_df['start'].values

# ✅ 이진 탐색 기반 겹침 검색

def find_overlap(chrom, pos):
    if chrom not in reference_dict:
        return pd.DataFrame()

    starts = reference_starts[chrom]
    idx = bisect_left(starts, pos)
    # idx가 0보다 작을 경우를 방지하기 위해 max(0, idx - 50)으로 설정
    # idx가 reference_dict[chrom]의 길이보다 클 경우를 방지하기 위해 idx + 50으로 설정
    # 50개의 범위를 설정하여 성능을 개선
    # 이진 탐색을 통해 pos보다 크거나 같은 첫 번째 위치를 찾음
    # 그 위치의 인덱스를 기준으로 앞뒤 50개 범위를 가져옴
    # 이 범위에서 pos와 겹치는 구간을 찾음
    window = reference_dict[chrom].iloc[max(0, idx - 50): idx + 50]
    return window[(window['start'] <= pos) & (window['end'] >= pos)]

# ✅ 병렬 처리 함수

def intersect_chunk(chunk):
    results = []
    for row in chunk:
        chrom, start, strand, context, *data = row
        if pd.isna(start):
            continue

        overlap = find_overlap(chrom, start)
        if not overlap.empty:
            results.append([chrom, start, strand, context, overlap.shape[0], *data])

    return results if results else [[]]


def read_and_split(chunk):
    position_split = chunk["Position"].str.split(":", expand=True)
    position_split.columns = ["chrom", "start", "strand", "context"]
    position_split["start"] = pd.to_numeric(position_split["start"], errors="coerce")
    chunk = pd.concat([position_split, chunk.iloc[:, 1:]], axis=1)
    return chunk


def process_data(input_file, output_dir, prefix, reference_file, chunk_size, n_cores):
    try:
        input_df = pd.read_csv(input_file, sep="\t", chunksize=chunk_size)
        ref_df = pd.read_csv(reference_file, sep="\t")
        ref_df["start"] = pd.to_numeric(ref_df["start"], errors="coerce").astype("Int64")
        ref_df["end"] = pd.to_numeric(ref_df["end"], errors="coerce").astype("Int64")

    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    output_path = os.path.join(output_dir, f"{prefix}_result.txt")
    total_chunks = 0

    print("Pre-processing completed. Starting parallel processing...")
    start_time = time.time()

    with open(output_path, "w") as out_file:
        with ProcessPoolExecutor(max_workers=n_cores, initializer=init_worker, initargs=(ref_df,)) as executor:
            for chunk in tqdm(input_df, desc="Processing", unit="chunk"):
                chunk_split = read_and_split(chunk)
                split_data = np.array_split(chunk_split.to_numpy(), n_cores)

                futures = {executor.submit(intersect_chunk, c): c for c in split_data}
                results = []
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.extend(result)

                results = [r for r in results if len(r) > 0]
                if len(results) == 0:
                    continue

                processed_data = np.vstack(results)
                columns = ["chrom", "start", "strand", "context", "overlap"] + list(chunk_split.columns[4:])
                result_df = pd.DataFrame(processed_data, columns=columns[:processed_data.shape[1]])
                result_df.to_csv(out_file, sep='\t', index=False, mode="a", header=False)

                total_chunks += 1
                elapsed = time.time() - start_time
                eta = (elapsed / total_chunks) * (1045 - total_chunks)  # 1,045 chunks 기준 ETA 예측
                print(f"Chunk {total_chunks}/1045 processed. Elapsed: {elapsed:.1f}s | ETA: {eta/60:.1f} min")

    print("\nProcessing completed.")
    total_time = time.time() - start_time
    print(f"Total elapsed time: {total_time:.1f} seconds (~{total_time/60:.1f} minutes)")


def main():
    parser = argparse.ArgumentParser(description="Efficient methylation-overlap analysis.")
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("output_dir", type=str, help="Directory to store the output results.")
    parser.add_argument("--prefix", type=str, default="output", help="Prefix for the output files.")
    parser.add_argument("--reference_df", type=str, help="Reference dataframe for intersection.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="Number of rows per chunk.")
    parser.add_argument("--n_cores", type=int, default=24, help="Number of parallel workers.")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    process_data(args.input_file, args.output_dir, args.prefix, args.reference_df, args.chunksize, args.n_cores)


if __name__ == "__main__":
    main()
