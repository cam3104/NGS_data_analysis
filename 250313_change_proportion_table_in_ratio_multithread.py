import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def compute_ratios_chunk(chunk):
    """각 데이터 청크에서 (짝수 열 / 홀수 열) 계산"""
    numerator = chunk[:, 1::2]  # 짝수 열 (나누어지는 값)
    denominator = chunk[:, :-1:2]  # 홀수 열 (나누는 값)

    # NaN 값을 0으로 변환하여 안정적인 계산 수행
    numerator = np.nan_to_num(numerator, nan=0.0)
    denominator = np.nan_to_num(denominator, nan=0.0)

    # 0으로 나누는 경우 방지
    return np.divide(numerator, denominator, where=(denominator != 0), out=np.zeros_like(numerator))

def change_data(input_file, output_dir, prefix, n_cores, chunk_size):
    """대용량 데이터를 청크 단위로 읽으며 멀티스레딩을 활용하여 처리"""

    # 입력 파일을 청크 단위로 읽음
    try:
        reader = pd.read_csv(input_file, sep="\t", skiprows=1, header=None, dtype=str, chunksize=chunk_size)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # 출력 파일 경로 설정
    output_path = os.path.join(output_dir, f"{prefix}_result.txt")
    
    # 결과 파일 쓰기 모드 시작
    with open(output_path, "wt") as f_out:
        first_chunk = True  # 첫 번째 청크에서만 헤더를 포함하기 위한 변수
        
        # tqdm을 사용하여 진행률 표시
        for chunk in tqdm(reader, desc="Processing", unit="chunk"):
            # 첫 번째 열(위치 정보)은 문자열로 유지
            first_col = chunk.iloc[:, 0].values
            data = chunk.iloc[:, 1:].replace(["", "NA", "NaN"], np.nan).astype(np.float32).values  # NaN 처리 및 float32 변환

            # 병렬 처리를 위해 데이터를 나누기
            split_data = np.array_split(data, n_cores)

            # 멀티스레딩 실행
            with ThreadPoolExecutor(max_workers=n_cores) as executor:
                results = list(executor.map(compute_ratios_chunk, split_data))

            # 결과를 NumPy 배열로 병합
            processed_data = np.vstack(results)

            # DataFrame으로 변환 및 첫 번째 열 삽입
            result_df = pd.DataFrame(processed_data)
            result_df.insert(0, "Position", first_col)

            # 파일에 점진적으로 저장 (gzip 압축 사용)
            result_df.to_csv(f_out, sep="\t", index=False, float_format="%.6f", header=first_chunk, mode="a")
            first_chunk = False  # 이후부터는 헤더 없이 저장

    print(f"Processed data saved at {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Process proportion table of dnmtools data in ratio value."
    )
    parser.add_argument("input_file", type=str, help="Path to the input file.")
    parser.add_argument("output_dir", type=str, help="Directory to store the output results.")
    parser.add_argument("--prefix", type=str, default="output", help="Prefix for the output files.")
    parser.add_argument("--n_cores", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--chunksize", type=int, default=50000, help="Number of rows per chunk.")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Program started at: {start_time}")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    try:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    except OSError as e:
        raise RuntimeError(f"Failed to create the output directory: {e}")
    
    change_data(args.input_file, args.output_dir, args.prefix, args.n_cores, args.chunksize)

    end_time = datetime.now()
    print(f"Program ended at: {end_time}")
    print(f"Total duration: {end_time - start_time}")

if __name__ == "__main__":
    main()
