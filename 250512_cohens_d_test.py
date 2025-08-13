import pandas as pd
import numpy as np
import warnings
import argparse
import os
# Cohen's d 계산 함수
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_std

# mann whitney test에서 유의한 subgroup끼리 Cohen's d 계산
def calculate_cohen_d(signal_data, mann_test_path):
    # Mann-Whitney U test 결과 불러오기
    mann_test_results = pd.read_csv(mann_test_path, sep="\t")

    # 유의한 p-value를 가진 그룹 쌍 필터링
    significant_pairs = mann_test_results[mann_test_results['adj_p_value'] < 0.05]

    # 필수 컬럼 확인
    required_cols = {'subgroup', 'Sample_4_avg', 'MDV_index'}
    signal_data.columns = signal_data.columns.str.strip()  # 공백 제거
    if not required_cols.issubset(signal_data.columns):
        raise ValueError(f"signal_data must include columns: {required_cols}")

    # Cohen's d 계산 결과를 저장할 리스트
    cohen_d_results = []

    # 각 MDV group 별로 반복
    for mdv_group in significant_pairs['MDV_level'].unique():
        mdv_significant = significant_pairs[significant_pairs['MDV_level'] == mdv_group]
        mdv_data = signal_data[signal_data['MDV_index'] == mdv_group]

        for index, row in mdv_significant.iterrows():
            group1 = row['Subgroup1']
            group2 = row['Subgroup2']

            data1 = mdv_data[mdv_data['subgroup'] == group1]['Sample_4_avg'].values
            data2 = mdv_data[mdv_data['subgroup'] == group2]['Sample_4_avg'].values

            d_value = cohen_d(data1, data2)

            cohen_d_results.append({
                'MDV_index': mdv_group,
                'group1': group1,
                'group2': group2,
                'cohen_d': d_value
            })

    return pd.DataFrame(cohen_d_results)

def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Calculate Cohen's d for significant groups from Mann-Whitney U test results.")
    parser.add_argument('--signal_data', type=str, required=True, help='Path to the signal data CSV file.')
    parser.add_argument('--mann_test_path', type=str, required=True, help='Path to the Mann-Whitney U test results CSV file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the Cohen\'s d results CSV file.')

    args = parser.parse_args()

    signal_data = pd.read_csv(args.signal_data, sep="\t")
    signal_data.columns = signal_data.columns.str.strip()

    cohen_d_results = calculate_cohen_d(signal_data, args.mann_test_path)
    cohen_d_results.to_csv(args.output_path, sep="\t", index=False)

if __name__ == "__main__":
    main()
