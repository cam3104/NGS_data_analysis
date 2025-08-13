import cooler
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os

def extract_hic(cool_path, bed_path, output_path, normalized=False):
    clr = cooler.Cooler(cool_path)
    
    # 🔹 BED 파일 읽기 (Enhancer - Promoter 위치 포함)
    bed_df = pd.read_csv(bed_path, sep="\t", header=None, 
                         names=["enh-chrom", "enh-start", "enh-end", "enh#", 
                                "promoter-chrom", "promoter-start", "promoter-end", "gene_name"])
    
    # 🔹 비정상적인 값 처리
    for col in ["enh-start", "enh-end", "promoter-start", "promoter-end"]:
        bed_df[col] = pd.to_numeric(bed_df[col], errors='coerce')
    bed_df = bed_df.dropna()
    for col in ["enh-start", "enh-end", "promoter-start", "promoter-end"]:
        bed_df[col] = bed_df[col].astype(int)
    
    # 🔹 Hi-C bins 데이터 불러오기
    bins_df = clr.bins()[:]
    bins_df["start"] = pd.to_numeric(bins_df["start"], errors='coerce').astype(int)
    bins_df["end"] = pd.to_numeric(bins_df["end"], errors='coerce').astype(int)
    
    def get_bin_indices(chrom, start, end, bins_df):
        matched_bins = bins_df[(bins_df["chrom"] == chrom) & (bins_df["start"] <= start) & (bins_df["end"] >= end)]
        return matched_bins.index.values
    
    # 🔹 사용할 매트릭스 설정 (정규화 여부에 따라)
    matrix = clr.matrix(balance=normalized)
    
    # 🔹 각 BED 행에서 bin index 찾기
    bed_df["enh_bin_indices"] = bed_df.apply(lambda row: get_bin_indices(row["enh-chrom"], row["enh-start"], row["enh-end"], bins_df), axis=1)
    bed_df["promoter_bin_indices"] = bed_df.apply(lambda row: get_bin_indices(row["promoter-chrom"], row["promoter-start"], row["promoter-end"], bins_df), axis=1)
    
    contacts = []
    for _, row in bed_df.iterrows():
        enh_num = row["enh#"]
        for i in row["enh_bin_indices"]:
            for j in row["promoter_bin_indices"]:
                contact_value = matrix[i, j]
                contacts.append((enh_num, i, j, contact_value))
    
    # 🔹 DataFrame으로 변환
    contact_df = pd.DataFrame(contacts, columns=["enh#", "enh_bin", "promoter_bin", "contact"])
    
    # 🔹 bin 좌표 추가
    contact_df = contact_df.merge(bins_df, left_on="enh_bin", right_index=True, suffixes=("", "_enh"))
    contact_df = contact_df.merge(bins_df, left_on="promoter_bin", right_index=True, suffixes=("", "_promoter"))
    
    # 🔹 최종 파일 저장 (Enhancer - Promoter 상호작용 포함)
    contact_df = contact_df[["enh#", "chrom", "start", "end", "chrom_promoter", "start_promoter", "end_promoter", "contact"]]
    contact_df.to_csv(output_path, sep="\t", index=False)
    
    print("extract_complete:", output_path)


def main():
    parser = argparse.ArgumentParser(description="Extract Hi-C contacts between enhancer and promoter regions from a cooler file")
    parser.add_argument("cool_path", help="Path to the cooler file")
    parser.add_argument("bed_path", help="Path to the bed file")
    parser.add_argument("output_path", help="Path to the output file")
    parser.add_argument("--normalized", action="store_true", dest="normalized",
                        help="If set, treat the input cooler as already normalized (use balance weights). Otherwise, process raw counts.")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Program started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(args.cool_path):
        print("Cooler file does not exist.")
        return
    if not os.path.exists(args.bed_path):
        print("BED file does not exist.")
        return

    extract_hic(args.cool_path, args.bed_path, args.output_path, args.normalized)
    end_time = datetime.now()
    print(f"Program completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {end_time - start_time}")

if __name__ == "__main__":
    main()
