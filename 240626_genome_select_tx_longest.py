import pandas as pd
import os
from datetime import datetime
import argparse

def read_and_sort(input_path, output_directory, prefix):
    df = pd.read_csv(input_path, sep="\t")
    df['sub'] = df['txEnd'] - df['txStart']

    # '+' strand 데이터에서 각 gene별로 'txstart' 최대값 행 추출
    longest = df.loc[df.groupby('name2')['sub'].idxmax()]

    # 결과 파일 저장
    output_path1 = os.path.join(output_directory, f"{prefix}_longest_data.txt")
    longest.to_csv(output_path1, index=False, sep='\t')

    longest_TSS = []
    for i in range(len(longest)):
        if longest.iloc[i]['strand'] == "+":
            longest_TSS.append({"chrom":longest.iloc[i]['chrom'],"Start":longest.iloc[i]['txStart'],"end":longest.iloc[i]['txStart'],"strand":longest.iloc[i]['strand'],
                                "gene_id":longest.iloc[i]['name'],"gene_name":longest.iloc[i]['name2']})
        else :
            longest_TSS.append({"chrom":longest.iloc[i]['chrom'],"Start":longest.iloc[i]['txEnd'],"end":longest.iloc[i]['txEnd'],"strand":longest.iloc[i]['strand'],
                                "gene_id":longest.iloc[i]['name'],"gene_name":longest.iloc[i]['name2']})
            
    df = pd.DataFrame(longest_TSS)
    output_path2 = os.path.join(output_directory, f"{prefix}_longest_data_TSS.bed")
    df.to_csv(output_path2, index=False, sep='\t')

def main():
    start_time = datetime.now()
    print(f"Program started at: {start_time}")

    parser = argparse.ArgumentParser(description="Process and analyze gene data.")
    parser.add_argument("input_path", type=str, help="Path to the input file containing gene data.")
    parser.add_argument("output_directory", type=str, help="Directory to store the output results.")
    parser.add_argument("--prefix", type=str, default="output", help="Prefix for the output files.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print("File does not exist.")
        return
    
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    read_and_sort(args.input_path, args.output_directory, args.prefix)

    end_time = datetime.now()
    print(f"Program ended at: {end_time}")
    duration = end_time - start_time
    print(f"Total duration: {duration}")

if __name__ == "__main__":
    main()
