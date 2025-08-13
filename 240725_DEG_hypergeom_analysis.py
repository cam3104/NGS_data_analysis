
# python 240725_DEG_hypergeom_analysis.py [gene_term 있는 파일(GOBP_gmt)] [선별한 유전자 파일(DEG파일)] [ouput 파일명]
# Description: calculate hypergeometic distribution with gmt files.

#######################################################################
import argparse
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def read_gmt(file_path, encoding='utf-8'):
    gmt_data = {}
    all_genes = set()
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                parts = line.split('\t')
                if len(parts) < 3:
                    print(f"Warning: Malformed line skipped: {line}")
                    continue  # Skip malformed lines
                term = parts[0].strip()
                genes = [gene.strip() for gene in parts[2:]]
                gmt_data[term] = genes
                all_genes.update(genes)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        exit(1)
    return gmt_data, all_genes


def calculate_hypergeometric(db_data, query_data, total_size):
    results = []
    p_values = []
    
    for term_db, genes_db in db_data.items():
        for term_query, genes_query in query_data.items():
            overlap = set(genes_db).intersection(genes_query)
            x = len(overlap)  # 공통 유전자의 수
            m = len(genes_db)  # 데이터베이스 유전자 세트의 크기
            n = total_size - m  # 전체 유전자 세트 크기에서 데이터베이스 유전자 세트 크기를 뺀 값
            k = len(genes_query)  # 쿼리 유전자 세트의 크기

            # Hypergeometric 분포를 이용해 성공 횟수가 x 이상일 확률 계산
            # 예외 처리를 추가하여 수치가 잘못된 경우 대비
            try:
                p_value = hypergeom.sf(x - 1, total_size, m, k)
            except ValueError as e:
                print(f"Error in hypergeometric calculation: {e}")
                continue
            
            results.append([term_db, term_query, total_size, x, m, n ,k, p_value])
            p_values.append(p_value)
    
    # Adjust p-values using Benjamini-Hochberg correction
    try:
        _, p_values_adjusted, _, _ = multipletests(p_values, method='fdr_bh')
        for i in range(len(results)):
            results[i].append(p_values_adjusted[i])
    except Exception as e:
        print(f"Error in p-value adjustment: {e}")
        return []
    
    return results

def save_results(results, output_file):
    # Create a DataFrame from the results
    df = pd.DataFrame(results, columns=[
        'term_db', 'term_geneset','total_size', 'overlapped_gene#', 'DB.gene_set_size' , '(total_size)-(DB.gene_set_size)', 'query.gene_set_size', 'p-value', 'adjusted p-value'
    ])
    # Save DataFrame to a TSV file
    df.to_csv(output_file, sep='\t', index=False)


def main(db_file, query_file, output_file):
    db_data, db_genes = read_gmt(db_file)
    if not db_data:
        print(f"Error: No valid data found in {db_file}.")
        exit(1)
    
    query_data, query_genes = read_gmt(query_file)
    if not query_data:
        print(f"Error: No valid data found in {query_file}.")
        exit(1)
    
    total_size = len(db_genes.union(query_genes)) # 두 집합(db.genes query_genes)의 합집합을 구하고, len 함수를 사용하여 전체 유전자 집합의 크기를 계산
    if total_size == 0:
        print("Error: No genes found in the union of database and query genes.")
        exit(1)
    
    results = calculate_hypergeometric(db_data, query_data, total_size)
    if not results:
        print("Error: No results generated from the hypergeometric calculation.")
        exit(1)
    
    print

    save_results(results, output_file)
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate hypergeometric distribution with GMT files.")
    parser.add_argument("db_file", type=str, help="Database file in GMT format")
    parser.add_argument("query_file", type=str, help="Query file in GMT format")
    parser.add_argument("output_file", type=str, help="Output file for results")
    args = parser.parse_args()
   
    main(args.db_file, args.query_file, args.output_file)
