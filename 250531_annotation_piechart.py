import pandas as pd
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt

def read_csv(file_path):

    input_df = pd.read_csv(file_path, sep = '\t')

    annotaions = input_df['Annotation']
    # cut value before"("
    annotaions = [x.split('(')[0] for x in annotaions]
    annotaions = [x.strip() for x in annotaions]
    input_df['Annotation'] = annotaions
    annotations_dict = {}
    for annotation in input_df['Annotation']:
        if annotation not in annotations_dict:
            annotations_dict[annotation] = 0
        annotations_dict[annotation] += 1
    
    annotations_list = list(annotations_dict.keys())
    annotations_list.sort()
    annotations_dict = {k: v for k, v in sorted(annotations_dict.items(), key=lambda item: item[1], reverse=True)}
    annotations_df = pd.DataFrame(list(annotations_dict.items()), columns=['Annotation', 'Count'])
    # plot with pie chart
    annotations_df.plot.pie(y='Count', labels=annotations_df['Annotation'], autopct='%1.1f%%', figsize=(6, 6), legend=False)
    plt.ylabel('')
    output_file = os.path.splitext(file_path)[0] + '_annotations_pie_chart.png'
    plt.savefig(output_file)
    print(f"Pie chart saved to {output_file}")
    # save annotations_df to csv
    output_csv_file = os.path.splitext(file_path)[0] + '_annotations.csv'
    annotations_df.to_csv(output_csv_file, index=False)
    print(f"Annotations saved to {output_csv_file}")

def main():
    parser = argparse.ArgumentParser(description='Process a CSV file and generate a pie chart of annotations.')
    parser.add_argument('file_path', type=str, help='Path to the input CSV file')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.file_path):
        print(f"Error: The file {args.file_path} does not exist.")
        sys.exit(1)
    
    read_csv(args.file_path)

if __name__ == "__main__":
    main()
 
