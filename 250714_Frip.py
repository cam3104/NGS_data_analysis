import argparse
import subprocess
from datetime import datetime

def count_mapped_reads(bam_file):
    # samtools view -c -F 4 sample.bam
    cmd = ["samtools", "view", "-c", "-F", "4", bam_file]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(result.stdout.strip())

def count_peak_reads(bam_file, peak_bed):
    # bedtools intersect -a sample.bam -b peaks.bed -u | samtools view -c
    bedtools_cmd = ["bedtools", "intersect", "-a", bam_file, "-b", peak_bed, "-u"]
    samtools_cmd = ["samtools", "view", "-c", "-"]

    # bedtools와 samtools 파이프 연결
    bedtools_proc = subprocess.Popen(bedtools_cmd, stdout=subprocess.PIPE)
    samtools_proc = subprocess.run(samtools_cmd, stdin=bedtools_proc.stdout, capture_output=True, text=True, check=True)
    bedtools_proc.stdout.close()
    bedtools_proc.wait()
    return int(samtools_proc.stdout.strip())

def main():
    parser = argparse.ArgumentParser(description="Calculate FRiP from BAM and peak BED.")
    parser.add_argument("--bam", required=True, help="Input BAM file (sorted and indexed).")
    parser.add_argument("--bed", required=True, help="Peak regions in BED format.")
    parser.add_argument("--out_prefix", required=True, help="Output file prefix.")
    args = parser.parse_args()
    print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("Calculating FRiP...")
    print(f"Input BAM file: {args.bam}")
    print(f"Input BED file: {args.bed}")
    total_reads = count_mapped_reads(args.bam)
    print(f"Total mapped reads: {total_reads}")
    peak_reads = count_peak_reads(args.bam, args.bed)
    print(f"Peak reads: {peak_reads}")
    frip = peak_reads / total_reads if total_reads > 0 else 0
    print(f"FRiP: {frip:.4f}")
    
    # 출력 파일 작성
    output_file = f"{args.out_prefix}_frip.txt"
    with open(output_file, "w") as f:
        f.write(f"Total mapped reads: {total_reads}\n")
        f.write(f"Peak reads: {peak_reads}\n")
        f.write(f"FRiP: {frip:.4f}\n")
    
    print(f"Done! Result saved to: {output_file}")

if __name__ == "__main__":
    main()
