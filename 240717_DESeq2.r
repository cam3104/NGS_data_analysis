library("DESeq2")
library("argparse")

parser <- ArgumentParser(description = "DESeq2 analysis script")
parser$add_argument("--read", required=TRUE, help="Path to the read count table file")
parser$add_argument("--meta", required=TRUE, help="Path to the expression meta file")
parser$add_argument("--output", required=TRUE, help="Path to the output directory")
parser$add_argument("--prefix", required=TRUE, help="Prefix for output files")
parser$add_argument("--group_a", required=TRUE, help="Group variable name in metadata")
parser$add_argument("--group_b", required=TRUE, help="Group variable name in metadata")

args <- parser$parse_args()

# 파일 존재 여부 확인
if (!file.exists(args$read)) {
    stop("Read count data file does not exist.")
}
if (!file.exists(args$meta)) {
    stop("Metadata file does not exist.")
}

# 출력 디렉터리 확인 및 생성
if (!dir.exists(args$output)) {
    dir.create(args$output, recursive = TRUE)
}

# 데이터 읽기
x <- read.table(args$read, header=TRUE, row.names=1, sep="\t")
roundx <- round(x)
coldata <- read.table(args$meta, header=TRUE, row.names=1, sep="\t")

# DESeq2 데이터 세트 준비
dds <- DESeqDataSetFromMatrix(countData = roundx, colData = coldata, design = ~group)
dds <- estimateSizeFactors(dds)

# 정규화된 카운트 계산 및 저장
normalized_counts <- counts(dds, normalized=TRUE)
norm_output_path <- paste(args$output, paste(args$prefix, "deseq2_norm.txt", sep="_"),sep="/")
write.table(data.frame("id"=rownames(normalized_counts), normalized_counts), file=norm_output_path, sep="\t", quote=F, col.names=TRUE, row.names=FALSE)

# 차등 발현 분석 수행
des <- DESeq(dds)
res <- results(des, contrast=c("group",args$group_a,args$group_b)) # (A - B)

# 결과 저장
deg_output_path <- paste(args$output, paste(args$prefix, "deg.txt", sep="_"),sep="/")
write.table(data.frame("id"=rownames(res), res), file=deg_output_path, sep="\t", quote=F, col.names=TRUE, row.names=FALSE)
summary(res)