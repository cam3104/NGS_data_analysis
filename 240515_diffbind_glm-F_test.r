#Rscript 250520_diffbind_glm-F_test.r --sample_csv [path] --output [path] --prefix [word] 
#=============================================================
# Load DiffBind library
library(DiffBind)
library(argparse)


parser <- ArgumentParser(description = "DiffBind analysis script")
parser$add_argument("--sample_csv", required=TRUE, help="Path to the sample sheet CSV file")
parser$add_argument("--output", required=TRUE, help="Path to the output directory")
parser$add_argument("--prefix", required=TRUE, help="Prefix for output files")

args <- parser$parse_args()

sample_sheet <- read.csv(args$sample_csv)

# 피크 파일 경로가 올바른지 확인
if (!all(file.exists(sample_sheet$Peaks))) {
    stop("some peak file doesn't exist. check CSV file column 'Peaks' .")
}

# BAM 파일 경로가 올바른지 확인
if (!all(file.exists(sample_sheet$bamReads))) {
    stop("some bam file doesn't exist. check CSV file column 'bamReads'.")
}

# 피크 파일의 내용 확인
check_peak_file <- function(file) {
    peaks <- read.table(file, header = FALSE)
    if (!is.numeric(peaks$V2) || !is.numeric(peaks$V3)) {
        stop(paste("peak file ", file, " start or end position contains a non-numeric value."))
    }
}
sapply(sample_sheet$Peaks, check_peak_file)

# DiffBind 분석 시작
dba_object <- dba(sampleSheet = sample_sheet)
# read count  
dba_object <- dba.count(dba_object,bParallel=TRUE,bUseSummarizeOverlaps = TRUE)
dba.show(dba_object, bContrasts=TRUE)
#=================
# Peak count matrix 추출
peakset_gr <- dba.peakset(dba_object, bRetrieve=TRUE)

# metadata (count 값)만 데이터프레임으로 변환
count_mat <- as.data.frame(mcols(peakset_gr))  # ← 반드시 mcols(...) 로 호출

# 그룹 정보 설정
group <- factor(dba_object$samples$Condition)


# edgeR glm-F_test 분석 시작
library(edgeR)
dge <- DGEList(counts = count_mat, group = group)
dge <- calcNormFactors(dge)
design <- model.matrix(~group)
dge <- estimateDisp(dge, design)
fit <- glmQLFit(dge, design)
qlf <- glmQLFTest(fit)
res <- topTags(qlf, n = Inf, sort.by = "none")

 # GRanges에 통계 결과 붙이기
mcols(peakset_gr) <- cbind(mcols(peakset_gr), res$table)

# 데이터프레임으로 변환 및 정렬
df <- data.frame(
  chrom = as.character(seqnames(peakset_gr)),
  start = start(peakset_gr),
  end   = end(peakset_gr),
  mcols(peakset_gr)
)

df <- df[order(df$FDR), ]  # FDR 기준 정렬

# ---결과 저장 --- #

output_file <- file.path(args$output, paste0(args$prefix, "_edgeR_glm-F_test_diffbind_results_with_coords.txt"))
write.table(df, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)


