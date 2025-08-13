
#Rscript 240515_Diffbind.r --sample_csv [path] --output [path] --prefix [word] 
#결과 threshold를 수동으로 변환중. 
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

# Create DiffBind object
dba_object <- dba(sampleSheet = sample_sheet)

# read count  
dba_object <- dba.count(dba_object,bParallel=TRUE,bUseSummarizeOverlaps = TRUE)
#=================
#bParallel = TRUE: 다중코어를 이용할 수 있다.
#bUseSummarizeOverlaps = TRUE : SummarizeOverlaps 기능을 사용하여 더 정밀한 리드 카운트를 수행
#bRemoveDuplicates=TRUE: 중복된 read들을 제거하여 카운트
#=================

# Establish contrasts
dba_object <- dba.contrast(dba_object,categories=DBA_CONDITION)
# RLE/TMM using reads in peaks
dba_object = dba.normalize(dba_object,method=DBA_ALL_METHODS,
                           normalize=DBA_NORM_NATIVE)
dba.normalize(dba_object, method=DBA_DESEQ2, bRetrieve=TRUE)
dba.normalize(dba_object, method=DBA_EDGER, bRetrieve=TRUE)
# Analyze contrasts
dba_object <- dba.analyze(dba_object, method = DBA_ALL_METHODS)
dba.show(dba_object,bContrasts=TRUE)
#=================

# Get results
diff_bind_results1 <- dba.report(dba_object,th = 1,method=DBA_DESEQ2)
diff_bind_results2 <- dba.report(dba_object,th = 1,method=DBA_EDGER)

# Save results to a file
output_file1 <- file.path(args$output, paste0(args$prefix, "_DESeq2_diffbind_results.txt"))
write.table(as.data.frame(diff_bind_results1), file = output_file1, sep = "\t", row.names = FALSE, quote = FALSE)

output_file2 <- file.path(args$output, paste0(args$prefix, "_edgeR_diffbind_results.txt"))
write.table(as.data.frame(diff_bind_results2), file = output_file2, sep = "\t", row.names = FALSE, quote = FALSE)

pca_plot_file <- file.path(args$output, paste0(args$prefix, "_DEseq2_PCA_plot.png"))
png(filename = pca_plot_file, width = 800, height = 600)
dba.plotPCA(dba_object, method = DBA_DESEQ2, attributes = DBA_CONDITION, label = DBA_ID)
dev.off()
pca_plot_file <- file.path(args$output, paste0(args$prefix, "_edgeR_PCA_plot.png"))
png(filename = pca_plot_file, width = 800, height = 600)
dba.plotPCA(dba_object, method = DBA_EDGER, attributes = DBA_CONDITION, label = DBA_ID)
dev.off()
heatmap_plot_file <- file.path(args$output, paste0(args$prefix, "_heatmap.png"))
png(filename = heatmap_plot_file, width = 1000, height = 800)
dba.plotHeatmap(dba_object,contrast=1,th=1)
dev.off()