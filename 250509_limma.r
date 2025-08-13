library(limma)

# 1. 데이터 불러오기
data <- read.table("CT_K27me3_MDV_limma.txt", header=TRUE, sep="\t")
gene_names <- make.names(data$gene_name, unique=TRUE)
data <- data[, -1]
rownames(data) <- gene_names

# 2. 그룹 지정
group <- factor(c("EAE", "EAE", "EAE", "Control", "Control", "Control"))
design <- model.matrix(~ group)

# 3. limma-trend 분석
fit <- lmFit(data, design)
fit <- eBayes(fit, trend=TRUE)

# 4. 결과 출력
res <- topTable(fit, coef=2, number=Inf, adjust="BH")
write.table(res, file="limma_EAE_vs_Control_results.txt", sep="\t", quote=FALSE, row.names=TRUE)
