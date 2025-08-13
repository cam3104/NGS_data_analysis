library("edgeR")
library("argparse")

parser <- ArgumentParser(description = "edgeR analysis script")
parser$add_argument("--read", required=TRUE, help="Path to the read count table file")
parser$add_argument("--meta", required=TRUE, help="Path to the expression meta file")
parser$add_argument("--output", required=TRUE, help="Path to the output directory")
parser$add_argument("--prefix", required=TRUE, help="Prefix for output files")

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

Counts <- read.table(args$read, header=TRUE, row.names=1, sep="\t")
dgList <- DGEList(counts=Counts, genes=rownames(Counts))

#filtering 단계, retain only those genes that are represented at least 1cpm reads in at least two samples
countsPerMillion <- cpm(dgList)
summary(countsPerMillion)

countCheck <- countsPerMillion > 1
head(countCheck)
keep <- which(rowSums(countCheck) >= 2)
dgList <- dgList[keep,]
summary(cpm(dgList)) 

# Trimmed mean method로 norm.
dgList <- calcNormFactors(dgList, method="TMM")

metadata <- read.table(args$meta,header=TRUE, row.names=1 , sep="\t")
group <- factor(metadata$Condition)
designMat <- model.matrix(~group)

#구분이 control treatment1 treatment2 와 같이 2개 이상일때..
#group <- factor(metaData$Condition)
#design <- model.matrix(~0 + group)  # Intercept를 없애고 그룹을 기준으로 설계
#colnames(design) <- levels(group)   # 그룹 이름을 열 이름으로 사용

# estimate dispersion
dgList <- estimateGLMCommonDisp(dgList, design=designMat)
dgList <- estimateGLMTrendedDisp(dgList, design=designMat)
dgList <- estimateGLMTagwiseDisp(dgList, design=designMat)

#DEG 계산
fit <- glmFit(dgList, designMat)
lrt <- glmLRT(fit) # 구분이 2개 이상일 경우에는.. # Treatment1과 Treatment2의 차이를 정의
#contrast <- makeContrasts(groupTreatment1 - groupTreatment2, levels=design)
#lrt <- glmLRT(fit, contrast=contrast)
edgeR_result <- topTags(lrt,n=20000)

edgeR_output_path <- paste(args$output, paste(args$prefix, "edgeR_deg.txt", sep="_"),sep="/")
write.table(data.frame(edgeR_result), file=edgeR_output_path, sep="\t", quote=F, col.names=TRUE, row.names=FALSE)
