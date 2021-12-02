#This is a script for plotting umaps of the query data using umaps created for the four different assays: ATAC,RNA (actual gene expression), Imputed RNA, and ATAC. 
# Read in the query assay 

atac <- readRDS("/home/madads/FinalScripts/query/queryRerun.rds")

DefaultAssay(atac) <- "RNA"
atac<- ScaleData(atac)
atac<- FindVariableFeatures(atac, assay = "RNA")
features= VariableFeatures(atac)
atac<- RunPCA(atac, features = features, reduction.name = "PCA.AcRNA")
atac<- RunUMAP(object = atac, reduction = "PCA.AcRNA", dims = 1:50, reduction.name = "AcRNA")

DefaultAssay(atac) <- "GeneActivity"
atac<- ScaleData(atac)
atac<- RunPCA(atac, features = features, reduction.name = "PCA.ATAC")
atac<- RunUMAP(object = atac, reduction = "PCA.ATAC", dims = 1:50, reduction.name = "GeneAct.")
# read in imputed RNA assay 
FullImputedAssay<- readRDS("/home/madads/FinalScripts/Imputed/Imputed_Rna_FullCoassay.rds")
atac[["ImputedRNA"]]<- FullImputedAssay
# computed umap with imputed RNA 
DefaultAssay(atac) <- "ImputedRNA"  # create a new PCA dim reduction  scale data 
atac<- ScaleData(atac)
atac<- RunPCA(atac, features = features, reduction.name = "PCA.ImputedRNA")
atac<- RunUMAP(object = atac, reduction = "PCA.ImputedRNA", dims = 1:50, reduction.name = "imputed")
# compute umap with Arch R 
atac2<- readRDS("/home/madads/FinalScripts/ArchR/queryArchRAssay.rds")
atac <- NormalizeData(
  object = atac ,
  assay = 'ArchRAssay',
  normalization.method = 'LogNormalize',
  scale.factor = median(atac$nCount_ArchRAssay)
)
DefaultAssay(atac2) <- "ArchRAssay"
atac2<- ScaleData(atac2)
atac2<- RunPCA(atac2, features = features, reduction.name = "PCA.ArchR")
atac2<- RunUMAP(object = atac2, reduction = "PCA.ArchR", dims = 1:50, reduction.name = "ArchR")
library(ggpubr)
# pdf(file = "/home/madads/FinalScripts/Figure6/Figure6.pdf",   # The directory you want to save the file in
#     width = 8, # The width of the plot in inches
#     height = 5) 
# p1=DimPlot(object = atac, reduction = 'AcRNA') + ggtitle("Actual Gene Expression")
# p2=DimPlot(object = atac, reduction = 'GeneAct.') + ggtitle("Gene Activity")
# p3=DimPlot(object = atac, reduction = 'imputed') + ggtitle("Imputed RNA")
# p4=DimPlot(object = atac2, reduction = 'ArchR' ) + ggtitle("ArchR")
# 
# CombinePlots(
#   plots = list(p1,p2,p3,p4),
#   legend = 'none')
# 
# dev.off()
# 



p1=DimPlot(object = gastrulation, reduction = 'PCA', group.by = "celltype") + ggtitle("PCA")
p2=DimPlot(object = gastrulation, reduction = 'gplvm_PCAinit', group.by = "celltype") + ggtitle("gplvm_PCA")
p3=DimPlot(object = gastrulation, reduction = 'gplvm_rnadomInit',group.by = "celltype") + ggtitle("randomInitialisation")

Figure1A= CombinePlots(
  plots = list(p1,p2,p3),
  legend = 'bottom')


barplots_list <- readRDS("~/barplots_knnpurity.rds")

g_legend<-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)}


mylegend<-g_legend(barplot_knnpurity])

p10k <- ggarrange(barplot_list[[1]] + theme(legend.position="none"),
                  
                  barplot_list[[2]] + theme(legend.position="none"),
                  
                  
                  barplot_list[[3]] + theme(legend.position="none"),
                  
                  barplot_list[[4]] + theme(legend.position="none"),
                  barplot_list[[5]] + theme(legend.position="none"),
                  barplots_knnpurity[[1]] + theme(legend.position="none"),
                  
                  barplots_knnpurity[[2]] + theme(legend.position="none"),
                  
                  
                  barplots_knnpurity[[3]] + theme(legend.position="none"),
                  
                  barplots_knnpurity[[4]] + theme(legend.position="none"),
                  barplots_knnpurity[[5]] + theme(legend.position="none"),
                  
                  nrow=3,ncol= 5, mylegend,font.label = list(size = 20, color = "black"))


#barplots_knnpurity <- readRDS("~/FinalScripts/query2/Figures/barplots_knnpurity.rds")


Figure1B=ggarrange(p10k, nrow = 1,common.legend = TRUE )
Figure1B



Figure1= ggarrange(Figure1A, Figure1B, labels = c("A", "B"), nrow = 2,heights = c(2,1))


#FinalFigure= ggarrange(Figure1A,fig2, nrow = 2, ncol = 1,  heights = c(1,3)  )

ggsave(filename = "~/UMAPandPurity.png", plot = Figure1, height = 12, width = 12, dpi = 200)

pdf(file = "~/Figure1.pdf",   # The directory you want to save the file in
    width = 8, # The width of the plot in inches
    height = 5) 

Figure1
dev.off()
