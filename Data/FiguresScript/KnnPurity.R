
library(Seurat)
library(DropletUtils)
library(ggplot2)
library(RANN)

# KNN purity function which takes in low-dim matrix and returns the  nn_purity metric 

knn_purity <- function(embeddings, clusters, k = 100) {
  nn <- nn2(data = embeddings, k = k + 1)$nn.idx[, 2:k] # remove self-neighbor
  # find percentage of neighbors that are of the same cluster
  nn_purity <- vector(mode = "numeric", length = length(x = clusters))
  for (i in seq_len(length.out = nrow(x = nn))) {
    nn_purity[i] <- sum(clusters[nn[i, ]] == clusters[i]) / k
  }
  return(nn_purity)
}


# Read in the dataset 

library(SeuratDisk)

Convert("/home/jovyan/GPLVM_Shaista/TrainedModels/gastrulation_PCA.h5ad", ".h5seurat")
Convert("/home/jovyan/GPLVM_Shaista/TrainedModels/gastrulation_random.h5ad", ".h5seurat")
# This creates a copy of this .h5ad object reformatted into .h5seurat inside the example_dir directory

# This .d5seurat object can then be read in manually
gastrulation <- LoadH5Seurat("/home/jovyan/GPLVM_Shaista/TrainedModels/gastrulation_PCA.h5Seurat")
gastrulation_random <-LoadH5Seurat("/home/jovyan/GPLVM_Shaista/TrainedModels/gastrulation_random.h5Seurat")


# Get embeddings for the three methods: PCA, gplvm_PCAinit and gplvm_randomIni


# Get embeddings 
purity_list <- list()
embeddings_list<-list()

emb_PCA <- Embeddings(object = gastrulation, reduction = "X_pca")
emb_gplvm_pca <- Embeddings(object = gastrulation, reduction = "X_BGPLVM_latent")
emb_gplvm_random<- Embeddings(object = gastrulation_random, reduction = "X_BGPLVM_latent")

embeddings_list[[1]]=emb_PCA 
embeddings_list[[2]]=emb_gplvm_pca
embeddings_list[[3]]= emb_gplvm_random

clustering.use= "celltype"
clusters <- gastrulation[[clustering.use]][[1]]

for (i in 1:length(embeddings_list)){
  nn <- knn_purity(embeddings = embeddings_list[[i]], clusters = clusters, k = 30)
  purity_list[[i]] <- nn
}


names(purity_list)[1]= "X_pca"
names(purity_list)[2]= "X_GPLVM_PCA"
names(purity_list)[3]= "X_GPLVM_Random"


knn_df <- data.frame()
for (i in seq_along(purity_list)) {
  ds_level <- as.factor(unlist(strsplit(names(purity_list)[i], "_"))[1])
  frc <- purity_list[[i]]
  dft <- data.frame("Reduction" = ds_level, "knn" = frc) 
  knn_df <- rbind(knn_df, dft)
}


savRDS(knn_df,'/barplots_knnpurity.rds')

 ggplot(
  data = knn_df,
  mapping = aes(x = as.factor(Assay), y = knn, fill = Assay)) +
  geom_boxplot(outlier.shape = NA) +
   stat_summary(fun.y="mean", color="black", shape=15) +
  theme_classic() + 
  scale_x_discrete(limits = levels(knn_df$Assay)) +
  ylab("KNN purity") +
  xlab("Reduction")
