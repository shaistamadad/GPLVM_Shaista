
library(Seurat)
library(ggplot2)
library(patchwork)
library(SeuratDisk)

counts <- Read10X_h5("/home/jovyan/mount/gdrive/BridgeIntegrationData/Data/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5")

# create object
pbmc <- CreateSeuratObject(
  counts = counts$`Gene Expression`,
  project = "pbmc",
  assay = "RNA"
)



reference <- LoadH5Seurat("/home/jovyan/mount/gdrive/BridgeIntegrationData/Data/pbmc_multimodal.h5seurat")

anchors <- FindTransferAnchors(
  reference = reference,
  query = pbmc,
  query.assay= "RNA",
  normalization.method = "SCT",
  reference.reduction = "spca",
  dims = 1:50
)

pbmc <- MapQuery(
  anchorset = anchors,
  query = pbmc,
  reference = reference,
  refdata = list(
    celltype.l1 = "celltype.l1",
    celltype.l2 = "celltype.l2",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)



saveRDS(object = pbmc, file = "/home/jovyan/mount/gdrive/BridgeIntegrationData/Data/pbmc_unsorted_10k.rds")

SaveH5Seurat(pbmc, filename = "/home/jovyan/mount/gdrive/BridgeIntegrationData/Data/pbmc10k.h5Seurat")

Convert("/home/jovyan/mount/gdrive/BridgeIntegrationData/Data/pbmc10k.h5Seurat", dest = "h5ad")
