# validation/r/cases/mcp_ml_spatial.R -- references for the MCP ML surface that no other case covers
# (TOOL-18). ml_pca's output_column writes the principal-component scores (stats::prcomp()$x,
# centred, not scaled, as ml_pca's pca(.., transform) computes them) into the dataset; the case
# kmeans_lloyd_pca_mds pins sdev and |rotation| but not the scores. The data are hclust_60
# (60 x 3; three blobs of 20). Signs of principal components are arbitrary (LAPACK), so the Rust
# test allows one sign flip per component.
case("mcp_pca_scores", "ml/pca", "prcomp scores (centred, unscaled) on hclust_60 for ml_pca's output_column", function(ctx) {
  d <- ctx$dataset("datasets/generated/hclust_60.csv"); X <- as.matrix(d[, c("a", "b", "c")])
  p <- stats::prcomp(X, center = TRUE, scale. = FALSE)
  stopifnot(max(abs(scale(X, scale = FALSE) %*% p$rotation - p$x)) < 1e-12)
  list(values = list(pca_sdev = vec(p$sdev), pca_x_pc1 = vec(p$x[, 1]), pca_x_pc2 = vec(p$x[, 2]), pca_x_pc3 = vec(p$x[, 3])),
       reference = "stats::prcomp(center = TRUE, scale. = FALSE)$x and $sdev",
       call = "prcomp(as.matrix(hclust_60[, c('a', 'b', 'c')]))", packages = character(0), dataset = "validation/datasets/generated/hclust_60.csv")
})
