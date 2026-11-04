# validation/r/cases/ml_hierarchical.R -- one hierarchical-clustering core validated against
# stats::hclust (T3-WP42: ML-55, ML-57, ML-115, ML-81, ML-84; moved in ML-14, ML-36, ML-51; closes
# ML-12, ML-13, ML-56 and, as side effects of the same rewrite, ML-25, ML-82, ML-83).
#  * hclust_methods: stats::hclust merge (column-major), height and order for all eight methods
#    (centroid and median on squared distances, as R documents), cutree k = 3, k = 2/4/6 and h = 1.5;
#    the same merges/heights/orders on a 0.5-grid dataset with exact ties (hclust_ties: R's tie-breaking);
#    cluster::agnes ac, sorted heights and banner order; cluster::diana dc, banner, order, merge and
#    cutree k = 3; the small hand cases of ML-13 (average/mcquitty), ML-14 (cutree), ML-25 (ward.D2).
#  * kmeans_lloyd_pca_mds: kmeans(algorithm = "Lloyd") from rows 1, 21, 41 at unit and 1e-5 scale and
#    with iter.max = 2 (not converged); prcomp scaled and unscaled; cmdscale(eurodist, k = 2) and of three
#    points on a line (ML-36); dbscan::dbscan(eps = 1, minPts = 5) (ML-84).
#  * dynamic_tree_cut: dynamicTreeCut::cutreeDynamic on hclust(dist(X), "average"): method 'tree'
#    (deepSplit FALSE/TRUE, cutHeight 1.5) and 'hybrid' (deepSplit 0-4, pamStage = FALSE, cutHeight 1.5,
#    maxPamDist = 0, the cutreeDynamic defaults), and 'hybrid' on the ward.D2 tree (ML-57).
suppressPackageStartupMessages({ library(cluster); library(dynamicTreeCut) })

.hc_data <- function(ctx) {
  set.seed(20261110)
  m <- rbind(cbind(rnorm(20, 0, 1), rnorm(20, 0, 1), rnorm(20, 0, 1)),
             cbind(rnorm(20, 4, 1), rnorm(20, 1, 1), rnorm(20, 0, 1)),
             cbind(rnorm(20, 2, 0.7), rnorm(20, 5, 0.7), rnorm(20, 2, 0.7)))
  ctx$gen_dataset("hclust_60", data.frame(a = round(m[, 1], 10), b = round(m[, 2], 10), c = round(m[, 3], 10)))
}
.hc_methods <- c("single", "complete", "average", "mcquitty", "ward.D", "ward.D2", "centroid", "median")
.hc_tree <- function(D, m) hclust(if (m %in% c("centroid", "median")) D^2 else D, method = m)  # LW on squared distances, as R documents

case("hclust_methods", "ml/clustering", "stats::hclust merges, heights and order (eight methods, with and without ties), cutree, cluster::agnes ac and order, cluster::diana dc and banner", function(ctx) {
  d <- .hc_data(ctx); X <- as.matrix(d); D <- dist(X)
  set.seed(7)
  g <- ctx$gen_dataset("hclust_ties", data.frame(u = sample(0:9, 40, replace = TRUE) * 0.5, v = sample(0:9, 40, replace = TRUE) * 0.5))
  G <- as.matrix(g); DG <- dist(G)
  stopifnot(anyDuplicated(as.numeric(DG)) > 0)                           # the grid has exactly tied distances
  v <- list()
  for (m in .hc_methods) {
    h <- .hc_tree(D, m); tag <- gsub("\\.", "", m)
    v[[paste0(tag, "_merge")]] <- vec(as.numeric(h$merge)); v[[paste0(tag, "_height")]] <- vec(h$height)
    v[[paste0(tag, "_order")]] <- vec(h$order); v[[paste0(tag, "_cutree3")]] <- vec(cutree(h, k = 3))
    ht <- .hc_tree(DG, m)
    v[[paste0("ties_", tag, "_merge")]] <- vec(as.numeric(ht$merge)); v[[paste0("ties_", tag, "_height")]] <- vec(ht$height)
    v[[paste0("ties_", tag, "_order")]] <- vec(ht$order)
  }
  ha <- hclust(D, "average")
  v$average_cutree_k246 <- vec(as.numeric(cutree(ha, k = c(2, 4, 6))))   # column-major, one column per k
  v$average_cutree_h15 <- vec(cutree(ha, h = 1.5)); v$average_k_h15 <- max(cutree(ha, h = 1.5))
  for (m in c("average", "single", "complete", "ward", "weighted")) {
    a <- agnes(X, method = m)
    v[[paste0("agnes_", m, "_ac")]] <- a$ac; v[[paste0("agnes_", m, "_height")]] <- vec(sort(a$height))
    v[[paste0("agnes_", m, "_order")]] <- vec(a$order); v[[paste0("agnes_", m, "_cutree3")]] <- vec(cutree(as.hclust(a), k = 3))
  }
  dv <- diana(X)
  v$diana_dc <- dv$dc; v$diana_banner <- vec(dv$height); v$diana_order <- vec(dv$order)
  v$diana_merge <- vec(as.numeric(dv$merge)); v$diana_cutree3 <- vec(cutree(as.hclust(dv), k = 3))
  # small hand cases from the findings (inline data in the Rust tests)
  v$ml13_average_height <- vec(hclust(dist(c(0, 1, 3, 7, 15)), "average")$height)
  v$ml13_mcquitty_height <- vec(hclust(dist(c(0, 1, 3, 7, 15)), "mcquitty")$height)
  v$ml14_single_cutree4 <- vec(cutree(hclust(dist(c(0, 100, 1, 1.1, 300)), "single"), k = 4))
  v$ml14_complete_cutree2 <- vec(cutree(hclust(dist(c(1000, 1, 2, 3, 4)), "complete"), k = 2))
  v$ml14_ties_cutree4 <- vec(cutree(hclust(dist(c(0, 0, 5, 5, 10)), "complete"), k = 4))
  v$ml14_chain_cutree4 <- vec(cutree(hclust(dist(c(0, 1, 10, 10.5, 30)), "single"), k = 4))
  v$ml25_wardD2_height <- vec(hclust(dist(rbind(c(0, 0), c(1, 0), c(10, 0))), "ward.D2")$height)
  list(dataset = "validation/datasets/generated/hclust_60.csv", values = v,
       reference = "stats::hclust(method; centroid/median on squared distances); stats::cutree; cluster::agnes ($ac, $height, $order); cluster::diana ($dc, $height, $order, $merge)",
       call = "hclust(dist(hclust_60), method)", packages = "cluster")
})

case("kmeans_lloyd_pca_mds", "ml/clustering", "k-means (Lloyd, fixed centres, unit and small scale, not converged), prcomp, cmdscale(eurodist) and a line, dbscan", function(ctx) {
  d <- ctx$dataset("datasets/generated/hclust_60.csv"); X <- as.matrix(d)
  v <- list()
  cen <- X[c(1, 21, 41), ]
  k1 <- kmeans(X, centers = cen, algorithm = "Lloyd", iter.max = 100)
  v$km_unit_cluster <- vec(k1$cluster); v$km_unit_centers <- vec(k1$centers); v$km_unit_tot_withinss <- k1$tot.withinss; v$km_unit_iter <- k1$iter
  Xs <- X * 1e-5                                                          # ML-81: same partition at any scale
  k2 <- kmeans(Xs, centers = cen * 1e-5, algorithm = "Lloyd", iter.max = 100)
  v$km_small_cluster <- vec(k2$cluster); v$km_small_tot_withinss <- k2$tot.withinss; v$km_small_iter <- k2$iter
  k3 <- suppressWarnings(kmeans(X, centers = cen, algorithm = "Lloyd", iter.max = 2))   # R warns "did not converge in 2 iterations"
  stopifnot(k3$iter > 2)
  v$km_iter2_cluster <- vec(k3$cluster); v$km_iter2_centers <- vec(k3$centers); v$km_iter2_tot_withinss <- k3$tot.withinss; v$km_iter2_iter <- k3$iter
  p <- prcomp(X, scale. = TRUE)
  v$pca_sdev <- vec(p$sdev); v$pca_rotation_abs <- vec(abs(p$rotation)); v$pca_var_ratio <- vec(p$sdev^2 / sum(p$sdev^2))
  pu <- prcomp(X)
  v$pca_unscaled_sdev <- vec(pu$sdev); v$pca_unscaled_rotation_abs <- vec(abs(pu$rotation)); v$pca_unscaled_var_ratio <- vec(pu$sdev^2 / sum(pu$sdev^2))
  e <- ctx$gen_dataset("eurodist_matrix", as.data.frame(as.matrix(eurodist)))
  mds <- cmdscale(as.dist(as.matrix(e)), k = 2, eig = TRUE)
  v$mds_points_abs <- vec(abs(mds$points)); v$mds_eig <- vec(mds$eig); v$mds_gof <- vec(mds$GOF)
  line <- cmdscale(dist(c(0, -1, 1)), k = 1, eig = TRUE)                  # ML-36: obs 1 at the centroid
  v$mds_line_points_abs <- vec(abs(line$points)); v$mds_line_eig <- vec(line$eig)
  v$dbscan_eps1_minpts5_cluster <- vec(dbscan::dbscan(X, eps = 1, minPts = 5)$cluster)   # 0 = noise
  list(dataset = "validation/datasets/generated/hclust_60.csv", values = v,
       reference = "stats::kmeans(algorithm = 'Lloyd', centers = rows 1, 21, 41); stats::prcomp(scale. = TRUE / FALSE); stats::cmdscale(k, eig = TRUE); dbscan::dbscan(eps = 1, minPts = 5)",
       call = "kmeans(hclust_60, centers, algorithm = 'Lloyd')", packages = "dbscan")
})

case("dynamic_tree_cut", "ml/clustering", "dynamicTreeCut::cutreeDynamic, method 'tree' and 'hybrid' (deepSplit, pamStage, cutHeight, maxPamDist, defaults)", function(ctx) {
  d <- ctx$dataset("datasets/generated/hclust_60.csv"); X <- as.matrix(d); D <- dist(X); M <- as.matrix(D)
  h <- hclust(D, method = "average")
  cd <- function(...) as.numeric(cutreeDynamic(h, verbose = 0, ...))
  v <- list()
  v$tree_deep_false <- vec(cd(method = "tree", deepSplit = FALSE, minClusterSize = 5))
  v$tree_deep_true <- vec(cd(method = "tree", deepSplit = TRUE, minClusterSize = 5))
  v$tree_deep_true_cut15 <- vec(cd(method = "tree", deepSplit = TRUE, minClusterSize = 5, cutHeight = 1.5))
  for (ds in 0:4) v[[paste0("hybrid_ds", ds)]] <- vec(cd(method = "hybrid", distM = M, deepSplit = ds, minClusterSize = 5))
  v$hybrid_ds2_nopam <- vec(cd(method = "hybrid", distM = M, deepSplit = 2, minClusterSize = 5, pamStage = FALSE))
  v$hybrid_ds2_cut15 <- vec(cd(method = "hybrid", distM = M, deepSplit = 2, minClusterSize = 5, cutHeight = 1.5))
  v$hybrid_ds2_maxpam0 <- vec(cd(method = "hybrid", distM = M, deepSplit = 2, minClusterSize = 5, maxPamDist = 0))
  v$hybrid_defaults <- vec(cd(distM = M))                                 # method 'hybrid', deepSplit 1, minClusterSize 20
  hw <- hclust(D, method = "ward.D2")
  v$ward_hybrid_ds1 <- vec(as.numeric(cutreeDynamic(hw, method = "hybrid", distM = M, deepSplit = 1, minClusterSize = 5, verbose = 0)))
  v$hclust_average_height <- vec(h$height)
  list(values = v, reference = "dynamicTreeCut::cutreeDynamic(hclust(dist(X), 'average' | 'ward.D2'), method = 'tree' | 'hybrid', ...)",
       call = "cutreeDynamic(h, method = 'hybrid', distM = as.matrix(dist(X)), deepSplit, minClusterSize = 5)", packages = "dynamicTreeCut")
})
