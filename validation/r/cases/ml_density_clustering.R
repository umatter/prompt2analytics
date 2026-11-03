# validation/r/cases/ml_density_clustering.R -- HDBSCAN, OPTICS and silhouette references (T3:
# ML-7, ML-45, ML-46, ML-48, ML-78, ML-61; moved in ML-6, ML-47, ML-49). dbscan::hdbscan (condensed
# tree, excess-of-mass selection, membership probabilities, GLOSH outlier scores), dbscan::optics
# (ordering, reachability and core distances; minPts counts the point itself, as dbscan/sklearn),
# extractDBSCAN and extractXi, dbscan::kNNdist, cluster::silhouette (singletons score 0).
# Package copy of design Appendix A.36 (T3-WP41; canonical). Added to the design's values:
#  * hdbscan_blobs: the cluster tree of computeStability (eps_birth, eps_death, stability, n_children,
#    number of points leaving, parent; by dbscan cluster id) and the salient ids, for minPts 5 and 10;
#    the minPts 5 fit at 1000 times the scale (ML-49: probabilities and GLOSH are scale-free); rows 1-9
#    (no split is possible with minPts 5: every point is noise, GLOSH 1); rows 1-5 and 41-45 (a split
#    into two children of exactly minPts points); hclust(dist(X), "single") heights, the merge heights
#    p2a reports for min_samples = 1 (dbscan needs minPts >= 2);
#  * optics_blobs: dbscan's predecessor, extractXi with correctPredecessors = FALSE, extractDBSCAN and
#    extractXi on the eps = 0.5 ordering, extractDBSCAN(eps_cl = 0.35) on an eps = 0.35 ordering;
#  * silhouette_singletons: the neighbour cluster of every observation.
suppressPackageStartupMessages({ library(dbscan); library(cluster) })

.blobs <- function(ctx) {
  set.seed(20261109)
  a <- cbind(rnorm(40, 0, 0.3), rnorm(40, 0, 0.3)); b <- cbind(rnorm(35, 3, 0.4), rnorm(35, 0, 0.4))
  cc <- cbind(rnorm(30, 1.5, 0.25), rnorm(30, 2.5, 0.25)); noise <- cbind(runif(10, -2, 5), runif(10, -2, 4.5))
  m <- round(rbind(a, b, cc, noise), 10)
  ctx$gen_dataset("blobs_noise", data.frame(x = m[, 1], y = m[, 2]))
}

# The cluster tree of dbscan's computeStability, by cluster id (0 = root), and the parent of each id.
.tree <- function(h) {
  res <- attr(h, "hdbscan"); ids <- sort(as.integer(names(res))); hier <- attr(res, "cl_hierarchy")
  parent <- rep(NA_real_, length(ids))
  for (p in names(hier)) for (ch in hier[[p]]) parent[match(ch, ids)] <- as.numeric(p)
  get <- function(f) vapply(ids, function(i) as.numeric(res[[as.character(i)]][[f]]), 0)
  list(id = ids, parent = parent, eps_birth = get("eps_birth"), eps_death = get("eps_death"),
       stability = get("stability"), n_children = get("n_children"),
       n_contains = vapply(ids, function(i) length(res[[as.character(i)]]$contains), 0),
       salient = attr(res, "salient_clusters"))
}

case("hdbscan_blobs", "ml/clustering", "dbscan::hdbscan on three blobs plus uniform noise, minPts 5 and 10", function(ctx) {
  d <- .blobs(ctx); X <- as.matrix(d)
  v <- list()
  for (mp in c(5, 10)) {
    h <- hdbscan(X, minPts = mp)
    v[[paste0("mp", mp, "_cluster")]] <- vec(h$cluster)               # 0 = noise
    v[[paste0("mp", mp, "_membership_prob")]] <- vec(h$membership_prob)
    v[[paste0("mp", mp, "_outlier_scores")]] <- vec(h$outlier_scores)
    v[[paste0("mp", mp, "_cluster_scores")]] <- vec(unname(h$cluster_scores))   # stability of the selected clusters
    v[[paste0("mp", mp, "_coredist")]] <- vec(h$coredist)
    v[[paste0("mp", mp, "_hc_height")]] <- vec(h$hc$height)             # single-linkage merge heights (mutual reachability)
    t <- .tree(h)
    for (f in c("id", "parent", "eps_birth", "eps_death", "stability", "n_children", "n_contains"))
      v[[paste0("mp", mp, "_tree_", f)]] <- vec(t[[f]])
    v[[paste0("mp", mp, "_salient")]] <- vec(t$salient)
  }
  v$knndist_k4 <- vec(kNNdist(X, k = 4))                                # 4th neighbour excluding self = core distance for minPts 5
  v$single_linkage_height <- vec(hclust(dist(X), method = "single")$height)   # min_samples = 1: zero core distances, MST of dist(X)
  hs <- hdbscan(X * 1000, minPts = 5)                                  # ML-49: scale-free probabilities and GLOSH
  v$mp5_x1000_cluster <- vec(hs$cluster); v$mp5_x1000_membership_prob <- vec(hs$membership_prob)
  v$mp5_x1000_outlier_scores <- vec(hs$outlier_scores); v$mp5_x1000_cluster_scores <- vec(unname(hs$cluster_scores))
  h9 <- hdbscan(X[1:9, ], minPts = 5)                                   # no split into two clusters of >= 5 points
  stopifnot(all(h9$cluster == 0))
  v$nosplit_cluster <- vec(h9$cluster); v$nosplit_outlier_scores <- vec(h9$outlier_scores)
  v$nosplit_membership_prob_dbscan <- vec(h9$membership_prob)          # dbscan's root-based values; p2a reports 0 (noise)
  hb <- hdbscan(X[c(1:5, 41:45), ], minPts = 5)                         # a split into two children of exactly minPts points
  stopifnot(length(unique(hb$cluster)) == 2, all(hb$cluster > 0))
  v$boundary_cluster <- vec(hb$cluster); v$boundary_cluster_scores <- vec(unname(hb$cluster_scores))
  list(values = v, reference = "dbscan::hdbscan(minPts); dbscan::kNNdist", call = "hdbscan(blobs_noise, minPts = 5 | 10)", packages = "dbscan")
})

case("optics_blobs", "ml/clustering", "dbscan::optics (eps = Inf and eps = 0.5), extractDBSCAN(eps_cl = 0.35), extractXi(xi = 0.05)", function(ctx) {
  d <- ctx$dataset("datasets/generated/blobs_noise.csv"); X <- as.matrix(d)
  v <- list()
  o <- optics(X, minPts = 5)
  v$inf_order <- vec(o$order); v$inf_reachdist <- vec(ifelse(is.infinite(o$reachdist), NA, o$reachdist)); v$inf_coredist <- vec(o$coredist)
  v$inf_predecessor <- vec(o$predecessor)                               # NA for the first point of each component
  e <- extractDBSCAN(o, eps_cl = 0.35); v$inf_dbscan_cluster <- vec(e$cluster)
  x <- extractXi(o, xi = 0.05); v$inf_xi_cluster <- vec(x$cluster)
  v$inf_xi_clusters_start <- vec(x$clusters_xi$start); v$inf_xi_clusters_end <- vec(x$clusters_xi$end)
  xn <- extractXi(o, xi = 0.05, correctPredecessors = FALSE)
  stopifnot(!identical(xn$clusters_xi$end, x$clusters_xi$end))       # the predecessor correction matters here
  v$inf_xi_nocorrect_cluster <- vec(xn$cluster)
  v$inf_xi_nocorrect_clusters_start <- vec(xn$clusters_xi$start); v$inf_xi_nocorrect_clusters_end <- vec(xn$clusters_xi$end)
  o2 <- optics(X, eps = 0.5, minPts = 5)
  v$eps05_order <- vec(o2$order); v$eps05_reachdist <- vec(ifelse(is.infinite(o2$reachdist), NA, o2$reachdist))
  v$eps05_coredist <- vec(ifelse(is.infinite(o2$coredist), NA, o2$coredist))
  v$eps05_predecessor <- vec(o2$predecessor)
  v$eps05_dbscan_cluster <- vec(extractDBSCAN(o2, eps_cl = 0.35)$cluster)
  x2 <- extractXi(o2, xi = 0.05); v$eps05_xi_cluster <- vec(x2$cluster)
  v$eps05_xi_clusters_start <- vec(x2$clusters_xi$start); v$eps05_xi_clusters_end <- vec(x2$clusters_xi$end)
  o3 <- optics(X, eps = 0.35, minPts = 5)                               # the legacy optics(.., max_eps = 0.35, xi = None) call
  v$eps035_order <- vec(o3$order); v$eps035_dbscan_cluster <- vec(extractDBSCAN(o3, eps_cl = 0.35)$cluster)
  list(values = v, reference = "dbscan::optics(minPts = 5, eps); dbscan::extractDBSCAN; dbscan::extractXi",
       call = "optics(blobs_noise, minPts = 5)", packages = "dbscan")
})

case("silhouette_singletons", "ml/clustering", "cluster::silhouette with singleton clusters (s(i) = 0 for singletons)", function(ctx) {
  d <- ctx$dataset("datasets/generated/blobs_noise.csv"); X <- as.matrix(d)
  cl <- cutree(hclust(dist(X), method = "average"), k = 6)
  s <- silhouette(cl, dist(X))
  v <- list(labels = vec(cl), sil_width = vec(s[, "sil_width"]), avg_width = mean(s[, "sil_width"]),
            cluster_avg = vec(summary(s)$clus.avg.widths), n_singletons = sum(table(cl) == 1),
            neighbor = vec(s[, "neighbor"]))
  list(values = v, reference = "cluster::silhouette(cutree(hclust(dist(X), 'average'), 6), dist(X))",
       call = "silhouette(cutree(hclust(dist(blobs_noise), 'average'), 6), dist(blobs_noise))", packages = "cluster")
})
