# validation/r/cases/ml_cluster_t4.R -- clustering references (ML-9, ML-10, ML-11, ML-44, ML-54, ML-79,
# ML-80, ML-127), R 4.3.3.
#  * fcm_cmeans_from_start   -- e1071::cmeans from fixed centres, Euclidean and Manhattan, m = 2 and 1.5. Exact
#    e1071 update: exponent 1/(m-1) on squared Euclidean / L1 distances, Manhattan centres = u^m-weighted medians,
#    withinerror with weights 1/n.
#  * fanny_blobs_more        -- cluster::fanny k = 2 (m = 2) and k = 3 (m = 1.5); the k = 3, m = 2 values are in
#    fuzzy_reference_blobs (ml_partition_t3.R). Memberships in clusters ordered by mean x.
#  * spectral_njw            -- exact Ng-Jordan-Weiss on ap_blobs (RBF, p2a's default gamma) and on spectral_pair
#    (two blobs and a far tight pair): eigen() of D^-1/2 A D^-1/2 with zero diagonal, the L_sym eigenvalues
#    1 - mu, and the k-means partition of the row-normalised embedding (nstart = 50; labels compared up to
#    permutation).
#  * cluster_stats_blobs     -- clusterCrit::intCriteria and fpc::cluster.stats on ap_blobs with the PAM k = 3
#    partition (DB with mean distance to the centroid as scatter).
#  * partition_agreement     -- aricode NMI (max, min, sqrt, sum) and ARI on fixed label vectors.
#  * gap_logw_blobs          -- cluster::clusGap(ap_blobs, kmeans, K.max = 5, d.power = 2, spaceH0 = "original")
#    logW (deterministic once kmeans finds the optimal partition; nstart = 50) and cluster::maxSE(...,
#    "Tibs2001SEmax") on fixed gap/SE vectors.
#  * pvclust_msfit           -- pvclust:::msfit (Shimodaira's multiscale fit) on fixed bootstrap counts, the
#    degenerate-scale rule, and the realised scale sizes round(p * r) of pvclust:::boot.hclust.
suppressPackageStartupMessages({ library(cluster); library(e1071); library(clusterCrit); library(fpc); library(aricode) })

.order_by_x <- function(C) order(C[, 1])

case("fcm_cmeans_from_start", "ml/clustering", "e1071::cmeans from fixed centres on ap_blobs: Euclidean and Manhattan, m = 2 and 1.5", function(ctx) {
  X <- as.matrix(ctx$dataset("datasets/generated/ap_blobs.csv"))
  c0 <- rbind(c(0, 0), c(3, 0), c(1.5, 2.5))
  v <- list(start = vec(c0))
  wmed <- function(x, w) { o <- order(x); x <- x[o]; w <- w[o]; x[which(cumsum(w) >= sum(w) / 2)[1]] }
  for (dist in c("euclidean", "manhattan")) for (m in c(2, 1.5)) {
    r <- cmeans(X, centers = c0, iter.max = 10000, dist = dist, m = m, control = list(reltol = 1e-15))
    if (dist == "manhattan") {   # e1071's Manhattan centres are the u^m-weighted medians
      W <- r$membership^m
      wm <- sapply(1:2, function(j) sapply(1:3, function(k) wmed(X[, j], W[, k])))
      stopifnot(max(abs(wm - r$centers)) == 0)
    }
    tg <- paste0(substr(dist, 1, 3), if (m == 2) "_m2" else "_m15")
    o <- .order_by_x(r$centers)   # c0 is already ordered by x; kept for robustness
    v[[paste0(tg, "_centers")]] <- vec(r$centers[o, ])
    v[[paste0(tg, "_membership")]] <- vec(r$membership[, o])
    v[[paste0(tg, "_withinerror")]] <- r$withinerror
    v[[paste0(tg, "_cluster")]] <- vec(match(r$cluster, o))
  }
  list(values = v, reference = "e1071::cmeans(x, centers = start, m, dist, control = list(reltol = 1e-15))",
       call = "cmeans(ap_blobs, centers = rbind(c(0,0), c(3,0), c(1.5,2.5)), m = 2 | 1.5, dist = 'euclidean' | 'manhattan')",
       packages = "e1071")
})

case("fanny_blobs_more", "ml/clustering", "cluster::fanny on ap_blobs: k = 2 (memb.exp = 2) and k = 3 (memb.exp = 1.5)", function(ctx) {
  X <- as.matrix(ctx$dataset("datasets/generated/ap_blobs.csv"))
  v <- list()
  for (cfg in list(c(2, 2), c(3, 1.5))) {
    k <- cfg[1]; m <- cfg[2]
    f <- fanny(X, k, memb.exp = m, maxit = 5000, tol = 1e-15)
    stopifnot(f$convergence[["converged"]] == 1)
    o <- order(sapply(1:k, function(j) mean(X[f$clustering == j, 1])))
    tg <- sprintf("k%d_m%s", k, if (m == 2) "2" else "15")
    v[[paste0(tg, "_membership")]] <- vec(f$membership[, o]); v[[paste0(tg, "_coeff")]] <- vec(unname(f$coeff))
    v[[paste0(tg, "_objective")]] <- f$objective[["objective"]]; v[[paste0(tg, "_clustering")]] <- vec(match(f$clustering, o))
  }
  list(values = v, reference = "cluster::fanny(x, k, memb.exp, tol = 1e-15)", call = "fanny(ap_blobs, 2); fanny(ap_blobs, 3, memb.exp = 1.5)",
       packages = "cluster")
})

.njw <- function(X, gamma, k) {
  A <- exp(-gamma * as.matrix(dist(X))^2); diag(A) <- 0
  d <- 1 / sqrt(rowSums(A)); M <- d * A %*% diag(d)
  e <- eigen(M, symmetric = TRUE)
  U <- e$vectors[, 1:k, drop = FALSE]; Y <- U / sqrt(rowSums(U^2))
  set.seed(1); km <- kmeans(Y, k, nstart = 50, iter.max = 100)
  first <- match(unique(km$cluster), km$cluster)        # relabel clusters in order of first appearance
  list(mu = e$values[1:k], lsym = 1 - e$values[1:k], min_mu = min(e$values), cl = match(km$cluster, km$cluster[first]))
}

case("spectral_njw", "ml/clustering", "Exact NJW spectral clustering: ap_blobs (default gamma) and spectral_pair (gamma = 0.5)", function(ctx) {
  X <- as.matrix(ctx$dataset("datasets/generated/ap_blobs.csv"))
  cm <- colMeans(X); gamma <- 1 / (ncol(X) * (sum(sweep(X, 2, cm)^2) / length(X)))   # p2a's default gamma
  a <- .njw(X, gamma, 3)
  set.seed(20261115)
  P <- rbind(cbind(rnorm(20, 0, 0.3), rnorm(20, 0, 0.3)), cbind(rnorm(20, 4, 0.3), rnorm(20, 0, 0.3)),
             cbind(40 + c(0, 0.1) + rnorm(2, 0, 0.005), 40 + rnorm(2, 0, 0.005)))
  d <- ctx$gen_dataset("spectral_pair", data.frame(x = P[, 1], y = P[, 2])); P <- as.matrix(d)
  b2 <- .njw(P, 0.5, 2); b3 <- .njw(P, 0.5, 3)
  list(values = list(blobs_gamma = gamma, blobs_mu = vec(a$mu), blobs_lsym = vec(a$lsym), blobs_cluster = vec(a$cl),
                     pair_k2_mu = vec(b2$mu), pair_k2_cluster = vec(b2$cl), pair_min_mu = b2$min_mu,
                     pair_k3_mu = vec(b3$mu), pair_k3_cluster = vec(b3$cl)),
       reference = "eigen(D^-1/2 A D^-1/2, symmetric = TRUE) with A = exp(-gamma |x - y|^2), diag 0; kmeans(rows / norms, k, nstart = 50) (kernlab::specc's embedding)",
       call = "NJW(ap_blobs, gamma = 1/(p var), k = 3); NJW(spectral_pair, gamma = 0.5, k = 2, 3)", packages = character(0),
       dataset = "validation/datasets/generated/spectral_pair.csv")
})

case("cluster_stats_blobs", "ml/clustering", "clusterCrit::intCriteria and fpc::cluster.stats on ap_blobs, PAM k = 3 partition", function(ctx) {
  X <- as.matrix(ctx$dataset("datasets/generated/ap_blobs.csv"))
  cl <- pam(X, 3)$clustering
  ic <- intCriteria(X, as.integer(cl), c("Calinski_Harabasz", "Davies_Bouldin", "Dunn", "Silhouette"))
  cs <- cluster.stats(dist(X), cl)
  # DB with the mean within-cluster pairwise distance as scatter (p2a's cluster_stats before the fix), for the record
  cen <- t(sapply(1:3, function(k) colMeans(X[cl == k, , drop = FALSE]))); D <- as.matrix(dist(X))
  s_pair <- sapply(1:3, function(k) { I <- which(cl == k); mean(sapply(I, function(i) sum(D[i, setdiff(I, i)]) / (length(I) - 1))) })
  cd <- as.matrix(dist(cen)); diag(cd) <- Inf
  db_pair <- mean(sapply(1:3, function(k) max((s_pair[k] + s_pair[-k]) / cd[k, -k])))
  within_pairs <- unlist(lapply(1:3, function(k) { I <- which(cl == k); D[I, I][upper.tri(D[I, I])] }))
  between_pairs <- D[outer(cl, cl, "!=") & upper.tri(D)]
  list(values = list(labels = vec(cl), ch = ic$calinski_harabasz, db = ic$davies_bouldin, dunn = ic$dunn,
                     silhouette = ic$silhouette, within_ss = cs$within.cluster.ss, avg_silwidth = cs$avg.silwidth,
                     fpc_ch = cs$ch, fpc_dunn = cs$dunn, fpc_average_within = cs$average.within,
                     fpc_average_between = cs$average.between, fpc_separation = min(cs$separation),
                     fpc_max_diameter = max(cs$diameter), fpc_cluster_avg_silwidths = vec(cs$clus.avg.silwidths),
                     pairwise_average_within = mean(within_pairs),
                     pairwise_average_between = mean(between_pairs), db_pairwise_scatter_old = db_pair),
       reference = "clusterCrit::intCriteria(x, cl, c('Calinski_Harabasz','Davies_Bouldin','Dunn','Silhouette')); fpc::cluster.stats(dist(x), cl)",
       call = "intCriteria(ap_blobs, pam(ap_blobs, 3)$clustering, ...); cluster.stats(dist(ap_blobs), cl)",
       packages = c("cluster", "clusterCrit", "fpc"))
})

case("partition_agreement", "ml/clustering", "aricode NMI variants and ARI on fixed partitions (0-based ids with gaps in p2a)", function(ctx) {
  ctx$dataset("datasets/generated/ap_blobs.csv")   # no data needed; the labels are below
  u <- c(1,1,1,1,2,2,2,2,3,3,3,3,3,4,4); w <- c(1,1,2,1,2,2,3,2,3,3,3,1,3,3,4)
  v <- list(u = vec(u), w = vec(w), ari = ARI(u, w))
  for (vv in c("max", "min", "sqrt", "sum")) v[[paste0("nmi_", vv)]] <- NMI(u, w, variant = vv)
  list(values = v, reference = "aricode::NMI(u, w, variant = 'max'|'min'|'sqrt'|'sum'); aricode::ARI(u, w)",
       call = "NMI(u, w, variant); ARI(u, w)", packages = "aricode")
})

case("gap_logw_blobs", "ml/clustering", "cluster::clusGap logW (d.power = 2) on ap_blobs and maxSE('Tibs2001SEmax')", function(ctx) {
  X <- as.matrix(ctx$dataset("datasets/generated/ap_blobs.csv"))
  set.seed(1)
  g <- clusGap(X, FUNcluster = function(x, k) kmeans(x, k, nstart = 50, iter.max = 100), K.max = 5, B = 20,
               d.power = 2, spaceH0 = "original", verbose = FALSE)
  # logW is deterministic given the partition. clusGap's W.k sums each pair once, so with d.power = 2 it is half
  # the within-cluster sum of squares (the paper's W_k = p2a's log_w): logW = log(WSS) - log(2).
  wss <- sapply(1:5, function(k) if (k == 1) sum(scale(X, scale = FALSE)^2) else { set.seed(1); kmeans(X, k, nstart = 50)$tot.withinss })
  stopifnot(max(abs(log(wss / 2) - g$Tab[, "logW"])) < 1e-12)
  f1 <- c(0.10, 0.40, 0.55, 0.52, 0.60); s1 <- c(0.02, 0.03, 0.05, 0.04, 0.03)
  f2 <- c(0.10, 0.20, 0.30, 0.40, 0.50); s2 <- rep(0.01, 5)
  list(values = list(logW = vec(g$Tab[, "logW"]), log_wss = vec(log(wss)), f1 = vec(f1), se1 = vec(s1), f2 = vec(f2), se2 = vec(s2),
                     tibs_k1 = maxSE(f1, s1, "Tibs2001SEmax"), tibs_k2 = maxSE(f2, s2, "Tibs2001SEmax"),
                     sd_example = sd(c(1.0, 1.5, 0.7, 1.2))),
       reference = "cluster::clusGap(x, kmeans, K.max = 5, d.power = 2, spaceH0 = 'original')$Tab[, 'logW']; cluster::maxSE(f, SE, 'Tibs2001SEmax'); stats::sd",
       call = "clusGap(ap_blobs, kmeans, 5, d.power = 2, spaceH0 = 'original'); maxSE(f, se, 'Tibs2001SEmax')", packages = "cluster")
})

case("pvclust_msfit", "ml/clustering", "pvclust:::msfit on fixed bootstrap counts; realised scale sizes", function(ctx) {
  ctx$dataset("datasets/generated/ap_blobs.csv")   # no data needed; the counts are below
  suppressPackageStartupMessages(library(pvclust))
  r <- seq(0.5, 1.4, by = 0.1)
  v <- list(r = vec(r))
  # exact-model counts out of 1000: bp(r) = pnorm(-(v sqrt(r) + c / sqrt(r)))
  for (i in 1:4) {
    vc <- list(c(1, 0.3), c(-1, 0.2), c(2, 0), c(0.5, -0.4))[[i]]
    cnt <- round(1000 * pnorm(-(vc[1] * sqrt(r) + vc[2] / sqrt(r))))
    f <- pvclust:::msfit(cnt / 1000, r, 1000)
    tg <- paste0("fit", i)
    v[[paste0(tg, "_counts")]] <- vec(cnt)
    v[[paste0(tg, "_au")]] <- f$p[["au"]]; v[[paste0(tg, "_bp")]] <- f$p[["bp"]]
    v[[paste0(tg, "_se_au")]] <- f$se[["au"]]; v[[paste0(tg, "_v")]] <- f$coef[["v"]]; v[[paste0(tg, "_c")]] <- f$coef[["c"]]
  }
  # fewer than three scales strictly inside (0.001, 0.999): AU = BP = 0 if mean(bp) < 0.5, else 1
  d1 <- c(0, 0, 0, 0, 0, 0, 0, 0, 1, 2); d2 <- c(1000, 1000, 1000, 1000, 1000, 1000, 1000, 999, 998, 997)
  d3 <- c(998, 999, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000)
  for (j in 1:3) {
    cnt <- list(d1, d2, d3)[[j]]; f <- pvclust:::msfit(cnt / 1000, r, 1000)
    v[[paste0("degenerate", j, "_counts")]] <- vec(cnt)
    v[[paste0("degenerate", j, "_au")]] <- f$p[["au"]]; v[[paste0("degenerate", j, "_bp")]] <- f$p[["bp"]]
  }
  for (p in c(2, 5, 20)) v[[paste0("sizes_p", p)]] <- vec(round(p * r))
  list(values = v, reference = "pvclust:::msfit(bp, r, nboot); round(p * r) as pvclust:::boot.hclust",
       call = "pvclust:::msfit(counts / 1000, seq(0.5, 1.4, by = 0.1), 1000)", packages = "pvclust")
})
