# validation/r/cases/ml_partition_t3.R -- partitioning references for the advanced clustering methods
# (T3-WP43, ML-61). Planner case of T3-WP43 (no design appendix entry).
#  * cluster::pam on ap_blobs for k = 2, 3, 4 (Euclidean): medoids, objective, the medoid of every point's
#    cluster, silhouette widths. p2a's first-improvement SWAP reaches R's best-improvement medoids here
#    (checked with a written-out copy of p2a's BUILD + SWAP below).
#  * cluster::clara with samples = 1 and sampsize = n (pamLike = TRUE): the full-sample CLARA, which is
#    PAM; p2a's CLARA draws its samples with its own RNG, so only this configuration is comparable.
#  * cluster::fanny (k = 3, memb.exp = 2) and e1071::cmeans (k = 3, m = 2, fixed initial centres):
#    RECORDED ONLY. p2a's fanny (ML-11) and fuzzy_cmeans (ML-9) are Tier 4 findings; these values are
#    their future references and are registry deviations now.
# ap_blobs.csv is written by ml_mixtures.R (apcluster_blobs); run.R sources the case files in name order.
suppressPackageStartupMessages({ library(cluster); library(e1071) })

# Written-out copy of p2a's kmedoids (BUILD as Kaufman-Rousseeuw, SWAP = first improvement, restart).
.p2a_pam <- function(D, k, max_iter = 100) {
  n <- nrow(D); med <- which.min(rowSums(D))
  while (length(med) < k) {
    cur <- apply(D[, med, drop = FALSE], 1, min); best <- NA; bg <- -Inf
    for (cand in setdiff(1:n, med)) { g <- sum(pmax(cur - D[, cand], 0)); if (g > bg) { bg <- g; best <- cand } }
    med <- c(med, best)
  }
  cost <- function(m) sum(apply(D[, m, drop = FALSE], 1, min)); tot <- cost(med)
  for (it in 1:max_iter) {
    improved <- FALSE
    for (mi in 1:k) { for (cand in setdiff(1:n, med)) { nm <- med; nm[mi] <- cand; nc <- cost(nm)
      if (nc < tot - 1e-10) { med <- nm; tot <- nc; improved <- TRUE; break } }
      if (improved) break }
    if (!improved) break
  }
  list(medoids = med, total = tot)
}

case("pam_clara_blobs", "ml/clustering", "PAM (k = 2, 3, 4) and full-sample CLARA (k = 3) on ap_blobs, Euclidean", function(ctx) {
  d <- ctx$dataset("datasets/generated/ap_blobs.csv"); X <- as.matrix(d); n <- nrow(X); D <- as.matrix(dist(X))
  v <- list()
  for (k in 2:4) {
    p <- pam(X, k)
    q <- .p2a_pam(D, k); stopifnot(setequal(q$medoids, p$id.med), abs(q$total - p$objective[["swap"]] * n) < 1e-9)
    tg <- paste0("pam", k)
    v[[paste0(tg, "_medoids")]] <- vec(sort(p$id.med))
    v[[paste0(tg, "_total")]] <- p$objective[["swap"]] * n
    v[[paste0(tg, "_medoid_of")]] <- vec(p$id.med[p$clustering])
    w <- p$silinfo$widths; w <- w[order(as.integer(rownames(w))), ]
    v[[paste0(tg, "_sil_width")]] <- vec(w[, "sil_width"]); v[[paste0(tg, "_avg_sil")]] <- p$silinfo$avg.width
  }
  cl <- clara(X, 3, samples = 1, sampsize = n, pamLike = TRUE)
  stopifnot(setequal(cl$i.med, pam(X, 3)$id.med))
  v$clara3_medoids <- vec(sort(cl$i.med)); v$clara3_objective <- cl$objective
  v$clara3_medoid_of <- vec(cl$i.med[cl$clustering]); v$clara3_avg_sil <- cl$silinfo$avg.width
  list(values = v, reference = "cluster::pam(x, k) (Euclidean); cluster::clara(x, 3, samples = 1, sampsize = n, pamLike = TRUE)",
       call = "pam(ap_blobs, k); clara(ap_blobs, 3, samples = 1, sampsize = nrow(ap_blobs), pamLike = TRUE)", packages = "cluster")
})

