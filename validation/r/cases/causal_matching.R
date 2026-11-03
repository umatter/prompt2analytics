# validation/r/cases/causal_matching.R -- MatchIt / optmatch references (T3-WP25: CAUS-20, CAUS-67,
# CAUS-70; contributes to CAUS-92). Package copy (canonical; supersedes T3 design Appendix A.22).
#  * nearest-neighbour matching (MatchIt 4.8.0 method = 'nearest'): logit PS with m.order 'largest'
#    (MatchIt's default for PS distances), 'smallest' and 'data'; probit PS (distance = 'glm',
#    link = 'probit'); 1:2 matching in rounds; a caliper of 0.1 SD of the PS; matching with
#    replacement; Mahalanobis (pooled within-group covariance) and Euclidean distances; the
#    summary.matchit balance table (standardize = TRUE) before and after matching;
#  * optimal full matching (method = 'full', optmatch 0.10.8): MatchIt's default tol = 0.001 accepts a
#    solution within tol * n of the optimum, so the exact optimum is taken from optmatch at tol = 1e-8
#    (checked against an independent LP during planning; the PS-distance optimum is not unique, the
#    Mahalanobis one is); bounded (min.controls 0.5, max.controls 2), the omit-fraction recovery
#    (max.controls 1) and the flipped problem (max.controls 0.5 on a subset with nt >= 2 nc);
#  * CEM (method = 'cem'): Sturges cut points (MatchIt's default), 'q4' quantile cut points and
#    explicit cut points;
#  * subclassification (method = 'subclass', subclass = 5, logit and probit PS);
#  * minimum-cost flow by LEMON's network simplex (rlemon 0.2.1), cross-checked with lpSolve.
suppressPackageStartupMessages({ library(MatchIt); library(optmatch) })
# Helpers are prefixed .mt_ because run.R sources every case file into one environment.

.mt_sim <- function(ctx) {
  set.seed(20261128)
  n <- 240; x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rbinom(n, 1, 0.4)
  tr <- rbinom(n, 1, plogis(-0.8 + 0.8 * x1 - 0.5 * x2 + 0.5 * x3))
  y <- 1 + 1.2 * tr + x1 + 0.5 * x2 + rnorm(n)
  ctx$gen_dataset("matching_sim", data.frame(id = seq_len(n), treat = tr, x1 = round(x1, 10), x2 = round(x2, 10), x3 = x3, y = round(y, 10)))
}

.mt_fml <- treat ~ x1 + x2 + x3
.mt_pairs <- function(m) { mm <- m$match.matrix; tr <- as.integer(rownames(mm)); vec(as.numeric(cbind(tr, as.integer(mm[, 1])))) }
# Partition labels: each unit's label is the smallest 1-based row index in its matched set (NA if unmatched).
.mt_canon <- function(s) { s <- as.character(s); out <- rep(NA_real_, length(s)); ok <- !is.na(s)
  out[ok] <- as.numeric(ave(which(ok), s[ok], FUN = min)); out }
# Sum over matched sets of the treated-control distances inside the set (the full-matching objective).
.mt_set_cost <- function(sub, treat, Dfull) { ok <- !is.na(sub)
  sum(vapply(split(which(ok), as.character(sub[ok])), function(ix) {
    t <- ix[treat[ix] == 1]; c <- ix[treat[ix] == 0]; sum(Dfull[t, c, drop = FALSE]) }, 0)) }
.mt_quiet <- function(expr) withCallingHandlers(expr, warning = function(w) invokeRestart("muffleWarning"))
.mt_maha_full <- function(X, treat) {
  t1 <- treat == 1
  Sp <- (cov(X[t1, ]) * (sum(t1) - 1) + cov(X[!t1, ]) * (sum(!t1) - 1)) / (nrow(X) - 2)
  Si <- solve(Sp); D <- matrix(0, nrow(X), nrow(X))
  for (a in which(t1)) for (b in which(!t1)) { df <- X[a, ] - X[b, ]; D[a, b] <- D[b, a] <- sqrt(sum(df * (Si %*% df))) }
  D
}
.mt_bal <- function(v, s, prefix) {
  for (part in c("all", "matched")) {
    tab <- s[[paste0("sum.", part)]][c("x1", "x2", "x3"), ]
    stage <- if (part == "all") "before" else "after"
    v[[paste0(prefix, "_", stage, "_mean_t")]] <- vec(unname(tab[, "Means Treated"]))
    v[[paste0(prefix, "_", stage, "_mean_c")]] <- vec(unname(tab[, "Means Control"]))
    v[[paste0(prefix, "_", stage, "_smd")]] <- vec(unname(tab[, "Std. Mean Diff."]))
    v[[paste0(prefix, "_", stage, "_var_ratio")]] <- vec(unname(tab[, "Var. Ratio"]))     # NA for the binary x3
    v[[paste0(prefix, "_", stage, "_ecdf_mean")]] <- vec(unname(tab[, "eCDF Mean"]))
    v[[paste0(prefix, "_", stage, "_ecdf_max")]] <- vec(unname(tab[, "eCDF Max"]))
  }
  v
}

case("matchit_nearest", "econometrics/matching", "MatchIt nearest neighbour (logit largest/smallest/data, probit, 1:2 rounds, caliper, replacement, Mahalanobis, Euclidean), PS, balance table", function(ctx) {
  d <- .mt_sim(ctx)
  v <- list()
  X <- as.matrix(d[, c("x1", "x2", "x3")]); t1 <- d$treat == 1
  m1 <- matchit(.mt_fml, data = d, method = "nearest", distance = "glm", m.order = "largest")
  v$logit_ps <- vec(head(m1$distance, 12)); v$logit_largest_pairs <- .mt_pairs(m1)
  v$logit_coef <- named(coef(m1$model))
  # Tight refit (glm epsilon 1e-14): the reference for fit_propensity (epsilon 1e-10); the pairs are
  # unchanged when MatchIt matches on the tight PS (checked).
  gl <- glm(.mt_fml, data = d, family = binomial("logit"), control = glm.control(epsilon = 1e-14, maxit = 100))
  v$logit_ps_tight <- vec(unname(fitted(gl)))
  stopifnot(identical(unname(m1$match.matrix), unname(matchit(.mt_fml, data = d, method = "nearest", distance = unname(fitted(gl)), m.order = "largest")$match.matrix)))
  m2 <- matchit(.mt_fml, data = d, method = "nearest", distance = "glm", m.order = "data")
  v$logit_data_pairs <- .mt_pairs(m2)
  m2s <- matchit(.mt_fml, data = d, method = "nearest", distance = "glm", m.order = "smallest")
  v$logit_smallest_pairs <- .mt_pairs(m2s)
  m3 <- matchit(.mt_fml, data = d, method = "nearest", distance = "glm", link = "probit", m.order = "largest")
  v$probit_ps <- vec(head(m3$distance, 12)); v$probit_coef <- named(coef(glm(.mt_fml, data = d, family = binomial("probit"))))
  gp <- glm(.mt_fml, data = d, family = binomial("probit"), control = glm.control(epsilon = 1e-14, maxit = 100))
  v$probit_coef_tight <- named(coef(gp)); v$probit_ps_tight <- vec(unname(fitted(gp)))
  v$probit_largest_pairs <- .mt_pairs(m3)
  # MatchIt's probit PS (glm epsilon 1e-8) differs from the tight fit by about 1e-7; the pairs do not.
  stopifnot(identical(unname(m3$match.matrix), unname(matchit(.mt_fml, data = d, method = "nearest", distance = unname(fitted(gp)), m.order = "largest")$match.matrix)))
  # 1:2 without replacement: every treated unit gets its first match before any gets a second (MatchIt
  # warns "Not all treated units will get 2 matches": 151 controls < 178).
  r2 <- .mt_quiet(matchit(.mt_fml, data = d, method = "nearest", distance = "glm", ratio = 2))
  v$ratio2_mm <- vec(as.numeric(cbind(as.integer(rownames(r2$match.matrix)), as.integer(r2$match.matrix[, 1]), as.integer(r2$match.matrix[, 2]))))
  v$ratio2_weights <- vec(r2$weights)
  cal <- matchit(.mt_fml, data = d, method = "nearest", distance = "glm", caliper = 0.1)
  v$caliper_pairs <- .mt_pairs(cal); v$caliper_abs <- 0.1 * sd(cal$distance); v$caliper_weights <- vec(cal$weights)
  rp <- matchit(.mt_fml, data = d, method = "nearest", distance = "glm", replace = TRUE)
  v$replace_pairs <- .mt_pairs(rp); v$replace_weights <- vec(rp$weights)
  # The 1:2, caliper and replacement matches are the same on the tight PS (p2a's glm is tighter than MatchIt's).
  pt <- unname(fitted(gl))
  stopifnot(identical(unname(r2$match.matrix), unname(.mt_quiet(matchit(.mt_fml, data = d, method = "nearest", distance = pt, ratio = 2))$match.matrix)),
            identical(unname(cal$match.matrix), unname(matchit(.mt_fml, data = d, method = "nearest", distance = pt, caliper = 0.1)$match.matrix)),
            identical(unname(rp$match.matrix), unname(matchit(.mt_fml, data = d, method = "nearest", distance = pt, replace = TRUE)$match.matrix)))
  m4 <- matchit(.mt_fml, data = d, method = "nearest", distance = "mahalanobis")
  v$maha_pairs <- .mt_pairs(m4)
  Sp <- (cov(X[t1, ]) * (sum(t1) - 1) + cov(X[!t1, ]) * (sum(!t1) - 1)) / (nrow(X) - 2)
  v$maha_pooled_cov <- vec(Sp)
  m5 <- matchit(.mt_fml, data = d, method = "nearest", distance = "euclidean")
  v$euclid_pairs <- .mt_pairs(m5)
  s1 <- summary(m1, standardize = TRUE)
  v$logit_largest_smd_after <- vec(unname(s1$sum.matched[c("x1", "x2", "x3"), "Std. Mean Diff."]))
  v$logit_largest_smd_before <- vec(unname(s1$sum.all[c("x1", "x2", "x3"), "Std. Mean Diff."]))
  v <- .mt_bal(v, s1, "logit_largest")
  v <- .mt_bal(v, summary(r2, standardize = TRUE), "ratio2")
  # Kish effective sample sizes of the matched sample, (sum w)^2 / sum w^2 per group (summary.matchit $nn)
  v$ratio2_ess <- vec(unname(summary(r2)$nn["Matched (ESS)", c("Control", "Treated")]))
  v$n_treated <- sum(t1); v$n_control <- sum(!t1)
  list(values = v, reference = "MatchIt::matchit(method = 'nearest', distance = 'glm' (logit | probit) | 'mahalanobis' | 'euclidean', m.order, ratio, caliper, replace); summary.matchit(standardize = TRUE)",
       call = "matchit(treat ~ x1 + x2 + x3, data = matching_sim, method = 'nearest', ...)", packages = c("MatchIt"))
})

case("matchit_full_cem", "econometrics/matching", "Optimal full matching (optmatch: default tol and exact, bounded, recovery, flipped) and CEM (Sturges, q4, explicit cut points)", function(ctx) {
  d <- ctx$dataset("datasets/generated/matching_sim.csv")
  v <- list()
  X <- as.matrix(d[, c("x1", "x2", "x3")])
  f <- matchit(.mt_fml, data = d, method = "full", distance = "glm", estimand = "ATT")
  v$full_subclass <- vec(as.numeric(f$subclass)); v$full_weights <- vec(f$weights)
  f2 <- matchit(.mt_fml, data = d, method = "full", distance = "glm", estimand = "ATT", min.controls = 0.5, max.controls = 2)
  v$full_bounded_subclass <- vec(as.numeric(f2$subclass)); v$full_bounded_weights <- vec(f2$weights)
  ps <- f$distance
  Dps <- abs(outer(ps, ps, "-"))
  v$full_total_distance <- .mt_set_cost(f$subclass, d$treat, Dps)
  v$full_bounded_total_distance <- .mt_set_cost(f2$subclass, d$treat, Dps)
  ft <- matchit(.mt_fml, data = d, method = "full", distance = "glm", estimand = "ATT", tol = 1e-8)
  v$full_tight_total_distance <- .mt_set_cost(ft$subclass, d$treat, Dps)
  f2t <- matchit(.mt_fml, data = d, method = "full", distance = "glm", estimand = "ATT", min.controls = 0.5, max.controls = 2, tol = 1e-8)
  v$full_bounded_tight_total_distance <- .mt_set_cost(f2t$subclass, d$treat, Dps)
  stopifnot(v$full_tight_total_distance < v$full_total_distance, v$full_bounded_tight_total_distance < v$full_bounded_total_distance)
  # probit PS from the tight refit (MatchIt's default glm epsilon leaves the probit PS about 1e-7 off)
  pp <- unname(fitted(glm(.mt_fml, data = d, family = binomial("probit"), control = glm.control(epsilon = 1e-14, maxit = 100))))
  fp <- matchit(.mt_fml, data = d, method = "full", distance = pp, estimand = "ATT", tol = 1e-8)
  v$full_probit_tight_total_distance <- .mt_set_cost(fp$subclass, d$treat, abs(outer(pp, pp, "-")))
  # Mahalanobis distance: the optimum is unique (planning: optmatch tol 1e-8 = independent LP), so the
  # partition and the weights are pinned.
  Dm <- .mt_maha_full(X, d$treat)
  fm <- matchit(.mt_fml, data = d, method = "full", distance = "mahalanobis", estimand = "ATT", tol = 1e-8)
  v$full_maha_tight_sets <- vec(.mt_canon(fm$subclass)); v$full_maha_tight_weights <- vec(fm$weights)
  v$full_maha_tight_total_distance <- .mt_set_cost(fm$subclass, d$treat, Dm)
  v$full_maha_tight_ess <- vec(unname(summary(fm)$nn["Matched (ESS)", c("Control", "Treated")]))
  fmd <- matchit(.mt_fml, data = d, method = "full", distance = "mahalanobis", estimand = "ATT")
  v$full_maha_default_total_distance <- .mt_set_cost(fmd$subclass, d$treat, Dm)
  fmb <- matchit(.mt_fml, data = d, method = "full", distance = "mahalanobis", estimand = "ATT", min.controls = 0.5, max.controls = 2, tol = 1e-8)
  v$full_maha_bounded_sets <- vec(.mt_canon(fmb$subclass)); v$full_maha_bounded_weights <- vec(fmb$weights)
  v$full_maha_bounded_total_distance <- .mt_set_cost(fmb$subclass, d$treat, Dm)
  # max.controls = 1 < nc / nt: optmatch's recovery matches only as many controls as the unconstrained
  # solution can use (here 79 of 151).
  fm1 <- matchit(.mt_fml, data = d, method = "full", distance = "mahalanobis", estimand = "ATT", max.controls = 1, tol = 1e-8)
  v$full_maha_max1_sets <- vec(.mt_canon(fm1$subclass)); v$full_maha_max1_weights <- vec(fm1$weights)
  v$full_maha_max1_total_distance <- .mt_set_cost(fm1$subclass, d$treat, Dm)
  # max.controls = 0.5 is solved by optmatch on the transposed problem; it needs nt >= 2 nc, so all 89
  # treated units and the first 40 controls (id <= flip_max_id).
  ctl <- which(d$treat == 0); flip_max_id <- d$id[ctl[40]]
  ds <- d[d$treat == 1 | d$id <= flip_max_id, ]
  Xs <- as.matrix(ds[, c("x1", "x2", "x3")])
  ff <- matchit(.mt_fml, data = ds, method = "full", distance = "mahalanobis", estimand = "ATT", max.controls = 0.5, tol = 1e-8)
  v$flip_max_id <- flip_max_id
  v$full_maha_flip_sets <- vec(.mt_canon(ff$subclass)); v$full_maha_flip_weights <- vec(ff$weights)
  v$full_maha_flip_total_distance <- .mt_set_cost(ff$subclass, ds$treat, .mt_maha_full(Xs, ds$treat))
  ce <- matchit(.mt_fml, data = d, method = "cem", estimand = "ATT")
  v$cem_weights <- vec(ce$weights); v$cem_n_matched_t <- sum(ce$weights[d$treat == 1] > 0); v$cem_n_matched_c <- sum(ce$weights[d$treat == 0] > 0)
  v$cem_n_strata <- nlevels(ce$subclass)
  cq <- matchit(.mt_fml, data = d, method = "cem", estimand = "ATT", cutpoints = "q4")
  v$cem_q4_weights <- vec(cq$weights); v$cem_q4_n_strata <- nlevels(cq$subclass)
  cx <- matchit(.mt_fml, data = d, method = "cem", estimand = "ATT", cutpoints = list(x1 = c(-1, 0, 1), x2 = c(-0.5, 0.5), x3 = c(0.2, 0.8)))
  v$cem_cut_weights <- vec(cx$weights); v$cem_cut_n_strata <- nlevels(cx$subclass)
  list(values = v, reference = "MatchIt::matchit(method = 'full', estimand = 'ATT') via optmatch::fullmatch (tol 0.001 and 1e-8); matchit(method = 'cem') with Sturges, 'q4' and explicit cut points",
       call = "matchit(treat ~ x1 + x2 + x3, data = matching_sim, method = 'full' | 'cem', ...)", packages = c("MatchIt", "optmatch"))
})

case("matchit_subclass", "econometrics/matching", "MatchIt subclassification (5 subclasses on the treated PS quantiles; logit and probit)", function(ctx) {
  d <- ctx$dataset("datasets/generated/matching_sim.csv")
  v <- list()
  for (lk in c("logit", "probit")) {
    sc <- matchit(.mt_fml, data = d, method = "subclass", subclass = 5, distance = "glm", link = lk)
    # every subclass holds both groups, so MatchIt's min.n = 1 scoot does not move anything
    stopifnot(all(table(d$treat, sc$subclass) >= 1),
              identical(as.integer(findInterval(sc$distance, sc$q.cut, all.inside = TRUE)), as.integer(as.character(sc$subclass))))
    v[[paste0("subclass5_", lk, "_q_cut")]] <- vec(unname(sc$q.cut))
    # the same cut points on the tight-refit PS (glm epsilon 1e-14; quantile type 7 of the treated PS)
    pt <- fitted(glm(.mt_fml, data = d, family = binomial(lk), control = glm.control(epsilon = 1e-14, maxit = 100)))
    v[[paste0("subclass5_", lk, "_q_cut_tight")]] <- vec(unname(quantile(pt[d$treat == 1], probs = seq(0, 1, length.out = 6), type = 7)))
    v[[paste0("subclass5_", lk, "_subclass")]] <- vec(as.numeric(as.character(sc$subclass)))
    v[[paste0("subclass5_", lk, "_weights")]] <- vec(sc$weights)
  }
  list(values = v, reference = "MatchIt::matchit(method = 'subclass', subclass = 5, distance = 'glm', link = 'logit' | 'probit')",
       call = "matchit(treat ~ x1 + x2 + x3, data = matching_sim, method = 'subclass', subclass = 5, link = ...)", packages = c("MatchIt"))
})

case("mcf_networks", "econometrics/matching", "Minimum-cost flow (LEMON network simplex via rlemon; lpSolve cross-check) on seeded random networks", function(ctx) {
  set.seed(20261023)
  rows <- list()
  # Networks 1-4: random arcs (costs -3..20, so negative-cost cycles occur), three supply and three
  # demand nodes, plus a backbone path through every node (capacity = total supply, cost 30) that makes
  # the problem feasible. Network 5: no backbone and a supply far beyond the capacity (infeasible).
  for (k in 1:5) {
    nn <- c(8, 12, 20, 40, 6)[k]; na <- c(20, 40, 90, 200, 10)[k]
    tl <- sample.int(nn, na, replace = TRUE); hd <- sample.int(nn, na, replace = TRUE)
    keep <- tl != hd; tl <- tl[keep]; hd <- hd[keep]
    cap <- sample.int(9, length(tl), replace = TRUE); cost <- sample(-3:20, length(tl), replace = TRUE)
    sup <- integer(nn); nodes6 <- sample.int(nn, 6)
    amt <- if (k == 5) c(50L, 0L, 0L) else c(7L, 4L, 2L)
    sup[nodes6[1:3]] <- amt; sup[nodes6[4:6]] <- -amt
    if (k < 5) { perm <- sample.int(nn); tl <- c(tl, perm[-nn]); hd <- c(hd, perm[-1]); cap <- c(cap, rep(sum(amt), nn - 1)); cost <- c(cost, rep(30L, nn - 1))
                 tl <- c(tl, perm[-1]); hd <- c(hd, perm[-nn]); cap <- c(cap, rep(sum(amt), nn - 1)); cost <- c(cost, rep(30L, nn - 1)) }
    rows[[k]] <- rbind(data.frame(network = k, kind = 1, a = tl - 1, b = hd - 1, cap = cap, cost = cost),
                       data.frame(network = k, kind = 2, a = seq_len(nn) - 1, b = -1, cap = sup, cost = 0))
  }
  df <- ctx$gen_dataset("mcf_networks", do.call(rbind, rows))
  v <- list()
  for (k in 1:5) {
    g <- df[df$network == k, ]; arcs <- g[g$kind == 1, ]; nodes <- g[g$kind == 2, ]
    out <- rlemon::MinCostFlow(arcSources = arcs$a + 1, arcTargets = arcs$b + 1, arcCapacities = arcs$cap,
                               arcCosts = arcs$cost, nodeSupplies = nodes$cap, numNodes = nrow(nodes), algorithm = "NetworkSimplex")
    feasible <- out[[4]] == "OPTIMAL"
    v[[paste0("net", k, "_feasible")]] <- as.numeric(feasible)
    if (feasible) {
      v[[paste0("net", k, "_cost")]] <- out[[3]]
      # lpSolve cross-check (arc flows in [0, cap], conservation flow_out - flow_in = supply)
      A <- matrix(0, nrow(nodes), nrow(arcs)); for (e in seq_len(nrow(arcs))) { A[arcs$a[e] + 1, e] <- 1; A[arcs$b[e] + 1, e] <- A[arcs$b[e] + 1, e] - 1 }
      lp <- lpSolve::lp("min", arcs$cost, rbind(A, diag(nrow(arcs))), c(rep("=", nrow(nodes)), rep("<=", nrow(arcs))), c(nodes$cap, arcs$cap))
      stopifnot(lp$status == 0, abs(lp$objval - out[[3]]) < 1e-9)
    }
  }
  list(values = v, reference = "rlemon::MinCostFlow(algorithm = 'NetworkSimplex') (LEMON 1.3, the solver optmatch uses through rlemon); lpSolve::lp cross-check",
       call = "MinCostFlow(arcSources, arcTargets, arcCapacities, arcCosts, nodeSupplies, numNodes, 'NetworkSimplex')", packages = c("rlemon", "lpSolve"))
})
