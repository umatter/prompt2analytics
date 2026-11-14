# validation/r/cases/ml_bart_t4.R -- T4-WP17 BART references (ML-15, ML-16, ML-64, ML-120).
#  * ml_bart_wbart_calibration -- BART::wbart's data-driven prior: fmean = mean(y), sigest = the lm
#    residual SD (sd(y) when p >= n), lambda = sigest^2 qchisq(1 - sigquant, sigdf) / sigdf, the leaf
#    prior SD tau = (max(y) - min(y)) / (2 k sqrt(ntree)), and the cutpoint grid of
#    BART::bartModelMatrix(numcut = 100): numcut equally spaced interior points of [min, max], or the
#    midpoints of the distinct values when a column has fewer than numcut of them. Exact.
#  * ml_bart_wbart_posterior -- BART::wbart on the same data (ntree 50, nskip 300, ndpost 1000, other
#    arguments at their defaults): posterior summaries averaged over 100 seeded chains, with the SD of
#    each summary across the chains (the Monte Carlo SD a single p2a chain is held to).
#  * ml_bart_pbart_posterior -- BART::pbart on bart_probit_sim (written by ml_bart_probit_glm in
#    ml_boosting_t3.R) with ntree 50, nskip 200, ndpost 300: the same kind of summaries.
# The chains' RNG is R's (set.seed(seed) before each call), so --check reproduces them exactly.
suppressPackageStartupMessages(library(BART))

.bart_popsd <- function(v) sqrt(mean((v - mean(v))^2))   # p2a's prediction_sd divides by the draw count

.bart_draw_summary <- function(draws, varcount, ntree, truth = NULL) {
  # draws: ndpost x n posterior draws of f (or of P(y = 1)); varcount: ndpost x p split counts
  m <- colMeans(draws)
  ci <- apply(draws, 2, function(c) quantile(c, c(0.025, 0.975), type = 7, names = FALSE))
  vc <- colMeans(varcount)
  out <- list(pred_mean_avg = mean(m),
              post_sd_avg = mean(apply(draws, 2, .bart_popsd)),
              width95_avg = mean(ci[2, ] - ci[1, ]),
              splits_per_tree = mean(rowSums(varcount)) / ntree,
              varprop = vc / sum(vc))
  if (!is.null(truth)) {
    out$rmse_f <- sqrt(mean((m - truth)^2))
    out$cover95_f <- mean(ci[1, ] <= truth & truth <= ci[2, ])
  }
  out
}

.bart_mc <- function(reps, one) {
  # one(seed) returns a named list of scalars and vectors; mean and SD across seeds, element-wise
  rs <- lapply(reps, one)
  keys <- names(rs[[1]])
  vals <- list()
  for (k in keys) {
    M <- do.call(rbind, lapply(rs, `[[`, k))
    vals[[k]] <- colMeans(M)
    vals[[paste0(k, "_mc_sd")]] <- apply(M, 2, sd)
  }
  vals
}

.bart_x_cols <- c("x1", "x2", "x3", "x4", "x5", "x6")

case("ml_bart_wbart_calibration", "ml/bart", "BART::wbart prior calibration (fmean, sigest, lambda, tau) and cutpoint grid", function(ctx) {
  set.seed(20261113)
  n <- 200
  X <- matrix(runif(n * 5), n, 5, dimnames = list(NULL, paste0("x", 1:5)))
  x6 <- sample(1:5, n, replace = TRUE)                     # five distinct values: the midpoint branch
  f <- 100 + 10 * sin(pi * X[, 1] * X[, 2]) + 20 * (X[, 3] - 0.5)^2 + 10 * X[, 4] + 5 * X[, 5]
  d <- ctx$gen_dataset("bart_friedman", data.frame(X, x6 = x6, f = f, y = f + rnorm(n)))
  x <- as.matrix(d[, .bart_x_cols]); y <- d$y
  sigest <- summary(lm(y ~ ., data.frame(x, y = y - mean(y))))$sigma
  qchi <- qchisq(1 - 0.9, 3)
  mm <- BART::bartModelMatrix(x, numcut = 100L, usequants = FALSE, rm.const = TRUE)
  list(values = list(fmean = mean(y), sigest = sigest, qchisq_0_1_3 = qchi, lambda = sigest^2 * qchi / 3,
                     tau_m50 = (max(y) - min(y)) / (2 * 2 * sqrt(50)),
                     tau_m200 = (max(y) - min(y)) / (2 * 2 * sqrt(200)),
                     numcut = vec(mm$numcut),
                     cut_x1 = vec(mm$xinfo[1, 1:mm$numcut[1]]),
                     cut_x6 = vec(mm$xinfo[6, 1:mm$numcut[6]])),
       reference = "BART::wbart (2.9.10) prior calibration as its R code computes it: y centred by fmean = mean(y); sigest = summary(lm(y ~ .))$sigma when p < n; lambda = sigest^2 * qchisq(1 - sigquant, sigdf) / sigdf (sigdf 3, sigquant 0.9); tau = (max(y) - min(y)) / (2 k sqrt(ntree)), k 2; cutpoints BART::bartModelMatrix(numcut = 100, usequants = FALSE)",
       call = "lm(y ~ x1 + ... + x6); qchisq(0.1, 3); bartModelMatrix(x, numcut = 100L, usequants = FALSE, rm.const = TRUE)",
       packages = c("BART"))
})

case("ml_bart_wbart_posterior", "ml/bart", "BART::wbart posterior summaries over 100 seeded chains", function(ctx) {
  d <- ctx$dataset("datasets/generated/bart_friedman.csv")
  x <- as.matrix(d[, .bart_x_cols]); y <- d$y; f <- d$f
  one <- function(seed) {
    set.seed(seed)
    invisible(capture.output(r <- BART::wbart(x, y, ntree = 50L, nskip = 300L, ndpost = 1000L, printevery = 1e6L)))
    s <- .bart_draw_summary(r$yhat.train, r$varcount, 50, truth = f)
    c(list(sigma_mean = mean(r$sigma[-(1:300)])), s)
  }
  vals <- .bart_mc(1:100, one)
  list(values = lapply(vals, function(v) if (length(v) > 1) vec(v) else v),
       reference = "BART::wbart (2.9.10), ntree 50, nskip 300, ndpost 1000, numcut 100, k 2, power 2, base 0.95, sigdf 3, sigquant 0.9; 100 chains (set.seed(1..100)); each value is the mean over the chains of a per-chain summary, and *_mc_sd its SD across the chains",
       call = "wbart(x, y, ntree = 50, nskip = 300, ndpost = 1000); summaries of yhat.train (population SD, quantile type 7), sigma[-(1:300)], varcount",
       packages = c("BART"))
})

case("ml_bart_pbart_posterior", "ml/bart", "BART::pbart posterior summaries over 100 seeded chains", function(ctx) {
  d <- ctx$dataset("datasets/generated/bart_probit_sim.csv")
  x <- as.matrix(d[, c("x1", "x2", "x3")]); y <- d$y
  one <- function(seed) {
    set.seed(seed)
    invisible(capture.output(r <- BART::pbart(x, y, ntree = 50L, nskip = 200L, ndpost = 300L, printevery = 1e6L)))
    c(list(binary_offset = r$binaryOffset), .bart_draw_summary(r$prob.train, r$varcount, 50))
  }
  vals <- .bart_mc(1:100, one)
  list(values = lapply(vals, function(v) if (length(v) > 1) vec(v) else v),
       reference = "BART::pbart (2.9.10), ntree 50, nskip 200, ndpost 300, numcut 100, k 2, power 2, base 0.95; 100 chains (set.seed(1..100)); each value is the mean over the chains of a per-chain summary of prob.train and varcount, and *_mc_sd its SD across the chains",
       call = "pbart(x, y, ntree = 50, nskip = 200, ndpost = 300)",
       packages = c("BART"))
})
