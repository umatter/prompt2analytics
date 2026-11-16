# validation/r/cases/spatial_diag_t4.R -- T4-WP26 references (spdep 1.4.2 installed; spatialprobit is not).
#  * spatial_lm_diag_t4: spdep::lm.morantest (OLS residuals; W symmetrised by listw2U, zero.policy = TRUE so N counts
#    the linked units) and spdep::lm.RStests(test = "all") (Anselin 1988; Anselin, Bera, Florax & Yoon 1996) on
#    lm(y ~ x1 + x2) of spatial_sim, 4-NN row-standardised W (knearneigh(use_kd_tree = FALSE)), a distance-band
#    W with varying row sums (style W and B) and a band with islands.
#  * spatial_localmoran_t4: spdep::localmoran(conditional = TRUE | FALSE, mlvar = TRUE, two-sided); the global I as
#    sum(Ii) / S0; and the exact limits of the conditional permutation p-values (spdep localmoran_perm with
#    no_repeat_in_row = TRUE, "Pr(folded) Sim") on a 7-unit fixture, by enumerating every neighbour draw.
#  * spatial_probit_pieces_t4: the deterministic pieces of the SAR/SEM probit sampler and summaries, written out
#    (LeSage & Pace 2009, ch. 10; spatialprobit::sarprobit is not installed): the rho full-conditional kernel
#    ln|I - rho W| - ||A z - c||^2 / 2 on a grid, the latent full conditional (precision H = A'A), exact impacts
#    (phi(mu), mu = A^-1 X b, as spatialprobit's direct/total effects but with the exact inverse), the marginal
#    probabilities Phi(mu_i / sigma_i), sigma_i^2 = [(A'A)^-1]_ii, and the composite log-likelihood.

.sd4_data <- function(ctx) ctx$dataset("datasets/generated/spatial_sim.csv")
.sd4_knn <- function(xy, k) spdep::knn2nb(spdep::knearneigh(xy, k, use_kd_tree = FALSE))

case("spatial_lm_diag_t4", "spatial/diagnostics", "Moran's I for OLS residuals (lm.morantest) and the five Rao score tests (lm.RStests) on 4-NN and distance-band W, with islands", function(ctx) {
  suppressPackageStartupMessages(library(spdep))
  d <- .sd4_data(ctx); xy <- as.matrix(d[, c("cx", "cy")])
  m <- lm(y ~ x1 + x2, data = d)
  lw <- nb2listw(.sd4_knn(xy, 4), style = "W")
  nbb <- dnearneigh(xy, 0, 0.2); stopifnot(all(card(nbb) > 0))
  nbi <- suppressWarnings(dnearneigh(xy, 0, 0.12)); isl <- which(card(nbi) == 0)
  stopifnot(length(isl) == 4)
  D <- as.matrix(dist(xy)); Dut <- D[upper.tri(D)]
  stopifnot(min(abs(Dut - 0.2)) > 1e-6, min(abs(Dut - 0.12)) > 1e-6)
  mt <- function(listw, alt) { r <- lm.morantest(m, listw, alternative = alt, zero.policy = TRUE)
    c(I = unname(r$estimate[1]), E = unname(r$estimate[2]), V = unname(r$estimate[3]), z = unname(r$statistic), p = r$p.value) }
  v <- list(
    lm_coef = named(coef(m)), residuals = vec(resid(m)),
    lmmoran_knn_greater = named(mt(lw, "greater")), lmmoran_knn_two_sided = named(mt(lw, "two.sided")),
    lmmoran_knn_less = named(mt(lw, "less")),
    lmmoran_band_W = named(mt(nb2listw(nbb, style = "W"), "greater")),
    lmmoran_band_B = named(mt(nb2listw(nbb, style = "B"), "greater")),
    lmmoran_islands_W = named(mt(nb2listw(nbi, style = "W", zero.policy = TRUE), "greater")),
    islands = vec(isl - 1))
  rs_vals <- function(listw, mod) { rs <- suppressWarnings(lm.RStests(mod, listw, test = "all"))
    list(stat = sapply(rs, function(t) unname(t$statistic)), p = sapply(rs, function(t) unname(t$p.value))) }
  r <- rs_vals(lw, m); v$rstests_knn_stat <- named(r$stat); v$rstests_knn_p <- named(r$p)
  rb <- rs_vals(nb2listw(nbb, style = "B"), m); v$rstests_band_B_stat <- named(rb$stat); v$rstests_band_B_p <- named(rb$p)
  # scale invariance in x and the identity SARMA = RSerr + adjRSlag = RSlag + adjRSerr
  d2 <- d; d2$x1 <- d2$x1 * 1000; r2 <- rs_vals(lw, lm(y ~ x1 + x2, data = d2))
  stopifnot(max(abs(r2$stat / r$stat - 1)) < 1e-10,
            abs(r$stat["SARMA"] - r$stat["RSerr"] - r$stat["adjRSlag"]) < 1e-10 * r$stat["SARMA"],
            abs(r$stat["SARMA"] - r$stat["RSlag"] - r$stat["adjRSerr"]) < 1e-10 * r$stat["SARMA"])
  list(values = v, reference = "spdep::lm.morantest(alternative, zero.policy = TRUE); spdep::lm.RStests(test = 'all') (spdep 1.4.2)",
       call = "lm.morantest(lm(y ~ x1 + x2), nb2listw(knn2nb(knearneigh(xy, 4, use_kd_tree = FALSE)))); lm.RStests(..., test = 'all')",
       packages = c("spdep"))
})

case("spatial_localmoran_t4", "spatial/diagnostics", "Local Moran's I (localmoran, conditional and total randomisation) and exact conditional-permutation limits", function(ctx) {
  suppressPackageStartupMessages(library(spdep))
  d <- .sd4_data(ctx); xy <- as.matrix(d[, c("cx", "cy")])
  lw <- nb2listw(.sd4_knn(xy, 4), style = "W"); W <- listw2mat(lw)
  v <- list()
  for (cond in c(TRUE, FALSE)) { L <- localmoran(d$y, lw, conditional = cond); tag <- if (cond) "cond" else "total"
    v[[paste0(tag, "_Ii")]] <- vec(L[, 1]); v[[paste0(tag, "_E")]] <- vec(L[, 2]); v[[paste0(tag, "_Var")]] <- vec(L[, 3])
    v[[paste0(tag, "_Z")]] <- vec(L[, 4]); v[[paste0(tag, "_p")]] <- vec(L[, 5]) }
  v$global_I <- sum(v$cond_Ii) / Szero(lw)
  stopifnot(abs(v$global_I - unname(moran.test(d$y, lw)$estimate[1])) < 1e-12)
  # 7 units, 2-NN row-standardised W (equal weights within a row, so a draw is an unordered neighbour set):
  # the exact distribution of I_i under conditional permutation without repeats in a row (neighbours drawn
  # from the other n - 1 values), and the limit of spdep's folded p, min(P(I >= I_i), P(I < I_i)).
  x7 <- d$x1[1:7]; lw7 <- nb2listw(.sd4_knn(xy[1:7, ], 2), style = "W"); W7 <- listw2mat(lw7)
  z7 <- x7 - mean(x7); s2 <- sum(z7^2) / 7; Ii7 <- (z7 / s2) * drop(W7 %*% z7)
  pge <- numeric(7)
  for (i in 1:7) { k <- sum(W7[i, ] > 0); others <- z7[-i]; cm <- combn(6, k)
    sims <- apply(cm, 2, function(s) (z7[i] / s2) * sum(others[s]) / k)
    pge[i] <- mean(sims >= Ii7[i] - 1e-12 * abs(Ii7[i])) }
  v$perm7_x <- vec(x7); v$perm7_Ii <- vec(Ii7); v$perm7_p_ge <- vec(pge); v$perm7_folded_limit <- vec(pmin(pge, 1 - pge))
  list(values = v, reference = "spdep::localmoran(conditional = TRUE | FALSE) (spdep 1.4.2); conditional permutation enumerated in base R",
       call = "localmoran(y, nb2listw(knn2nb(knearneigh(xy, 4, use_kd_tree = FALSE))), conditional = TRUE | FALSE)",
       packages = c("spdep"))
})

case("spatial_probit_pieces_t4", "econometrics/spatialprobit", "SAR/SEM probit full conditionals, exact impacts, marginal probabilities and composite log-likelihood, written out", function(ctx) {
  d <- .sd4_data(ctx); xy <- as.matrix(d[, c("cx", "cy")]); n <- nrow(d)
  Dm <- as.matrix(dist(xy)); diag(Dm) <- Inf; B <- matrix(0, n, n)
  for (i in 1:n) B[i, order(Dm[i, ])[1:4]] <- 1
  W <- B / rowSums(B); X <- cbind(1, d$x1, d$x2); yb <- as.numeric(d$y > 1.9)
  b <- c(-0.3, 0.8, -0.5); ev <- eigen(W, only.values = TRUE)$values
  logdet <- function(r) Re(sum(log(1 - r * ev)))
  # a fixed latent vector consistent with yb (positive where yb = 1)
  z <- ifelse(yb == 1, 0.5 + abs(d$x1) / 2, -0.5 - abs(d$x2) / 2)
  grid <- c(-0.9, -0.3, 0, 0.3, 0.6, 0.9)
  ker <- function(r, model) { A <- diag(n) - r * W
    res <- if (model == "sar") drop(A %*% z - X %*% b) else drop(A %*% (z - X %*% b)); logdet(r) - sum(res^2) / 2 }
  v <- list(beta = vec(b), yb = vec(yb), z = vec(z), grid = vec(grid),
            sar_rho_kernel = vec(sapply(grid, ker, model = "sar")), sem_rho_kernel = vec(sapply(grid, ker, model = "sem")),
            logdet_grid = vec(sapply(grid, logdet)))
  # latent full conditional at rho = 0.6: mean m_i = mu_i - sum_{j != i} H_ij (z_j - mu_j) / H_ii, var 1 / H_ii
  r <- 0.6; A <- diag(n) - r * W; H <- crossprod(A)
  for (model in c("sar", "sem")) { mu <- if (model == "sar") drop(solve(A, X %*% b)) else drop(X %*% b)
    cm <- sapply(1:n, function(i) mu[i] - sum(H[i, -i] * (z[-i] - mu[-i])) / H[i, i])
    alt <- z - drop(H %*% z - H %*% mu) / diag(H)                     # the solve-free form the sampler uses
    stopifnot(max(abs(cm - alt)) < 1e-12)
    v[[paste0(model, "_latent_mean_rho06")]] <- vec(cm) }
  v$latent_var_rho06 <- vec(1 / diag(H))
  # impacts (one draw), marginal probabilities and composite log-likelihood
  for (r in c(0.5, 0.9)) { A <- diag(n) - r * W; Ai <- solve(A); mu <- drop(Ai %*% X %*% b); phi <- dnorm(mu)
    sig <- sqrt(rowSums(Ai^2)); tag <- sub("\\.", "", sprintf("%.1f", r))
    dir <- b[2:3] * mean(phi * diag(Ai)); tot <- b[2:3] * mean(phi * rowSums(Ai))
    v[[paste0("sar_direct_rho", tag)]] <- vec(dir); v[[paste0("sar_total_rho", tag)]] <- vec(tot)
    v[[paste0("sar_indirect_rho", tag)]] <- vec(tot - dir)
    v[[paste0("sar_mu_rho", tag)]] <- vec(mu); v[[paste0("sar_diag_rho", tag)]] <- vec(diag(Ai))
    # the truncated series the code uses before the fix (powers up to 10, stopping when |rho^p| < 1e-10)
    S <- diag(n); P <- diag(n); rp <- 1
    for (p in 1:10) { P <- W %*% P; rp <- rp * r; S <- S + rp * P; if (abs(rp * r) < 1e-10) break }
    mut <- drop(S %*% X %*% b)
    v[[paste0("sar_total_neumann10_rho", tag)]] <- vec(b[2:3] * mean(dnorm(mut) * rowSums(S)))
    pr <- pnorm(mu / sig); v[[paste0("sar_prob_rho", tag)]] <- vec(pr); v[[paste0("sar_sigma_rho", tag)]] <- vec(sig)
    v[[paste0("sar_cll_rho", tag)]] <- sum(ifelse(yb == 1, pnorm(mu / sig, log.p = TRUE), pnorm(-mu / sig, log.p = TRUE))) }
  # SEM: mu = X b, sigma_i^2 = [(A'A)^-1]_ii; dP_i/dx_ik = b_k phi(mu_i / sigma_i) / sigma_i, no spillover
  r <- 0.6; Ai <- solve(diag(n) - r * W); sig <- sqrt(rowSums(Ai^2)); mu <- drop(X %*% b)
  v$sem_sigma_rho06 <- vec(sig); v$sem_prob_rho06 <- vec(pnorm(mu / sig))
  v$sem_direct_rho06 <- vec(b[2:3] * mean(dnorm(mu / sig) / sig))
  v$sem_cll_rho06 <- sum(ifelse(yb == 1, pnorm(mu / sig, log.p = TRUE), pnorm(-mu / sig, log.p = TRUE)))
  list(values = v, reference = "LeSage & Pace (2009, ch. 10) written out in base R (spatialprobit not installed)",
       call = "solve(diag(n) - rho * W); pnorm(mu / sigma); log|I - rho W| = Re(sum(log(1 - rho * eigen(W)$values)))",
       packages = character(0))
})
