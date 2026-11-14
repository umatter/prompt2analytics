# validation/r/cases/causal_hettx_t4.R -- T4-WP13 references for the hettx rewrite (ECON-23, ECON-24, ECON-25).
# hettx (Ding, Feller & Miratrix) is not installed on the reference machine, so the procedures are
# written out in base R and sandwich from the papers:
#  * Neyman ATE: difference in means and sqrt(s1^2 / n1 + s0^2 / n0).
#  * DFM (2016) Fisher randomization test of a constant effect, H0: tau_i = tau for every i, with the
#    shifted Kolmogorov-Smirnov statistic SKS(Y, Z) = max_u |F1(u) - F0(u)| of the arm-centred outcomes
#    (ecdf of Y[Z = 1] - mean, of Y[Z = 0] - mean, evaluated at every pooled point), computed on integer
#    counts as max |a n0 - b n1| / (n1 n0). For each tau on an evenly spaced grid of G points over the
#    1 - gamma Neyman interval, Y(0) = Y - tau Z is imputed, every stored assignment Zb (a permutation
#    of Z: complete randomization) gives Yb = Y(0) + tau Zb and t_b = SKS(Yb, Zb); p(tau) =
#    (1 + #{t_b >= t_obs}) / (B + 1) (the same B assignments for every tau). The test's p-value is
#    min(1, max_tau p(tau) + gamma) (Berger-Boos); the plug-in p-value is p(te_hat) with no gamma.
#    The B = 199 assignments are stored in the case (assign_*: row-major B x n, 0/1) so that Rust
#    replays the same draws; R's sample() is not reproduced.
#  * DFM (2019) systematic variation: lm(y ~ treat * (x1 + x2)); the interaction coefficients
#    beta = beta1 - beta0 (separate-arm slopes), their HC2 covariance (sandwich::vcovHC, Lin 2013's
#    Neyman-type variance), the Wald chi-squared on 2 df and its upper-tail p, and the systematic
#    variance beta' S_xx beta (S_xx the sample covariance of (x1, x2), divisor n - 1).
#  * DFM (2016) conditional test of no idiosyncratic variation given (x1, x2), plug-in beta: the KS
#    distance between the arms' own regression residuals, on the same stored assignments.
# Data hettx_frt_sim: n = 40, 20 treated (fixed design), x1, x2 normal; y_het = y0 + treat (1 + 1.5 x1),
# y_homo = y0 + 2 treat (a constant effect).
case("hettx_dfm_t4", "econometrics/hettx",
     "Neyman ATE, DFM (2016) SKS randomization test of a constant effect over the gamma interval (stored assignments), DFM (2019) systematic-variation Wald test (HC2), conditional KS test of idiosyncratic variation (plug-in)",
     function(ctx) {
  set.seed(20261113)
  n <- 40; n1 <- 20
  x1 <- rnorm(n); x2 <- rnorm(n); y0 <- 0.5 * x2 + rnorm(n)
  treat <- rep(c(1, 0), each = n1)
  d <- ctx$gen_dataset("hettx_frt_sim", data.frame(
    y_het = round(y0 + treat * (1 + 1.5 * x1), 10), y_homo = round(y0 + 2 * treat, 10),
    treat = treat, x1 = round(x1, 10), x2 = round(x2, 10)))
  Z <- d$treat; n0 <- n - n1; B <- 199; G <- 21; gamma <- 1e-4
  A <- t(replicate(B, sample(Z)))                       # B x n assignments, complete randomization
  sks_num <- function(y, z) {
    a <- y[z == 1] - mean(y[z == 1]); b <- y[z == 0] - mean(y[z == 0])
    u <- c(a, b); F1 <- ecdf(a); F0 <- ecdf(b)
    max(abs(round(F1(u) * n1) * n0 - round(F0(u) * n0) * n1))   # exact integers
  }
  v <- list(n = n, n1 = n1, B = B, grid_size = G, gamma = gamma, assignments = vec(as.vector(t(A))))
  for (out in c("het", "homo")) {
    Y <- d[[paste0("y_", out)]]
    te <- mean(Y[Z == 1]) - mean(Y[Z == 0])
    se <- sqrt(var(Y[Z == 1]) / n1 + var(Y[Z == 0]) / n0)
    lo <- te - qnorm(1 - gamma / 2) * se; hi <- te + qnorm(1 - gamma / 2) * se
    grid <- seq(lo, hi, length.out = G)
    t_obs <- sks_num(Y, Z)
    p_at <- function(tau) {
      Y0 <- Y - tau * Z
      tb <- apply(A, 1, function(zb) sks_num(Y0 + tau * zb, zb))
      (1 + sum(tb >= t_obs)) / (B + 1)
    }
    p_tau <- vapply(grid, p_at, 0)
    v[[paste0(out, "_ate")]] <- te
    v[[paste0(out, "_ate_se")]] <- se
    v[[paste0(out, "_ci")]] <- vec(c(lo, hi))
    v[[paste0(out, "_grid")]] <- vec(grid)
    v[[paste0(out, "_sks_num")]] <- t_obs
    v[[paste0(out, "_sks")]] <- t_obs / (n1 * n0)
    v[[paste0(out, "_p_by_tau")]] <- vec(p_tau)
    v[[paste0(out, "_p")]] <- min(1, max(p_tau) + gamma)
    v[[paste0(out, "_p_plugin")]] <- p_at(te)
    # systematic variation (DFM 2019): interaction Wald test, HC2
    f <- lm(Y ~ treat * (x1 + x2), data = d)
    ib <- c("treat:x1", "treat:x2")
    beta <- coef(f)[ib]; V <- sandwich::vcovHC(f, type = "HC2")[ib, ib]
    w <- drop(t(beta) %*% solve(V, beta))
    v[[paste0(out, "_sys_beta")]] <- named(setNames(beta, c("x1", "x2")))
    v[[paste0(out, "_sys_se")]] <- named(setNames(sqrt(diag(V)), c("x1", "x2")))
    v[[paste0(out, "_sys_wald")]] <- w
    v[[paste0(out, "_sys_p")]] <- pchisq(w, 2, lower.tail = FALSE)
    v[[paste0(out, "_sys_var")]] <- drop(t(beta) %*% cov(cbind(d$x1, d$x2)) %*% beta)
    # idiosyncratic variation (DFM 2016, section 6; hettx detect_idiosyncratic with interaction.formula,
    # plug-in beta): H0 tau_i = b_treat + x_i' beta. Y(0) = Y - Z tau_i, Yb = Y(0) + Zb tau_i, and the
    # statistic is the KS distance between the two arms' residuals of their own lm(y ~ x1 + x2)
    # (the SKS statistic with covariate adjustment), on integer counts; p = (1 + #{t_b >= t_obs}) / (B + 1).
    tau_i <- drop(coef(f)[["treat"]] + cbind(d$x1, d$x2) %*% beta)
    X <- cbind(d$x1, d$x2)
    rks_num <- function(y, z) {
      a <- resid(lm(y[z == 1] ~ X[z == 1, ])); b <- resid(lm(y[z == 0] ~ X[z == 0, ]))
      u <- c(a, b); F1 <- ecdf(a); F0 <- ecdf(b)
      max(abs(round(F1(u) * n1) * n0 - round(F0(u) * n0) * n1))
    }
    r_obs <- rks_num(Y, Z)
    Y0 <- Y - Z * tau_i
    rb <- apply(A, 1, function(zb) rks_num(Y0 + zb * tau_i, zb))
    v[[paste0(out, "_idio_tau")]] <- vec(tau_i)
    v[[paste0(out, "_idio_ks_num")]] <- r_obs
    v[[paste0(out, "_idio_p")]] <- (1 + sum(rb >= r_obs)) / (B + 1)
  }
  list(values = v,
       reference = "base R (mean, var, ecdf, qnorm, pchisq) and sandwich::vcovHC(type = 'HC2') on lm(y ~ treat * (x1 + x2)); DFM (2016) and DFM (2019) written out (hettx not installed)",
       call = "SKS FRT over seq(te -/+ qnorm(1 - gamma/2) se, length.out = 21), gamma = 1e-4, B = 199 stored assignments; lm(y ~ treat * (x1 + x2)) + vcovHC(HC2) Wald on the interactions",
       packages = "sandwich")
})
