# validation/r/cases/econ_iv.R -- AER::ivreg first-stage (weak-instrument) F references
# (T2-WP08: ECON-3), and 2SLS on data with missing cells (ECON-71).
suppressPackageStartupMessages({ library(AER); library(sandwich) })

.iv_values <- function(m, d, endog, exog, instr) {
  s <- summary(m, diagnostics = TRUE)
  hc1 <- function(x) sandwich::vcovHC(x, type = "HC1")
  sr <- summary(m, vcov. = hc1, diagnostics = TRUE)
  dg <- s$diagnostics; dgr <- sr$diagnostics
  out <- list(coef = named(coef(m)), se = named(sqrt(diag(vcov(m)))),
              se_hc1 = named(sqrt(diag(hc1(m)))), se_hc0 = named(sqrt(diag(sandwich::vcovHC(m, type = "HC0")))))
  wk <- grep("^Weak", rownames(dg))
  out$weak_f <- vec(dg[wk, "statistic"]); out$weak_df1 <- vec(dg[wk, "df1"]); out$weak_df2 <- vec(dg[wk, "df2"])
  out$weak_p <- vec(dg[wk, "p-value"])
  out$weak_f_hc1 <- vec(dgr[wk, "statistic"]); out$weak_p_hc1 <- vec(dgr[wk, "p-value"])
  out$wu_hausman <- dg["Wu-Hausman", "statistic"]; out$wu_hausman_p <- dg["Wu-Hausman", "p-value"]
  if ("Sargan" %in% rownames(dg) && !is.na(dg["Sargan", "statistic"])) {
    out$sargan <- dg["Sargan", "statistic"]; out$sargan_p <- dg["Sargan", "p-value"]
  }
  # overall F of each first stage (what p2a reported before the fix), for the record
  out$overall_first_stage_f <- vec(sapply(endog, function(e) {
    f <- summary(lm(reformulate(c(exog, instr), e), data = d))$fstatistic; unname(f["value"]) }))
  out$n_obs <- nobs(m)
  out
}

case("iv_weak_controls", "econometrics/iv", "2SLS with an exogenous control that predicts the endogenous regressor; weak excluded instruments", function(ctx) {
  set.seed(20261021)
  n <- 500
  w <- rnorm(n); z1 <- rnorm(n); z2 <- rnorm(n); u <- rnorm(n)
  x <- 2 * w + 0.08 * z1 + 0.05 * z2 + u + rnorm(n)
  y <- 1 + 0.5 * x + 0.5 * w + u + rnorm(n)
  d <- ctx$gen_dataset("iv_weak_controls", data.frame(y = y, x = x, w = w, z1 = z1, z2 = z2))
  m <- ivreg(y ~ x + w | w + z1 + z2, data = d)
  v <- .iv_values(m, d, "x", "w", c("z1", "z2"))
  # run_first_stage_diagnostics with controls: partial F = anova(restricted, full)
  a <- anova(lm(x ~ w, data = d), lm(x ~ w + z1 + z2, data = d))
  v$fs_partial_f <- a$F[2]; v$fs_partial_p <- a$`Pr(>F)`[2]; v$fs_partial_df1 <- a$Df[2]; v$fs_partial_df2 <- a$Res.Df[2]
  fs <- summary(lm(x ~ z1 + z2 + w, data = d))
  v$fs_r2 <- fs$r.squared; v$fs_adj_r2 <- fs$adj.r.squared
  v$fs_partial_r2 <- 1 - sum(residuals(lm(x ~ w + z1 + z2, data = d))^2) / sum(residuals(lm(x ~ w, data = d))^2)
  v$fs_instr_coef <- named(coef(fs)[c("z1", "z2"), 1]); v$fs_instr_se <- named(coef(fs)[c("z1", "z2"), 2])
  stopifnot(v$weak_f < 10, v$overall_first_stage_f > 10)
  list(values = v, reference = "AER::ivreg; summary.ivreg(diagnostics = TRUE) (weak-instrument F = partial F of the excluded instruments); vcov. = sandwich::vcovHC(type = 'HC1') for the robust F",
       call = "ivreg(y ~ x + w | w + z1 + z2)", packages = c("AER", "sandwich"))
})

case("iv_two_endog", "econometrics/iv", "2SLS with two endogenous regressors, two controls and three instruments", function(ctx) {
  set.seed(77)
  n <- 400
  w1 <- rnorm(n); w2 <- runif(n); z1 <- rnorm(n); z2 <- rnorm(n); z3 <- rnorm(n); u <- rnorm(n)
  x1 <- 0.6 * z1 + 0.3 * z3 + 0.5 * w1 + u + rnorm(n)
  x2 <- 0.4 * z2 - 0.2 * z3 + 0.8 * w2 + 0.5 * u + rnorm(n)
  y <- 2 + 1.0 * x1 - 0.5 * x2 + 0.3 * w1 + 0.2 * w2 + u + rnorm(n)
  d <- ctx$gen_dataset("iv_two_endog", data.frame(y = y, x1 = x1, x2 = x2, w1 = w1, w2 = w2, z1 = z1, z2 = z2, z3 = z3))
  m <- ivreg(y ~ x1 + x2 + w1 + w2 | w1 + w2 + z1 + z2 + z3, data = d)
  list(values = .iv_values(m, d, c("x1", "x2"), c("w1", "w2"), c("z1", "z2", "z3")),
       reference = "AER::summary.ivreg(diagnostics = TRUE): one weak-instrument F per endogenous regressor",
       call = "ivreg(y ~ x1 + x2 + w1 + w2 | w1 + w2 + z1 + z2 + z3)", packages = c("AER", "sandwich"))
})

case("iv_no_controls", "econometrics/iv", "2SLS without exogenous controls (first-stage F was already correct)", function(ctx) {
  set.seed(5)
  n <- 300
  z1 <- rnorm(n); z2 <- rnorm(n); u <- rnorm(n)
  x <- 0.3 * z1 + 0.2 * z2 + u + rnorm(n)
  y <- 1 + 0.7 * x + u + rnorm(n)
  d <- ctx$gen_dataset("iv_no_controls", data.frame(y = y, x = x, z1 = z1, z2 = z2))
  m <- ivreg(y ~ x | z1 + z2, data = d)
  list(values = .iv_values(m, d, "x", character(0), c("z1", "z2")),
       reference = "AER::summary.ivreg(diagnostics = TRUE)", call = "ivreg(y ~ x | z1 + z2)", packages = c("AER", "sandwich"))
})

case("iv_na_rows", "econometrics/iv", "2SLS and first-stage diagnostics on data with missing cells, an integer control and a text column (ivreg na.omit)", function(ctx) {
  set.seed(4242)
  n <- 300
  k <- sample(1:6, n, replace = TRUE)                    # integer control: written without a decimal point, read as Int64
  w <- rnorm(n); z1 <- rnorm(n); z2 <- rnorm(n); u <- rnorm(n)
  x <- 0.8 * z1 - 0.5 * z2 + 0.3 * w + 0.2 * k + u + rnorm(n)
  y <- 1 + 0.5 * x + 0.4 * w - 0.3 * k + u + rnorm(n)
  y[c(7, 140)] <- NA; x[33] <- NA; w[90] <- NA; z1[90] <- NA; z2[201] <- NA; k[260] <- NA
  d <- ctx$gen_dataset("iv_na_rows", data.frame(id = seq_len(n), label = sprintf("u%03d", seq_len(n)),
                                                y = y, x = x, w = w, k = k, z1 = z1, z2 = z2))
  stopifnot(is.integer(d$k), is.integer(d$id), is.character(d$label))
  m <- ivreg(y ~ x + w + k | w + k + z1 + z2, data = d)  # default na.action: na.omit
  v <- .iv_values(m, d[complete.cases(d), ], "x", c("w", "k"), c("z1", "z2"))
  v$n_dropped <- n - nobs(m)
  v$missing_y <- sum(is.na(d$y)); v$missing_x <- sum(is.na(d$x)); v$missing_w <- sum(is.na(d$w))
  v$missing_k <- sum(is.na(d$k)); v$missing_z1 <- sum(is.na(d$z1)); v$missing_z2 <- sum(is.na(d$z2))
  # run_first_stage_diagnostics(x; z1, z2; controls w, k) does not use y: its complete rows are those of (x, z1, z2, w, k)
  f <- d[complete.cases(d[, c("x", "z1", "z2", "w", "k")]), ]
  a <- anova(lm(x ~ w + k, data = f), lm(x ~ w + k + z1 + z2, data = f))
  fs <- summary(lm(x ~ z1 + z2 + w + k, data = f))
  v$fsd_n_obs <- nrow(f); v$fsd_partial_f <- a$F[2]; v$fsd_partial_p <- a[["Pr(>F)"]][2]
  v$fsd_partial_df1 <- a$Df[2]; v$fsd_partial_df2 <- a$Res.Df[2]
  v$fsd_partial_r2 <- 1 - a$RSS[2] / a$RSS[1]
  v$fsd_r2 <- fs$r.squared; v$fsd_adj_r2 <- fs$adj.r.squared
  v$fsd_instr_coef <- named(coef(fs)[c("z1", "z2"), 1]); v$fsd_instr_se <- named(coef(fs)[c("z1", "z2"), 2])
  stopifnot(v$n_dropped == 6, v$fsd_n_obs == 296, v$weak_f > 10)
  list(values = v, reference = "AER::ivreg (na.action = na.omit); summary.ivreg(diagnostics = TRUE), vcov. = sandwich::vcovHC(type = 'HC1'); first-stage partial F by anova(lm(x ~ w + k), lm(x ~ w + k + z1 + z2)) on the complete rows of the first-stage columns",
       call = "ivreg(y ~ x + w + k | w + k + z1 + z2)", packages = c("AER", "sandwich"))
})
