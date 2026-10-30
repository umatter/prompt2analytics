# validation/r/cases/causal_mediation_na.R -- T2-WP13: missing values in the mediation and weighting families.
# R fits every model of a mediation analysis on the rows complete in all of its variables
# (lm/glm na.omit; mediation::mediate refuses models fitted on different rows). The case
# pins those rows, the lm component models of run_medflex_dataset, and the glm propensity
# scores behind run_mediation_analysis's total effect (Hajek IPW with p2a's trimming rule;
# causalweight is not installed, so the total is a formula cross-check, not a package value).
# The weighting/matching/TMLE tests use the same file for their refusal counts (missing_*):
# MatchIt and tmle refuse missing treatment or covariates, and so does p2a for all of them.
case("causal_mediation_na", "econometrics/mediation",
     "Mediation with missing values: complete cases, lm mediator/outcome models, glm propensity-score total effect",
     function(ctx) {
  set.seed(20261021)
  n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n)
  d <- rbinom(n, 1, plogis(0.2 + 0.9 * x1 - 0.6 * x2))            # confounded treatment
  m <- 0.5 + 0.6 * d + 0.4 * x1 + 0.2 * x2 + rnorm(n, sd = 0.5)
  y <- 1 + 0.4 * d + 0.5 * m + 0.3 * d * m + 0.3 * x1 - 0.2 * x2 + rnorm(n, sd = 0.5)
  y[sample(n, 15)] <- NA; m[sample(n, 25)] <- NA; x1[sample(n, 10)] <- NA
  d_na <- d; d_na[sample(n, 6)] <- NA
  dat <- ctx$gen_dataset("causal_na", data.frame(
    id = seq_len(n), y = y, d = as.integer(d), m = m, x1 = x1, x2 = x2,
    d_na = as.integer(d_na), d_txt = ifelse(d == 1, "treated", "control")))
  # Integers are written without a decimal point, so polars reads d, d_na and id as Int64
  # (design 2.2); d_txt is text; id is never missing.
  stopifnot(is.integer(dat$d), is.integer(dat$d_na), is.integer(dat$id), is.character(dat$d_txt),
            !anyNA(dat$d), !anyNA(dat$x2), !anyNA(dat$d_txt))
  vars <- c("y", "d", "m", "x1", "x2")
  cc <- complete.cases(dat[, vars])
  k <- dat[cc, ]
  fm <- lm(m ~ d + x1 + x2, data = k)
  fy0 <- lm(y ~ d + m + x1 + x2, data = k)
  fy1 <- lm(y ~ d + m + d:m + x1 + x2, data = k)
  se <- function(f) sqrt(diag(vcov(f)))
  # p2a's propensity models: logit of d on (1, x1, x2) and on (1, x1, x2, m), fitted tightly.
  ctl <- glm.control(epsilon = 1e-14, maxit = 100)
  ps1 <- fitted(glm(d ~ x1 + x2, family = binomial, data = k, control = ctl))
  ps2 <- fitted(glm(d ~ x1 + x2 + m, family = binomial, data = k, control = ctl))
  hajek <- function(trim) {
    keep <- ps1 > trim & ps1 < 1 - trim & ps2 > trim & ps2 < 1 - trim
    dd <- k$d[keep]; yy <- k$y[keep]; p <- ps1[keep]
    c(total = sum(dd * yy / p) / sum(dd / p) - sum((1 - dd) * yy / (1 - p)) / sum((1 - dd) / (1 - p)),
      kept = sum(keep), margin = min(abs(c(ps1, ps2) - trim), abs(c(ps1, ps2) - (1 - trim))))
  }
  h0 <- hajek(0); h5 <- hajek(0.05)
  stopifnot(h5["margin"] > 1e-4)   # no propensity score sits at the trimming bound
  list(values = list(
    complete_rows = vec(which(cc) - 1), n_complete = sum(cc), n_dropped = sum(!cc),
    missing_y = sum(is.na(dat$y)), missing_m = sum(is.na(dat$m)), missing_x1 = sum(is.na(dat$x1)),
    missing_d_na = sum(is.na(dat$d_na)),
    n_treated = sum(k$d == 1), n_control = sum(k$d == 0),
    mean_m_control = mean(k$m[k$d == 0]), mean_m_treated = mean(k$m[k$d == 1]),
    med_coef = named(coef(fm)), med_se = named(se(fm)), med_r2 = summary(fm)$r.squared,
    out_coef = named(coef(fy0)), out_se = named(se(fy0)), out_r2 = summary(fy0)$r.squared,
    outi_coef = named(coef(fy1)), outi_se = named(se(fy1)), outi_r2 = summary(fy1)$r.squared,
    ipw_total_trim0 = unname(h0["total"]), ipw_kept_trim0 = unname(h0["kept"]),
    ipw_total_trim05 = unname(h5["total"]), ipw_kept_trim05 = unname(h5["kept"])),
    reference = paste("stats::complete.cases; stats::lm (na.omit); stats::glm(binomial, epsilon = 1e-14);",
                      "Hajek IPW total effect with p2a's trimming rule (cross-check, causalweight not installed)"),
    call = paste("lm(m ~ d + x1 + x2); lm(y ~ d + m + x1 + x2); lm(y ~ d + m + d:m + x1 + x2);",
                 "glm(d ~ x1 + x2, binomial); glm(d ~ x1 + x2 + m, binomial); all on complete.cases(y, d, m, x1, x2)"),
    packages = character(0))
})
