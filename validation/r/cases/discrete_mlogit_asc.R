# validation/r/cases/discrete_mlogit_asc.R -- discrete-choice references for T3-WP18 (ECON-13, ECON-58,
# ECON-59, ECON-60, ECON-61, X-19; the logit/probit/multinom part of X-10). Package copy (canonical; it
# supersedes design Appendix A.14: every Appendix A.14 key is kept with the same value, keys are added,
# and three cases are added).
#  * disc_mlogit_fishing: mlogit::mlogit(choice ~ generic | individual) includes J - 1 alternative-
#    specific constants (reference = first alternative, or reflevel) by default; `| 0 + income` drops
#    them (the model p2a fits before ECON-13). Data: mlogit's Fishing (1182 anglers, 4 modes), long
#    format. mlogit's own optimum is tight for `| income` (1e-14) but stops 1e-9 to 3e-8 relative from
#    the MLE for the other specifications, so every specification is re-solved by Newton's method on
#    the analytic conditional-logit score and information (`*_tight`); SEs from that information.
#  * disc_binary_glm: stats::glm(binomial) logit and probit at p2a's control (epsilon 1e-10, maxit 100):
#    coefficients, SEs, z, p, logLik, AIC, iterations; aliased regressors (NA), a dummy trap,
#    multivariate complete separation (glm's warning; also at R's default control), a quasi-separated
#    dummy, a near-collinear pair (the fit the old Newton/line-search loop stopped short of) and an
#    outcome outside {0, 1} (glm refuses it).
#  * disc_multinom_overlap: nnet::multinom on overlapping three-category data (finite MLE), re-solved
#    by Newton's method on the analytic multinomial-logit score and information (SEs from it; nnet's
#    vcov agrees to 1e-6); reference categories A (default) and B.
#  * disc_hurdle_zero: the binomial zero hurdle of pscl::hurdle is glm(I(y > 0) ~ ., binomial) on
#    T2-WP10's zeroinfl.csv; pscl and MASS refuse negative counts (and pscl non-integer ones).
#  * disc_polr_aliased: MASS::polr with an aliased regressor (xb = 2 x) on T1's ordered_j5.csv: polr
#    warns and drops xb; the fit on x alone is the reference for the identified coefficients.
suppressPackageStartupMessages({ library(mlogit); library(nnet) })

# Newton's method for the conditional logit on a long-format design (rows = situation x alternative):
# score sum_i (d_ij - p_ij) x_ij, information sum_i sum_j p_ij (x_ij - xbar_i)(x_ij - xbar_i)'.
.clogit_newton <- function(X, sit, chosen, b) {
  ll_at <- function(b) {
    v <- drop(X %*% b); mv <- ave(v, sit, FUN = max); s <- ave(exp(v - mv), sit, FUN = sum)
    sum((v - mv - log(s))[chosen == 1])
  }
  info_at <- function(b) {
    v <- drop(X %*% b); mv <- ave(v, sit, FUN = max); e <- exp(v - mv); s <- ave(e, sit, FUN = sum); p <- e / s
    xd <- X - rowsum(p * X, sit)[sit, , drop = FALSE]
    list(g = colSums((chosen - p) * X), I = crossprod(xd * sqrt(p)))
  }
  for (it in 1:50) {
    gi <- info_at(b); sc <- 1 / sqrt(diag(gi$I))
    step <- sc * solve(gi$I * outer(sc, sc), sc * gi$g)
    b <- b + step
    if (max(abs(step) / pmax(abs(b), 1e-300)) < 1e-15) break
  }
  gi <- info_at(b)
  stopifnot(max(abs(gi$g) * sqrt(diag(solve(gi$I)))) < 1e-9)   # score at the optimum, in SE units
  names(b) <- colnames(X)
  list(coef = b, se = setNames(sqrt(diag(solve(gi$I))), colnames(X)), loglik = ll_at(b))
}

case("disc_mlogit_fishing", "econometrics/discrete", "mlogit with and without alternative-specific constants, Fishing (mode ~ price + catch | income)", function(ctx) {
  data("Fishing", package = "mlogit")
  long <- mlogit::dfidx(Fishing, varying = 2:9, shape = "wide", choice = "mode")
  ld <- as.data.frame(long)
  ld$id <- ld$idx$id1; ld$alt <- as.character(ld$idx$id2); ld$idx <- NULL
  ld$chosen <- as.integer(ld$mode)
  d <- ctx$gen_dataset("mlogit_fishing", ld[, c("id", "alt", "chosen", "price", "catch", "income")])
  dl <- mlogit::dfidx(d, idx = c("id", "alt"), choice = "chosen")
  m1 <- mlogit(chosen ~ price + catch | income, data = dl, reflevel = "beach", tol = 1e-12, ftol = 1e-14)
  m0 <- mlogit(chosen ~ price + catch | 0 + income, data = dl, reflevel = "beach", tol = 1e-12, ftol = 1e-14)
  m2 <- mlogit(chosen ~ price + catch, data = dl, reflevel = "beach", tol = 1e-12, ftol = 1e-14)
  m3 <- mlogit(chosen ~ price + catch | 0, data = dl, reflevel = "beach", tol = 1e-12, ftol = 1e-14)
  m4 <- mlogit(chosen ~ price + catch | income, data = dl, reflevel = "pier", tol = 1e-12, ftol = 1e-14)
  # mlogit refuses a regressor that is a multiple of another (singular information)
  d$price2 <- 2 * d$price
  dl2 <- mlogit::dfidx(d, idx = c("id", "alt"), choice = "chosen")
  aliased_refused <- inherits(tryCatch(mlogit(chosen ~ price + price2 + catch, data = dl2), error = function(e) e), "error")
  stopifnot(aliased_refused)
  # the same models in mlogit's coefficient names, solved by Newton's method
  sit <- match(d$id, unique(d$id)); chosen <- d$chosen
  design <- function(ref, asc, ind) {
    alts <- setdiff(sort(unique(d$alt)), ref); cols <- list()
    if (asc) for (a in alts) cols[[paste0("(Intercept):", a)]] <- as.numeric(d$alt == a)
    cols$price <- d$price; cols$catch <- d$catch
    if (ind) for (a in alts) cols[[paste0("income:", a)]] <- d$income * (d$alt == a)
    do.call(cbind, cols)
  }
  fit <- function(m, ref, asc, ind) { X <- design(ref, asc, ind); .clogit_newton(X, sit, chosen, coef(m)[colnames(X)]) }
  t1 <- fit(m1, "beach", TRUE, TRUE); t0 <- fit(m0, "beach", FALSE, TRUE)
  t2 <- fit(m2, "beach", TRUE, FALSE); t3 <- fit(m3, "beach", FALSE, FALSE); t4 <- fit(m4, "pier", TRUE, TRUE)
  stopifnot(abs(t1$loglik - as.numeric(logLik(m1))) < 1e-9, abs(t4$loglik - t1$loglik) < 1e-9)
  v <- list(asc_coef = named(coef(m1)), asc_se = named(sqrt(diag(vcov(m1)))), asc_loglik = as.numeric(logLik(m1)),
            noasc_coef = named(coef(m0)), noasc_se = named(sqrt(diag(vcov(m0)))), noasc_loglik = as.numeric(logLik(m0)),
            generic_asc_coef = named(coef(m2)), generic_asc_loglik = as.numeric(logLik(m2)),
            n_choices = length(unique(d$id)),
            # WP18 package copy: Newton-polished optima and the other specifications
            asc_coef_tight = named(t1$coef), asc_se_tight = named(t1$se), asc_aic = AIC(m1),
            noasc_coef_tight = named(t0$coef), noasc_se_tight = named(t0$se),
            generic_asc_coef_tight = named(t2$coef), generic_asc_se_tight = named(t2$se),
            generic_noasc_coef_tight = named(t3$coef), generic_noasc_se_tight = named(t3$se),
            generic_noasc_loglik = t3$loglik,
            pier_coef_tight = named(t4$coef), pier_se_tight = named(t4$se), pier_loglik = t4$loglik,
            choice_counts = named(c(table(d$alt[d$chosen == 1]))),
            aliased_refused = as.numeric(aliased_refused))
  list(values = v, reference = "mlogit::mlogit(chosen ~ price + catch | income) (ASCs, reflevel = 'beach'); '| 0 + income' without ASCs; Newton-polished optima (*_tight)",
       call = "mlogit(chosen ~ price + catch | income, data = dfidx(Fishing long), reflevel = 'beach')",
       packages = c("mlogit", "dfidx"), dataset = "validation/datasets/generated/mlogit_fishing.csv")
})

case("disc_binary_glm", "econometrics/discrete", "glm binomial logit and probit: pins, aliasing, separation, near-collinearity, outcome domain", function(ctx) {
  set.seed(20261021)
  n <- 400
  x1 <- round(rnorm(n), 6); x2 <- round(rnorm(n), 6)
  d <- rbinom(n, 1, 0.4)
  y <- rbinom(n, 1, plogis(-0.4 + 0.9 * x1 - 0.6 * x2 + 0.5 * d))
  xnc <- x1 + 1e-7 * rnorm(n)                  # near-collinear with x1: not aliased at glm's QR tolerance
  y012 <- y; y012[seq(5, n, by = 40)] <- 2L     # outside {0, 1}
  g <- ctx$gen_dataset("binary_glm", data.frame(
    y = y, x1 = x1, x2 = x2, d = d, x1b = 2 * x1, nd = 1L - d, xnc = xnc,
    ysep = as.integer(x1 + x2 > 0.5),           # separated by x1 + x2, by neither alone
    yq = ifelse(d == 1, 1L, y),                 # quasi-complete separation by the dummy d
    y012 = y012))
  stopifnot(max(g$x1[g$ysep == 0]) > min(g$x1[g$ysep == 1]), max(g$x2[g$ysep == 0]) > min(g$x2[g$ysep == 1]))
  # p2a's run_logit/run_probit control: glm.fit at epsilon = 1e-10, maxit = 100 (GlmSpec's defaults;
  # R's glm.control defaults are 1e-8 and 25). glm.fit's QR tolerance is min(1e-7, epsilon / 1000).
  pc <- glm.control(epsilon = 1e-10, maxit = 100)
  tight <- glm.control(epsilon = 1e-14, maxit = 100)
  catch <- function(expr) { w <- character(0); r <- withCallingHandlers(expr, warning = function(m) { w <<- c(w, conditionMessage(m)); invokeRestart("muffleWarning") }); list(fit = r, warnings = w) }
  lc <- glm(y ~ x1 + x2 + d, family = binomial("logit"), data = g, control = pc)
  pcf <- glm(y ~ x1 + x2 + d, family = binomial("probit"), data = g, control = pc)
  lt <- glm(y ~ x1 + x2 + d, family = binomial("logit"), data = g, control = tight)
  pt <- glm(y ~ x1 + x2 + d, family = binomial("probit"), data = g, control = tight)
  al <- catch(glm(y ~ x1 + x1b + x2 + d, family = binomial, data = g, control = pc))   # x1b = 2 x1 is aliased (NA)
  tr <- catch(glm(y ~ x1 + x2 + d + nd, family = binomial, data = g, control = pc))    # dummy trap: nd is aliased
  stopifnot(is.na(coef(al$fit)["x1b"]), is.na(coef(tr$fit)["nd"]), !length(al$warnings), !length(tr$warnings))
  sep <- catch(glm(ysep ~ x1 + x2, family = binomial, data = g, control = pc))
  sepd <- catch(glm(ysep ~ x1 + x2, family = binomial, data = g))                       # R's default control
  stopifnot(identical(sep$warnings, "glm.fit: fitted probabilities numerically 0 or 1 occurred"),
            setequal(sepd$warnings, c("glm.fit: algorithm did not converge", "glm.fit: fitted probabilities numerically 0 or 1 occurred")))
  eps10 <- 10 * .Machine$double.eps
  qs <- catch(glm(yq ~ x1 + x2 + d, family = binomial, data = g, control = pc))
  nc <- catch(glm(y ~ x1 + xnc + x2, family = binomial, data = g, control = pc))
  nct <- glm(y ~ x1 + xnc + x2, family = binomial, data = g, control = tight)
  stopifnot(!length(nc$warnings), nc$fit$converged, !anyNA(coef(nct)))
  restricted <- glm(y ~ I(x1 + xnc) + x2, family = binomial, data = g, control = tight)   # x1 and xnc forced equal
  qlim <- glm(yq ~ x1 + x2, family = binomial, data = g[g$d == 0, ], control = tight)      # limit of the identified coefficients as beta_d -> Inf
  stopifnot(pt$converged, qlim$converged)
  y012_refused <- tryCatch({ glm(y012 ~ x1, family = binomial, data = g); 0 }, error = function(e) as.numeric(conditionMessage(e) == "y values must be 0 <= y <= 1"))
  stopifnot(y012_refused == 1)
  zs <- function(f) summary(f)$coefficients
  v <- list(n = n,
            logit_coef = named(coef(lc)), logit_se = named(zs(lc)[, 2]), logit_z = named(zs(lc)[, 3]), logit_p = named(zs(lc)[, 4]),
            logit_loglik = as.numeric(logLik(lc)), logit_aic = AIC(lc), logit_null_loglik = -lc$null.deviance / 2,
            logit_iter = lc$iter, logit_coef_tight = named(coef(lt)), logit_se_tight = named(zs(lt)[, 2]),
            probit_coef = named(coef(pcf)), probit_se = named(zs(pcf)[, 2]), probit_p = named(zs(pcf)[, 4]),
            probit_loglik = as.numeric(logLik(pcf)), probit_aic = AIC(pcf), probit_iter = pcf$iter,
            probit_coef_tight = named(coef(pt)), probit_se_tight = named(zs(pt)[, 2]),
            probit_z_tight = named(zs(pt)[, 3]), probit_p_tight = named(zs(pt)[, 4]),
            probit_loglik_tight = as.numeric(logLik(pt)), probit_aic_tight = AIC(pt),
            quasi_limit_coef = named(coef(qlim)), quasi_limit_se = named(zs(qlim)[, 2]),
            aliased_coef = named(coef(al$fit)), aliased_se = named(zs(al$fit)[, 2]), aliased_rank = al$fit$rank,
            trap_coef = named(coef(tr$fit)), trap_se = named(zs(tr$fit)[, 2]),
            sep_converged = as.numeric(sep$fit$converged), sep_iter = sep$fit$iter,
            sep_fitted_extreme = sum(fitted(sep$fit) < eps10 | fitted(sep$fit) > 1 - eps10),
            sep_default_converged = as.numeric(sepd$fit$converged), sep_default_iter = sepd$fit$iter,
            quasi_coef = named(coef(qs$fit)), quasi_se = named(zs(qs$fit)[, 2]), quasi_iter = qs$fit$iter,
            quasi_converged = as.numeric(qs$fit$converged), quasi_r_warned = as.numeric(length(qs$warnings) > 0),
            nc_coef = named(coef(nc$fit)), nc_loglik = as.numeric(logLik(nc$fit)), nc_iter = nc$fit$iter,
            nc_coef_tight = named(coef(nct)), nc_loglik_tight = as.numeric(logLik(nct)),
            nc_restricted_loglik = as.numeric(logLik(restricted)),
            y012_refused = y012_refused)
  list(values = v, reference = "stats::glm(family = binomial('logit' | 'probit')) at glm.control(epsilon = 1e-10, maxit = 100) (p2a's control); summary.glm; *_tight at epsilon = 1e-14; sep_default_* at R's default control",
       call = "glm(y ~ x1 + x2 + d, family = binomial, data = binary_glm, control = glm.control(epsilon = 1e-10, maxit = 100))", packages = character(0))
})

case("disc_multinom_overlap", "econometrics/discrete", "nnet::multinom on overlapping three-category data (finite MLE)", function(ctx) {
  set.seed(20261022)
  n <- 300
  x1 <- round(rnorm(n), 6); x2 <- round(rnorm(n), 6)
  eta <- cbind(0, 0.5 + 1.0 * x1 - 0.5 * x2, -0.3 + 0.3 * x1 + 0.8 * x2)
  p <- exp(eta) / rowSums(exp(eta))
  y <- c("A", "B", "C")[apply(p, 1, function(pr) sample.int(3, 1, prob = pr))]
  d <- ctx$gen_dataset("multinom_overlap", data.frame(y = y, x1 = x1, x2 = x2))
  X <- cbind("(Intercept)" = 1, x1 = d$x1, x2 = d$x2)
  k <- ncol(X)
  # Newton's method on the multinomial logit (reference = lev[1]) from nnet's estimate; analytic
  # score X'(Y - P) and information sum_i (diag(p_i) - p_i p_i') (x) x_i x_i'.
  newton <- function(lev, B) {
    Y <- sapply(lev[-1], function(l) as.numeric(d$y == l)); J <- length(lev) - 1
    info <- function(P) {
      I <- matrix(0, k * J, k * J)
      for (a in 1:J) for (b in 1:J) {
        w <- if (a == b) P[, a] * (1 - P[, a]) else -P[, a] * P[, b]
        I[(a - 1) * k + 1:k, (b - 1) * k + 1:k] <- crossprod(X * w, X)
      }
      I
    }
    for (it in 1:50) {
      E <- exp(X %*% B); P <- E / (1 + rowSums(E))
      step <- solve(info(P), as.vector(crossprod(X, Y - P)))
      B <- B + matrix(step, k, J)
      if (max(abs(step) / pmax(abs(as.vector(B)), 1e-300)) < 1e-15) break
    }
    E <- exp(X %*% B); P <- E / (1 + rowSums(E))
    g <- as.vector(crossprod(X, Y - P)); I <- info(P)
    se <- sqrt(diag(solve(I)))
    stopifnot(max(abs(g) * se) < 1e-9)
    ll <- sum(log(cbind(1 - rowSums(P), P))[cbind(seq_len(n), match(d$y, lev))])
    nm <- as.vector(outer(colnames(X), lev[-1], function(v, l) paste0(l, ":", v)))
    list(coef = setNames(as.vector(B), nm), se = setNames(se, nm), loglik = ll)
  }
  fit_ref <- function(ref) {
    lev <- c(ref, setdiff(c("A", "B", "C"), ref))
    m <- multinom(y ~ x1 + x2, data = transform(d, y = factor(y, levels = lev)), trace = FALSE, Hess = TRUE,
                  maxit = 1000, reltol = 1e-16, abstol = 1e-300)
    tn <- newton(lev, t(coef(m)))
    se_nnet <- sqrt(diag(vcov(m)))                          # nnet's analytic Hessian at its own optimum
    stopifnot(max(abs(se_nnet[names(tn$se)] / tn$se - 1)) < 1e-6)
    tn
  }
  a <- fit_ref("A"); b <- fit_ref("B")
  mdef <- multinom(y ~ x1 + x2, data = transform(d, y = factor(y)), trace = FALSE, Hess = TRUE)   # nnet defaults
  stopifnot(abs(a$loglik - b$loglik) < 1e-9, mdef$convergence == 0)
  pk <- t(coef(mdef))
  v <- list(n = n, counts = named(c(table(d$y))),
            coef_tight = named(a$coef), se_tight = named(a$se), loglik = a$loglik,
            aic = -2 * a$loglik + 2 * length(a$coef), null_loglik = sum(table(d$y) * log(table(d$y) / n)),
            coef_pkg = named(setNames(as.vector(pk), as.vector(outer(rownames(pk), colnames(pk), function(v, l) paste0(l, ":", v))))),
            loglik_pkg = as.numeric(logLik(mdef)),
            refB_coef_tight = named(b$coef), refB_se_tight = named(b$se))
  list(values = v, reference = "nnet::multinom(y ~ x1 + x2) re-solved by Newton's method on the analytic score and information (*_tight; nnet's vcov at reltol 1e-16 agrees to 1e-6); *_pkg = nnet at its default control",
       call = "multinom(y ~ x1 + x2, data = multinom_overlap, Hess = TRUE)", packages = "nnet")
})

case("disc_hurdle_zero", "econometrics/discrete", "hurdle model zero part = binomial logit glm on 1(y > 0); count-outcome domain (pscl, MASS)", function(ctx) {
  suppressPackageStartupMessages({ library(pscl); library(MASS) })
  d <- ctx$dataset("datasets/generated/zeroinfl.csv")          # committed by T2-WP10 (disc_zeroinfl)
  pc <- glm.control(epsilon = 1e-10, maxit = 100)              # p2a's GlmSpec defaults
  z <- glm(I(as.numeric(y > 0)) ~ x1 + x2, family = binomial, data = d, control = pc)
  zt <- glm(I(as.numeric(y > 0)) ~ x1 + x2, family = binomial, data = d, control = glm.control(epsilon = 1e-14, maxit = 100))
  h <- hurdle(y ~ x1 + x2, data = d, dist = "poisson", zero.dist = "binomial")
  hz <- coef(h, model = "zero")
  stopifnot(max(abs(hz / coef(zt) - 1)) < 1e-4)                 # pscl's optim stops near the glm MLE
  refused <- function(expr, msg) tryCatch({ expr; 0 }, error = function(e) as.numeric(grepl(msg, conditionMessage(e), fixed = TRUE)))
  warned <- function(expr, msg) { w <- 0; withCallingHandlers(expr, warning = function(m) { if (grepl(msg, conditionMessage(m), fixed = TRUE)) w <<- 1; invokeRestart("muffleWarning") }); w }
  yneg <- d$y; yneg[c(3, 17)] <- c(-1, -2)
  yfrac <- d$y; yfrac[c(4, 9)] <- c(1.5, 2.25)
  v <- list(n = nrow(d), n_zeros = sum(d$y == 0), n_positive = sum(d$y > 0),
            zero_coef = named(coef(z)), zero_se = named(summary(z)$coefficients[, 2]), zero_loglik = as.numeric(logLik(z)),
            zero_iter = z$iter, zero_coef_pkg = named(hz),
            zero_coef_tight = named(coef(zt)), zero_se_tight = named(summary(zt)$coefficients[, 2]),
            hurdle_neg_refused = refused(hurdle(yneg ~ x1 + x2, data = d), "invalid dependent variable, negative counts"),
            hurdle_frac_refused = refused(hurdle(yfrac ~ x1 + x2, data = d), "invalid dependent variable, non-integer values"),
            zeroinfl_neg_refused = refused(zeroinfl(yneg ~ x1 + x2, data = d), "invalid dependent variable, negative counts"),
            zeroinfl_frac_refused = refused(zeroinfl(yfrac ~ x1 + x2, data = d), "invalid dependent variable, non-integer values"),
            negbin_neg_refused = refused(glm.nb(yneg ~ x1 + x2, data = d), "negative values not allowed for the 'Poisson' family"),
            negbin_frac_warned = warned(glm.nb(yfrac ~ x1 + x2, data = d), "non-integer x"))
  stopifnot(v$hurdle_neg_refused == 1, v$hurdle_frac_refused == 1, v$zeroinfl_neg_refused == 1, v$zeroinfl_frac_refused == 1,
            v$negbin_neg_refused == 1, v$negbin_frac_warned == 1)
  list(values = v, reference = "glm(I(y > 0) ~ x1 + x2, binomial) at glm.control(epsilon = 1e-10, maxit = 100) = pscl::hurdle's binomial zero hurdle (zero_coef_pkg, optim); pscl/MASS errors for negative and non-integer counts",
       call = "glm(I(y > 0) ~ x1 + x2, family = binomial, data = zeroinfl)", packages = c("pscl", "MASS"),
       dataset = "validation/datasets/generated/zeroinfl.csv")
})

case("disc_polr_aliased", "econometrics/discrete", "MASS::polr with an aliased regressor (xb = 2 x) on ordered_j5", function(ctx) {
  suppressPackageStartupMessages(library(MASS))
  d <- ctx$dataset("datasets/generated/ordered_j5.csv"); d$xb <- 2 * d$x
  yf <- factor(d$y, levels = sort(unique(d$y)), ordered = TRUE)
  v <- list(n = nrow(d))
  for (m in c("logistic", "probit")) {
    w <- NULL
    f <- withCallingHandlers(polr(yf ~ x + xb, data = d, method = m, Hess = TRUE, control = list(reltol = 1e-14, maxit = 10000)),
                             warning = function(c) { w <<- conditionMessage(c); invokeRestart("muffleWarning") })
    g <- polr(yf ~ x, data = d, method = m, Hess = TRUE, control = list(reltol = 1e-14, maxit = 10000))
    tag <- if (m == "logistic") "logit" else "probit"
    v[[paste0(tag, "_warned")]] <- as.numeric(!is.null(w))
    v[[paste0(tag, "_kept")]] <- length(coef(f))           # 1 if polr drops xb
    v[[paste0(tag, "_coef")]] <- named(coef(g)); v[[paste0(tag, "_zeta")]] <- named(g$zeta)
  }
  list(values = v, reference = "MASS::polr on x + 2x (rank-deficient design) against polr on x", call = "polr(y ~ x + xb, ordered_j5)", packages = "MASS")
})
