# validation/r/cases/discrete_t4.R -- Tier 4 T4-WP12 references (ECON-16, ECON-62, ECON-63, ECON-64, ECON-65;
# the WP07 probit hand-off).
#  * disc_hurdle_t4: pscl::hurdle count part (zero-truncated Poisson and NB) on T2-WP10's zeroinfl.csv. pscl's
#    optim stops early, so the count part is polished by Newton steps on the analytic truncated score
#    (numDeriv::jacobian of it) until max |gradient| < 1e-9; SEs from numDeriv::hessian of the truncated
#    log-likelihood at that point (pscl: optim's Hessian). The zero part is glm(I(y > 0) ~ ., binomial) at
#    epsilon 1e-14 (disc_hurdle_zero pins it at p2a's control); the total log-likelihood adds the two.
#  * disc_negbin_null_t4: glm.nb(y ~ 1) on the model's rows (pscl::pR2: update(object, ~ 1, data = model.frame))
#    with its own theta, polished as disc_negbin; McFadden R2 = 1 - ll / ll0.
#  * disc_clogit_unbalanced_t4: Fishing with 'charter' removed from the first 300 situations that did not choose
#    it. mlogit with and without ASCs (Newton-polished); null log-likelihoods: equal shares per situation
#    -sum ln J_i, mlogit's "null" (market shares sum n_j ln(n_j / N)) and the constants-only MLE; the same three
#    on the balanced data.
#  * disc_mixed_tri_unif_t4: mlogit rpar = c(x = 't') and c(x = 'u'), R = 50 Halton draws per situation, on
#    T2-WP18's mixed_logit.csv: the reconstruction equals mlogit's simulated log-likelihood; re-solved as
#    discrete_mixed.R does (optim BFGS + numDeriv Newton); the log-likelihood and gradient at a fixed point.
#  * me_probit_ctl_t4: glm probit at p2a's control (epsilon 1e-10, maxit 100) on WP07's me_mem_contrast.csv:
#    coefficients, MEM with the full delta-method gradient, average comparisons of x1.
suppressPackageStartupMessages({ library(pscl); library(MASS); library(mlogit); library(numDeriv) })

.t4_polish <- function(par, gr, tol = 1e-9, max_steps = 30) {
  for (s in seq_len(max_steps)) {
    g <- gr(par)
    if (max(abs(g)) < tol) break
    H <- numDeriv::jacobian(gr, par); H <- (H + t(H)) / 2
    par <- par - solve(H, g)
  }
  stopifnot(max(abs(gr(par))) < tol)
  par
}
.t4_zp <- function(est, se) { z <- est / se; list(z = z, p = 2 * pnorm(-abs(z))) }

# ---- hurdle count part: zero-truncated Poisson and NB (log link; NB in (beta, ln theta)) -----------------------
.tp_ll <- function(b, X, y) { mu <- exp(drop(X %*% b)); sum(dpois(y, mu, log = TRUE) - log(-expm1(-mu))) }
.tp_gr <- function(b, X, y) { mu <- exp(drop(X %*% b)); adj <- mu / expm1(mu); colSums((y - mu - adj) * X) }
.tnb_ll <- function(par, X, y) {
  k <- ncol(X); mu <- exp(drop(X %*% par[1:k])); th <- exp(par[k + 1])
  lf0 <- th * (log(th) - log(th + mu))
  sum(dnbinom(y, size = th, mu = mu, log = TRUE) - log(-expm1(lf0)))
}
.tnb_gr <- function(par, X, y) {
  k <- ncol(X); mu <- exp(drop(X %*% par[1:k])); th <- exp(par[k + 1])
  lf0 <- th * (log(th) - log(th + mu)); r <- exp(lf0) / (-expm1(lf0))   # f0 / (1 - f0)
  Lc <- -th * mu / (th + mu); Lt <- th * (log(th) - log(th + mu) + mu / (th + mu))
  sc <- y - mu * (y + th) / (mu + th) + r * Lc
  st <- th * (digamma(y + th) - digamma(th) + log(th) + 1 - log(mu + th) - (y + th) / (mu + th)) + r * Lt
  c(colSums(sc * X), sum(st))
}
.hurdle_values <- function(d, yname, xnames, znames, dist) {
  frm <- as.formula(paste(yname, "~", paste(xnames, collapse = " + "), "|", if (length(znames)) paste(znames, collapse = " + ") else "1"))
  m <- hurdle(frm, data = d, dist = dist, zero.dist = "binomial", control = hurdle.control(reltol = 1e-15, maxit = 100000))
  y <- d[[yname]]; pos <- y > 0
  X <- cbind(1, as.matrix(d[pos, xnames, drop = FALSE])); yp <- y[pos]; k <- ncol(X)
  Z <- cbind(1, as.matrix(d[, znames, drop = FALSE]))
  zf <- glm.fit(Z, as.numeric(y > 0), family = binomial(), control = glm.control(epsilon = 1e-14, maxit = 100))
  ll_zero <- sum(dbinom(as.numeric(y > 0), 1, zf$fitted.values, log = TRUE))
  cn <- names(coef(m, model = "count")); cn <- sub("^count_", "", cn)
  if (dist == "poisson") {
    par <- .t4_polish(coef(m, model = "count"), function(b) .tp_gr(b, X, yp))
    ll_c <- .tp_ll(par, X, yp); H <- numDeriv::hessian(function(b) .tp_ll(b, X, yp), par)
  } else {
    par <- .t4_polish(c(coef(m, model = "count"), log(m$theta[["count"]])), function(p) .tnb_gr(p, X, yp))
    ll_c <- .tnb_ll(par, X, yp); H <- numDeriv::hessian(function(p) .tnb_ll(p, X, yp), par)
  }
  se <- sqrt(diag(solve(-H))); b <- par[1:k]; zp <- .t4_zp(b, se[1:k])
  npar <- length(par) + ncol(Z); ll <- ll_zero + ll_c; n <- nrow(d)
  out <- list(count = named(setNames(b, cn)), count_se = named(setNames(se[1:k], cn)),
              count_z = named(setNames(zp$z, cn)), count_p = named(setNames(zp$p, cn)),
              ll_count = ll_c, ll_zero = ll_zero, loglik = ll, aic = -2 * ll + 2 * npar, bic = -2 * ll + log(n) * npar,
              count_pkg = named(setNames(coef(m, model = "count"), cn)),
              count_se_pkg = named(setNames(sqrt(diag(vcov(m)))[1:k], cn)), loglik_pkg = as.numeric(logLik(m)),
              n = n, n_positive = sum(pos))
  if (dist == "negbin") {
    out$log_theta <- par[k + 1]; out$theta <- exp(par[k + 1]); out$se_log_theta <- se[k + 1]
    out$se_theta <- exp(par[k + 1]) * se[k + 1]
    out$theta_pkg <- m$theta[["count"]]; out$se_log_theta_pkg <- m$SE.logtheta[["count"]]
  }
  out
}
case("disc_hurdle_t4", "econometrics/discrete", "pscl::hurdle count part (zero-truncated Poisson and negative binomial), Newton-polished", function(ctx) {
  d <- ctx$dataset("datasets/generated/zeroinfl.csv")               # committed by T2-WP10 (disc_zeroinfl)
  hp <- .hurdle_values(d, "y", c("x1", "x2"), c("x1", "x2"), "poisson"); names(hp) <- paste0("hp_", names(hp))
  hn <- .hurdle_values(d, "y_nb", c("x1", "x2"), c("x1", "x2"), "negbin"); names(hn) <- paste0("hnb_", names(hn))
  h1 <- .hurdle_values(d, "y", c("x1", "x2"), character(0), "poisson")   # zero part intercept only: y ~ x1 + x2 | 1
  # NB hurdle on the Poisson counts: little overdispersion in the positives, a flat profile in ln theta
  # (theta about 174, SE(ln theta) about 10) but a finite MLE
  hpn <- .hurdle_values(d, "y", c("x1", "x2"), c("x1", "x2"), "negbin"); names(hpn) <- paste0("hpnb_", names(hpn))
  # the truncated-Poisson information weight mu/(1-p0) - mu^2 p0/(1-p0)^2 at the polished estimate equals
  # -d2 ll / d eta2 (the formula p2a implements; checked against numDeriv)
  Xp <- cbind(1, as.matrix(d[d$y > 0, c("x1", "x2")])); mu <- exp(drop(Xp %*% hp$hp_count$values))
  p0 <- exp(-mu); w <- mu / (1 - p0) - mu^2 * p0 / (1 - p0)^2
  stopifnot(max(abs(sqrt(diag(solve(crossprod(Xp * sqrt(w))))) / hp$hp_count_se$values - 1)) < 1e-6)
  list(values = c(hp, hn, list(hz1_count = h1$count, hz1_count_se = h1$count_se, hz1_ll_zero = h1$ll_zero,
                               hz1_loglik = h1$loglik), hpn),
       reference = "pscl::hurdle(y ~ x1 + x2, dist = 'poisson' | 'negbin', zero.dist = 'binomial'), count part polished by Newton steps on the analytic zero-truncated score (numDeriv::jacobian); SEs from numDeriv::hessian of the truncated log-likelihood; zero part glm(binomial) at epsilon 1e-14",
       call = "hurdle(y ~ x1 + x2, data = zeroinfl, dist = ...)", packages = c("pscl", "MASS", "numDeriv"),
       dataset = "validation/datasets/generated/zeroinfl.csv")
})

# ---- negative binomial null model (ECON-64) -----------------------------------------------------------------------
.t4_nb_grad <- function(p, X, y) {
  k <- ncol(X); b <- p[1:k]; th <- p[k + 1]; mu <- exp(drop(X %*% b))
  c(colSums(X * (y - mu) * th / (th + mu)),
    sum(digamma(th + y) - digamma(th) + log(th) + 1 - log(th + mu) - (y + th) / (mu + th)))
}
.t4_nb_ll <- function(th, mu, y) sum(lgamma(th + y) - lgamma(th) - lgamma(y + 1) + th * log(th) + y * log(mu + (y == 0)) - (th + y) * log(th + mu))
.t4_nb_fit <- function(dd, yname, xnames) {
  frm <- as.formula(paste(yname, "~", if (length(xnames)) paste(xnames, collapse = " + ") else "1"))
  m <- glm.nb(frm, data = dd, control = glm.control(epsilon = 1e-14, maxit = 200))
  X <- cbind(1, as.matrix(dd[, xnames, drop = FALSE])); y <- dd[[yname]]
  p <- .t4_polish(c(coef(m), m$theta), function(q) .t4_nb_grad(q, X, y)); k <- ncol(X)
  list(theta = p[k + 1], loglik = .t4_nb_ll(p[k + 1], exp(drop(X %*% p[1:k])), y), coef = p[1:k],
       loglik_pkg = as.numeric(logLik(m)))
}
case("disc_negbin_null_t4", "econometrics/discrete", "glm.nb(y ~ 1) on the model's rows, its own theta; McFadden R2 (pscl::pR2)", function(ctx) {
  d <- ctx$dataset("datasets/generated/negbin.csv")                  # committed by T2-WP10 (disc_negbin)
  full <- .t4_nb_fit(d, "y", c("x1", "x2")); null <- .t4_nb_fit(d, "y", character(0))
  stopifnot(abs(null$coef[[1]] - log(mean(d$y))) < 1e-12)          # the intercept-only mean is ybar
  dd <- d[complete.cases(d[, c("y_na", "x1_na", "x2")]), ]
  nfull <- .t4_nb_fit(dd, "y_na", c("x1_na", "x2")); nnull <- .t4_nb_fit(dd, "y_na", character(0))
  # the full model's theta held fixed (the other convention, for the record)
  ll0_fixed <- .t4_nb_ll(full$theta, rep(mean(d$y), nrow(d)), d$y)
  m <- glm.nb(y ~ x1 + x2, data = d, control = glm.control(epsilon = 1e-14, maxit = 200))
  pr2 <- suppressMessages(capture.output(pp <- pscl::pR2(m)))
  list(values = list(loglik = full$loglik, theta = full$theta, null_loglik = null$loglik, null_theta = null$theta,
                     pseudo_r2 = 1 - full$loglik / null$loglik, null_loglik_fixed_theta = ll0_fixed,
                     pseudo_r2_fixed_theta = 1 - full$loglik / ll0_fixed,
                     pseudo_r2_pkg = unname(pp["McFadden"]), null_loglik_pkg = unname(pp["llhNull"]),
                     na_loglik = nfull$loglik, na_null_loglik = nnull$loglik, na_null_theta = nnull$theta,
                     na_pseudo_r2 = 1 - nfull$loglik / nnull$loglik, na_n_used = nrow(dd)),
       reference = "MASS::glm.nb(y ~ 1) on the model's complete rows with its own theta (pscl::pR2: update(object, ~ 1, data = model.frame(object))), polished by Newton steps on the joint score; McFadden R2 = 1 - ll / ll0",
       call = "pscl::pR2(glm.nb(y ~ x1 + x2, data = negbin))", packages = c("MASS", "pscl", "numDeriv"),
       dataset = "validation/datasets/generated/negbin.csv")
})

# ---- conditional logit on unbalanced choice sets (ECON-63) --------------------------------------------------------
.t4_clogit_newton <- function(X, sit, chosen, b) {
  ll_at <- function(b) { v <- drop(X %*% b); mv <- ave(v, sit, FUN = max); s <- ave(exp(v - mv), sit, FUN = sum); sum((v - mv - log(s))[chosen == 1]) }
  info_at <- function(b) {
    v <- drop(X %*% b); mv <- ave(v, sit, FUN = max); e <- exp(v - mv); s <- ave(e, sit, FUN = sum); p <- e / s
    xd <- X - rowsum(p * X, sit)[sit, , drop = FALSE]
    list(g = colSums((chosen - p) * X), I = crossprod(xd * sqrt(p)))
  }
  for (it in 1:50) {
    gi <- info_at(b); sc <- 1 / sqrt(diag(gi$I)); step <- sc * solve(gi$I * outer(sc, sc), sc * gi$g); b <- b + step
    if (max(abs(step) / pmax(abs(b), 1e-300)) < 1e-15) break
  }
  gi <- info_at(b); stopifnot(max(abs(gi$g) * sqrt(diag(solve(gi$I)))) < 1e-9)
  list(coef = setNames(b, colnames(X)), se = setNames(sqrt(diag(solve(gi$I))), colnames(X)), loglik = ll_at(b))
}
.t4_clogit_values <- function(d, tag) {
  dl <- mlogit::dfidx(d, idx = c("id", "alt"), choice = "chosen")
  m_asc <- mlogit(chosen ~ price + catch, data = dl, reflevel = "beach", tol = 1e-12, ftol = 1e-14)
  m_noasc <- mlogit(chosen ~ price + catch | 0, data = dl, tol = 1e-12, ftol = 1e-14)
  m_const <- mlogit(chosen ~ 1, data = dl, reflevel = "beach", tol = 1e-12, ftol = 1e-14)
  sit <- match(d$id, unique(d$id)); chosen <- d$chosen
  alts <- c("boat", "charter", "pier")
  A <- sapply(alts, function(a) as.numeric(d$alt == a)); colnames(A) <- paste0("(Intercept):", alts)
  t_asc <- .t4_clogit_newton(cbind(A, price = d$price, catch = d$catch), sit, chosen, coef(m_asc)[c(colnames(A), "price", "catch")])
  t_noasc <- .t4_clogit_newton(cbind(price = d$price, catch = d$catch), sit, chosen, coef(m_noasc)[c("price", "catch")])
  t_const <- .t4_clogit_newton(A, sit, chosen, coef(m_const)[colnames(A)])
  J_i <- as.vector(table(sit)); n_j <- as.vector(table(d$alt[chosen == 1]))
  equal <- -sum(log(J_i)); market <- sum(n_j * log(n_j / sum(n_j)))
  stopifnot(abs(market - m_asc$logLik[["null"]]) < 1e-9)
  v <- list(asc_coef_tight = named(t_asc$coef), asc_se_tight = named(t_asc$se), asc_loglik = t_asc$loglik,
            noasc_coef_tight = named(t_noasc$coef), noasc_se_tight = named(t_noasc$se), noasc_loglik = t_noasc$loglik,
            null_equal_shares = equal, null_market_shares = market, null_constants_only = t_const$loglik,
            asc_pr2_equal = 1 - t_asc$loglik / equal, asc_pr2_market = 1 - t_asc$loglik / market,
            asc_pr2_constants = 1 - t_asc$loglik / t_const$loglik, asc_pr2_pkg = unname(summary(m_asc)$mfR2),
            noasc_pr2_equal = 1 - t_noasc$loglik / equal,
            n_situations = length(J_i), n_rows = nrow(d), choice_counts = vec(n_j))
  names(v) <- paste0(tag, "_", names(v))
  v
}
case("disc_clogit_unbalanced_t4", "econometrics/discrete", "mlogit on Fishing with and without 'charter' in 300 situations: null log-likelihoods for unbalanced choice sets", function(ctx) {
  f <- ctx$dataset("datasets/generated/mlogit_fishing.csv")          # committed by T3-WP18 (disc_mlogit_fishing)
  chose_charter <- unique(f$id[f$alt == "charter" & f$chosen == 1])
  drop_ids <- head(setdiff(sort(unique(f$id)), chose_charter), 300)
  u <- ctx$gen_dataset("mlogit_fishing_unbalanced", f[!(f$alt == "charter" & f$id %in% drop_ids), c("id", "alt", "chosen", "price", "catch")])
  stopifnot(length(drop_ids) == 300, nrow(u) == nrow(f) - 300)
  vb <- .t4_clogit_values(f, "bal"); vu <- .t4_clogit_values(u, "unbal")
  # malformed choice sets (ECON-62): mlogit refuses a situation with no chosen row, one with two, and a
  # duplicated alternative (dfidx: "the two indexes don't define unique observations")
  refused <- function(dd) as.numeric(inherits(tryCatch(mlogit(chosen ~ price + catch | 0,
                 data = mlogit::dfidx(dd, idx = c("id", "alt"), choice = "chosen")), error = function(e) e), "error"))
  f0 <- f; f0$chosen[f0$id == 1] <- 0L
  f2 <- f; f2$chosen[f2$id == 1 & f2$alt %in% c("beach", "boat")] <- 1L
  fd <- rbind(f, f[f$id == 2 & f$alt == "boat", ]); fd <- fd[order(fd$id), ]
  rf <- c(refused(f0), refused(f2), refused(fd)); stopifnot(all(rf == 1))
  list(values = c(vb, vu, list(dropped_ids_first = vec(head(drop_ids, 5)), dropped_ids_last = vec(tail(drop_ids, 1)),
                               refused_no_chosen = rf[1], refused_two_chosen = rf[2], refused_duplicate_alt = rf[3])),
       reference = "mlogit::mlogit(chosen ~ price + catch[ | 0]) and mlogit(chosen ~ 1), Newton-polished (*_tight); null log-likelihoods: equal shares -sum ln J_i, mlogit's logLik['null'] = sum n_j ln(n_j / N) (summary.mlogit mfR2), and the constants-only MLE",
       call = "mlogit(chosen ~ price + catch, data = dfidx(fishing_unbalanced), reflevel = 'beach')", packages = c("mlogit", "dfidx"),
       dataset = "validation/datasets/generated/mlogit_fishing_unbalanced.csv")
})

# ---- mixed logit, triangular and uniform random coefficient (ECON-65) ---------------------------------------------
.t4_factor <- function(z, dist) {
  u <- pnorm(z)
  if (dist == "u") 2 * u - 1 else ifelse(u < 0.5, sqrt(2 * u) - 1, 1 - sqrt(2 * (1 - u)))
}
.t4_mxl_ll <- function(par, Xl, chosen, R, e) {     # par = (x, w, sd.x); e = factor draws, unit-major
  N <- nrow(Xl[[1]])
  bx <- matrix(par[1] + par[3] * e, N, R, byrow = TRUE)
  ex <- lapply(Xl, function(X) exp(X[, 1] * bx + X[, 2] * par[2]))
  S <- Reduce(`+`, ex)
  pch <- Reduce(`+`, mapply(function(ee, j) ee * (chosen == j), ex, seq_along(Xl), SIMPLIFY = FALSE)) / S
  sum(log(rowMeans(pch)))
}
.t4_newton_polish <- function(f, p, max_it = 10) {
  for (i in seq_len(max_it)) {
    step <- solve(numDeriv::hessian(f, p), numDeriv::grad(f, p)); p <- p - step
    if (max(abs(step) / pmax(1, abs(p))) < 1e-12) break
  }
  p
}
case("disc_mixed_tri_unif_t4", "econometrics/discrete", "Mixed logit with a triangular and a uniform coefficient on x, R = 50 Halton draws (mlogit rpar 't', 'u')", function(ctx) {
  d <- ctx$dataset("datasets/generated/mixed_logit.csv")              # committed by T2-WP18 (disc_mixed_logit)
  md <- dfidx(d, idx = c("id", "alt"), choice = "choice")
  Xl <- lapply(c("a", "b", "c"), function(a) as.matrix(d[d$alt == a, c("x", "w")]))
  chosen <- sapply(split(d, d$id), function(s) which(s$choice == 1))
  R <- 50; N <- length(chosen); z <- qnorm(mlogit:::halton(2, N * R, 100))
  v <- list()
  pt <- c(x = 0.9, w = -0.7, sd.x = 1.2)
  for (dist in c("t", "u")) {
    mm <- mlogit(choice ~ x + w | 0, data = md, rpar = setNames(dist, "x"), R = R, halton = NA)
    e <- .t4_factor(z, dist)
    f <- function(p) .t4_mxl_ll(p, Xl, chosen, R, e)
    stopifnot(abs(f(coef(mm)[c("x", "w", "sd.x")]) - as.numeric(logLik(mm))) < 1e-8)   # mlogit's likelihood
    opt <- optim(coef(mm)[c("x", "w", "sd.x")], function(p) -f(p), method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
    p <- .t4_newton_polish(f, opt$par); H <- numDeriv::hessian(f, p)
    stopifnot(p[3] > 0, max(abs(numDeriv::grad(f, p))) < 1e-8)
    tag <- if (dist == "t") "tri" else "unif"
    v[[paste0(tag, "_coef")]] <- named(c(mean_x = p[[1]], mean_w = p[[2]], sd_x = p[[3]]))
    v[[paste0(tag, "_se")]] <- named(setNames(sqrt(diag(solve(-H))), c("mean_x", "mean_w", "sd_x")))
    v[[paste0(tag, "_loglik")]] <- f(p)
    v[[paste0(tag, "_coef_pkg")]] <- named(coef(mm)[c("x", "w", "sd.x")]); v[[paste0(tag, "_loglik_pkg")]] <- as.numeric(logLik(mm))
    v[[paste0(tag, "_point_loglik")]] <- f(pt)
    v[[paste0(tag, "_point_grad")]] <- named(setNames(numDeriv::grad(f, pt), names(pt)))
  }
  u <- c(0.25, 0.4999, 0.5, 0.5001, 0.75); v$tri_factor_u <- vec(u)
  v$tri_factor <- vec(ifelse(u < 0.5, sqrt(2 * u) - 1, 1 - sqrt(2 * (1 - u))))
  v$point <- named(pt); v$n_draws <- R; v$n_situations <- N
  list(values = v,
       reference = "mlogit::mlogit(choice ~ x + w | 0, rpar = c(x = 't' | 'u'), R = 50, halton = NA) simulated log-likelihood (make.beta: triangular sqrt(2 u) - 1 below 1/2, 1 - sqrt(2 (1 - u)) above; uniform 2 u - 1; u = pnorm of the normal Halton draw), re-solved with optim(BFGS, reltol 1e-15) and numDeriv Newton steps; SEs from numDeriv::hessian",
       call = "mlogit(choice ~ x + w | 0, data = dfidx(mixed_logit), rpar = c(x = 't'), R = 50, halton = NA)", packages = c("mlogit", "numDeriv"),
       dataset = "validation/datasets/generated/mixed_logit.csv")
})

# ---- probit at p2a's control (WP07 hand-off) ----------------------------------------------------------------------
case("me_probit_ctl_t4", "regression/marginal_effects", "glm probit at p2a's control (epsilon 1e-10, maxit 100): MEM and average comparisons", function(ctx) {
  d <- ctx$dataset("datasets/generated/me_mem_contrast.csv")          # committed by T4-WP07 (me_mem_contrast)
  fit <- glm(y ~ x1 + x2, family = binomial("probit"), data = d, control = glm.control(epsilon = 1e-10, maxit = 100))
  b <- coef(fit); V <- vcov(fit); X <- model.matrix(fit)
  xb <- colMeans(X); zb <- sum(xb * b)
  mem <- b * dnorm(zb); G <- diag(dnorm(zb), length(b)) + outer(b * (-zb * dnorm(zb)), xb)
  mem_se <- sqrt(diag(G %*% V %*% t(G))); mz <- mem / mem_se
  vals <- c(0, 1, 2.5); est <- se <- numeric(0)
  for (i in 1:2) for (j in (i + 1):3) {
    X1 <- X; X1[, "x1"] <- vals[i]; X2 <- X; X2[, "x1"] <- vals[j]
    z1 <- drop(X1 %*% b); z2 <- drop(X2 %*% b)
    e <- mean(pnorm(z2) - pnorm(z1)); g <- colMeans(dnorm(z2) * X2 - dnorm(z1) * X1)
    est <- c(est, e); se <- c(se, sqrt(drop(t(g) %*% V %*% g)))
  }
  list(values = list(probit_coef = named(b), probit_se = named(sqrt(diag(V))), probit_iter = fit$iter,
                     probit_mem = named(mem), probit_mem_se = named(mem_se), probit_mem_z = named(mz),
                     probit_mem_p = named(2 * pnorm(-abs(mz))), probit_contrast_est = vec(est), probit_contrast_se = vec(se),
                     probit_contrast_p = vec(2 * pnorm(-abs(est / se))), contrast_values = vec(vals), n = nrow(d)),
       reference = "stats::glm(binomial('probit')) at glm.control(epsilon = 1e-10, maxit = 100) (p2a's MleSettings default); MEM and SE by the delta method with G = f(zbar) I + beta f'(zbar) xbar'; average comparisons of x1",
       call = "glm(y ~ x1 + x2, binomial('probit'), data = me_mem_contrast, control = glm.control(epsilon = 1e-10, maxit = 100))",
       packages = character(0), dataset = "validation/datasets/generated/me_mem_contrast.csv")
})
