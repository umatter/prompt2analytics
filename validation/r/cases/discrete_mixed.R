# validation/r/cases/discrete_mixed.R -- mixed logit with mlogit's Halton draws, cross-section and
# panel, and missing cells in discrete-choice data (T2-WP18: ECON-5 mixed part, ECON-12, ECON-17, ECON-84).
#
# mlogit's own optimum is loose (BFGS on a numerical gradient; its conditional logit on this data stops
# with |gradient| ~ 1e-3), so every case re-solves the same simulated log-likelihood to full precision:
# optim(BFGS, reltol = 1e-15) from mlogit's estimate, then Newton steps on numDeriv's gradient and
# Hessian. SEs come from numDeriv::hessian at that point. The package's own numbers are kept as *_pkg.
suppressPackageStartupMessages({ library(mlogit); library(numDeriv) })

# Simulated log-likelihood with mlogit's draw scheme (panel = FALSE: draws per choice situation;
# panel = TRUE: draws per individual, choice probabilities multiplied over the individual's
# situations). x has a normal random coefficient, w is fixed. Units are in id order.
.mxl_ll <- function(par, Xl, chosen, R, draws, unit_of_obs = NULL) {
  N <- nrow(Xl[[1]]); J <- length(Xl)
  n_units <- if (is.null(unit_of_obs)) N else max(unit_of_obs)
  bxm_u <- matrix(par[1] + par[3] * draws, n_units, R, byrow = TRUE)     # units x draws
  bxm <- if (is.null(unit_of_obs)) bxm_u else bxm_u[unit_of_obs, , drop = FALSE]
  ex <- lapply(Xl, function(X) exp(X[, 1] * bxm + X[, 2] * par[2]))
  S <- Reduce(`+`, ex)
  pch <- Reduce(`+`, mapply(function(e, j) e * (chosen == j), ex, seq_len(J), SIMPLIFY = FALSE)) / S
  if (!is.null(unit_of_obs)) pch <- apply(pch, 2, function(col) tapply(col, unit_of_obs, prod))
  sum(log(rowMeans(pch)))
}

# Two random coefficients, x log-normal (prime 2) and w normal (prime 3), cross-section;
# par = c(x, w, sd.x, sd.w) as mlogit names them.
.mxl_ll_ln_n <- function(par, Xl, chosen, R) {
  N <- nrow(Xl[[1]])
  zx <- qnorm(mlogit:::halton(2, N * R, 100)); zw <- qnorm(mlogit:::halton(3, N * R, 100))
  bx <- matrix(exp(par[1] + par[3] * zx), N, R, byrow = TRUE)
  bw <- matrix(par[2] + par[4] * zw, N, R, byrow = TRUE)
  ex <- lapply(Xl, function(X) exp(X[, 1] * bx + X[, 2] * bw))
  S <- Reduce(`+`, ex)
  pch <- Reduce(`+`, mapply(function(e, j) e * (chosen == j), ex, seq_along(Xl), SIMPLIFY = FALSE)) / S
  sum(log(rowMeans(pch)))
}

# Newton steps on numDeriv's gradient and Hessian until the step is below 1e-12 (relative).
.newton_polish <- function(f, p, max_it = 10) {
  for (i in seq_len(max_it)) {
    step <- solve(numDeriv::hessian(f, p), numDeriv::grad(f, p))
    p <- p - step
    if (max(abs(step) / pmax(1, abs(p))) < 1e-12) break
  }
  p
}

.mxl_fit <- function(Xl, chosen, R, draws, start, unit_of_obs = NULL) {
  f <- function(p) .mxl_ll(p, Xl, chosen, R, draws, unit_of_obs)
  opt <- optim(start, function(p) -f(p), method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  p <- .newton_polish(f, opt$par)
  H <- numDeriv::hessian(f, p)
  list(par = p, se = sqrt(diag(solve(-H))), ll = f(p), grad = max(abs(numDeriv::grad(f, p))))
}

.mxl_data <- function(ctx, name, n_ind, n_sit) {
  set.seed(1234)
  J <- 3
  ind <- rep(seq_len(n_ind), each = n_sit * J)
  chid <- rep(seq_len(n_ind * n_sit), each = J)
  alt <- rep(c("a", "b", "c"), n_ind * n_sit)
  x <- round(rnorm(length(alt)), 6); w <- round(rnorm(length(alt)), 6)
  bx <- (1.0 + 0.8 * rnorm(n_ind))[ind]
  util <- bx * x - 0.7 * w - log(-log(runif(length(alt))))
  ch <- as.integer(ave(util, chid, FUN = function(u) u == max(u)))
  ctx$gen_dataset(name, data.frame(person = ind, id = chid, alt = alt, choice = ch, x = x, w = w))
}

.mxl_case <- function(d, R, panel) {
  stopifnot(is.integer(d$id), is.integer(d$person), is.integer(d$choice))
  md <- if (panel) dfidx(d, idx = list(c("id", "person"), "alt"), choice = "choice") else dfidx(d, idx = c("id", "alt"), choice = "choice")
  mm <- mlogit(choice ~ x + w | 0, data = md, rpar = c(x = "n"), R = R, halton = NA, panel = panel)
  Xl <- lapply(c("a", "b", "c"), function(a) as.matrix(d[d$alt == a, c("x", "w")]))
  chosen <- sapply(split(d, d$id), function(s) which(s$choice == 1))
  unit <- if (panel) d$person[d$alt == "a"] else NULL
  n_units <- if (panel) length(unique(d$person)) else length(chosen)
  draws <- qnorm(mlogit:::halton(2, n_units * R, 100))
  ll_pkg <- .mxl_ll(coef(mm)[c("x", "w", "sd.x")], Xl, chosen, R, draws, unit)
  stopifnot(abs(ll_pkg - as.numeric(logLik(mm))) < 1e-8)      # the reconstruction is mlogit's likelihood
  fit <- .mxl_fit(Xl, chosen, R, draws, coef(mm)[c("x", "w", "sd.x")], unit)
  stopifnot(fit$par[3] > 0, fit$grad < 1e-8)
  list(coef = named(c(mean_x = fit$par[[1]], mean_w = fit$par[[2]], sd_x = abs(fit$par[[3]]))),
       se = named(c(mean_x = fit$se[1], mean_w = fit$se[2], sd_x = fit$se[3])), loglik = fit$ll, max_abs_grad = fit$grad,
       coef_pkg = named(coef(mm)[c("x", "w", "sd.x")]), se_pkg = named(sqrt(diag(vcov(mm)))[c("x", "w", "sd.x")]),
       loglik_pkg = as.numeric(logLik(mm)), first_draws = vec(draws[1:5]), unit1_first_draws = vec(draws[R + 1:3]),
       n_units = n_units, n_draws = R, n_situations = length(chosen))
}

case("disc_mixed_logit", "econometrics/discrete", "Mixed logit, normal coefficient on x, R = 50 Halton draws per choice situation (mlogit)", function(ctx) {
  d <- .mxl_data(ctx, "mixed_logit", n_ind = 300, n_sit = 1)
  v <- .mxl_case(d, R = 50, panel = FALSE)
  Xl <- lapply(c("a", "b", "c"), function(a) as.matrix(d[d$alt == a, c("x", "w")]))
  chosen <- sapply(split(d, d$id), function(s) which(s$choice == 1))
  # Conditional logit. mlogit stops with |gradient| ~ 1e-3 here (coefficients 1.5e-5 from the MLE):
  # the same likelihood (sd fixed at 0), Newton-polished, is the tight reference.
  cl <- mlogit(choice ~ x + w | 0, data = dfidx(d, idx = c("id", "alt"), choice = "choice"))
  v$clogit_coef <- named(coef(cl)); v$clogit_se <- named(sqrt(diag(vcov(cl)))); v$clogit_loglik <- as.numeric(logLik(cl))
  fcl <- function(b) .mxl_ll(c(b, 0), Xl, chosen, 1, 0)
  bcl <- .newton_polish(fcl, coef(cl))
  v$clogit_coef_tight <- named(setNames(bcl, names(coef(cl))))
  v$clogit_se_tight <- named(setNames(sqrt(diag(solve(-numDeriv::hessian(fcl, bcl)))), names(coef(cl))))
  v$clogit_loglik_tight <- fcl(bcl)
  # Draw layout for two random coefficients (x log-normal: prime 2; w normal: prime 3), R = 20:
  # the reconstruction equals mlogit's simulated log-likelihood at mlogit's estimate; the
  # log-likelihood and its gradient at a fixed point pin the Rust evaluator.
  m2 <- mlogit(choice ~ x + w | 0, data = dfidx(d, idx = c("id", "alt"), choice = "choice"),
               rpar = c(x = "ln", w = "n"), R = 20, halton = NA)
  stopifnot(abs(.mxl_ll_ln_n(coef(m2)[c("x", "w", "sd.x", "sd.w")], Xl, chosen, 20) - as.numeric(logLik(m2))) < 1e-8)
  pt <- c(x = -0.1, w = -0.7, sd.x = 1.0, sd.w = 0.4)
  f2 <- function(p) .mxl_ll_ln_n(p, Xl, chosen, 20)
  v$ln_n_point <- named(pt); v$ln_n_loglik <- f2(pt); v$ln_n_grad <- named(setNames(numDeriv::grad(f2, pt), names(pt)))
  v$halton_p3_drop100 <- vec(mlogit:::halton(3, 6, 100)); v$halton_p2_drop100 <- vec(mlogit:::halton(2, 4, 100))
  v$halton_p7_3000_tail <- vec(tail(mlogit:::halton(7, 3000, 100), 3))
  list(values = v,
       reference = "mlogit::mlogit(rpar = c(x = 'n'), R = 50, halton = NA) simulated log-likelihood, re-solved with optim(BFGS, reltol = 1e-15) and Newton steps (numDeriv); SEs from numDeriv::hessian; conditional logit mlogit(choice ~ x + w | 0) likewise; mlogit:::halton",
       call = "mlogit(choice ~ x + w | 0, data = dfidx(d, idx = c('id', 'alt')), rpar = c(x = 'n'), R = 50, halton = NA)",
       packages = c("mlogit", "numDeriv"))
})

case("disc_mixed_logit_panel", "econometrics/discrete", "Mixed logit, 100 people x 4 situations, draws per person (mlogit panel = TRUE)", function(ctx) {
  d <- .mxl_data(ctx, "mixed_logit_panel", n_ind = 100, n_sit = 4)
  list(values = .mxl_case(d, R = 50, panel = TRUE),
       reference = "mlogit::mlogit(rpar = c(x = 'n'), R = 50, halton = NA, panel = TRUE), re-solved tightly (optim + numDeriv Newton steps); SEs from numDeriv::hessian",
       call = "mlogit(choice ~ x + w | 0, data = dfidx(d, idx = list(c('id', 'person'), 'alt')), rpar = c(x = 'n'), R = 50, halton = NA, panel = TRUE)",
       packages = c("mlogit", "numDeriv"))
})

case("disc_choice_na", "econometrics/discrete", "Missing cells in discrete-choice data: glm logit/probit and nnet::multinom drop incomplete rows (na.omit); mlogit refuses a missing id, choice or attribute", function(ctx) {
  set.seed(4321)
  n_sit <- 100; J <- 3; n <- n_sit * J
  id <- rep(seq_len(n_sit), each = J); alt <- rep(c("a", "b", "c"), n_sit)
  x <- round(rnorm(n), 6); w <- round(rnorm(n), 6)
  util <- 0.8 * x - 0.5 * w - log(-log(runif(n)))
  choice <- as.integer(ave(util, id, FUN = function(u) u == max(u)))
  g <- c("A", "B", "C")[1 + (x + rnorm(n) > 0) + (w + rnorm(n) > 0.3)]
  d0 <- data.frame(row = seq_len(n), id = id, alt = alt, choice = choice, x = x, w = w,
                   x_int = as.integer(round(3 * x)), g = g, k = c(2L, 10L, 33L)[match(g, c("A", "B", "C"))])
  d0$choice_na <- d0$choice; d0$choice_na[c(5, 101)] <- NA
  d0$x_na <- d0$x; d0$x_na[17] <- NA
  d0$id_na <- d0$id; d0$id_na[44] <- NA
  d0$g[c(23, 240)] <- NA; d0$k[c(23, 240)] <- NA
  d <- ctx$gen_dataset("choice_na", d0)
  stopifnot(is.integer(d$choice), is.integer(d$choice_na), is.integer(d$id_na), is.integer(d$x_int), is.integer(d$k))
  fit <- function(fm, link) glm(fm, family = binomial(link), data = d, control = glm.control(epsilon = 1e-14, maxit = 100))
  ly <- fit(choice_na ~ x + w, "logit")                        # missing outcome only (2 rows)
  lo <- fit(choice_na ~ x_na + w, "logit"); pr <- fit(choice_na ~ x_na + w, "probit")   # outcome and regressor (3 rows)
  cc_g <- complete.cases(d$g, d$x, d$w)
  refuses <- function(dd, ch) inherits(try(mlogit(as.formula(paste(ch, "~ x + w | 0")), data = dfidx(dd, idx = c("id", "alt"), choice = ch)), silent = TRUE), "try-error")
  dx <- d; dx$x <- dx$x_na; di <- d; di$id <- di$id_na
  list(values = list(
         logit_y_coef = named(coef(ly)), logit_y_se = named(sqrt(diag(vcov(ly)))), logit_y_loglik = as.numeric(logLik(ly)), logit_y_nobs = nobs(ly),
         logit_coef = named(coef(lo)), logit_se = named(sqrt(diag(vcov(lo)))), logit_loglik = as.numeric(logLik(lo)), logit_nobs = nobs(lo),
         probit_coef = named(coef(pr)), probit_se = named(sqrt(diag(vcov(pr)))), probit_loglik = as.numeric(logLik(pr)), probit_nobs = nobs(pr),
         n_rows = nrow(d), missing_choice_na = sum(is.na(d$choice_na)), missing_x_na = sum(is.na(d$x_na)),
         missing_id_na = sum(is.na(d$id_na)), missing_g = sum(is.na(d$g)),
         multinom_n = sum(cc_g), multinom_counts = vec(as.numeric(table(d$g[cc_g]))),
         k_levels = vec(as.numeric(levels(factor(d$k)))),
         mlogit_refuses = vec(as.numeric(c(refuses(d, "choice_na"), refuses(dx, "choice"), refuses(di, "choice"))))),
       reference = "stats::glm(binomial('logit' | 'probit'), na.action = na.omit, epsilon = 1e-14); table(g) on complete.cases(g, x, w) (nnet::multinom's na.omit sample); mlogit::mlogit errors on a missing choice, attribute or id",
       call = "glm(choice_na ~ x + w, family = binomial, data = d); glm(choice_na ~ x_na + w, family = binomial(link), data = d)", packages = c("mlogit"))
})
