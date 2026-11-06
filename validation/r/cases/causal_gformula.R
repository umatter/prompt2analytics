# validation/r/cases/causal_gformula.R -- parametric g-formula references (T3-WP28: CAUS-64, CAUS-65;
# moved in CAUS-16, CAUS-105). gfoRmula is not installed, so every estimator is written out from its
# published formula (Hernan & Robins 2020, ch. 17 and 21; Keil et al. 2014) and the Monte Carlo
# integral of the g-formula is replaced by exact quadrature:
#  * gformula_hazard: pooled logistic discrete-time hazard on person-periods, time-fixed covariates and a
#    time-fixed (point) treatment; S(t | a, L) = prod_{s <= t} (1 - h(s, a, L)); risks under always /
#    never treat standardised over the empirical covariates (deterministic, no Monte Carlo); the natural
#    course of the point treatment; the end-of-follow-up logistic model the old code ran instead.
#  * gformula_hazard_tv: three periods, one continuous time-varying covariate L_t, a time-varying
#    treatment and loss to follow-up after period 2. Covariate, treatment and hazard models are fitted on
#    the subjects at risk; the g-formula integral over the Gaussian covariate paths is a 16-point
#    Gauss-Hermite product rule (converged to < 1e-13), the natural course sums the 8 treatment paths.
#  * gformula_natural_course: one period, a continuous covariate L_t0 ~ Normal(lm), confounded treatment,
#    binary and continuous outcomes; exact natural-course and static risks by 120-point Gauss-Hermite
#    (agrees with integrate(rel.tol = 1e-12) to < 1e-9); CAUS-16's natural-course check.

# Probabilists' Gauss-Hermite rule (Golub-Welsch): integral f(z) dnorm(z) dz ~ sum(w * f(x)).
.gf_gh <- function(k) {
  i <- seq_len(k - 1); J <- matrix(0, k, k); J[cbind(i, i + 1)] <- sqrt(i); J[cbind(i + 1, i)] <- sqrt(i)
  e <- eigen(J, symmetric = TRUE); list(x = e$values, w = e$vectors[1, ]^2)
}
# Discrete-time Kaplan-Meier cumulative incidence at periods 1..TT.
.gf_km <- function(time, event, TT) {
  nt <- vapply(seq_len(TT), function(t) sum(time >= t), 0); dt <- vapply(seq_len(TT), function(t) sum(time == t & event == 1), 0)
  1 - cumprod(1 - dt / nt)
}
.gf_ctl <- glm.control(epsilon = 1e-14, maxit = 100)

case("gformula_hazard", "causal/gformula", "Discrete-time hazard g-formula (pooled logistic, 5 periods, time-fixed L): risks under always / never treat", function(ctx) {
  set.seed(20261207)
  n <- 400; TT <- 5
  L <- rnorm(n); Lb <- rbinom(n, 1, 0.4); A <- rbinom(n, 1, plogis(0.4 * L))
  ev <- rep(NA, n)
  for (i in seq_len(n)) for (t in 1:TT) { h <- plogis(-2.5 + 0.15 * t - 0.6 * A[i] + 0.5 * L[i] + 0.4 * Lb[i]); if (runif(1) < h) { ev[i] <- t; break } }
  time <- ifelse(is.na(ev), TT, ev); event <- as.integer(!is.na(ev))
  d <- ctx$gen_dataset("gformula_surv", data.frame(id = seq_len(n), L = round(L, 10), Lb = Lb, A = A, time = time, event = event))
  pp <- do.call(rbind, lapply(seq_len(n), function(i) data.frame(id = i, t = 1:d$time[i], L = d$L[i], Lb = d$Lb[i], A = d$A[i], y = as.integer(1:d$time[i] == d$time[i] & d$event[i] == 1))))
  hm <- glm(y ~ t + A + L + Lb, data = pp, family = binomial, control = .gf_ctl)
  stopifnot(hm$converged)
  surv <- function(model, a) { S <- matrix(1, n, TT); s <- rep(1, n)
    for (t in 1:TT) { s <- s * (1 - predict(model, newdata = data.frame(t = t, A = a, L = d$L, Lb = d$Lb), type = "response")); S[, t] <- s }; S }
  risk <- function(model, a) 1 - colMeans(surv(model, a))
  r1 <- risk(hm, 1); r0 <- risk(hm, 0)
  # Natural course of the point treatment: A drawn once from the period-1 model and carried forward.
  g1 <- fitted(glm(A ~ L + Lb, data = d, family = binomial, control = .gf_ctl))
  r_nat <- 1 - colMeans(g1 * surv(hm, 1) + (1 - g1) * surv(hm, 0))
  # What outcome_type 'survival' computed before (CAUS-64): the end-of-follow-up logistic model.
  eof <- glm(event ~ L + Lb + A, data = d, family = binomial, control = .gf_ctl)
  v <- list(hazard_coef = named(coef(hm)), hazard_se = named(summary(hm)$coefficients[, 2]),
            risk_always = vec(r1), risk_never = vec(r0), risk_diff_T = r1[TT] - r0[TT], risk_ratio_T = r1[TT] / r0[TT],
            risk_natural_point = vec(r_nat), observed_risk_curve = vec(.gf_km(d$time, d$event, TT)),
            observed_risk_T = mean(d$event), n_person_periods = nrow(pp),
            binary_eof_always = mean(predict(eof, newdata = transform(d, A = 1), type = "response")),
            binary_eof_never = mean(predict(eof, newdata = transform(d, A = 0), type = "response")))
  list(values = v, reference = "parametric g-formula with a pooled logistic discrete-time hazard and time-fixed covariates (Hernan & Robins 2020, ch. 17), written out (gfoRmula not installed)",
       call = "glm(y ~ t + A + L + Lb, person-period data, binomial); standardise prod (1 - h) over L", packages = character(0))
})

case("gformula_hazard_tv", "causal/gformula", "Discrete-time hazard g-formula with a time-varying covariate and treatment (3 periods, loss to follow-up): models on the risk sets, exact g-formula risks", function(ctx) {
  set.seed(20261208)
  n <- 1000; TT <- 3
  B <- rnorm(n)
  Lm <- matrix(NA_real_, n, TT); Am <- matrix(NA_integer_, n, TT)
  time <- rep(TT, n); event <- rep(0L, n); cens2 <- runif(n) < 0.1
  for (i in seq_len(n)) for (t in seq_len(TT)) {
    Lm[i, t] <- if (t == 1) 0.5 * B[i] + rnorm(1) else 0.3 * B[i] + 0.6 * Lm[i, t - 1] - 0.4 * Am[i, t - 1] + rnorm(1)
    Am[i, t] <- rbinom(1, 1, plogis(-0.2 + 0.6 * Lm[i, t] + if (t > 1) 1.0 * Am[i, t - 1] else 0))
    if (runif(1) < plogis(-2.2 + 0.1 * t - 0.7 * Am[i, t] + 0.6 * Lm[i, t] + 0.3 * B[i])) { time[i] <- t; event[i] <- 1L; break }
    if (t == 2 && cens2[i]) { time[i] <- 2L; break }   # lost to follow-up after period 2
  }
  # Wide layout as the MCP tool reads it (varname_t0, varname_t1, ...); cells after a subject's last
  # period are missing (empty in the CSV).
  d <- ctx$gen_dataset("gformula_surv_tv", data.frame(id = seq_len(n), B = round(B, 10),
         L_t0 = round(Lm[, 1], 10), L_t1 = round(Lm[, 2], 10), L_t2 = round(Lm[, 3], 10),
         A_t0 = Am[, 1], A_t1 = Am[, 2], A_t2 = Am[, 3], time = time, event = event))
  stopifnot(sum(is.na(d$L_t1)) == sum(d$time < 2), sum(is.na(d$L_t2)) == sum(d$time < 3), sum(d$time == 2 & d$event == 0) > 50)
  rs <- lapply(seq_len(TT), function(t) d$time >= t)                          # risk sets: at risk at the start of period t
  mL <- list(lm(L_t0 ~ B, data = d[rs[[1]], ]), lm(L_t1 ~ B + L_t0 + A_t0, data = d[rs[[2]], ]), lm(L_t2 ~ B + L_t1 + A_t1, data = d[rs[[3]], ]))
  mA <- list(glm(A_t0 ~ B + L_t0, data = d[rs[[1]], ], family = binomial, control = .gf_ctl),
             glm(A_t1 ~ B + L_t1 + A_t0, data = d[rs[[2]], ], family = binomial, control = .gf_ctl),
             glm(A_t2 ~ B + L_t2 + A_t1, data = d[rs[[3]], ], family = binomial, control = .gf_ctl))
  pp <- do.call(rbind, lapply(seq_len(n), function(i) { tt <- seq_len(d$time[i])
    data.frame(id = i, t = tt, A = unlist(d[i, paste0("A_t", tt - 1)]), B = d$B[i], L = unlist(d[i, paste0("L_t", tt - 1)]),
               y = as.integer(tt == d$time[i] & d$event[i] == 1)) }))
  hm <- glm(y ~ t + A + B + L, data = pp, family = binomial, control = .gf_ctl)
  stopifnot(hm$converged, all(vapply(mA, function(m) m$converged, TRUE)))
  c1 <- coef(mL[[1]]); c2 <- coef(mL[[2]]); c3 <- coef(mL[[3]]); s <- vapply(mL, sigma, 0)
  e1 <- coef(mA[[1]]); e2 <- coef(mA[[2]]); e3 <- coef(mA[[3]]); k <- coef(hm)
  haz <- function(t, a, b, l) plogis(k[1] + k[2] * t + k[3] * a + k[4] * b + k[5] * l)
  r <- .gf_gh(16); G <- expand.grid(i1 = seq_along(r$x), i2 = seq_along(r$x), i3 = seq_along(r$x))
  z1 <- r$x[G$i1]; z2 <- r$x[G$i2]; z3 <- r$x[G$i3]; W <- r$w[G$i1] * r$w[G$i2] * r$w[G$i3]
  exact <- function(regime) {
    paths <- if (regime == "natural") as.matrix(expand.grid(0:1, 0:1, 0:1)) else matrix(if (regime == "always") 1 else 0, 1, 3)
    risk <- c(0, 0, 0)
    for (i in seq_len(n)) { b <- d$B[i]; S <- c(0, 0, 0)
      for (p in seq_len(nrow(paths))) { a <- paths[p, ]
        l1 <- c1[1] + c1[2] * b + s[1] * z1
        l2 <- c2[1] + c2[2] * b + c2[3] * l1 + c2[4] * a[1] + s[2] * z2
        l3 <- c3[1] + c3[2] * b + c3[3] * l2 + c3[4] * a[2] + s[3] * z3
        pw <- if (regime != "natural") 1 else {
          g1 <- plogis(e1[1] + e1[2] * b + e1[3] * l1); g2 <- plogis(e2[1] + e2[2] * b + e2[3] * l2 + e2[4] * a[1])
          g3 <- plogis(e3[1] + e3[2] * b + e3[3] * l3 + e3[4] * a[2])
          (if (a[1] == 1) g1 else 1 - g1) * (if (a[2] == 1) g2 else 1 - g2) * (if (a[3] == 1) g3 else 1 - g3) }
        S1 <- 1 - haz(1, a[1], b, l1); S2 <- S1 * (1 - haz(2, a[2], b, l2)); S3 <- S2 * (1 - haz(3, a[3], b, l3))
        S <- S + c(sum(W * pw * S1), sum(W * pw * S2), sum(W * pw * S3)) }
      risk <- risk + (1 - S) }
    risk / n
  }
  ra <- exact("always"); rn <- exact("never"); rnat <- exact("natural")
  v <- list(cov_coef_p1 = vec(c1), cov_coef_p2 = vec(c2), cov_coef_p3 = vec(c3), cov_sigma = vec(s),
            trt_coef_p1 = vec(e1), trt_coef_p2 = vec(e2), trt_coef_p3 = vec(e3),
            n_at_risk = vec(vapply(rs, sum, 0)), n_person_periods = nrow(pp),
            hazard_coef = named(k), hazard_se = named(summary(hm)$coefficients[, 2]),
            observed_risk_curve = vec(.gf_km(d$time, d$event, TT)),
            risk_always = vec(ra), risk_never = vec(rn), risk_natural = vec(rnat))
  list(values = v, reference = "parametric g-formula (Hernan & Robins 2020, ch. 17 and 21; Keil et al. 2014): covariate, treatment and pooled logistic hazard models on the risk sets, the g-formula integral by Gauss-Hermite quadrature, written out (gfoRmula not installed)",
       call = "lm(L_t ~ B + L_(t-1) + A_(t-1)), glm(A_t ~ B + L_t + A_(t-1), binomial) on time >= t; glm(y ~ t + A + B + L, person-periods, binomial); E[prod (1 - h)] by 16^3-point Gauss-Hermite", packages = character(0))
})

case("gformula_natural_course", "causal/gformula", "Parametric g-formula, one period, continuous covariate: exact natural-course and static risks (CAUS-16 natural-course check)", function(ctx) {
  set.seed(20261209)
  n <- 2000
  B <- rnorm(n)
  L <- 0.5 * B + rnorm(n, sd = 2)
  A <- rbinom(n, 1, plogis(-0.3 + 0.5 * L))
  Y <- rbinom(n, 1, plogis(-2 + 1.5 * L + 0.4 * A + 0.3 * B))
  Yc <- 1 + 0.8 * L + 1.0 * A + 0.5 * B + rnorm(n)
  d <- ctx$gen_dataset("gformula_nc", data.frame(B = round(B, 10), L_t0 = round(L, 10), A_t0 = A, Y = Y, Yc = round(Yc, 10)))
  mL <- lm(L_t0 ~ B, data = d); s <- sigma(mL); mu <- fitted(mL)
  mA <- glm(A_t0 ~ B + L_t0, data = d, family = binomial, control = .gf_ctl)
  mY <- glm(Y ~ B + L_t0 + A_t0, data = d, family = binomial, control = .gf_ctl)
  mC <- lm(Yc ~ B + L_t0 + A_t0, data = d)
  eA <- coef(mA); eY <- coef(mY); eC <- coef(mC)
  r <- .gf_gh(120)
  nat <- alw <- nev <- Eg <- 0
  for (i in seq_len(n)) {
    l <- mu[i] + s * r$x; b <- d$B[i]
    g <- plogis(eA[1] + eA[2] * b + eA[3] * l)
    q1 <- plogis(eY[1] + eY[2] * b + eY[3] * l + eY[4]); q0 <- plogis(eY[1] + eY[2] * b + eY[3] * l)
    nat <- nat + sum(r$w * (g * q1 + (1 - g) * q0)); alw <- alw + sum(r$w * q1); nev <- nev + sum(r$w * q0); Eg <- Eg + sum(r$w * g)
  }
  Eg <- Eg / n; lin <- eC[1] + eC[2] * mean(d$B) + eC[3] * mean(mu)
  v <- list(cov_coef = vec(coef(mL)), cov_sigma = s, trt_coef = vec(eA),
            outcome_coef_binary = vec(eY), outcome_coef_cont = vec(eC), outcome_sigma_cont = sigma(mC),
            natural_binary = nat / n, always_binary = alw / n, never_binary = nev / n,
            natural_cont = lin + eC[4] * Eg, always_cont = lin + eC[4], never_cont = lin,
            observed_mean_binary = mean(d$Y), observed_mean_cont = mean(d$Yc))
  list(values = v, reference = "parametric g-formula (Hernan & Robins 2020, ch. 21): Normal covariate model, logistic treatment and outcome models; the natural-course integral by Gauss-Hermite quadrature, written out (gfoRmula not installed)",
       call = "lm(L_t0 ~ B); glm(A_t0 ~ B + L_t0, binomial); glm(Y ~ B + L_t0 + A_t0, binomial); lm(Yc ~ B + L_t0 + A_t0); integral over Normal(mu_i, sigma)", packages = character(0))
})
