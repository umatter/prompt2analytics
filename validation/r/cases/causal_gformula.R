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
