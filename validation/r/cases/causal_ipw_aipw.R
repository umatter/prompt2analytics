# validation/r/cases/causal_ipw_aipw.R -- IPW and AIPW references (T3-WP23: CAUS-92, CAUS-111, CAUS-113).
# Propensity: logistic regression (glm) of treat on x1, x2, x3 (WeightIt method = 'glm').
#  * WeightIt glm weights (ATE, ATT, ATC, stabilised ATE): the weightit() logistic path;
#  * IPW ATE, normalised (Hajek) and Horvitz-Thompson; ATT with control weights ps/(1 - ps);
#    three standard errors per estimator:
#      - "naive": propensity treated as known. Hajek and ATT: sandwich HC0 of the weighted lm y ~ treat
#        (= WeightIt::lm_weightit(vcov = "HC0")); HT: sqrt(sum IF^2) / n with IF = D y/e - (1-D) y/(1-e) - tau;
#      - "mest": M-estimation, the exact finite-sample sandwich A^-1 B A^-T / n of the stacked logit-score
#        and weighted-mean estimating equations (the propensity estimation is accounted for). For Hajek and
#        ATT this is WeightIt::lm_weightit(vcov = "asympt"), which is recorded as a cross-check
#        (*_weightit; WeightIt fits the propensity at glm's default epsilon and its Jacobian differs by
#        ~5e-10); Lunceford & Davidian's asymptotic form (2004, eq. 7) is recorded as *_ld;
#      - "eif": p2a's analytic default, sqrt(sum EIF^2) / n with the AIPW efficient influence function
#        centred at the IPW estimate (OLS outcome models per arm on (1, x1, x2, x3)); written out here.
#  * AIPW (doubly robust) ATE and ATT with OLS outcome models per arm on (1, x1, x2, x3):
#    ATE psi_i = m1 - m0 + D (Y - m1)/e - (1 - D)(Y - m0)/(1 - e), SE = sd(psi) / sqrt(n);
#    ATT IF_i = [D (Y - m0 - tau) - (1 - D) e/(1 - e) (Y - m0)] n / n1, SE = sd(IF) / sqrt(n)
#    (analytic; nuisance estimation ignored, the usual AIPW influence-function SE). Regression
#    adjustment ATE = mean(m1 - m0). No bootstrap is pinned.
suppressPackageStartupMessages({ library(WeightIt); library(sandwich) })

case("ipw_aipw_sim", "econometrics/treatment", "IPW (Hajek, HT, ATT) and AIPW on weighting_sim with logistic propensity scores", function(ctx) {
  d <- ctx$dataset("datasets/generated/weighting_sim.csv")
  X <- cbind(1, d$x1, d$x2, d$x3); D <- d$treat; Y <- d$y; n <- nrow(d); n1 <- sum(D); p <- ncol(X)
  g <- glm(treat ~ x1 + x2 + x3, data = d, family = binomial, control = glm.control(epsilon = 1e-14))
  e <- fitted(g)
  W_ate <- weightit(treat ~ x1 + x2 + x3, data = d, method = "glm", estimand = "ATE")
  W_att <- weightit(treat ~ x1 + x2 + x3, data = d, method = "glm", estimand = "ATT")
  W_atc <- weightit(treat ~ x1 + x2 + x3, data = d, method = "glm", estimand = "ATC")
  W_stab <- weightit(treat ~ x1 + x2 + x3, data = d, method = "glm", estimand = "ATE", stabilize = TRUE)
  w_ate <- W_ate$weights; w_att <- W_att$weights
  stopifnot(max(abs(w_ate - ifelse(D == 1, 1 / e, 1 / (1 - e)))) < 1e-8)
  hajek <- weighted.mean(Y[D == 1], w_ate[D == 1]) - weighted.mean(Y[D == 0], w_ate[D == 0])
  ht <- mean(D * Y / e) - mean((1 - D) * Y / (1 - e))
  att <- weighted.mean(Y[D == 1], w_att[D == 1]) - weighted.mean(Y[D == 0], w_att[D == 0])
  f_ate <- lm(y ~ treat, data = d, weights = w_ate); f_att <- lm(y ~ treat, data = d, weights = w_att)
  se_naive_ate <- sqrt(sandwich::vcovHC(f_ate, type = "HC0")[2, 2]); se_naive_att <- sqrt(sandwich::vcovHC(f_att, type = "HC0")[2, 2])
  se_naive_ht <- sqrt(sum((D * Y / e - (1 - D) * Y / (1 - e) - ht)^2)) / n
  # M-estimation, exact sandwich of the stacked equations [X (D - e); w Z (Y - Z theta)], Z = (1, D)
  mest_reg <- function(w, dw) {
    Z <- cbind(1, D)
    th <- solve(crossprod(Z * w, Z), crossprod(Z * w, Y)); r <- drop(Y - Z %*% th)
    psi <- cbind(X * (D - e), Z * (w * r))
    A <- matrix(0, p + 2, p + 2)
    A[1:p, 1:p] <- -crossprod(X * (e * (1 - e)), X)
    A[p + 1:2, 1:p] <- crossprod(Z * r, dw)
    A[p + 1:2, p + 1:2] <- -crossprod(Z * w, Z)
    A <- A / n; B <- crossprod(psi) / n; Ai <- solve(A)
    sqrt((Ai %*% B %*% t(Ai) / n)[p + 2, p + 2])
  }
  se_mest_ate <- mest_reg(w_ate, X * ifelse(D == 1, -(1 - e) / e, e / (1 - e)))
  se_mest_att <- mest_reg(w_att, X * ifelse(D == 1, 0, e / (1 - e)))
  psi_ht <- cbind(X * (D - e), D * Y / e - (1 - D) * Y / (1 - e) - ht)
  A <- matrix(0, p + 1, p + 1)
  A[1:p, 1:p] <- -crossprod(X * (e * (1 - e)), X)
  A[p + 1, 1:p] <- colSums(X * (-D * Y * (1 - e) / e - (1 - D) * Y * e / (1 - e)))
  A[p + 1, p + 1] <- -n
  A <- A / n; Ai <- solve(A)
  se_mest_ht <- sqrt((Ai %*% (crossprod(psi_ht) / n) %*% t(Ai) / n)[p + 1, p + 1])
  # Lunceford & Davidian (2004, eq. 7) asymptotic form for the Hajek ATE (cross-check only)
  mu1 <- sum(D * Y / e) / sum(D / e); mu0 <- sum((1 - D) * Y / (1 - e)) / sum((1 - D) / (1 - e))
  I <- D * (Y - mu1) / e - (1 - D) * (Y - mu0) / (1 - e)
  Hv <- colMeans(X * (D * (Y - mu1) * (1 - e) / e + (1 - D) * (Y - mu0) * e / (1 - e)))
  Egg <- crossprod(X * sqrt(e * (1 - e))) / n; S <- X * (D - e)
  se_mest_ld <- sqrt(sum((I - drop(S %*% solve(Egg, Hv)))^2)) / n
  # outcome models and the AIPW scores
  m1 <- drop(X %*% qr.coef(qr(X[D == 1, ]), Y[D == 1])); m0 <- drop(X %*% qr.coef(qr(X[D == 0, ]), Y[D == 0]))
  psi <- m1 - m0 + D * (Y - m1) / e - (1 - D) * (Y - m0) / (1 - e)
  phi <- D * (Y - m0) - (1 - D) * e / (1 - e) * (Y - m0)
  aipw_att <- sum(phi) / n1
  if_att <- (phi - D * aipw_att) * n / n1
  # p2a's analytic IPW default: the efficient influence function centred at the IPW estimate
  se_eif_hajek <- sqrt(sum((psi - hajek)^2)) / n
  se_eif_ht <- sqrt(sum((psi - ht)^2)) / n
  p1 <- n1 / n
  se_eif_att <- sqrt(sum((ifelse(D == 1, (Y - m0 - att) / p1, -(e / (1 - e)) * (Y - m0) / p1))^2)) / n
  r2 <- function(s) summary(lm(y ~ x1 + x2 + x3, data = d, subset = s))$r.squared
  v <- list(ps_coef = named(coef(g)), ps_head = vec(head(e, 10)),
            glm_ate_weights_head = vec(head(W_ate$weights, 12)), glm_att_weights_head = vec(head(W_att$weights, 12)),
            glm_atc_weights_head = vec(head(W_atc$weights, 12)), glm_ate_stab_weights_head = vec(head(W_stab$weights, 12)),
            ipw_ate_hajek = hajek, ipw_ate_ht = ht, ipw_att = att,
            ipw_ate_se_naive = se_naive_ate, ipw_att_se_naive = se_naive_att, ipw_ate_ht_se_naive = se_naive_ht,
            ipw_ate_se_mest = se_mest_ate, ipw_att_se_mest = se_mest_att, ipw_ate_ht_se_mest = se_mest_ht,
            ipw_ate_se_mest_weightit = sqrt(vcov(lm_weightit(y ~ treat, data = d, weightit = W_ate))[2, 2]),
            ipw_att_se_mest_weightit = sqrt(vcov(lm_weightit(y ~ treat, data = d, weightit = W_att))[2, 2]),
            ipw_ate_se_mest_ld = se_mest_ld,
            ipw_ate_se_eif = se_eif_hajek, ipw_ate_ht_se_eif = se_eif_ht, ipw_att_se_eif = se_eif_att,
            aipw_ate = mean(psi), aipw_ate_se = sd(psi) / sqrt(n),
            aipw_att = aipw_att, aipw_att_se = sd(if_att) / sqrt(n),
            ra_ate = mean(m1 - m0), outcome_r2_treated = r2(D == 1), outcome_r2_control = r2(D == 0), n = n)
  list(values = v, reference = "glm(binomial) propensity (WeightIt method = 'glm'); Hajek / Horvitz-Thompson IPW; sandwich::vcovHC(HC0) of the weighted lm and WeightIt::lm_weightit(vcov = 'asympt' | 'HC0'); stacked M-estimation sandwich; AIPW with OLS outcome models",
       call = "weightit(treat ~ x1 + x2 + x3, data = weighting_sim, method = 'glm', estimand); lm_weightit(y ~ treat, weightit = W)", packages = c("WeightIt", "sandwich"))
})
