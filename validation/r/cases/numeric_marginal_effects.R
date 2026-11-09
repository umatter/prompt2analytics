# validation/r/cases/numeric_marginal_effects.R -- (X-17): delta-method average marginal
# effects of a logit, written out in R as p2a computes them (G = d AME / d beta, V = vcov(glm)) and
# cross-checked against marginaleffects::avg_slopes; the slope's p-value is 1.06e-170, where 1 - pnorm is 0;
# the same fit with an aliased regressor (x3 = 2 x1; glm reports NA).

.ame_logit <- function(fit) {
  b <- coef(fit); ok <- !is.na(b); bb <- b[ok]
  V <- vcov(fit, complete = TRUE)[ok, ok, drop = FALSE]   # identified block only
  X <- model.matrix(fit)[, ok, drop = FALSE]
  z <- drop(X %*% bb); lam <- plogis(z) * (1 - plogis(z)); lamp <- lam * (1 - 2 * plogis(z))
  k <- length(bb); G <- matrix(0, k, k)
  for (j in 1:k) for (l in 1:k) G[j, l] <- mean((j == l) * lam + bb[j] * lamp * X[, l])
  ame <- bb * mean(lam); se <- sqrt(diag(G %*% V %*% t(G))); zz <- ame / se
  list(names = names(bb), ame = unname(ame), se = unname(se), z = unname(zz), p = 2 * pnorm(-abs(unname(zz))))
}

case("tail_marginal_logit", "regression/marginal_effects", "Logit average marginal effects (delta method) with p far below 1e-16; aliased regressor", function(ctx) {
  set.seed(20261023)
  n <- 2000
  x1 <- round(rnorm(n), 6); x2 <- round(rnorm(n), 6)
  y <- rbinom(n, 1, plogis(-0.4 + 1.2 * x1 + 0.15 * x2))
  d <- ctx$gen_dataset("tail_marginal_logit", data.frame(y = y, x1 = x1, x2 = x2, x3 = 2 * x1))
  ctl <- glm.control(epsilon = 1e-14, maxit = 100)
  fit <- glm(y ~ x1 + x2, family = binomial, data = d, control = ctl)
  # glm.fit's QR tolerance is min(1e-7, epsilon / 1000): the aliased fit needs epsilon = 1e-10 (tol 1e-13)
  fa <- glm(y ~ x1 + x2 + x3, family = binomial, data = d, control = glm.control(epsilon = 1e-10, maxit = 100))
  stopifnot(is.na(coef(fa)["x3"]), isTRUE(all.equal(coef(fa)[1:3], coef(fit), tolerance = 1e-8)))
  a <- .ame_logit(fit)
  stopifnot(1 - pnorm(abs(a$z[2])) == 0, a$p[2] > 1e-300)
  me <- suppressWarnings(marginaleffects::avg_slopes(fit))
  me <- as.data.frame(me)[match(c("x1", "x2"), me$term), ]
  stopifnot(max(abs(me$estimate / a$ame[2:3] - 1)) < 1e-6, max(abs(me$std.error / a$se[2:3] - 1)) < 1e-6)
  aa <- .ame_logit(fa)
  stopifnot(max(abs(aa$ame / a$ame - 1)) < 1e-8)
  list(values = list(coef = named(coef(fit)), coef_aliased = named(coef(fa)),
                     ame = named(setNames(a$ame, a$names)), ame_se = named(setNames(a$se, a$names)),
                     ame_z = named(setNames(a$z, a$names)), ame_p = named(setNames(a$p, a$names))),
       reference = "stats::glm(binomial); delta-method AME as p2a computes it (G = d AME / d beta), cross-checked against marginaleffects::avg_slopes",
       call = "glm(y ~ x1 + x2, binomial, epsilon = 1e-14); avg_slopes(fit)", packages = c("marginaleffects"))
})
