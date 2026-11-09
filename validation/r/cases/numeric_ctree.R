# validation/r/cases/numeric_ctree.R -- (X-17): ctree root association statistics far below 1e-16.
# partykit is not installed: the quadratic rank statistic of ml/ctree.rs is written out,
# z = cor(rank(x), y) sqrt(n - 1), p = pchisq(z^2, 1, lower.tail = FALSE), Bonferroni over the 3 inputs.

case("tail_ctree_root", "ml/ctree", "ctree root association statistics below 1e-16 (formula; partykit not installed)", function(ctx) {
  set.seed(20261025)
  n <- 300
  x2 <- round(rnorm(n), 6)
  y <- round(x2 + 0.6 * rnorm(n), 6)
  x1 <- round(y + 1.8 * rnorm(n), 6)                  # weaker than x2, still p < 1e-16
  x3 <- round(rnorm(n), 6)
  d <- ctx$gen_dataset("tail_ctree_root", data.frame(x1 = x1, x2 = x2, x3 = x3, y = y))
  z2 <- sapply(c("x1", "x2", "x3"), function(v) (n - 1) * cor(rank(d[[v]]), d$y)^2)
  p <- pchisq(z2, 1, lower.tail = FALSE)
  stopifnot(1 - pchisq(z2[1], 1) == 0, 1 - pchisq(z2[2], 1) == 0, p[2] < p[1], p[3] > 1e-3,
            !anyDuplicated(d$x1), !anyDuplicated(d$x2), !anyDuplicated(d$x3))
  list(values = list(z2 = vec(unname(z2)), p = vec(unname(p)), p_bonferroni = vec(pmin(1, 3 * unname(p)))),
       reference = "ctree quadratic rank statistic written out: z = cor(rank(x), y) sqrt(n - 1); pchisq(z^2, 1, lower.tail = FALSE)",
       call = "(n - 1) * cor(rank(x), y)^2", packages = character(0))
})
