# validation/r/cases/numeric_ctree.R -- (X-17, rewritten by T4-WP18): ctree root statistics far below 1e-16, now from partykit.
suppressPackageStartupMessages(library(partykit))
case("tail_ctree_root", "ml/ctree", "partykit::ctree root statistics with p-values below 1e-16 (Bonferroni tie on the log scale)", function(ctx) {
  set.seed(20261025)
  n <- 300
  x2 <- round(rnorm(n), 6)
  y <- round(x2 + 0.6 * rnorm(n), 6)
  x1 <- round(y + 1.8 * rnorm(n), 6)                  # weaker than x2, still p < 1e-16
  x3 <- round(rnorm(n), 6)
  d <- ctx$gen_dataset("tail_ctree_root", data.frame(x1 = x1, x2 = x2, x3 = x3, y = y))
  ct <- partykit::ctree(y ~ x1 + x2 + x3, data = d, control = partykit::ctree_control(stump = TRUE))
  cr <- info_node(node_party(ct))$criterion
  st <- cr["statistic", c("x1", "x2", "x3")]
  p <- pchisq(st, 1, lower.tail = FALSE)
  stopifnot(1 - pchisq(st[1], 1) == 0, 1 - pchisq(st[2], 1) == 0, p[2] < p[1], p[3] > 1e-3,
            varid_split(split_node(node_party(ct))) == 3L)
  list(values = list(statistic = vec(unname(st)), p = vec(unname(p)), p_adjusted = vec(unname(cr["p.value", c("x1", "x2", "x3")])),
                     split_break = breaks_split(split_node(node_party(ct)))),
       reference = "partykit::ctree(y ~ x1 + x2 + x3, ctree_control(stump = TRUE)): root criterion; p = pchisq(statistic, 1, lower.tail = FALSE)",
       call = "ctree(y ~ x1 + x2 + x3, control = ctree_control(stump = TRUE))", packages = c("partykit", "libcoin"))
})
