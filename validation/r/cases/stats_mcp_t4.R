# validation/r/cases/stats_mcp_t4.R -- T4-WP31 references: model.tables (STAT-75, TOOL-22) one-way and
# two-way, balanced, unbalanced and with an empty cell, means and effects with their standard errors;
# cov.wt without centring and with cor = TRUE (TOOL-23).
#
# model.tables semantics pinned here (R 4.3.3, stats:::model.tables.aov):
#  * effects are the projections of the sequential fit (proj(aov)): for y ~ a * b the a effect is
#    fit(a) - grand, the b effect fit(a + b) - fit(a), the a:b effect fit(a * b) - fit(a + b), each
#    averaged over the term's cells; means are grand + the effects of the term and its margins.
#  * se = TRUE: "means" gives the standard errors for differences of means, sqrt(2 MSE / r);
#    "effects" the standard errors of effects, sqrt(MSE / r); r is the replication of the term.
#    An unbalanced design gives no se (the message "Design is unbalanced - use se.contrast() for se's").
#  * an empty cell: its mean is NA (reported as null); its a:b effect is NA.

.mt_term <- function(tab) as.numeric(if (length(dim(tab)) == 2L) t(tab) else tab)   # row-major over (a, b)

.mt_values <- function(fit, p, two) {
  v <- list()
  m <- model.tables(fit, "means"); e <- model.tables(fit, "effects")
  v[[paste0(p, "grand")]] <- m$tables[["Grand mean"]]
  terms <- if (two) c("a", "b", "a:b") else "g"
  for (t in terms) {
    k <- gsub(":", "", t)
    v[[paste0(p, "means_", k)]] <- vec(.mt_term(m$tables[[t]]))
    v[[paste0(p, "effects_", k)]] <- vec(.mt_term(e$tables[[t]]))
  }
  v[[paste0(p, "mse")]] <- sum(residuals(fit)^2) / df.residual(fit); v[[paste0(p, "df_resid")]] <- df.residual(fit)
  v
}

case("stats_mcp_t4", "stats/modeltables", "model.tables(aov) means and effects with se: one-way balanced and unbalanced, two-way balanced, unbalanced and with an empty cell; cov.wt(center = FALSE, cor = TRUE)", function(ctx) {
  set.seed(20261117)
  g <- rep(c("a", "b", "c"), each = 4); y <- rep(c(5, 7, 9), each = 4) + round(rnorm(12, 0, 0.7), 3)
  gu <- c(rep("a", 3), rep("b", 4), rep("c", 5)); yu <- c(5, 7, 9)[match(gu, c("a", "b", "c"))] + round(rnorm(12, 0, 0.7), 3)
  O <- ctx$gen_dataset("modeltables_t4_oneway", data.frame(y = y, g = g, yu = yu, gu = gu))
  set.seed(20261118)
  a <- rep(c("a1", "a2"), each = 9); b <- rep(rep(c("b1", "b2", "b3"), each = 3), 2)
  mu <- outer(c(0, 1.5), c(0, 2, 3), "+") + 10; mu[2, 3] <- mu[2, 3] + 1
  y2 <- mu[cbind(match(a, c("a1", "a2")), match(b, c("b1", "b2", "b3")))] + round(rnorm(18, 0, 0.8), 3)
  B <- ctx$gen_dataset("modeltables_t4_bal", data.frame(y = y2, a = a, b = b))
  U <- ctx$gen_dataset("modeltables_t4_unbal", B[-c(1, 2, 13), ])
  E <- ctx$gen_dataset("modeltables_t4_empty", B[!(B$a == "a2" & B$b == "b3"), ])
  v <- list()
  f1 <- aov(y ~ g, data = O)
  v <- c(v, .mt_values(f1, "ow_", FALSE))
  v$ow_se_means <- c(model.tables(f1, "means", se = TRUE)$se$g)
  v$ow_se_effects <- c(model.tables(f1, "effects", se = TRUE)$se)[[1]]
  v$ow_n <- 4
  fu <- aov(yu ~ gu, data = O)
  vu <- .mt_values(fu, "owu_", FALSE); names(vu) <- sub("_gu$", "_g", names(vu))
  mu1 <- model.tables(fu, "means"); vu$owu_means_g <- vec(c(mu1$tables$gu)); vu$owu_effects_g <- vec(c(model.tables(fu, "effects")$tables$gu))
  vu$owu_n <- vec(c(table(O$gu)))
  stopifnot(is.null(suppressMessages(model.tables(fu, "means", se = TRUE))$se))       # unbalanced: no se
  v <- c(v, vu)
  for (cfg in list(list("bal_", B), list("unbal_", U), list("empty_", E))) {
    fit <- aov(y ~ a * b, data = cfg[[2]])
    v <- c(v, .mt_values(fit, cfg[[1]], TRUE))
    v[[paste0(cfg[[1]], "cell_n")]] <- vec(as.numeric(t(table(cfg[[2]]$a, cfg[[2]]$b))))
  }
  fb <- aov(y ~ a * b, data = B)
  sm <- model.tables(fb, "means", se = TRUE)$se; se <- model.tables(fb, "effects", se = TRUE)$se
  v$bal_se_means <- vec(c(sm$a, sm$b, sm$`a:b`)); v$bal_se_effects <- vec(c(se))
  stopifnot(abs(v$bal_se_effects[3] - sqrt(v$bal_mse / 3)) < 1e-12, abs(v$bal_se_means[3] - sqrt(2 * v$bal_mse / 3)) < 1e-12)
  # projection check for the unbalanced b effect: fit(a + b) - fit(a), averaged over the b levels
  fa <- fitted(lm(y ~ a, U)); fab <- fitted(lm(y ~ a + b, U))
  stopifnot(max(abs(tapply(fab - fa, U$b, mean) - v$unbal_effects_b)) < 1e-12)
  v$unbal_b_raw_means <- vec(tapply(U$y, U$b, mean))                                     # what a raw margin gives
  # cov.wt
  set.seed(20261119)
  n <- 25; x1 <- round(rnorm(n, 2, 1), 4); x2 <- round(0.5 * x1 + rnorm(n, 1, 0.8), 4); x3 <- round(runif(n, -1, 3), 4)
  w <- round(runif(n, 0.2, 2), 3)
  W <- ctx$gen_dataset("cov_wt_t4", data.frame(x1 = x1, x2 = x2, x3 = x3, w = w))
  X <- as.matrix(W[, c("x1", "x2", "x3")])
  for (meth in c("unbiased", "ML")) {
    for (cen in c(FALSE, TRUE)) {
      r <- cov.wt(X, wt = W$w, cor = TRUE, center = cen, method = meth)
      k <- paste0("covwt_", tolower(meth), if (cen) "_centered" else "_zero")
      v[[paste0(k, "_cov")]] <- vec(as.numeric(t(r$cov))); v[[paste0(k, "_cor")]] <- vec(as.numeric(t(r$cor)))
      v[[paste0(k, "_n_obs")]] <- r$n.obs
      if (cen) v[[paste0(k, "_center")]] <- vec(r$center)
    }
  }
  list(values = v, dataset = "validation/datasets/generated/modeltables_t4_bal.csv",
       reference = "stats::model.tables(aov(...), 'means' | 'effects', se = TRUE) (projections of the sequential fit; se of differences of means / of effects; none for an unbalanced design); stats::cov.wt(x, wt, cor = TRUE, center = FALSE | TRUE, method = 'unbiased' | 'ML')",
       call = "model.tables(aov(y ~ g), type, se = TRUE); model.tables(aov(y ~ a * b), type, se = TRUE); cov.wt(X, w, cor = TRUE, center, method)", packages = character(0))
})
