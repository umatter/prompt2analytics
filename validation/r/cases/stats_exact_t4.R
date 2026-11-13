# Tier 4 exact and count tests (T4-WP03): binom.test and poisson.test two-sided minlike p-values, one-sided
# intervals of poisson.test / mantelhaen.test / prop.test, prop.test continuity-corrected intervals, contr.poly
# and se.contrast, and the additive two-way ANOVA on a subject x condition layout.
# An infinite interval bound cannot be stored: it is NA here, and the Rust tests assert it is infinite.
.na_inf <- function(x) { x <- as.numeric(x); x[!is.finite(x)] <- NA; x }

case("stat_binom_minlike_t4", "stats/binom", "binom.test two-sided p (relErr 1 + 1e-7 minlike rule) at x = m and at n = 1,000 and 10,000", function(ctx) {
  g <- data.frame(label = sprintf("m%02d", 1:6),
                  x = c(5, 4950, 5100, 52, 1, 20), n = c(10, 10000, 10000, 1000, 10000, 10000),
                  p = c(0.5, 0.5, 0.5, 0.04, 5e-4, 0.001))
  d <- ctx$gen_dataset("binom_minlike_t4", g)
  pv <- setNames(vapply(seq_len(nrow(d)), function(i) binom.test(d$x[i], d$n[i], d$p[i])$p.value, 0), d$label)
  list(values = list(p_two = named(pv)),
       reference = "stats::binom.test(x, n, p) two-sided: sum of dbinom(i) <= dbinom(x) (1 + 1e-7) over the far tail",
       call = "binom.test(x, n, p)", packages = character(0))
})

case("stat_poisson_minlike_t4", "stats/poisson_test", "poisson.test: two-sided minlike p (one and two samples), one-sided intervals at 0.95 and 0.90", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")   # no data needed
  v <- list()
  one <- list(z0 = c(0, 2), lo2 = c(2, 10), hi25 = c(25, 10), eq10 = c(10, 10), lo3 = c(3, 0.5),
              k1100 = c(1100, 1000), k10300 = c(10300, 10000), k940 = c(940, 1000))
  v$one_two_p <- named(vapply(one, function(a) poisson.test(a[1], a[2])$p.value, 0))
  two <- list(z1 = c(0, 5, 1, 1), z2 = c(5, 0, 1, 1), r230 = c(2, 30, 10, 20))
  for (nm in names(two)) {
    a <- two[[nm]]; t <- poisson.test(a[1:2], a[3:4])
    v[[paste0("two_", nm, "_p")]] <- t$p.value; v[[paste0("two_", nm, "_ci")]] <- vec(.na_inf(t$conf.int))
  }
  for (lv in c(95, 90)) for (alt in c("less", "greater")) {
    tag <- paste0(alt, lv)
    t <- poisson.test(10, 2, r = 3, alternative = alt, conf.level = lv / 100)
    v[[paste0("x10r3_", tag, "_p")]] <- t$p.value; v[[paste0("x10r3_", tag, "_ci")]] <- vec(.na_inf(t$conf.int))
    v[[paste0("x0_", tag, "_ci")]] <- vec(.na_inf(poisson.test(0, 2, alternative = alt, conf.level = lv / 100)$conf.int))
    for (nm in c("rr", "z1", "z2")) {
      a <- switch(nm, rr = c(11, 21, 800, 3011), z1 = c(0, 5, 1, 1), z2 = c(5, 0, 1, 1))
      t <- poisson.test(a[1:2], a[3:4], alternative = alt, conf.level = lv / 100)
      v[[paste0(nm, "_", tag, "_p")]] <- t$p.value; v[[paste0(nm, "_", tag, "_ci")]] <- vec(.na_inf(t$conf.int))
    }
  }
  list(dataset = "validation/datasets/grunfeld.csv", values = v,
       reference = "stats::poisson.test (one sample: minlike p, qgamma bounds at alpha for one-sided; two samples: binom.test)",
       call = "poisson.test(x, T, r, alternative, conf.level)", packages = character(0))
})

case("stat_mantelhaen_alt_t4", "stats/mantelhaen_test", "mantelhaen.test 2 x 2 x K: one-sided p and intervals, conf.level 0.90, and the method label at DELTA = 0", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")   # no data needed: the Rabbits array of ?mantelhaen.test
  rab <- array(c(0, 0, 6, 5, 3, 0, 3, 6, 6, 2, 0, 4, 5, 6, 1, 0, 2, 5, 0, 0), dim = c(2, 2, 5))
  v <- list(rab = vec(as.numeric(rab)))
  for (cor in c(TRUE, FALSE)) for (alt in c("two.sided", "less", "greater")) {
    m <- mantelhaen.test(rab, alternative = alt, correct = cor)
    tag <- paste0(c(two.sided = "two", less = "less", greater = "greater")[[alt]], if (cor) "_cc" else "_nocc")
    v[[paste0("stat_", tag)]] <- unname(m$statistic); v[[paste0("p_", tag)]] <- m$p.value
  }
  for (lv in c(95, 90)) for (alt in c("two.sided", "less", "greater")) {
    tag <- paste0(c(two.sided = "two", less = "less", greater = "greater")[[alt]], lv)
    v[[paste0("ci_", tag)]] <- vec(.na_inf(mantelhaen.test(rab, alternative = alt, conf.level = lv / 100)$conf.int))
  }
  v$estimate <- unname(mantelhaen.test(rab)$estimate)
  bal <- array(rep(5, 8), c(2, 2, 2)); mb <- mantelhaen.test(bal)
  stopifnot(grepl("without continuity correction", mb$method))
  v$bal_stat <- unname(mb$statistic); v$bal_p <- mb$p.value; v$bal_ci <- vec(mb$conf.int)
  list(dataset = "validation/datasets/grunfeld.csv", values = v,
       reference = "stats::mantelhaen.test (2 x 2 x K; YATES only when |DELTA| >= 0.5; one-sided CI ESTIMATE exp(+-qnorm(conf.level) sd))",
       call = "mantelhaen.test(array(c(...), c(2, 2, 5)), alternative, correct, conf.level)", packages = character(0))
})

case("stat_prop_ci_t4", "stats/prop_test", "prop.test intervals: Yates capped at |x - n p| and |DELTA| / sum(1/n), boundaries, conf.level 0.90", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")   # no data needed
  v <- list()
  one <- list(o10 = list(x = 10, n = 100, p = 0.1, lv = 0.95), o15 = list(x = 15, n = 100, p = 0.1, lv = 0.90),
              o0 = list(x = 0, n = 20, p = 0.2, lv = 0.95))
  two <- list(t1525 = list(x = c(15, 25), n = c(50, 50), lv = 0.95), t3030 = list(x = c(30, 30), n = c(100, 100), lv = 0.95),
              t3031 = list(x = c(30, 31), n = c(100, 100), lv = 0.95), t119 = list(x = c(1, 19), n = c(20, 20), lv = 0.90))
  for (cor in c(TRUE, FALSE)) for (alt in c("two.sided", "less", "greater")) {
    tag <- paste0(c(two.sided = "two", less = "less", greater = "greater")[[alt]], if (cor) "_cc" else "_nocc")
    for (nm in names(one)) {
      a <- one[[nm]]; r <- suppressWarnings(prop.test(a$x, a$n, p = a$p, alternative = alt, correct = cor, conf.level = a$lv))
      v[[paste0(nm, "_", tag, "_stat")]] <- unname(r$statistic); v[[paste0(nm, "_", tag, "_p")]] <- r$p.value
      v[[paste0(nm, "_", tag, "_ci")]] <- vec(r$conf.int)
    }
    for (nm in names(two)) {
      a <- two[[nm]]; r <- suppressWarnings(prop.test(a$x, a$n, alternative = alt, correct = cor, conf.level = a$lv))
      v[[paste0(nm, "_", tag, "_stat")]] <- unname(r$statistic); v[[paste0(nm, "_", tag, "_p")]] <- r$p.value
      v[[paste0(nm, "_", tag, "_ci")]] <- vec(r$conf.int)
    }
  }
  list(dataset = "validation/datasets/grunfeld.csv", values = v,
       reference = "stats::prop.test (k = 1: Wilson with p.c = ESTIMATE +- YATES/n, YATES = min(0.5, |x - n p|); k = 2: Wald +- YATES sum(1/n), YATES = min(0.5, |DELTA| / sum(1/n)))",
       call = "prop.test(x, n[, p], alternative, correct, conf.level)", packages = character(0))
})

case("stat_secontrast_poly_t4", "stats/secontrast", "contr.poly(k) for k = 2, 3, 4, 6 and se.contrast of the contr.poly(4) columns on numeric dose levels", function(ctx) {
  set.seed(20261111)
  dose <- rep(c(5, 10, 20, 40), times = c(6, 5, 7, 6))
  y <- round(1 + 0.08 * dose - 0.0012 * dose^2 + rnorm(length(dose), 0, 0.3), 4)
  d <- ctx$gen_dataset("secontrast_poly_t4", data.frame(dose = dose, y = y))
  f <- factor(d$dose); fit <- aov(y ~ f, data = d)
  stopifnot(identical(levels(f), c("5", "10", "20", "40")))
  cp <- contr.poly(4); ns <- as.vector(table(f))
  W <- cp[as.integer(f), ] / ns[as.integer(f)]   # observation-level contrasts: se.contrast's matrix form
  list(values = list(poly2 = vec(contr.poly(2)), poly3 = vec(contr.poly(3)), poly4 = vec(cp), poly6 = vec(contr.poly(6)),
                     se4 = vec(se.contrast(fit, W)), est4 = vec(colSums(cp * as.vector(tapply(d$y, f, mean)))),
                     mse = sum(residuals(fit)^2) / fit$df.residual, df_mse = fit$df.residual),
       reference = "stats::contr.poly (orthonormal, column-major), stats::se.contrast(aov(y ~ factor(dose)), contrast matrix)",
       call = "contr.poly(k); se.contrast(fit, contr.poly(4)[g, ] / n[g])", packages = character(0))
})

case("stat_anova2_rcbd_t4", "stats/anova", "Additive two-way ANOVA on a subject x condition layout (40 x 3, 5 rows dropped): Type I both orders and Type II", function(ctx) {
  set.seed(4411)
  S <- 40
  g <- expand.grid(subj = sprintf("s%02d", 1:S), cond = c("c1", "c2", "c3"), stringsAsFactors = FALSE)
  se <- rnorm(S, 0, 1.5); ce <- c(c1 = 0, c2 = 0.6, c3 = 1.1)
  g$y <- round(10 + se[as.integer(factor(g$subj))] + ce[g$cond] + rnorm(nrow(g), 0, 0.8), 4)
  d <- ctx$gen_dataset("anova2_rcbd_t4", g[-c(7, 51, 88, 102, 119), ])
  a1 <- anova(lm(y ~ subj + cond, data = d)); a2 <- anova(lm(y ~ cond + subj, data = d))
  dr <- drop1(lm(y ~ subj + cond, data = d), test = "F")
  list(values = list(ss_subj_cond = vec(a1[["Sum Sq"]]), df_subj_cond = vec(a1[["Df"]]),
                     f_subj_cond = vec(a1[["F value"]][1:2]), p_subj_cond = vec(a1[["Pr(>F)"]][1:2]),
                     ss_cond_subj = vec(a2[["Sum Sq"]]), f_cond_subj = vec(a2[["F value"]][1:2]), p_cond_subj = vec(a2[["Pr(>F)"]][1:2]),
                     ss2 = vec(dr[["Sum of Sq"]][2:3]), f2 = vec(dr[["F value"]][2:3]), p2 = vec(dr[["Pr(>F)"]][2:3])),
       reference = "stats::anova(lm(y ~ subj + cond)) and (y ~ cond + subj) for Type I; drop1(test = 'F') for Type II",
       call = "anova(lm(y ~ subj + cond)); drop1(lm(y ~ subj + cond), test = 'F')", packages = character(0))
})
