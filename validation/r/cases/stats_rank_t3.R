# validation/r/cases/stats_rank_t3.R -- rank-test references for T3-WP39
# (STAT-53 exact-equality midranks; STAT-44 nonparametric pins; X-16 Wilcoxon exact flag;
#  contributes STAT-62 and X-13 for wilcoxon.rs).
#
# Every case computes with the R function named in `reference`. Values that p2a does not yet
# reproduce because of a Tier 4 finding are kept under their own keys (suffix or comment names the
# finding) so that the Tier 4 fix can pin them without regenerating anything:
#   STAT-12 Ansari one-sided tails swapped      STAT-13 Ansari exact distribution truncated
#   STAT-15 binom two-sided rule                STAT-29 Mood scores under ties
#   STAT-42 Wilcoxon tie variance               STAT-52 isoreg tied x, iKnots and yc definitions
#   STAT-73 Wilcoxon exact p-values for 21 <= N < 50 (p2a falls back to the normal approximation)
#   STAT-79 Ansari normal approximation ignores ties

# ---------------------------------------------------------------------------------------------
# STAT-53: rank tests on measurements recorded in SI base units. Every value lies within 1e-10
# of every other, so a midrank helper that treats |a - b| < 1e-10 as a tie puts the whole sample
# into one tie group (H = 0, p = 1). R's rank() uses exact equality.
case("stat_rank_small_groups", "stats/ranks", "Kruskal-Wallis, Fligner-Killeen, Mood, Ansari-Bradley and Wilcoxon on values of order 1e-12 (exact-equality midranks)", function(ctx) {
  set.seed(20261022)
  g <- rep(c("a", "b", "c"), each = 7)
  y_unit <- round(rnorm(21, rep(c(5, 6, 7.5), each = 7), 1), 1)              # one decimal: genuine ties
  ynt_unit <- round(rnorm(21, 6, rep(c(0.6, 2.5, 1), each = 7)), 4)          # four decimals: tie-free
  d <- ctx$gen_dataset("rank_small_groups", data.frame(g = g, y = y_unit * 1e-12, y_unit = y_unit, ynt = ynt_unit * 1e-12))
  stopifnot(anyDuplicated(d$y) > 0, !anyDuplicated(d$ynt), diff(range(d$y)) < 1e-10, diff(range(d$ynt)) < 1e-10,
            identical(rank(d$y), rank(d$y_unit)))
  kw <- kruskal.test(y ~ g, data = d); kwu <- kruskal.test(y_unit ~ g, data = d)
  tt <- as.vector(table(d$y)); n <- nrow(d)
  fl <- fligner.test(y ~ g, data = d)
  x <- d$ynt[d$g == "a"]; yv <- d$ynt[d$g == "b"]; m <- length(x); N <- m + length(yv)
  mt <- mood.test(x, yv)
  an <- ansari.test(x, yv, exact = FALSE); ae <- ansari.test(x, yv, exact = TRUE)
  wx <- wilcox.test(x, yv)                                                    # exact: N < 50, no ties
  stopifnot(grepl("exact", wx$method))
  list(values = list(
         midranks_y = vec(rank(d$y)), tie_sizes_y = vec(tt), midranks_ynt = vec(rank(d$ynt)),
         kw_chisq = unname(kw$statistic), kw_df = unname(kw$parameter), kw_p = kw$p.value,
         kw_unit_chisq = unname(kwu$statistic),
         kw_tie_correction = 1 - sum(tt^3 - tt) / (n^3 - n),
         kw_rank_sums = named(tapply(rank(d$y), d$g, sum)),
         fligner_chisq = unname(fl$statistic), fligner_df = unname(fl$parameter), fligner_p = fl$p.value,
         mood_z = unname(mt$statistic), mood_p = mt$p.value,
         mood_T = sum((rank(c(x, yv))[seq_len(m)] - (N + 1) / 2)^2),
         ansari_ab = unname(an$statistic), ansari_p_normal = an$p.value,
         ansari_p_exact = ae$p.value,                                            # deviation STAT-13
         wilcox_W = unname(wx$statistic), wilcox_rank_sum = unname(wx$statistic) + m * (m + 1) / 2,
         wilcox_p_exact = wx$p.value),
       reference = "stats::rank, kruskal.test, fligner.test, mood.test, ansari.test(exact = FALSE | TRUE), wilcox.test",
       call = "kruskal.test(y ~ g); fligner.test(y ~ g); mood.test(x, y); ansari.test(x, y); wilcox.test(x, y) with x = ynt[g == 'a'], y = ynt[g == 'b']",
       packages = character(0))
})

case("stat_rank_small_blocks", "stats/ranks", "Friedman and Quade tests on an 8 x 4 block design with values of order 1e-12 and within-block ties", function(ctx) {
  set.seed(20261023)
  b <- 8; k <- 4
  block <- rep(sprintf("B%d", 1:b), each = k); treat <- rep(sprintf("t%d", 1:k), b)
  y_unit <- round(rep(rnorm(b, 10, 2), each = k) + rep(c(0, 0.4, 0.9, 1.2), b) + rnorm(b * k, 0, 0.5), 1)
  d <- ctx$gen_dataset("rank_small_blocks", data.frame(block = block, treat = treat, y = y_unit * 1e-12, y_unit = y_unit))
  ym <- matrix(d$y, nrow = b, byrow = TRUE, dimnames = list(sprintf("B%d", 1:b), sprintf("t%d", 1:k)))
  ymu <- matrix(d$y_unit, nrow = b, byrow = TRUE)
  stopifnot(any(apply(ym, 1, anyDuplicated) > 0), all(apply(ym, 1, function(u) diff(range(u))) < 1e-10),
            identical(unname(t(apply(ym, 1, rank))), unname(t(apply(ymu, 1, rank)))))
  fr <- friedman.test(ym); fru <- friedman.test(ymu)
  qd <- quade.test(ym)
  # Quade's intermediate quantities, written out exactly as stats:::quade.test.default computes them.
  r <- t(apply(ym, 1L, rank)); q <- rank(apply(ym, 1, function(u) max(u) - min(u))); s <- q * (r - (k + 1) / 2)
  list(values = list(
         friedman_chisq = unname(fr$statistic), friedman_df = unname(fr$parameter), friedman_p = fr$p.value,
         friedman_unit_chisq = unname(fru$statistic),
         friedman_rank_sums = named(colSums(r)),
         quade_f = unname(qd$statistic), quade_df1 = unname(qd$parameter[1]), quade_df2 = unname(qd$parameter[2]),
         quade_p = qd$p.value, quade_treatment_sums = named(colSums(s)),
         quade_a = sum(s^2), quade_b = sum(colSums(s)^2) / b),
       reference = "stats::friedman.test, stats::quade.test (A, B and treatment sums as in quade.test.default)",
       call = "friedman.test(y_matrix); quade.test(y_matrix), y_matrix = 8 blocks x 4 treatments", packages = character(0))
})

# ---------------------------------------------------------------------------------------------
# STAT-44: the R examples behind the loose tests, pinned.
case("stat_rank_kruskal_hw", "stats/kruskal", "Kruskal-Wallis test, Hollander & Wolfe (1973) p. 116 (the ?kruskal.test example)", function(ctx) {
  x <- c(2.9, 3.0, 2.5, 2.6, 3.2); y <- c(3.8, 2.7, 4.0, 2.4); z <- c(2.8, 3.4, 3.7, 2.2, 2.0)
  d <- ctx$gen_dataset("rank_kruskal_hw", data.frame(value = c(x, y, z),
                                                     group = rep(c("asbestosis", "normal", "obstructive")[c(2, 3, 1)], c(5, 4, 5))))
  kw <- kruskal.test(value ~ group, data = d)
  list(values = list(chisq = unname(kw$statistic), df = unname(kw$parameter), p = kw$p.value,
                     rank_sums = named(tapply(rank(d$value), d$group, sum))),
       reference = "stats::kruskal.test", call = "kruskal.test(value ~ group)", packages = character(0))
})

case("stat_rank_friedman_rounding", "stats/friedman", "Friedman test on RoundingTimes (Hollander & Wolfe 1973 p. 140; ?friedman.test): all 22 players and the first 12", function(ctx) {
  rt <- matrix(c(5.40, 5.50, 5.55, 5.85, 5.70, 5.75, 5.20, 5.60, 5.50, 5.55, 5.50, 5.40, 5.90, 5.85, 5.70,
                 5.45, 5.55, 5.60, 5.40, 5.40, 5.35, 5.45, 5.50, 5.35, 5.25, 5.15, 5.00, 5.85, 5.80, 5.70,
                 5.25, 5.20, 5.10, 5.65, 5.55, 5.45, 5.60, 5.35, 5.45, 5.05, 5.00, 4.95, 5.50, 5.50, 5.40,
                 5.45, 5.55, 5.50, 5.55, 5.55, 5.35, 5.45, 5.50, 5.55, 5.50, 5.45, 5.25, 5.65, 5.60, 5.40,
                 5.70, 5.65, 5.55, 6.30, 6.30, 6.25),
               nrow = 22, byrow = TRUE, dimnames = list(1:22, c("Round Out", "Narrow Angle", "Wide Angle")))
  d <- ctx$gen_dataset("rank_friedman_rounding", data.frame(
    block = rep(sprintf("P%02d", 1:22), each = 3), method = rep(colnames(rt), 22), time = c(t(rt)),
    first12 = rep(as.integer(1:22 <= 12), each = 3)))
  m22 <- matrix(d$time, nrow = 22, byrow = TRUE, dimnames = list(NULL, colnames(rt)))
  m12 <- m22[1:12, ]
  f22 <- friedman.test(m22); f12 <- friedman.test(m12)
  stopifnot(abs(f22$statistic - 11.142857142857142) < 1e-12)
  rs <- function(m) { r <- t(apply(m, 1L, rank)); setNames(colSums(r), colnames(m)) }
  list(values = list(chisq_22 = unname(f22$statistic), df_22 = unname(f22$parameter), p_22 = f22$p.value,
                     rank_sums_22 = named(rs(m22)),
                     chisq_12 = unname(f12$statistic), df_12 = unname(f12$parameter), p_12 = f12$p.value,
                     rank_sums_12 = named(rs(m12))),
       reference = "stats::friedman.test", call = "friedman.test(RoundingTimes); friedman.test(RoundingTimes[1:12, ])",
       packages = character(0))
})

case("stat_rank_quade_store", "stats/quade", "Quade test, Conover (1999) p. 375 (the ?quade.test hand-lotion example)", function(ctx) {
  y <- matrix(c( 5,  4,  7, 10, 12,  1,  3,  1,  0,  2, 16, 12, 22, 22, 35,  5,  4,  3,  5,  4,
                10,  9,  7, 13, 10, 19, 18, 28, 37, 58, 10,  7,  6,  8,  7),
              nrow = 7, byrow = TRUE, dimnames = list(Store = as.character(1:7), Brand = LETTERS[1:5]))
  d <- ctx$gen_dataset("rank_quade_store", data.frame(store = rep(sprintf("S%d", 1:7), each = 5),
                                                      brand = rep(LETTERS[1:5], 7), sales = c(t(y))))
  ym <- matrix(d$sales, nrow = 7, byrow = TRUE, dimnames = list(NULL, LETTERS[1:5]))
  qd <- quade.test(ym); k <- 5; b <- 7
  r <- t(apply(ym, 1L, rank)); q <- rank(apply(ym, 1, function(u) max(u) - min(u))); s <- q * (r - (k + 1) / 2)
  list(values = list(f = unname(qd$statistic), df1 = unname(qd$parameter[1]), df2 = unname(qd$parameter[2]),
                     p = qd$p.value, treatment_sums = named(colSums(s)), a = sum(s^2), b = sum(colSums(s)^2) / b,
                     block_ranges = vec(apply(ym, 1, function(u) max(u) - min(u)))),
       reference = "stats::quade.test (A, B, treatment sums and ranges as in quade.test.default)",
       call = "quade.test(y)", packages = character(0))
})

case("stat_rank_fligner", "stats/fligner", "Fligner-Killeen test: the two examples the old tests quoted and InsectSprays (?fligner.test)", function(ctx) {
  sets <- list(
    three = list(g = rep(c("A", "B", "C"), c(7, 7, 6)),
                 v = c(2.1, 2.5, 2.3, 2.8, 2.6, 2.4, 2.7, 3.2, 3.5, 3.1, 3.8, 3.4, 3.6, 3.3, 4.1, 5.2, 3.9, 4.5, 4.8, 5.0)),
    two = list(g = rep(c("A", "B"), each = 5), v = c(1:5, 10 * (1:5))),
    insect = list(g = as.character(datasets::InsectSprays$spray), v = datasets::InsectSprays$count))
  d <- ctx$gen_dataset("rank_fligner", do.call(rbind, lapply(names(sets), function(s)
    data.frame(set = s, group = sets[[s]]$g, value = as.numeric(sets[[s]]$v)))))
  ft <- lapply(names(sets), function(s) { e <- d[d$set == s, ]; fligner.test(e$value, factor(e$group)) })
  names(ft) <- names(sets)
  list(values = list(chisq = named(sapply(ft, function(f) unname(f$statistic))),
                     df = named(sapply(ft, function(f) unname(f$parameter))),
                     p = named(sapply(ft, function(f) f$p.value))),
       reference = "stats::fligner.test", call = "fligner.test(value, group) per set", packages = "datasets")
})

case("stat_rank_two_sample", "stats/scale_tests", "Ansari-Bradley and Mood tests of scale on six two-sample sets (tie-free and tied)", function(ctx) {
  set.seed(20261024)
  sim_x <- round(rnorm(20, 0, 1), 3); sim_y <- round(rnorm(20, 0, 3), 3)
  sets <- list(
    basic  = list(x = 1:5, y = 10 * (1:5)),
    scale  = list(x = c(5.5, 4.8, 5.2, 4.6, 5.8, 5.1, 4.9, 5.3, 5.0, 4.7), y = c(2.1, 7.8, 5.05, 1.5, 9.2, 6.3, 3.8, 8.1, 4.5, 0.8)),
    manual = list(x = 1:3, y = 4:6),
    sim    = list(x = sim_x, y = sim_y),
    ties   = list(x = c(1, 2, 2, 3, 4), y = c(2, 3, 3, 4, 5)),
    serum  = list(x = c(111, 107, 100, 99, 102, 106, 109, 108, 104, 99, 101, 96, 97, 102, 107, 113, 116, 113, 110, 98),
                  y = c(107, 108, 106, 98, 105, 103, 110, 105, 104, 100, 96, 108, 103, 104, 114, 114, 113, 108, 106, 99)))
  d <- ctx$gen_dataset("rank_two_sample", do.call(rbind, lapply(names(sets), function(s)
    data.frame(set = s, group = rep(c("x", "y"), c(length(sets[[s]]$x), length(sets[[s]]$y))),
               value = as.numeric(c(sets[[s]]$x, sets[[s]]$y))))))
  get <- function(s, gr) d$value[d$set == s & d$group == gr]
  tie_free <- c("basic", "scale", "manual", "sim"); tied <- c("ties", "serum")
  for (s in tie_free) stopifnot(!anyDuplicated(c(get(s, "x"), get(s, "y"))))
  for (s in tied) stopifnot(anyDuplicated(c(get(s, "x"), get(s, "y"))) > 0)
  per <- function(sets_, f) setNames(vapply(sets_, f, 0), sets_)
  ab <- function(s) suppressWarnings(ansari.test(get(s, "x"), get(s, "y"), exact = FALSE))
  mo <- function(s, alt = "two.sided") mood.test(get(s, "x"), get(s, "y"), alternative = alt)
  moodT <- function(s) { x <- get(s, "x"); z <- c(x, get(s, "y")); N <- length(z); sum((rank(z)[seq_along(x)] - (N + 1) / 2)^2) }
  list(values = list(
         ab = named(per(names(sets), function(s) unname(ab(s)$statistic))),
         ab_p_normal = named(per(tie_free, function(s) ab(s)$p.value)),
         ab_p_exact = named(per(tie_free, function(s) ansari.test(get(s, "x"), get(s, "y"), exact = TRUE)$p.value)),        # deviation STAT-13
         ab_p_less = named(per(tie_free, function(s) ansari.test(get(s, "x"), get(s, "y"), exact = FALSE, alternative = "less")$p.value)),        # deviation STAT-12
         ab_p_greater = named(per(tie_free, function(s) ansari.test(get(s, "x"), get(s, "y"), exact = FALSE, alternative = "greater")$p.value)),  # deviation STAT-12
         ab_p_ties = named(per(tied, function(s) ab(s)$p.value)),                                                                                 # deviation STAT-79
         mood_z = named(per(tie_free, function(s) unname(mo(s)$statistic))),
         mood_p = named(per(tie_free, function(s) mo(s)$p.value)),
         mood_p_less = named(per(tie_free, function(s) mo(s, "less")$p.value)),
         mood_p_greater = named(per(tie_free, function(s) mo(s, "greater")$p.value)),
         mood_T = named(per(tie_free, moodT)),
         mood_z_ties = named(per(tied, function(s) unname(mo(s)$statistic))),                                                                     # deviation STAT-29
         mood_p_ties = named(per(tied, function(s) mo(s)$p.value))),                                                                              # deviation STAT-29
       reference = "stats::ansari.test(exact = FALSE | TRUE, alternative), stats::mood.test(alternative)",
       call = "ansari.test(x, y, exact = FALSE); mood.test(x, y) per set", packages = character(0))
})

case("stat_rank_wilcoxon", "stats/wilcoxon", "Wilcoxon rank-sum and signed-rank tests: N = 30 and n = 25 (normal approximation in p2a), the quoted examples, a paired set with a zero and ties", function(ctx) {
  set.seed(20261025)
  n30x <- round(rnorm(15, 0, 1), 3); n30y <- round(rnorm(15, 0.6, 1), 3)
  sr25 <- round(rnorm(25, 0.45, 1), 4)
  sets <- list(
    n30 = list(x = n30x, y = n30y), sr25 = list(x = sr25),
    x13a = list(x = c(1.2, 2.3, 3.1, 4.5, 5.2), y = c(2.1, 3.4, 4.2, 5.8, 6.1, 7.2)),
    x13b = list(x = c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30)),
    exact3 = list(x = c(1, 2, 3), y = c(4, 5, 6)),
    paired = list(x = c(125, 115, 130, 140, 140, 115, 140, 125, 140, 135), y = c(110, 122, 125, 120, 140, 124, 123, 137, 135, 145)))
  d <- ctx$gen_dataset("rank_wilcoxon", do.call(rbind, lapply(names(sets), function(s) {
    st <- sets[[s]]; data.frame(set = s, group = rep(names(st), lengths(st)), value = as.numeric(unlist(st)))
  })))
  get <- function(s, gr) d$value[d$set == s & d$group == gr]
  x <- get("n30", "x"); y <- get("n30", "y")
  stopifnot(!anyDuplicated(c(x, y)), !anyDuplicated(abs(get("sr25", "x"))), all(get("sr25", "x") != 0))
  rs_n <- wilcox.test(x, y, exact = FALSE, correct = TRUE); rs_e <- wilcox.test(x, y)
  stopifnot(grepl("exact", rs_e$method))
  sr_n <- wilcox.test(get("sr25", "x"), exact = FALSE, correct = TRUE); sr_e <- wilcox.test(get("sr25", "x"))
  a <- wilcox.test(get("x13a", "x"), get("x13a", "y"), exact = FALSE, correct = TRUE)
  bb <- wilcox.test(get("x13b", "x"), mu = 1.5, exact = FALSE, correct = TRUE)
  e3 <- wilcox.test(get("exact3", "x"), get("exact3", "y"), exact = TRUE)
  pr <- suppressWarnings(wilcox.test(get("paired", "x"), get("paired", "y"), paired = TRUE, exact = FALSE, correct = TRUE))
  rank_sum <- function(x, y) sum(rank(c(x, y))[seq_along(x)])
  list(values = list(
         n30_p_normal = rs_n$p.value, n30_W = unname(rs_n$statistic), n30_rank_sum = rank_sum(x, y),
         n30_p_exact = rs_e$p.value,                                                   # deviation STAT-73
         sr25_p_normal = sr_n$p.value, sr25_V = unname(sr_n$statistic),
         sr25_p_exact = sr_e$p.value,                                                  # deviation STAT-73
         x13a_W = unname(a$statistic), x13a_rank_sum = rank_sum(get("x13a", "x"), get("x13a", "y")), x13a_p = a$p.value,
         x13b_V = unname(bb$statistic), x13b_p = bb$p.value,
         exact3_W = unname(e3$statistic), exact3_rank_sum = rank_sum(get("exact3", "x"), get("exact3", "y")), exact3_p = e3$p.value,
         paired_V = unname(pr$statistic), paired_p = pr$p.value),                      # paired_p: deviation STAT-42
       reference = "stats::wilcox.test(exact = TRUE | FALSE, correct = TRUE, mu, paired)",
       call = "wilcox.test(x, y, exact = FALSE, correct = TRUE); wilcox.test(x, y); wilcox.test(x, mu = 1.5, ...); wilcox.test(x, y, paired = TRUE, exact = FALSE)",
       packages = character(0))
})

case("stat_rank_binom", "stats/binom", "Exact binomial test (two-sided relative-likelihood rule, one-sided tails, Clopper-Pearson intervals) on 23 (x, n, p) cells", function(ctx) {
  g <- data.frame(
    x = c(15, 0, 10, 3, 7, 1, 2, 5, 6, 3, 1, 10, 95, 7, 12, 2, 60, 4, 9, 0, 30, 18, 40),
    n = c(100, 10, 20, 10, 10, 10, 10, 100, 50, 10, 10, 10, 100, 20, 20, 20, 100, 25, 25, 20, 60, 60, 60),
    p = c(0.1, 0.3, 0.5, 0.37, 0.75, 0.05, 0.05, 0.01, 0.05, 0.1, 0.19, 0.05, 0.5, 0.3, 0.3, 0.3, 0.5, 0.3, 0.3, 0.05, 0.4, 0.4, 0.4))
  # Cells of the former STAT-15 difference (fixed in Tier 4): x = floor(np) < np, or no lower-tail
  # cell qualifies (the former spurious P(X = 0)), or x = n.
  g$stat15 <- as.integer(seq_len(nrow(g)) %in% c(4:12))
  g$label <- sprintf("r%02d", seq_len(nrow(g)))
  d <- ctx$gen_dataset("rank_binom_grid", g[, c("label", "x", "n", "p", "stat15")])
  bt <- function(i, alt) binom.test(d$x[i], d$n[i], d$p[i], alternative = alt)
  col <- function(rows, f) setNames(vapply(rows, f, 0), d$label[rows])
  all_rows <- seq_len(nrow(d)); ok <- which(d$stat15 == 0); dev <- which(d$stat15 == 1)
  list(values = list(
         p_two = named(col(ok, function(i) bt(i, "two.sided")$p.value)),
         p_two_stat15 = named(col(dev, function(i) bt(i, "two.sided")$p.value)),        # cells of the former STAT-15 difference
         p_less = named(col(all_rows, function(i) bt(i, "less")$p.value)),
         p_greater = named(col(all_rows, function(i) bt(i, "greater")$p.value)),
         ci_two_lower = named(col(all_rows, function(i) bt(i, "two.sided")$conf.int[1])),
         ci_two_upper = named(col(all_rows, function(i) bt(i, "two.sided")$conf.int[2])),
         ci_greater_lower = named(col(all_rows, function(i) bt(i, "greater")$conf.int[1])),
         ci_less_upper = named(col(all_rows, function(i) bt(i, "less")$conf.int[2])),
         estimate = named(col(all_rows, function(i) unname(bt(i, "two.sided")$estimate)))),
       reference = "stats::binom.test(alternative = two.sided | less | greater, conf.level = 0.95)",
       call = "binom.test(x, n, p, alternative)", packages = character(0))
})

case("stat_rank_isoreg", "stats/isoreg", "Isotonic regression: the ?isoreg examples, an unsorted tie-free x, and tied x (STAT-52)", function(ctx) {
  set.seed(20261026)
  xu <- sample(round(seq(0.5, 15, length.out = 15) + runif(15, -0.2, 0.2), 3))
  yu <- round(log(xu) + rnorm(15, 0, 0.4), 3)
  sets <- list(doc = c(1, 0, 4, 3, 3, 5, 4, 2, 0), doc3 = c(1, 0, 4, 3, 3, 5, 4, 2, 3),
               ir4 = c(5, 9, 1:2, 5:8, 3, 8), mono = c(1, 2, 3, 4, 5), decr = c(5, 4, 3, 2, 1))
  d <- ctx$gen_dataset("rank_isoreg", rbind(
    do.call(rbind, lapply(names(sets), function(s) data.frame(set = s, x = seq_along(sets[[s]]), y = sets[[s]]))),
    data.frame(set = "unsorted", x = xu, y = yu),
    data.frame(set = "tiedx", x = c(3, 1, 2, 2, 1), y = c(5, 1, 2, 4, 3))))
  get <- function(s) d[d$set == s, ]
  stopifnot(!anyDuplicated(get("unsorted")$x), is.unsorted(get("unsorted")$x))
  i_doc <- isoreg(get("doc")$y); i_doc3 <- isoreg(get("doc3")$y); i_ir4 <- isoreg(get("ir4")$x, get("ir4")$y)
  i_un <- isoreg(get("unsorted")$x, get("unsorted")$y); i_t <- isoreg(get("tiedx")$x, get("tiedx")$y)
  i_mono <- isoreg(get("mono")$y); i_decr <- isoreg(get("decr")$y)
  list(values = list(
         yf_doc = vec(i_doc$yf), yf_doc3 = vec(i_doc3$yf), yf_ir4 = vec(i_ir4$yf),
         yf_mono = vec(i_mono$yf), yf_decr = vec(i_decr$yf), n_knots_decr = length(i_decr$iKnots),
         yf_unsorted = vec(i_un$yf), x_sorted_unsorted = vec(sort(get("unsorted")$x)), ord_unsorted = vec(i_un$ord),
         iknots_doc = vec(i_doc$iKnots), iknots_ir4 = vec(i_ir4$iKnots), yc_doc = vec(i_doc$yc),   # deviation STAT-52 (definitions)
         yf_tiedx = vec(i_t$yf), iknots_tiedx = vec(i_t$iKnots)),                                 # deviation STAT-52 (tied x)
       reference = "stats::isoreg(x, y) (C_isoreg; order(x, -y) when x is not strictly increasing)",
       call = "isoreg(y); isoreg(x, y)", packages = character(0))
})
