# validation/r/cases/stats_rank_t4.R -- rank-test references for T4-WP02
# (STAT-41/42/72/73/74 Wilcoxon; STAT-12/13/14/79 Ansari-Bradley; STAT-29 Mood; STAT-27
#  Kolmogorov-Smirnov; STAT-64 Quade). The T3 cases in stats_rank_t3.R already hold the values
# the T3 deviations name (ab_p_exact, ab_p_less, mood_z_ties, n30_p_exact, paired_p, ...); the
# cases here add what those do not cover: non-zero mu, ties in the variance, per-sample exact
# rule, zeros, confidence intervals and estimates, Bauer intervals, KS exact and one-sided
# distributions, Quade's A == B branch.
#
# Confidence bounds and estimates that wilcox.test and ansari.test find with uniroot() on a step
# function are the jump points of that function: a pairwise difference x_i - y_j (rank sum), a
# Walsh average (x_i + x_j) / 2 (signed rank) or a ratio x_i / y_j (Ansari). Each such value is
# recorded twice: `*_r` is what R prints (uniroot at R's tolerance), `*` (no suffix) is the jump
# point, obtained by snapping a uniroot(tol.root = 1e-12) root (wilcox.test) or R's own root
# (ansari.test, tol 1e-4 fixed) to the nearest candidate, with a stopifnot() that the snap is
# unambiguous.
# ---------------------------------------------------------------------------------------------

.wx <- function(...) suppressWarnings(wilcox.test(...))
.snap <- function(root, cand, tol) {
  i <- which.min(abs(cand - root)); d <- abs(cand - root)
  near <- cand[d < 2 * tol]
  stopifnot(d[i] < tol, diff(range(near)) < 1e-12)                            # unambiguous
  cand[i]
}
.walsh <- function(x) { w <- outer(x, x, "+"); sort(w[!lower.tri(w)]) / 2 }

case("stat_rank_wilcoxon_t4", "stats/wilcoxon", "Wilcoxon rank-sum and signed-rank tests: non-zero mu, tie-corrected variance, per-sample exact rule, zeros, exact and asymptotic confidence intervals and estimates, pairwise tests with ties", function(ctx) {
  set.seed(20261111)
  tf_x <- round(rnorm(8, 0.9, 1), 3); tf_y <- round(rnorm(10, 0, 1), 3)
  n4530_x <- round(rnorm(45, 0.4, 1), 4); n4530_y <- round(rnorm(30, 0, 1), 4)
  n5510_x <- round(rnorm(55, 0.5, 1), 4); n5510_y <- round(rnorm(10, 0, 1), 4)
  n60_x <- round(rnorm(60, 0.35, 1), 4); n60_y <- round(rnorm(60, 0, 1), 4)
  sr40 <- round(rnorm(40, 0.35, 1), 4); sr60 <- round(rnorm(60, 0.2, 1), 4)
  pw_v <- round(rnorm(24, rep(c(3, 3.8, 4.6), each = 8), 1)); pw_g <- rep(c("A", "B", "C"), each = 8)
  sets <- list(
    depr  = list(x = c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30),
                 y = c(0.878, 0.647, 0.598, 2.05, 1.06, 1.29, 1.06, 3.14, 1.29)),
    tf    = list(x = tf_x, y = tf_y),
    lik   = list(x = c(1, 2, 2, 3, 3, 3, 4, 4, 5, 2, 3, 1), y = c(3, 3, 4, 4, 4, 5, 5, 2, 3, 4, 5, 5)),
    heavy = list(x = rep(1:3, c(10, 5, 3)), y = rep(1:3, c(4, 6, 8))),
    srt   = list(x = c(-2, 1, 1, 3, 3, 3, 4, 5, -1, 2, 6, 7, -3, 8)),
    n4530 = list(x = n4530_x, y = n4530_y),
    n5510 = list(x = n5510_x, y = n5510_y),
    n60   = list(x = n60_x, y = n60_y),
    sr40  = list(x = sr40),
    sr60  = list(x = sr60),
    srz   = list(x = c(1.2, 0.7, 2.1, 1, -0.4, 1.9, 2.6, 0.3, 1.5, 3.2, -0.9, 2.8)),
    sr12  = list(x = c(1.83, 0.50, 1.62, 2.48, 1.68, 1.88, 1.55, 3.06, 1.30, 0.70, 2.10, 1.95)))
  long <- do.call(rbind, lapply(names(sets), function(s) {
    st <- sets[[s]]; data.frame(set = s, group = rep(names(st), lengths(st)), value = as.numeric(unlist(st)))
  }))
  long <- rbind(long, data.frame(set = "pw", group = pw_g, value = as.numeric(pw_v)))
  d <- ctx$gen_dataset("rank_wilcoxon_t4", long)
  get <- function(s, g = "x") d$value[d$set == s & d$group == g]
  x <- function(s) get(s, "x"); y <- function(s) get(s, "y")
  # design checks
  stopifnot(anyDuplicated(y("depr")) > 0,                                     # ties: normal path
            !anyDuplicated(c(x("tf") - 0.3, y("tf"))), !anyDuplicated(c(x("tf"), y("tf"))),
            !anyDuplicated(c(x("n4530"), y("n4530"))), !anyDuplicated(c(x("n5510"), y("n5510"))),
            !anyDuplicated(c(x("n60"), y("n60"))),
            !anyDuplicated(abs(x("sr40"))), all(x("sr40") != 0), !anyDuplicated(abs(x("sr60"))), all(x("sr60") != 0),
            sum(x("srz") == 1) == 1, !anyDuplicated(abs(x("srz") - 1)),         # one zero at mu = 1, no ties
            anyDuplicated(abs(x("srt"))) > 0, anyDuplicated(abs(x("sr12") - 1)) > 0,
            anyDuplicated(d$value[d$set == "pw"]) > 0)
  diffs <- function(s) sort(outer(x(s), y(s), "-"))
  # ---- STAT-41: mu != 0 (depr: ties, normal path; tf: tie-free, exact path)
  a05 <- .wx(x("depr"), y("depr"), mu = 0.5, conf.int = TRUE)
  a05f <- .wx(x("depr"), y("depr"), mu = 0.5, conf.int = TRUE, tol.root = 1e-12)
  a0 <- .wx(x("depr"), y("depr"), conf.int = TRUE); a0f <- .wx(x("depr"), y("depr"), conf.int = TRUE, tol.root = 1e-12)
  a0nc <- .wx(x("depr"), y("depr"), conf.int = TRUE, correct = FALSE, tol.root = 1e-12)
  dd <- diffs("depr"); stopifnot(length(dd) %% 2 == 1)
  stopifnot(grepl("continuity", a05$method))
  t3 <- .wx(x("tf"), y("tf"), mu = 0.3, conf.int = TRUE)
  stopifnot(grepl("exact", t3$method))
  t3l <- .wx(x("tf"), y("tf"), mu = 0.3, alternative = "less", conf.int = TRUE)
  t3g <- .wx(x("tf"), y("tf"), mu = 0.3, alternative = "greater", conf.int = TRUE)
  t390 <- .wx(x("tf"), y("tf"), mu = 0.3, conf.int = TRUE, conf.level = 0.9)
  # ---- STAT-42: tie-corrected variance
  lk <- .wx(x("lik"), y("lik")); lknc <- .wx(x("lik"), y("lik"), correct = FALSE)
  lkf <- .wx(x("lik"), y("lik"), conf.int = TRUE, tol.root = 1e-12); dl <- diffs("lik"); ml <- length(dl)
  lz <- local({ r <- rank(c(x("lik"), y("lik"))); nx <- 12; ny <- 12; W <- sum(r[1:nx]) - nx * (nx + 1) / 2
    nt <- table(r); z <- W - nx * ny / 2
    s <- sqrt((nx * ny / 12) * ((nx + ny + 1) - sum(nt^3 - nt) / ((nx + ny) * (nx + ny - 1))))
    (z - sign(z) * 0.5) / s })
  lp <- .wx(x("lik"), y("lik"), paired = TRUE)
  hv <- .wx(x("heavy"), y("heavy"))
  st <- .wx(x("srt")); stg <- .wx(x("srt"), alternative = "greater")
  stf <- .wx(x("srt"), conf.int = TRUE, tol.root = 1e-12); ws <- .walsh(x("srt")); mws <- length(ws)
  # ---- STAT-73: per-sample exact rule, zeros, exact beyond R's default
  e4530 <- .wx(x("n4530"), y("n4530")); stopifnot(grepl("exact", e4530$method))
  e4530g <- .wx(x("n4530"), y("n4530"), alternative = "greater")
  e5510 <- .wx(x("n5510"), y("n5510")); stopifnot(grepl("continuity", e5510$method))
  e60 <- .wx(x("n60"), y("n60"), exact = TRUE); stopifnot(grepl("exact", e60$method))
  s40 <- .wx(x("sr40"), conf.int = TRUE); stopifnot(grepl("exact", s40$method))
  s40l <- .wx(x("sr40"), alternative = "less")
  s60 <- .wx(x("sr60"), exact = TRUE); stopifnot(grepl("exact", s60$method))
  sz <- .wx(x("srz"), mu = 1); stopifnot(grepl("continuity", sz$method))
  sze <- .wx(x("srz"), mu = 1, exact = TRUE, correct = FALSE)
  # ---- STAT-74: confidence intervals
  s12 <- .wx(x("sr12"), mu = 1, conf.int = TRUE); s12f <- .wx(x("sr12"), mu = 1, conf.int = TRUE, tol.root = 1e-12)
  w12 <- .walsh(x("sr12")[x("sr12") != 1]); m12 <- length(w12)
  # ---- pairwise with ties
  pv <- d$value[d$set == "pw"]; pg <- d$group[d$set == "pw"]
  pwn <- suppressWarnings(pairwise.wilcox.test(pv, pg, p.adjust.method = "none", exact = FALSE))$p.value
  pwh <- suppressWarnings(pairwise.wilcox.test(pv, pg, p.adjust.method = "holm"))$p.value
  low <- function(m) vec(c(m[1, 1], m[2, 1], m[2, 2]))                         # B-A, C-A, C-B
  mid2 <- function(v) { m <- length(v); if (m %% 2) rep(v[(m + 1) / 2], 2) else v[c(m / 2, m / 2 + 1)] }
  list(values = list(
    # STAT-41
    depr_mu05_W = unname(a05$statistic), depr_mu05_p = a05$p.value,
    depr_ci = vec(c(.snap(a05f$conf.int[1], dd, 1e-9), .snap(a05f$conf.int[2], dd, 1e-9))),
    depr_ci_r = vec(a05$conf.int), depr_est = median(dd), depr_est_r = unname(a05$estimate),
    depr_mu0_W = unname(a0$statistic), depr_mu0_p = a0$p.value,
    depr_mu0_ci = vec(c(.snap(a0f$conf.int[1], dd, 1e-9), .snap(a0f$conf.int[2], dd, 1e-9))),
    depr_mu0_ci_nc = vec(c(.snap(a0nc$conf.int[1], dd, 1e-9), .snap(a0nc$conf.int[2], dd, 1e-9))),
    tf_mu03_W = unname(t3$statistic), tf_mu03_p = t3$p.value,
    tf_mu03_p_less = t3l$p.value, tf_mu03_p_greater = t3g$p.value,
    tf_ci = vec(t3$conf.int), tf_ci_greater_lower = t3g$conf.int[1], tf_ci_less_upper = t3l$conf.int[2],
    tf_ci90 = vec(t390$conf.int), tf_est = unname(t3$estimate),
    # STAT-42
    lik_W = unname(lk$statistic), lik_p = lk$p.value, lik_p_nc = lknc$p.value, lik_z = lz,
    lik_ci = vec(c(.snap(lkf$conf.int[1], dl, 1e-9), .snap(lkf$conf.int[2], dl, 1e-9))),
    lik_est_mid = vec(mid2(dl)), lik_est_root = unname(lkf$estimate),
    lik_paired_V = unname(lp$statistic), lik_paired_p = lp$p.value, lik_paired_n_nonzero = sum(x("lik") != y("lik")),
    heavy_W = unname(hv$statistic), heavy_p = hv$p.value,
    srt_V = unname(st$statistic), srt_p = st$p.value, srt_p_greater = stg$p.value,
    srt_ci = vec(c(.snap(stf$conf.int[1], ws, 1e-9), .snap(stf$conf.int[2], ws, 1e-9))),
    srt_est_mid = vec(mid2(ws)), srt_est_root = unname(stf$estimate),
    # STAT-73
    n4530_W = unname(e4530$statistic), n4530_p_exact = e4530$p.value, n4530_p_greater_exact = e4530g$p.value,
    n5510_W = unname(e5510$statistic), n5510_p_normal = e5510$p.value,
    n60_W = unname(e60$statistic), n60_p_exact = e60$p.value,
    sr40_V = unname(s40$statistic), sr40_p_exact = s40$p.value, sr40_p_less_exact = s40l$p.value,
    sr40_ci = vec(s40$conf.int), sr40_est = unname(s40$estimate),
    sr60_V = unname(s60$statistic), sr60_p_exact = s60$p.value,
    srz_V = unname(sz$statistic), srz_p = sz$p.value, srz_p_exact_requested_nc = sze$p.value,
    # STAT-74
    sr12_V = unname(s12$statistic), sr12_p = s12$p.value,
    sr12_ci = vec(c(.snap(s12f$conf.int[1], w12 , 1e-9), .snap(s12f$conf.int[2], w12, 1e-9))),
    sr12_ci_r = vec(s12$conf.int), sr12_est_mid = vec(mid2(w12)), sr12_est_root = unname(s12f$estimate),
    # pairwise
    pw_p_none = low(pwn), pw_p_holm = low(pwh)),
    reference = "stats::wilcox.test(mu, paired, exact, correct, conf.int, tol.root); stats::pairwise.wilcox.test",
    call = "wilcox.test(x, y, mu = 0.5, conf.int = TRUE); wilcox.test(x, y, mu = 0.3, conf.int = TRUE) (exact); wilcox.test(x, mu = 1, conf.int = TRUE); pairwise.wilcox.test(v, g)",
    packages = character(0))
})

case("stat_rank_scale_t4", "stats/scale_tests", "Ansari-Bradley exact p (both tails, m = n = 45), Bauer exact and asymptotic intervals and estimates, tie-corrected normal approximation; Mood test with Mielke scores under heavy ties", function(ctx) {
  set.seed(20261112)
  cen_x <- round(rnorm(10, 0, 1), 3); cen_y <- round(rnorm(12, 0, 2.5), 3)
  big_x <- round(rnorm(30, 0, 1), 4); big_y <- round(rnorm(35, 0, 2), 4)
  sets <- list(
    scale = list(x = c(5.5, 4.8, 5.2, 4.6, 5.8, 5.1, 4.9, 5.3, 5.0, 4.7), y = c(2.1, 7.8, 5.05, 1.5, 9.2, 6.3, 3.8, 8.1, 4.5, 0.8)),
    odd   = list(x = c(4.6, 4.8, 5.05, 5.2, 5.5, 4.9, 5.1), y = c(1, 3, 5.3, 7, 9, 2.2, 8.1)),
    ab45  = list(x = seq(1, 89, by = 2), y = seq(2, 90, by = 2)),
    cen   = list(x = cen_x, y = cen_y),
    big   = list(x = big_x, y = big_y),
    ties  = list(x = c(1, 2, 2, 3, 4), y = c(2, 3, 3, 4, 5)),
    heavy = list(x = c(1, 1, 1, 5, 5, 5), y = c(3, 3, 2, 4, 3, 3)))
  d <- ctx$gen_dataset("rank_scale_t4", do.call(rbind, lapply(names(sets), function(s)
    data.frame(set = s, group = rep(c("x", "y"), c(length(sets[[s]]$x), length(sets[[s]]$y))),
               value = as.numeric(c(sets[[s]]$x, sets[[s]]$y))))))
  x <- function(s) d$value[d$set == s & d$group == "x"]; y <- function(s) d$value[d$set == s & d$group == "y"]
  for (s in c("scale", "odd", "ab45", "cen", "big")) stopifnot(!anyDuplicated(c(x(s), y(s))))
  stopifnot((length(x("odd")) + length(y("odd"))) %% 2 == 0, length(x("ab45")) == 45, length(y("ab45")) == 45)
  ab <- function(s, ...) suppressWarnings(ansari.test(x(s), y(s), ...))
  # exact p, both tails (STAT-12, STAT-13)
  pe <- function(s, alt) ab(s, exact = TRUE, alternative = alt)$p.value
  # Bauer exact interval and estimate (STAT-14)
  ce <- ab("cen", exact = TRUE, conf.int = TRUE); cs <- ab("scale", exact = TRUE, conf.int = TRUE)
  ceg <- ab("cen", exact = TRUE, conf.int = TRUE, alternative = "greater")
  # asymptotic interval and estimate: uniroot(ab2, tol = 1e-4) on a step function whose jumps are the
  # positive ratios x_i / y_j. The jump point is found here by evaluating ab2 (as in
  # stats:::ansari.test.default) between consecutive ratios; each zq must give exactly one sign change.
  ca <- ab("big", exact = FALSE, conf.int = TRUE)
  rat <- local({ r <- outer(x("big"), y("big"), "/"); sort(unique(r[r > 0])) })
  jump <- local({ xx <- x("big"); yy <- y("big"); m <- length(xx); n <- length(yy); N <- m + n; EVEN <- N %% 2L == 0L
    normalize <- function(s, r, TIES) {
      z <- if (EVEN) s - m * (N + 2) / 4 else s - m * (N + 1)^2 / (4 * N)
      SIGMA <- if (!TIES) { if (EVEN) sqrt((m * n * (N + 2) * (N - 2)) / (48 * (N - 1))) else sqrt((m * n * (N + 1) * (3 + N^2)) / (48 * N^2)) }
               else { r <- rle(sort(pmin(r, N - r + 1)))
                 if (EVEN) sqrt(m * n * (16 * sum(r$lengths * r$values^2) - N * (N + 2)^2) / (16 * N * (N - 1)))
                 else sqrt(m * n * (16 * N * sum(r$lengths * r$values^2) - (N + 1)^4) / (16 * N^2 * (N - 1))) }
      z / SIGMA }
    ab2 <- function(sig, zq) { r <- rank(c(xx / sig, yy)); s <- sum(pmin(r, N - r + 1)[seq_along(xx)])
      normalize(s, r, length(r) != length(unique(r))) - zq }
    mids <- c(rat[1] * 0.99, (rat[-1] + rat[-length(rat)]) / 2, rat[length(rat)] * 1.01)
    function(zq) { f <- vapply(mids, ab2, 0, zq = zq); sc <- which(diff(sign(f)) != 0); stopifnot(length(sc) == 1L); rat[sc] } })
  big_ci <- sort(c(jump(qnorm(0.025)), jump(qnorm(0.975)))); big_est <- jump(0)
  stopifnot(all(abs(big_ci - ca$conf.int) < 1e-4), abs(big_est - ca$estimate) < 1e-4)
  # tie-corrected normal approximation, one-sided (STAT-79 with STAT-12)
  tl <- ab("ties", exact = FALSE, alternative = "less"); tg <- ab("ties", exact = FALSE, alternative = "greater")
  # Mood with Mielke scores (STAT-29): T exactly as stats:::mood.test.default computes it
  mo <- function(alt) mood.test(x("heavy"), y("heavy"), alternative = alt)
  mT <- local({ xx <- x("heavy"); z <- c(xx, y("heavy")); N <- length(z); u <- sort(unique(z))
    a <- tabulate(match(xx, u), length(u)); t <- tabulate(match(z, u), length(u))
    p <- cumsum((seq_along(z) - (N + 1L) / 2)^2); sum(a * diff(c(0, p[cumsum(t)])) / t) })
  list(values = list(
    scale_p_exact_two = pe("scale", "two.sided"), scale_p_exact_less = pe("scale", "less"), scale_p_exact_greater = pe("scale", "greater"),
    odd_p_exact_two = pe("odd", "two.sided"), odd_p_exact_less = pe("odd", "less"),
    ab45_AB = unname(ab("ab45", exact = TRUE)$statistic), ab45_p_exact = pe("ab45", "two.sided"),
    cen_AB = unname(ce$statistic), cen_p_exact = ce$p.value,
    cen_ci_exact = vec(ce$conf.int), cen_est_exact = unname(ce$estimate),
    cen_ci_greater_lower = ceg$conf.int[1],
    scale_ci_exact = vec(cs$conf.int), scale_est_exact = unname(cs$estimate),
    big_AB = unname(ca$statistic), big_p_normal = ca$p.value,
    big_ci = vec(big_ci),
    big_ci_r = vec(ca$conf.int), big_est = big_est, big_est_r = unname(ca$estimate),
    ties_p_less = tl$p.value, ties_p_greater = tg$p.value,
    mood_heavy_z = unname(mo("two.sided")$statistic), mood_heavy_p = mo("two.sided")$p.value,
    mood_heavy_p_less = mo("less")$p.value, mood_heavy_p_greater = mo("greater")$p.value, mood_heavy_T = mT),
    reference = "stats::ansari.test(exact, alternative, conf.int); stats::mood.test(alternative)",
    call = "ansari.test(x, y, exact = TRUE, conf.int = TRUE); ansari.test(x, y, exact = FALSE, conf.int = TRUE); mood.test(x, y)",
    packages = character(0))
})

case("stat_rank_ks_t4", "stats/ks", "Kolmogorov-Smirnov: one-sample exact one-sided and two-sided (n = 30, 99), with ties, asymptotic tails; two-sample exact with ties (both tails), n x m = 9801 exact and 10000 asymptotic", function(ctx) {
  set.seed(20261113)
  o30 <- round(rnorm(30, 0.25, 1), 4); o99 <- round(rnorm(99, 0.45, 1), 4); o99b <- round(rnorm(99, 0, 1), 4)
  o20t <- round(rnorm(20, 0.2, 1), 1); a150 <- round(rnorm(150, 0, 1), 4); a400 <- round(rnorm(400, 0.4, 1), 4)
  e99x <- round(rnorm(99, 0, 1), 4); e99y <- round(rnorm(99, 0.3, 1), 4)
  a100x <- round(rnorm(100, 0, 1), 4); a100y <- round(rnorm(100, 0.3, 1), 4)
  sets <- list(o30 = list(x = o30), o99 = list(x = o99), o99b = list(x = o99b), o20t = list(x = o20t),
               a150 = list(x = a150), a400 = list(x = a400),
               t75 = list(x = c(1.5, 2, 2, 3.5, 5, 5.5, 8), y = c(2, 3, 3, 4)),
               t57 = list(x = c(1, 2, 2, 3, 5), y = c(2, 3, 3, 4, 6, 6, 7)),
               e99 = list(x = e99x, y = e99y), a100 = list(x = a100x, y = a100y))
  d <- ctx$gen_dataset("rank_ks_t4", do.call(rbind, lapply(names(sets), function(s) {
    st <- sets[[s]]; data.frame(set = s, group = rep(names(st), lengths(st)), value = as.numeric(unlist(st))) })))
  x <- function(s) d$value[d$set == s & d$group == "x"]; y <- function(s) d$value[d$set == s & d$group == "y"]
  stopifnot(!anyDuplicated(x("o30")), !anyDuplicated(x("o99")), !anyDuplicated(x("o99b")), anyDuplicated(x("o20t")) > 0,
            !anyDuplicated(c(x("e99"), y("e99"))), !anyDuplicated(c(x("a100"), y("a100"))))
  k <- function(...) suppressWarnings(ks.test(...))
  o1 <- function(s, alt) k(x(s), "pnorm", alternative = alt)
  stopifnot(o1("o30", "two.sided")$exact, !o1("o20t", "two.sided")$exact)
  a1 <- function(s, alt = "two.sided") k(x(s), "pnorm", alternative = alt, exact = FALSE)
  z150 <- sqrt(150) * unname(a1("a150")$statistic); z400 <- sqrt(400) * unname(a1("a400")$statistic)
  stopifnot(z150 < 1, z400 > 3)
  t2 <- function(s, alt) k(x(s), y(s), alternative = alt)
  # exact conditional permutation p of the statistic R reports, by enumeration (small sets only)
  brute <- function(s, alt) {
    xx <- x(s); yy <- y(s); w <- c(xx, yy); m <- length(xx); n <- length(yy); N <- m + n
    ends <- c(which(diff(sort(w)) != 0), N)
    stat <- function(isx) { z <- (cumsum(isx) / m - cumsum(!isx) / n)[ends]
      switch(alt, two.sided = max(abs(z)), greater = max(z), less = -min(z)) }
    obs <- stat(order(w) <= m)
    sims <- apply(combn(N, m), 2, function(cc) { v <- rep(FALSE, N); v[cc] <- TRUE; stat(v) })
    mean(sims >= obs - 1e-12)
  }
  for (s in c("t75", "t57")) for (alt in c("two.sided", "greater"))
    stopifnot(abs(t2(s, alt)$p.value - brute(s, alt)) < 1e-12)
  list(values = list(
    o30_D = unname(o1("o30", "two.sided")$statistic), o30_p_exact = o1("o30", "two.sided")$p.value,
    o30_Dplus = unname(o1("o30", "greater")$statistic), o30_p_greater_exact = o1("o30", "greater")$p.value,
    o30_Dminus = unname(o1("o30", "less")$statistic), o30_p_less_exact = o1("o30", "less")$p.value,
    o99_D = unname(o1("o99", "two.sided")$statistic), o99_p_exact = o1("o99", "two.sided")$p.value,
    o99b_p_exact = o1("o99b", "two.sided")$p.value,
    o20t_D = unname(o1("o20t", "two.sided")$statistic), o20t_p_asym = o1("o20t", "two.sided")$p.value,
    a150_p_asym = a1("a150")$p.value, a400_p_asym = a1("a400")$p.value, a400_p_greater_asym = a1("a400", "greater")$p.value,
    t75_D = unname(t2("t75", "two.sided")$statistic), t75_p_exact = t2("t75", "two.sided")$p.value,
    t75_p_greater_exact = t2("t75", "greater")$p.value,
    t75_Dminus = unname(t2("t75", "less")$statistic), t75_p_less_r = t2("t75", "less")$p.value,     # deviation: R uses the D+ distribution
    t75_p_less_exact = brute("t75", "less"),
    t57_p_exact = t2("t57", "two.sided")$p.value, t57_p_greater_exact = t2("t57", "greater")$p.value,
    e99_D = unname(t2("e99", "two.sided")$statistic), e99_p_exact = t2("e99", "two.sided")$p.value,
    e99_p_greater_exact = t2("e99", "greater")$p.value, e99_p_less_exact = t2("e99", "less")$p.value,
    a100_D = unname(t2("a100", "two.sided")$statistic), a100_p_asym = t2("a100", "two.sided")$p.value,
    a100_p_less_asym = t2("a100", "less")$p.value),
    reference = "stats::ks.test (exact = NULL: one-sample n < 100 without ties, two-sample n.x * n.y < 10000; psmirnov exact conditional on ties)",
    call = "ks.test(x, 'pnorm', alternative); ks.test(x, y, alternative)", packages = character(0))
})

case("stat_rank_quade_t4", "stats/quade", "Quade test when A == B: four blocks ranking three treatments identically with tied ranges (R: p = (k!)^(1 - b)), the same rankings with distinct ranges, and a design whose blocks are all constant", function(ctx) {
  conc <- rbind(c(1, 2, 3), c(2, 3, 4), c(3, 4, 5), c(5, 6, 7))
  dist <- rbind(c(1, 2, 3), c(2, 3, 5), c(3, 4, 8), c(5, 6, 12))
  const <- rbind(c(2, 2, 2), c(5, 5, 5), c(1, 1, 1))
  mk <- function(set, m) data.frame(set = set, block = rep(sprintf("B%d", seq_len(nrow(m))), each = ncol(m)),
                                    treat = rep(sprintf("t%d", seq_len(ncol(m))), nrow(m)), y = c(t(m)))
  d <- ctx$gen_dataset("rank_quade_t4", rbind(mk("conc", conc), mk("dist", dist), mk("const", const)))
  mat <- function(s) { e <- d[d$set == s, ]; matrix(e$y, ncol = 3, byrow = TRUE) }
  qc <- quade.test(mat("conc")); qd <- quade.test(mat("dist")); qk <- quade.test(mat("const"))
  stopifnot(is.nan(qc$statistic), is.na(qc$parameter[1]), is.nan(qk$statistic), is.finite(qd$statistic))
  list(values = list(conc_p = qc$p.value, conc_stat_is_nan = as.numeric(is.nan(qc$statistic)),
                     dist_f = unname(qd$statistic), dist_p = qd$p.value,
                     const_p_r = qk$p.value),                                           # deviation: p2a refuses
       reference = "stats::quade.test (A == B branch: STATISTIC NaN, df NA, PVAL = gamma(k + 1)^(1 - b))",
       call = "quade.test(y_matrix)", packages = character(0))
})

case("stat_ks_t3_exact_t4", "stats/ks", "ks.test on the stat_ks_t3 columns with R's default exact = NULL: one-sided exact p (one-sample Birnbaum-Tingey sum, two-sample psmirnov)", function(ctx) {
  d <- ctx$dataset("datasets/generated/ks_t3.csv")
  g <- function(k) d[[k]][!is.na(d[[k]])]
  xn <- g("xn"); xu <- g("xu"); xe <- g("xe"); x5 <- g("x5"); y5 <- g("y5"); xa <- g("xa"); ya <- g("ya")
  one <- function(x, alt, ...) { r <- ks.test(x, ..., alternative = alt); stopifnot(r$exact); r$p.value }
  two <- function(x, y, alt) { r <- ks.test(x, y, alternative = alt); stopifnot(r$exact); r$p.value }
  list(values = list(
    norm_p_greater_exact = one(xn, "greater", "pnorm", 2, 2), norm_p_less_exact = one(xn, "less", "pnorm", 2, 2),
    unif_p_greater_exact = one(xu, "greater", "punif", -1.5, 3), unif_p_less_exact = one(xu, "less", "punif", -1.5, 3),
    exp_p_greater_exact = one(xe, "greater", "pexp", 0.8), exp_p_less_exact = one(xe, "less", "pexp", 0.8),
    two5_p_greater_exact = two(x5, y5, "greater"), two5_p_less_exact = two(x5, y5, "less"),
    twol_p_greater_exact = two(xa, ya, "greater"), twol_p_less_exact = two(xa, ya, "less")),
    reference = "stats::ks.test(alternative = 'greater' | 'less'), exact = NULL (exact for these sizes, no ties)",
    call = "ks.test(x, 'pnorm', 2, 2, alternative = ...); ks.test(x, y, alternative = ...)", packages = character(0))
})
