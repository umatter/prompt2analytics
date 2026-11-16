# validation/r/cases/ts_changepoint_t4.R -- T4-WP25: binary segmentation (TS-8).
#  changepoint::cpt.mean / cpt.var / cpt.meanvar (method = "BinSeg", minseglen = 2): penalty "MBIC"
#  ((k + 2) ln n, the cost with ln(segment length) per segment) and "Manual". The greedy search splits
#  the segment with the largest cost reduction while that reduction exceeds the penalty, up to Q
#  changepoints. Each pinned series is also run through a written-out greedy search with the
#  symmetric candidate range s + minseglen <= t <= e - minseglen, which p2a uses: changepoint's C code
#  narrows the range by one at irregular places (left end always, right end only for the last segment
#  at minseglen 2), so the two can differ on series with a near-tie at a boundary; every pinned series
#  is one on which they agree (stopifnot).
.bs_cost <- function(z, type, mu) function(s, e) { v <- z[(s + 1):e]; m <- length(v)
  if (type == "mean") return(sum((v - mean(v))^2))
  if (type == "var") { q <- sum((v - mu)^2); if (q <= 0) q <- 1e-11; return(m * (log(2 * pi) + log(q / m) + 1)) }
  s2 <- mean((v - mean(v))^2); if (s2 <= 0) s2 <- 1e-11; m * (log(2 * pi) + log(s2) + 1) }
.bs_written <- function(z, type, pen, Q, ms, mbic) {
  n <- length(z); C0 <- .bs_cost(z, type, mean(z)); C <- function(s, e) C0(s, e) + if (mbic) log(e - s) else 0
  bnd <- c(0, n); cps <- integer(0)
  for (q in seq_len(Q)) { best <- -Inf; bt <- NA
    for (k in seq_len(length(bnd) - 1)) { s <- bnd[k]; e <- bnd[k + 1]; if (e - s < 2 * ms) next
      for (t in (s + ms):(e - ms)) { g <- C(s, e) - C(s, t) - C(t, e); if (g > best) { best <- g; bt <- t } } }
    if (is.na(bt) || !(best > pen)) break
    cps <- c(cps, bt); bnd <- sort(c(bnd, bt)) }
  sort(cps) }
case("cpt_binseg_t4", "forecasting/changepoint", "changepoint BinSeg (cpt.mean, cpt.var, cpt.meanvar; MBIC and Manual penalties; Q)", function(ctx) {
  set.seed(20261115)
  wn <- rnorm(300); reg <- c(rnorm(100, 0), rnorm(100, 1.5), rnorm(100, 0.3))
  vr <- c(rnorm(90, 0, 1), rnorm(80, 0, 2.5), rnorm(70, 0, 1))
  many <- c(rnorm(60, 0), rnorm(60, 3), rnorm(60, 0), rnorm(60, 3), rnorm(60, 0))
  mv <- c(rnorm(120, 0, 1), rnorm(120, 1, 3))
  # two datasets, no NA padding: ts_changepoint reads a column with NaPolicy::Fail
  d <- ctx$gen_dataset("cpt_binseg_t4", data.frame(t = 1:300, wn = round(wn, 10), reg = round(reg, 10),
                                                   many = round(many, 10)))
  dv <- ctx$gen_dataset("cpt_binseg_t4_var", data.frame(t = 1:240, vr = round(vr, 10), mv = round(mv, 10)))
  wn <- d$wn; reg <- d$reg; vr <- dv$vr; many <- d$many; mv <- dv$mv
  v <- list(); fn <- list(mean = cpt.mean, var = cpt.var, meanvar = cpt.meanvar)
  pin <- function(tag, x, type, penalty = "MBIC", pen.value = 0, Q = 5) {
    r <- suppressWarnings(fn[[type]](x, method = "BinSeg", penalty = penalty, pen.value = pen.value, Q = Q, minseglen = 2))
    cp <- cpts(r); pen <- pen.value(r)
    stopifnot(identical(as.numeric(cp), as.numeric(.bs_written(x, type, pen, Q, 2, penalty == "MBIC"))))
    if (length(cp)) v[[paste0(tag, "_cpts")]] <<- vec(cp)
    v[[paste0(tag, "_ncpts")]] <<- length(cp)
    v[[paste0(tag, "_q_reached")]] <<- as.numeric(length(cp) == Q)
    v[[paste0(tag, "_pen")]] <<- pen
    C0 <- .bs_cost(x, type, mean(x)); b <- c(0, cp, length(x))
    v[[paste0(tag, "_cost")]] <<- sum(sapply(seq_len(length(b) - 1), function(i) C0(b[i], b[i + 1]) + if (penalty == "MBIC") log(b[i + 1] - b[i]) else 0))
  }
  pin("wn_mean", wn, "mean"); pin("reg_mean", reg, "mean"); pin("reg_mean_q10", reg, "mean", Q = 10)
  pin("reg_mean_manual", reg, "mean", penalty = "Manual", pen.value = 10)
  pin("vr_var", vr - mean(vr), "var"); pin("vr_meanvar", vr - mean(vr), "meanvar")
  pin("mv_meanvar", mv - mean(mv), "meanvar"); pin("mv_var", mv - mean(mv), "var")
  pin("many_mean_q2", many, "mean", Q = 2); pin("many_mean", many, "mean")
  list(values = v, dataset = "validation/datasets/generated/cpt_binseg_t4.csv", reference = "changepoint::cpt.mean / cpt.var / cpt.meanvar (method = 'BinSeg', minseglen = 2)",
       call = "cpt.mean(reg, method = 'BinSeg', penalty = 'MBIC', Q = 5, minseglen = 2)", packages = "changepoint")
})
