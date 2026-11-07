# validation/r/cases/data_munging_t3.R -- munging references (T3-WP50: DATA-42, DATA-15, DATA-14,
# DATA-9, DATA-38; DATA-28 and DATA-65 core parts). Value-level pins of the munging layer:
#  * dplyr::lag / dplyr::lead (positional, within group_by groups, in data order), x - lag(x, k) and
#    (y - lag(y)) / lag(y) on an integer column with NA and an unsorted, interleaved group column;
#  * base::cut(x, breaks = n) (R's 0.1% widened range) and cut() on type-7 quantile edges with
#    open outer bins (p2a's Quantile strategy), on an integer column with NA;
#  * stats::model.matrix(~ 0 + factor(g)) on an integer-coded factor (full dummy set);
#  * tidyr::pivot_longer: row-major order, NA kept, vctrs common type (integer + double -> double;
#    integer + integer -> integer); integer + character is an error in tidyr ("Can't combine `a`
#    <integer> and `s` <character>."), which p2a's melt also refuses;
#  * the mode fill: names(which.max(table(x))) (ties go to the first level in R's sorted level
#    order, NA and NaN excluded by table()), then tidyr::replace_na;
#  * dplyr::na_if (replace a sentinel by NA) on integer, double and character columns;
#  * dplyr::count(sort = TRUE) with the NA group, and n / sum(n) (value_counts normalize);
#  * column-wise count of non-missing values, mean, sd, min, max with na.rm = TRUE (describe);
#  * dplyr::filter on integer and double columns with NA (a row whose condition is NA is dropped);
#  * dplyr joins with suffix = c("", "_right") (dplyr:::add_suffixes appends the suffix until the
#    name is unique) for chained joins whose names collide; full_join(keep = TRUE) because p2a's full
#    join keeps both key columns (DATA-43, Tier 4). Strings are carried in the `names` of named values.
# Calls go through the namespace: case files share one R session, and an attached dplyr would mask
# stats::lag for the plm formulas of the case files sourced after this one.

.munging_long <- function() {
  set.seed(5011)
  y <- round(rnorm(12, mean = 10, sd = 3), 4)
  y[c(4, 9)] <- NA
  data.frame(id = 1:12,
             g = c(2L, 1L, 2L, 1L, 1L, 2L, 3L, 3L, 1L, 2L, 3L, 1L),
             x = c(5L, 3L, NA, 8L, 2L, 7L, 4L, NA, 6L, 1L, 9L, 10L),
             y = y)
}
.munging_cut <- function() {
  data.frame(id = 1:12, x = c(12L, 3L, NA, 25L, 7L, 18L, 3L, 30L, NA, 15L, 22L, 9L))
}
.munging_factor <- function() {
  set.seed(5012)
  data.frame(id = 1:10, g = c(2L, 3L, 1L, 2L, 1L, 3L, 3L, 2L, 1L, 2L), w = round(rnorm(10), 4))
}
.munging_wide <- function() {
  data.frame(id = 1:4, a = c(1L, NA, 3L, 4L), b = c(0.5, 1.75, NA, 2.25), c = c(5L, 6L, 7L, 8L))
}
.munging_fill <- function() {
  data.frame(id = 1:10,
             k = c(3L, NA, 1L, 3L, 1L, 2L, NA, 5L, 4L, 6L),
             f = c(2.5, 1.25, NA, 2.5, 3.75, NA, 2.5, 1.25, 0.5, 4.125),
             s = c("b", "a", NA, "b", "a", "c", NA, "c", "d", "e"),
             stringsAsFactors = FALSE)
}
.munging_counts <- function() {
  data.frame(id = 1:10, g = c(2L, 7L, NA, 2L, 5L, 7L, 2L, NA, 7L, 2L))
}
.munging_join_frames <- function() {
  data.frame(frame = c("A", "A", "AB", "AB", "C", "C"),
             k = c(1L, 2L, 1L, 2L, 1L, 2L),
             value = c(1.5, 2.5, 1.25, 2.75, 100.5, 200.25),
             value_right = c(NA, NA, 10.5, 20.5, NA, NA),
             stringsAsFactors = FALSE)
}
# Strings with NA as a named value: name "" and value NA for a missing cell, the text and 1 otherwise.
.strings <- function(v) named(setNames(ifelse(is.na(v), NA_real_, 1), ifelse(is.na(v), "", v)))

case("data_munging_lag", "data/munging",
     "dplyr::lag / lead (positional, within groups), diff and pct_change on integer data with NA",
     function(ctx) {
  d <- ctx$gen_dataset("munging_long", .munging_long())
  stopifnot(is.integer(d$x), is.integer(d$g), is.integer(dplyr::lag(d$x)))
  by_g <- d |>
    dplyr::group_by(g) |>
    dplyr::mutate(lag2 = dplyr::lag(x, 2), lead1 = dplyr::lead(y, 1)) |>
    dplyr::ungroup()
  list(values = list(
         lag1_x = vec(dplyr::lag(d$x, 1)),
         lag2_x_by_g = vec(by_g$lag2),
         lead1_y_by_g = vec(by_g$lead1),
         diff1_y = vec(d$y - dplyr::lag(d$y, 1)),
         diff2_x = vec(d$x - dplyr::lag(d$x, 2)),
         pct_change1_y = vec((d$y - dplyr::lag(d$y, 1)) / dplyr::lag(d$y, 1))),
       reference = "dplyr::lag, dplyr::lead (group_by(g)); x - lag(x, k); (y - lag(y)) / lag(y)",
       call = "d %>% group_by(g) %>% mutate(lag2 = lag(x, 2), lead1 = lead(y, 1))",
       packages = c("dplyr"))
})

case("data_munging_cut", "data/munging",
     "base::cut(x, breaks = n) and cut() on type-7 quantile edges with open outer bins, integer data with NA",
     function(ctx) {
  d <- ctx$gen_dataset("munging_cut", .munging_cut())
  stopifnot(is.integer(d$x))
  nb <- 4L
  rx <- range(d$x, na.rm = TRUE); dx <- diff(rx)
  br <- seq.int(rx[1], rx[2], length.out = nb + 1L)
  br[c(1L, nb + 1L)] <- c(rx[1] - dx / 1000, rx[2] + dx / 1000)
  ew <- cut(d$x, nb)
  stopifnot(identical(as.integer(ew), as.integer(cut(d$x, br))))
  q <- quantile(d$x, c(1, 2) / 3, type = 7, na.rm = TRUE, names = FALSE)
  qc <- cut(d$x, c(-Inf, q, Inf))
  code <- function(f) { v <- as.integer(f) - 1L; v[is.na(v)] <- -1L; v }
  list(values = list(
         ew_breaks = vec(br),
         ew_codes = vec(code(ew)),
         q_edges = vec(q),
         q_codes = vec(code(qc)),
         n_na = sum(is.na(ew))),
       reference = "base::cut(x, breaks = 4); base::cut(x, c(-Inf, quantile(x, 1:2 / 3, type = 7, na.rm = TRUE), Inf))",
       call = "cut(x, 4); cut(x, c(-Inf, quantile(x, c(1, 2) / 3, type = 7, na.rm = TRUE), Inf))",
       packages = character(0))
})

case("data_munging_model_matrix", "data/munging",
     "stats::model.matrix(~ 0 + factor(g)): one dummy per level of an integer-coded factor",
     function(ctx) {
  d <- ctx$gen_dataset("munging_factor", .munging_factor())
  stopifnot(is.integer(d$g))
  mm <- model.matrix(~ 0 + factor(g), d)
  stopifnot(identical(colnames(mm), c("factor(g)1", "factor(g)2", "factor(g)3")))
  list(values = list(
         levels = vec(sort(unique(d$g))),
         dummy_1 = vec(mm[, "factor(g)1"]),
         dummy_2 = vec(mm[, "factor(g)2"]),
         dummy_3 = vec(mm[, "factor(g)3"]),
         level_counts = vec(colSums(mm))),
       reference = "stats::model.matrix(~ 0 + factor(g))",
       call = "model.matrix(~ 0 + factor(g), d)",
       packages = character(0))
})

case("data_munging_pivot_longer", "data/munging",
     "tidyr::pivot_longer: row-major long format, NA kept, integer + double -> double, integer + integer -> integer",
     function(ctx) {
  d <- ctx$gen_dataset("munging_wide", .munging_wide())
  stopifnot(is.integer(d$a), is.double(d$b), is.integer(d$c))
  long <- tidyr::pivot_longer(d, c(a, b, c), names_to = "var", values_to = "val")
  long_int <- tidyr::pivot_longer(d, c(a, c), names_to = "var", values_to = "val")
  stopifnot(is.double(long$val), is.integer(long_int$val))
  list(values = list(
         long_id = vec(long$id),
         long_val = named(setNames(long$val, long$var)),
         long_int_id = vec(long_int$id),
         long_int_val = named(setNames(as.numeric(long_int$val), long_int$var))),
       reference = "tidyr::pivot_longer(cols, names_to = 'var', values_to = 'val')",
       call = "pivot_longer(d, c(a, b, c), names_to = 'var', values_to = 'val'); pivot_longer(d, c(a, c), ...)",
       packages = c("tidyr"))
})

case("data_munging_fill_mode", "data/munging",
     "Mode fill: names(which.max(table(x))) (ties to the first sorted level), then tidyr::replace_na",
     function(ctx) {
  d <- ctx$gen_dataset("munging_fill", .munging_fill())
  stopifnot(is.integer(d$k), is.double(d$f), is.character(d$s))
  mode_of <- function(v) names(which.max(table(v)))
  k_mode <- as.integer(mode_of(d$k)); f_mode <- as.numeric(mode_of(d$f)); s_mode <- mode_of(d$s)
  # the tie rules the case exists for: k has 3 and 1 twice each (3 first in the data), s has a, b, c
  # twice each (b first); R takes the first level in sorted order
  stopifnot(k_mode == 1L, f_mode == 2.5, s_mode == "a")
  list(values = list(
         k_mode = k_mode,
         k_filled = vec(tidyr::replace_na(d$k, k_mode)),
         f_mode = f_mode,
         f_filled = vec(tidyr::replace_na(d$f, f_mode)),
         s_mode = named(setNames(1, s_mode)),
         s_filled = .strings(tidyr::replace_na(d$s, s_mode))),
       reference = "names(which.max(table(x))); tidyr::replace_na(x, mode)",
       call = "replace_na(d$k, as.integer(names(which.max(table(d$k)))))",
       packages = c("tidyr"))
})

case("data_munging_na_if", "data/munging",
     "dplyr::na_if: a sentinel value becomes NA in integer, double and character columns",
     function(ctx) {
  d <- ctx$gen_dataset("munging_fill", .munging_fill())
  list(values = list(
         k_na_if_3 = vec(dplyr::na_if(d$k, 3L)),
         f_na_if_1_25 = vec(dplyr::na_if(d$f, 1.25)),
         s_na_if_c = .strings(dplyr::na_if(d$s, "c"))),
       reference = "dplyr::na_if(x, y)",
       call = "na_if(d$k, 3L); na_if(d$f, 1.25); na_if(d$s, 'c')",
       packages = c("dplyr"))
})

case("data_munging_value_counts", "data/munging",
     "dplyr::count(d, g, sort = TRUE) with the NA group; proportion = n / sum(n)",
     function(ctx) {
  d <- ctx$gen_dataset("munging_counts", .munging_counts())
  cnt <- dplyr::count(d, g, sort = TRUE)
  stopifnot(!anyDuplicated(cnt$n))   # distinct counts: the order is unambiguous
  list(values = list(
         vc_g = vec(cnt$g),
         vc_n = vec(cnt$n),
         vc_proportion = vec(cnt$n / sum(cnt$n))),
       reference = "dplyr::count(sort = TRUE); n / sum(n)",
       call = "count(d, g, sort = TRUE) %>% mutate(proportion = n / sum(n))",
       packages = c("dplyr"))
})

case("data_munging_describe", "data/munging",
     "Column summaries with na.rm = TRUE: non-missing count, mean, sd, min, max (integer and double columns)",
     function(ctx) {
  d <- ctx$gen_dataset("munging_fill", .munging_fill())
  s <- function(v) c(count = sum(!is.na(v)), mean = mean(v, na.rm = TRUE), std = sd(v, na.rm = TRUE),
                     min = min(v, na.rm = TRUE), max = max(v, na.rm = TRUE))
  list(values = list(describe_k = named(s(d$k)), describe_f = named(s(d$f))),
       reference = "sum(!is.na(x)); mean, sd, min, max with na.rm = TRUE",
       call = "sapply(d[c('k', 'f')], function(v) c(sum(!is.na(v)), mean(v, na.rm = TRUE), sd(v, na.rm = TRUE), min(v, na.rm = TRUE), max(v, na.rm = TRUE)))",
       packages = character(0))
})

case("data_munging_join_suffix", "data/munging",
     "dplyr joins with suffix c('', '_right'): colliding right-hand names get the suffix until unique",
     function(ctx) {
  d <- ctx$gen_dataset("munging_join_frames", .munging_join_frames())
  A <- d[d$frame == "A", c("k", "value")]
  AB <- d[d$frame == "AB", c("k", "value", "value_right")]
  C <- d[d$frame == "C", c("k", "value")]
  sfx <- c("", "_right")
  out <- list(
    a = dplyr::left_join(AB, C, by = "k", suffix = sfx),
    b = dplyr::left_join(A, AB, by = "k", suffix = sfx),
    c = dplyr::full_join(AB, AB, by = "k", suffix = sfx, keep = TRUE),
    d = dplyr::right_join(AB, C, by = "k", suffix = sfx),
    e = dplyr::inner_join(AB, C, by = "k", suffix = sfx),
    f = dplyr::cross_join(AB, C, suffix = sfx),
    g = dplyr::left_join(AB, C, by = "k", suffix = c("", "_c")))
  stopifnot(identical(names(out$a), c("k", "value", "value_right", "value_right_right")),
            identical(names(out$b), c("k", "value", "value_right", "value_right_right")))
  values <- list()
  for (s in names(out)) {
    o <- as.data.frame(out[[s]])
    o <- if (s == "f") o[order(o$k, o$k_right), ] else o[order(o$k), ]
    for (nm in names(o)) values[[paste0(s, "__", nm)]] <- vec(o[[nm]])
  }
  list(values = values,
       reference = "dplyr::left_join / full_join(keep = TRUE) / right_join / inner_join / cross_join, suffix = c('', '_right'); dplyr:::add_suffixes",
       call = "left_join(AB, C, by = 'k', suffix = c('', '_right')) and the other joins; rows ordered by the keys",
       packages = c("dplyr"))
})

case("data_munging_filter", "data/munging",
     "dplyr::filter on an integer and a double column with NA (rows whose condition is NA are dropped)",
     function(ctx) {
  d <- ctx$gen_dataset("munging_long", .munging_long())
  list(values = list(ids_x_ge_5 = vec(dplyr::filter(d, x >= 5)$id), ids_y_lt_9 = vec(dplyr::filter(d, y < 9)$id)),
       reference = "dplyr::filter(d, x >= 5); dplyr::filter(d, y < 9)",
       call = "filter(d, x >= 5)$id; filter(d, y < 9)$id",
       packages = c("dplyr"))
})
