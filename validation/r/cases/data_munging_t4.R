# validation/r/cases/data_munging_t4.R -- munging references (T4-WP27: DATA-11, DATA-29, DATA-39, DATA-40,
# DATA-41, DATA-71, DATA-43, DATA-44, DATA-36).
#  * dplyr::lag within group_by groups in data order: x - lag(x) and (x - lag(x)) / lag(x) on an integer
#    column (double result, no integer division), (x - min) / (max - min) on an integer column;
#  * tidyr::fill(.direction = "down" / "up") within group_by groups;
#  * plm::lag / plm::lead / diff on a pdata.frame indexed by (firm, year): the value of the period t - k
#    of the same firm, NA when that period is not observed (gaps), in the data's own row order;
#  * stats::model.matrix(~ 0 + f) and model.matrix(~ f) (treatment contrasts drop the first sorted level)
#    on a text and a double factor with NA under na.action = na.pass (an NA row is NA in every dummy);
#  * base::rank(ties.method = "average", na.last = "keep"), ascending and descending (rank(-x));
#  * dplyr joins with NA keys (na_matches = "na": NA matches NA, as base::merge) and coalesced full-join
#    keys; dplyr::bind_rows (union of columns, integer + double -> double);
#  * dplyr::distinct(.keep_all = TRUE) and !duplicated(fromLast = TRUE) row order; dplyr::summarise
#    group order (sorted keys, NA last).
# Calls go through the namespace (see data_munging_t3.R).

.munging_panel <- function() {
  set.seed(5021)
  d <- data.frame(id = 1:11,
    firm = c(2L, 1L, 1L, 2L, 3L, 1L, 3L, 2L, 3L, 1L, 2L),
    year = c(2002L, 2003L, 2001L, 2001L, 2001L, 2002L, 2004L, 2003L, 2002L, 2004L, 2004L),
    x = c(7L, 12L, 10L, 5L, 3L, 11L, 9L, 8L, 6L, 15L, NA),
    z = c(NA, 2.5, 1.25, 4.75, NA, NA, 3.5, NA, 6.125, NA, 0.5))
  d$y <- round(rnorm(11, 10, 3), 4)
  d
}
.munging_regions <- function() {
  data.frame(id = 1:8,
             region = c("west", "south", NA, "north", "east", "south", "west", "east"),
             g = c(1.5, 3, 1.5, 2.5, 3, NA, 2.5, 1.5),
             r = c(10L, 20L, 20L, 30L, NA, 20L, 5L, 30L),
             stringsAsFactors = FALSE)
}
.munging_join_na <- function() {
  data.frame(id = 1:8, frame = c("L", "L", "L", "L", "R", "R", "R", "R"),
             k = c(1L, 2L, NA, 4L, 2L, 3L, NA, NA),
             val = c(10.5, 20.25, 30.125, 40.75, 200.5, 300.25, 400.125, 500.75),
             stringsAsFactors = FALSE)
}
.munging_bind <- function() {
  data.frame(frame = c("A", "A", "B", "B"), id = 1:4,
             xi = c(1L, 2L, NA, NA), xd = c(NA, NA, 1.5, 2.5),
             Income = c(100.5, 200.25, NA, NA), income = c(NA, NA, 300.5, 400.75),
             stringsAsFactors = FALSE)
}
.munging_order <- function() {
  data.frame(id = 1:9, key = c(5L, 3L, 5L, 1L, 4L, 3L, 2L, 1L, 6L),
             v = c(1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5),
             g = c(2L, NA, 1L, 3L, 2L, 1L, NA, 2L, 3L))
}
.strings4 <- function(v) named(setNames(ifelse(is.na(v), NA_real_, 1), ifelse(is.na(v), "", v)))

case("data_munging_panel_shift", "data/munging",
     "Grouped diff/pct_change/fill (dplyr, tidyr) and time-indexed lag/lead/diff (plm) on integer panel data",
     function(ctx) {
  d <- ctx$gen_dataset("munging_panel", .munging_panel())
  stopifnot(is.integer(d$x), is.integer(d$firm), is.integer(d$year), is.double(d$z), is.double(d$y))
  m <- dplyr::ungroup(dplyr::mutate(dplyr::group_by(d, firm),
         diff1_y = y - dplyr::lag(y, 1),
         pct1_x = (x - dplyr::lag(x, 1)) / dplyr::lag(x, 1)))
  stopifnot(is.double(m$pct1_x))
  down <- tidyr::fill(dplyr::group_by(d, firm), z, .direction = "down")$z
  up <- tidyr::fill(dplyr::group_by(d, firm), z, .direction = "up")$z
  pd <- plm::pdata.frame(d, index = c("firm", "year"), drop.index = FALSE, row.names = FALSE)
  o <- order(d$firm, d$year)
  stopifnot(identical(as.integer(as.character(pd$id)), d$id[o]))
  back <- function(v) { r <- numeric(nrow(d)); r[o] <- as.numeric(v); r }
  lag1_x <- back(plm::lag(pd$x, 1))
  list(values = list(
         grouped_diff1_y = vec(m$diff1_y),
         grouped_pct1_x = vec(m$pct1_x),
         pct1_x = vec((d$x - dplyr::lag(d$x, 1)) / dplyr::lag(d$x, 1)),
         normalize_x = vec((d$x - min(d$x, na.rm = TRUE)) / (max(d$x, na.rm = TRUE) - min(d$x, na.rm = TRUE))),
         fill_down_z_by_firm = vec(down),
         fill_up_z_by_firm = vec(up),
         time_lag1_y = vec(back(plm::lag(pd$y, 1))),
         time_lag2_y = vec(back(plm::lag(pd$y, 2))),
         time_lead1_y = vec(back(plm::lead(pd$y, 1))),
         time_diff1_y = vec(back(diff(pd$y, 1))),
         time_lag1_x = vec(lag1_x),
         time_pct1_x = vec((d$x - lag1_x) / lag1_x)),
       reference = "dplyr::lag within group_by(firm); tidyr::fill(.direction = 'down'/'up') within group_by(firm); plm::lag, plm::lead, diff on pdata.frame(index = c('firm', 'year'))",
       call = "group_by(d, firm) %>% mutate(y - lag(y), (x - lag(x)) / lag(x)); fill(group_by(d, firm), z); plm::lag(pdata.frame(d, c('firm', 'year'))$y, k)",
       packages = c("dplyr", "tidyr", "plm"))
})

case("data_munging_factor_t4", "data/munging",
     "One-hot dummies as model.matrix (full set and treatment contrasts, NA rows kept as NA); rank with average ties",
     function(ctx) {
  d <- ctx$gen_dataset("munging_regions", .munging_regions())
  stopifnot(is.character(d$region), is.double(d$g), is.integer(d$r))
  mf <- model.frame(~ region, d, na.action = na.pass)
  full <- model.matrix(~ 0 + region, mf)
  tr <- model.matrix(~ region, mf)
  stopifnot(identical(colnames(full), c("regioneast", "regionnorth", "regionsouth", "regionwest")),
            identical(colnames(tr), c("(Intercept)", "regionnorth", "regionsouth", "regionwest")))
  mg <- model.matrix(~ 0 + factor(g), model.frame(~ factor(g), d, na.action = na.pass))
  lv <- levels(factor(d$region)); lg <- levels(factor(d$g))
  stopifnot(identical(lg, c("1.5", "2.5", "3")))
  list(values = list(
         region_levels = named(setNames(seq_along(lv), lv)),
         region_east = vec(full[, "regioneast"]),
         region_north = vec(full[, "regionnorth"]),
         region_south = vec(full[, "regionsouth"]),
         region_west = vec(full[, "regionwest"]),
         treatment_dropped = named(setNames(1, lv[1])),
         g_levels = named(setNames(seq_along(lg), lg)),
         g_1_5 = vec(mg[, 1]), g_2_5 = vec(mg[, 2]), g_3 = vec(mg[, 3]),
         rank_r = vec(rank(d$r, na.last = "keep")),
         rank_r_desc = vec(rank(-d$r, na.last = "keep"))),
       reference = "stats::model.matrix(~ 0 + region) and model.matrix(~ region) under na.action = na.pass; base::rank(ties.method = 'average', na.last = 'keep')",
       call = "model.matrix(~ 0 + region, model.frame(~ region, d, na.action = na.pass)); rank(d$r, na.last = 'keep'); rank(-d$r, na.last = 'keep')",
       packages = character(0))
})

case("data_munging_combine_t4", "data/munging",
     "Joins with NA keys and coalesced full-join keys (dplyr), bind_rows type union, distinct and summarise row order",
     function(ctx) {
  j <- ctx$gen_dataset("munging_join_na", .munging_join_na())
  b <- ctx$gen_dataset("munging_bind", .munging_bind())
  o <- ctx$gen_dataset("munging_order", .munging_order())
  stopifnot(is.integer(j$k), is.integer(b$xi), is.double(b$xd), is.integer(o$key), is.integer(o$g))
  L <- data.frame(k = j$k[j$frame == "L"], a = j$val[j$frame == "L"])
  R <- data.frame(k = j$k[j$frame == "R"], b = j$val[j$frame == "R"])
  srt <- function(x) x[order(x$k, x[[2]], if (ncol(x) > 2) x[[3]] else rep(0, nrow(x)), na.last = TRUE), , drop = FALSE]
  fj <- srt(dplyr::full_join(L, R, by = "k"))
  stopifnot(identical(names(fj), c("k", "a", "b")),
            identical(merge(L, R, by = "k", all = TRUE)$k, fj$k))
  ij <- srt(dplyr::inner_join(L, R, by = "k"))
  aj <- srt(dplyr::anti_join(L, R, by = "k"))
  sj <- srt(dplyr::semi_join(L, R, by = "k"))
  A <- data.frame(id = b$id[b$frame == "A"], x = b$xi[b$frame == "A"], Income = b$Income[b$frame == "A"])
  B <- data.frame(id = b$id[b$frame == "B"], x = b$xd[b$frame == "B"], income = b$income[b$frame == "B"])
  br <- dplyr::bind_rows(A, B)
  stopifnot(is.integer(A$x), is.double(br$x), identical(names(br), c("id", "x", "Income", "income")))
  df1 <- dplyr::distinct(o, key, .keep_all = TRUE)
  kl <- o[!duplicated(o$key, fromLast = TRUE), ]
  kn <- o[!(duplicated(o$key) | duplicated(o$key, fromLast = TRUE)), ]
  s <- dplyr::summarise(dplyr::group_by(o, g), n = dplyr::n(), s = sum(v), .groups = "drop")
  list(values = list(
         full_k = vec(fj$k), full_a = vec(fj$a), full_b = vec(fj$b),
         inner_k = vec(ij$k), inner_a = vec(ij$a), inner_b = vec(ij$b),
         anti_k = vec(aj$k), semi_k = vec(sj$k), semi_a = vec(sj$a),
         bind_x = vec(br$x), bind_Income = vec(br$Income), bind_income = vec(br$income),
         distinct_key = vec(df1$key), distinct_id = vec(df1$id),
         keep_last_key = vec(kl$key), keep_last_id = vec(kl$id),
         keep_none_key = vec(kn$key),
         summarise_g = vec(s$g), summarise_n = vec(s$n), summarise_s = vec(s$s)),
       reference = "dplyr::full_join / inner_join / anti_join / semi_join (na_matches = 'na'), base::merge(all = TRUE); dplyr::bind_rows; dplyr::distinct(.keep_all = TRUE); !duplicated(fromLast = TRUE); dplyr::summarise(group_by(g))",
       call = "full_join(L, R, by = 'k'); bind_rows(A, B); distinct(o, key, .keep_all = TRUE); summarise(group_by(o, g), n = n(), s = sum(v)); rows of the joins ordered by k (NA last)",
       packages = c("dplyr"))
})
