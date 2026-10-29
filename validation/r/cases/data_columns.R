# validation/r/cases/data_columns.R -- the R conventions the shared column extractor follows
# (T2-WP01): which rows complete.cases keeps, factor() level order (numeric levels numerically,
# strings in C-locale order), logical as 0/1, and the na.strings used to read the CSV.

case("columns_conventions", "data/columns", "complete.cases, factor level order (C locale), logical to numeric", function(ctx) {
  old <- Sys.getlocale("LC_COLLATE"); on.exit(Sys.setlocale("LC_COLLATE", old))
  Sys.setlocale("LC_COLLATE", "C")
  d <- ctx$gen_dataset("columns_conventions", data.frame(
    id = 1:10,
    y = c(1.5, NA, 3.25, 4, 5.5, 6, NA, 8, 9.75, 10),
    n = c(10L, 2L, 2L, NA, 33L, 10L, 4L, 2L, 33L, 4L),        # integer codes; numeric level order 2 < 4 < 10 < 33
    s = c("b", "a", "B", "a", NA, "c", "b", "A", "c", "a"),   # C-locale order A < B < a < b < c
    flag = c(TRUE, FALSE, TRUE, TRUE, FALSE, NA, TRUE, FALSE, FALSE, TRUE)))
  stopifnot(is.integer(d$n), is.logical(d$flag))
  cc <- complete.cases(d$y, d$n, d$s)
  fn <- factor(d$n); fs <- factor(d$s)
  list(values = list(
    complete_rows = vec(which(cc) - 1), n_complete = sum(cc), n_dropped = sum(!cc),
    missing_y = sum(is.na(d$y)), missing_n = sum(is.na(d$n)), missing_s = sum(is.na(d$s)),
    n_levels_numeric = vec(as.numeric(levels(fn))), n_codes = vec(ifelse(is.na(fn), -1, as.integer(fn) - 1)),
    s_level_count = nlevels(fs), s_codes = vec(ifelse(is.na(fs), -1, as.integer(fs) - 1)),
    flag_numeric = vec(as.numeric(d$flag)), mean_y_complete = mean(d$y[cc])),
    reference = "stats::complete.cases; base::factor (levels = sort(unique(x)), LC_COLLATE = C); as.numeric(logical)",
    call = "complete.cases(y, n, s); factor(n); factor(s)", packages = character(0))
})
