# validation/r/cases/data_quality.R -- quantile / cut references for the data-layer package
# (T2-WP04: DATA-12, DATA-16, WT-9).

case("data_quality_profile_na", "data/quality", "Quality-profile statistics on a column with nulls and NaN (quantile type 7 on present values)", function(ctx) {
  set.seed(12)
  x <- round(rnorm(40, 10, 3), 4)
  x[c(3, 17, 29)] <- NA                       # three nulls
  z <- c(round(runif(37, 0, 5), 4), NaN, NaN, NaN)   # NaN cells: written as empty fields, read as null by both sides
  d <- ctx$gen_dataset("quality_na", data.frame(id = 1:40, x = x, z = z))
  q <- quantile(d$x, c(0.25, 0.5, 0.75), type = 7, na.rm = TRUE)
  iqr <- q[3] - q[1]
  out <- sum(d$x < q[1] - 1.5 * iqr | d$x > q[3] + 1.5 * iqr, na.rm = TRUE)
  list(values = list(x_q1 = unname(q[1]), x_median = unname(q[2]), x_q3 = unname(q[3]),
                     x_mean = mean(d$x, na.rm = TRUE), x_sd = sd(d$x, na.rm = TRUE),
                     x_min = min(d$x, na.rm = TRUE), x_max = max(d$x, na.rm = TRUE),
                     x_n_missing = sum(is.na(d$x)), x_n_outliers_iqr = out,
                     z_n_missing = sum(is.na(d$z)), z_median = median(d$z, na.rm = TRUE),
                     completeness = 1 - (sum(is.na(d$x)) + sum(is.na(d$z))) / (2 * 40)),
       reference = "stats::quantile(type = 7, na.rm = TRUE); sd; IQR outlier rule 1.5 * IQR",
       call = "quantile(x, c(.25, .5, .75), na.rm = TRUE)", packages = character(0))
})

case("data_bin_cut", "data/munging", "Binning with custom edges, nulls and out-of-range values (base::cut)", function(ctx) {
  x <- c(-5, 0, 0.5, 1, 2.5, 3, 4.9, 5, 7, NA, 2, NA)
  d <- ctx$gen_dataset("bin_cut", data.frame(id = seq_along(x), x = x))
  b_right <- cut(d$x, breaks = c(0, 1, 3, 5))                          # (0,1] (1,3] (3,5]
  b_low <- cut(d$x, breaks = c(0, 1, 3, 5), include.lowest = TRUE)     # [0,1] (1,3] (3,5]
  b_left <- cut(d$x, breaks = c(0, 1, 3, 5), right = FALSE)            # [0,1) [1,3) [3,5)
  code <- function(f) vec(ifelse(is.na(f), -1, as.integer(f) - 1))      # 0-based bin index, -1 = NA
  list(values = list(right_codes = code(b_right), include_lowest_codes = code(b_low), left_closed_codes = code(b_left),
                     n_na_right = sum(is.na(b_right))),
       reference = "base::cut(breaks = c(0, 1, 3, 5), right = TRUE | FALSE, include.lowest)",
       call = "cut(x, breaks = c(0, 1, 3, 5))", packages = character(0))
})
