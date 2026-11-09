# validation/r/cases/mcp_data_t3.R -- references for the MCP data surface (T3-WP59: LLM-6, DATA-25).
#  * describe_dataset (LLM-6): per column the number of non-missing values and of missing ones
#    (NA; the harness cannot write NaN or Inf, which the Rust fixtures cover); for numeric
#    columns R's summary() (min, 1st quartile, median, mean, 3rd quartile, max; quantile type 7)
#    and sd(), all with na.rm = TRUE; for text and logical columns the number of distinct
#    non-missing values and the five most frequent ones from sort(table(x), decreasing = TRUE)
#    (table() drops NA; ties keep table()'s level order, which is C-locale byte order for the
#    ASCII labels used here, and FALSE before TRUE). An all-missing column is read as text by
#    polars and as logical NA by R: both have no non-missing value.
#  * viz_dendrogram (DATA-25): stats::hclust(dist(x)) (complete linkage) and its merge matrix,
#    heights and cluster sizes, so the Rust test can check the documented conversion of R's
#    1-based merge matrix (negative = sample) into the 0-based [c1, c2, height, size] rows.
#  The fill, value-count and replace tools are pinned to T3-WP50's data_munging_t3.R cases.

.describe_columns <- function() {
  set.seed(5901)
  income <- round(rnorm(15, mean = 52000, sd = 9000), 2)
  income[c(3, 8, 11)] <- NA
  age <- sample(22:64, 15, replace = TRUE)
  age[7] <- NA
  region <- c("west", "east", "north", "east", NA, "west", "south", "north", "east",
              "west", "south", "east", "north", "west", "south")
  grade <- c("B", "A", "C", "A", "D", "B", "E", "F", "A", "G", "B", "C", "D", "E", "F")
  member <- c(TRUE, FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, NA, TRUE, FALSE, TRUE,
              TRUE, FALSE, TRUE)
  data.frame(id = 1:15, income = income, age = age, region = region, grade = grade,
             member = member, empty = rep(NA, 15), stringsAsFactors = FALSE)
}

case("data_describe_columns", "data/describe",
     "describe_dataset: n, missing, summary() and sd() per numeric column; distinct values and the most frequent ones per text/logical column",
     function(ctx) {
  old <- Sys.getlocale("LC_COLLATE"); on.exit(Sys.setlocale("LC_COLLATE", old))
  Sys.setlocale("LC_COLLATE", "C")                # table() level order = byte order, as p2a's factor_options
  d <- ctx$gen_dataset("describe_columns", .describe_columns())
  stopifnot(is.integer(d$id), is.integer(d$age), is.double(d$income), is.character(d$region),
            is.character(d$grade), is.logical(d$member), all(is.na(d$empty)))
  num <- function(v) {
    s <- summary(v)                               # full-precision summaryDefault (R >= 4.0.0)
    q <- quantile(v, c(0, 0.25, 0.5, 0.75, 1), type = 7, na.rm = TRUE, names = FALSE)
    stopifnot(identical(unname(as.numeric(s[c("Min.", "1st Qu.", "Median", "3rd Qu.", "Max.")])), q))
    c(n = sum(!is.na(v)), missing = sum(is.na(v)), mean = mean(v, na.rm = TRUE),
      sd = sd(v, na.rm = TRUE), min = q[1], q1 = q[2], median = q[3], q3 = q[4], max = q[5])
  }
  top <- function(v) {
    t5 <- utils::head(sort(table(v), decreasing = TRUE), 5L)
    setNames(as.numeric(t5), names(t5))
  }
  cnt <- function(v) c(n = sum(!is.na(v)), missing = sum(is.na(v)), distinct = length(table(v)))
  # The properties the case exists for: interpolated type-7 quartiles (n = 12, 14, 15 non-missing),
  # ties in the counts (east/west 4, north/south 3; A/B 3, then C..F 2 in level order), more than
  # five levels (grade has 7), and a logical column with TRUE more frequent than FALSE.
  stopifnot(identical(names(top(d$region)), c("east", "west", "north", "south")),
            identical(names(top(d$grade)), c("A", "B", "C", "D", "E")),
            identical(names(top(d$member)), c("TRUE", "FALSE")),
            sum(!is.na(d$income)) == 12L, sum(!is.na(d$age)) == 14L)
  list(values = list(
         id = named(num(d$id)),
         income = named(num(d$income)),
         age = named(num(d$age)),
         region_counts = named(cnt(d$region)),
         region_top = named(top(d$region)),
         grade_counts = named(cnt(d$grade)),
         grade_top = named(top(d$grade)),
         member_counts = named(cnt(d$member)),
         member_top = named(top(d$member)),
         empty_counts = named(cnt(d$empty))),
       reference = "summary(x); sd(x, na.rm = TRUE); quantile(x, type = 7, na.rm = TRUE); head(sort(table(x), decreasing = TRUE), 5)",
       call = "lapply(d[c('id', 'income', 'age')], function(v) c(sum(!is.na(v)), sum(is.na(v)), summary(v), sd(v, na.rm = TRUE))); head(sort(table(d$region), decreasing = TRUE), 5)",
       packages = character(0))
})

case("viz_dendrogram_hclust", "visualization/dendrogram",
     "stats::hclust(dist(x)) merge matrix, heights and cluster sizes (complete linkage), for the 0-based linkage conversion",
     function(ctx) {
  set.seed(5902)
  d <- ctx$gen_dataset("dendrogram_points", data.frame(id = 1:7, x = round(cumsum(runif(7, 0.5, 3)), 3)))
  hc <- stats::hclust(stats::dist(d$x), method = "complete")
  n <- nrow(d)
  size <- integer(n - 1L)
  for (i in seq_len(n - 1L)) {
    s <- vapply(hc$merge[i, ], function(m) if (m < 0) 1L else size[m], 1L)
    size[i] <- sum(s)
  }
  stopifnot(size[n - 1L] == n, all(diff(hc$height) >= 0))
  list(values = list(
         n = n,
         merge_left = vec(hc$merge[, 1]),
         merge_right = vec(hc$merge[, 2]),
         height = vec(hc$height),
         size = vec(size)),
       reference = "stats::hclust(stats::dist(x), method = 'complete'): merge, height; sizes from the merge matrix",
       call = "hclust(dist(d$x))",
       packages = character(0))
})
