# validation/r/cases/data_na_strings.R -- how R reads missing-value markers in a CSV (T2-WP03),
# the reference for `p2a data load` and its --na-strings option. The columns_conventions file
# (data_columns.R, which sorts first, so `--write` with no ids writes that file before this case
# reads it) is coded twice: every empty field becomes `NA` (write.csv's marker) or `.` (Stata
# and SAS exports). R reads the first with read.csv()'s default na.strings, the second with
# na.strings = "." and once more with the default, where "." is text. The p2a test derives the
# same two copies from the same file.

case("cli_na_strings", "data/na_strings", "read.csv na.strings on NA- and .-coded copies of a CSV", function(ctx) {
  canon <- ctx$dataset("datasets/generated/columns_conventions.csv")   # na.strings = c("NA", "")
  src <- tempfile(fileext = ".csv"); on.exit(unlink(src))
  .write_csv17(canon, src)   # the writer that wrote the case file: the same text
  coded <- function(marker) {
    lines <- readLines(src)
    body <- vapply(lines[-1], function(l) {
      f <- strsplit(paste0(l, ","), ",", fixed = TRUE)[[1]]   # the extra "," keeps a trailing empty field
      stopifnot(length(f) == ncol(canon))
      f[f == ""] <- marker
      paste(f, collapse = ",")
    }, "", USE.NAMES = FALSE)
    p <- tempfile(fileext = ".csv")
    writeLines(c(lines[1], body), p)
    p
  }
  na_file <- coded("NA"); dot_file <- coded(".")
  on.exit(unlink(c(na_file, dot_file)), add = TRUE)
  stopifnot(any(grepl(",NA(,|$)", readLines(na_file))), any(grepl(",\\.(,|$)", readLines(dot_file))))
  d_na  <- utils::read.csv(na_file, stringsAsFactors = FALSE)                     # na.strings = "NA"
  d_dot <- utils::read.csv(dot_file, stringsAsFactors = FALSE, na.strings = ".")
  d_raw <- utils::read.csv(dot_file, stringsAsFactors = FALSE)                    # "." is not missing
  # Column classes as codes: integer 0, numeric 1, character 2, logical 3.
  classes <- c("integer", "numeric", "character", "logical")
  class_codes <- function(d) {
    k <- match(vapply(d, function(x) class(x)[1], ""), classes) - 1
    stopifnot(!anyNA(k))
    named(stats::setNames(k, names(d)))
  }
  missing <- function(d) named(colSums(is.na(d)))
  list(values = list(
    canon_class = class_codes(canon), na_class = class_codes(d_na),
    dot_class = class_codes(d_dot), raw_class = class_codes(d_raw),
    canon_missing = missing(canon), na_missing = missing(d_na),
    dot_missing = missing(d_dot), raw_missing = missing(d_raw),
    na_identical = as.numeric(identical(d_na, canon)), dot_identical = as.numeric(identical(d_dot, canon))),
    reference = "utils::read.csv (default na.strings = \"NA\"; na.strings = \".\") on NA- and .-coded copies of the case file; identical() to read.csv(case file, na.strings = c(\"NA\", \"\"))",
    call = "read.csv(na_file); read.csv(dot_file, na.strings = \".\"); read.csv(dot_file)",
    packages = character(0))
})
