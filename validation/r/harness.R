# validation/r/harness.R -- shared helpers for R reference cases.
#
# A *case* is one R reference computation on one committed dataset. It is
# written to validation/expected/<id>.json (schema 1) and read by the Rust
# crate p2a-validation. Datasets that R simulates are written to
# validation/datasets/generated/<name>.csv with 17 significant digits and READ BACK
# before any reference value is computed, so R and Rust see identical doubles.
#
# Usage (from any directory; paths resolve relative to run.R):
#   Rscript validation/r/run.R --check            # recompute all cases into a temp dir, compare with committed files
#   Rscript validation/r/run.R --write [ids...]   # (re)write committed JSON and generated datasets
#   Rscript validation/r/run.R --list
suppressPackageStartupMessages(library(jsonlite))

.harness <- new.env()
.harness$cases <- list()

# Register a case. `fn` receives a context `ctx` with ctx$dataset(rel) (read a
# committed CSV), ctx$gen_dataset(name, df) (write + read back a simulated CSV)
# and must return list(values = <named list>, reference = <chr>, call = <chr>,
# packages = <chr vector; character(0) for base R only>).
case <- function(id, area, method, fn) {
  stopifnot(grepl("^[a-z0-9_]+$", id), is.null(.harness$cases[[id]]))
  .harness$cases[[id]] <- list(id = id, area = area, method = method, fn = fn)
  invisible(id)
}

# A vector value with names (coefficients, SEs, CIs).
named <- function(x) {
  stopifnot(is.numeric(x), !is.null(names(x)))
  list(names = I(unname(names(x))), values = I(unname(as.numeric(x))))   # I(): stay arrays at length 1
}
# An unnamed vector that must stay a JSON array even at length 1.
vec <- function(x) { stopifnot(is.numeric(x)); I(unname(as.numeric(x))) }

# Shortest of %.15g / %.16g / %.17g that reads back as the identical double.
.fmt_double <- function(x) {
  vapply(x, function(v) {
    if (is.na(v)) return(NA_character_)
    for (d in 15:17) { s <- sprintf(paste0("%.", d, "g"), v); if (as.numeric(s) == v) return(s) }
    sprintf("%.17g", v)
  }, "")
}
.write_csv17 <- function(df, path) {
  out <- as.data.frame(df, stringsAsFactors = FALSE)
  is_chr <- vapply(out, function(col) is.character(col) || is.factor(col), TRUE)
  for (nm in names(out)[!is_chr]) if (is.double(out[[nm]])) out[[nm]] <- .fmt_double(out[[nm]])
  # Quote only text columns, so numeric columns stay numeric for every CSV reader.
  utils::write.csv(out, path, row.names = FALSE, na = "", quote = which(is_chr))
}

.validate_values <- function(id, values) {
  if (!is.list(values) || length(values) == 0L || is.null(names(values)) || any(names(values) == ""))
    stop(sprintf("case %s: values must be a non-empty named list", id))
  for (k in names(values)) {
    v <- values[[k]]
    nums <- if (is.list(v)) v$values else v
    if (!is.numeric(nums) || length(nums) == 0L) stop(sprintf("case %s: value '%s' is not numeric", id, k))
    if (any(is.infinite(nums))) stop(sprintf("case %s: value '%s' is infinite", id, k))
    if (is.list(v) && length(v$names) != length(v$values)) stop(sprintf("case %s: '%s' names/values length differ", id, k))
  }
}

run_case <- function(c, out_dir, gen_dir, data_root) {
  used <- character(0)
  ctx <- list(
    dataset = function(rel) {
      p <- file.path(data_root, rel)
      if (!file.exists(p)) stop(sprintf("case %s: dataset %s not found", c$id, rel))
      used <<- c(used, file.path("validation", rel))
      utils::read.csv(p, stringsAsFactors = FALSE)
    },
    gen_dataset = function(name, df) {
      stopifnot(grepl("^[a-z0-9_]+$", name))
      p <- file.path(gen_dir, paste0(name, ".csv"))
      .write_csv17(df, p)
      used <<- c(used, file.path("validation/datasets/generated", paste0(name, ".csv")))
      utils::read.csv(p, stringsAsFactors = FALSE)   # compute on what Rust will read
    }
  )
  res <- c$fn(ctx)
  .validate_values(c$id, res$values)
  pk <- res$packages
  # NULL would be written as "packages": [], which Rust and the lint reject;
  # character(0) (a base-R-only case) is written as "packages": {}.
  if (is.null(pk) || !is.character(pk) || anyNA(pk) || !all(nzchar(pk)))
    stop(sprintf("case %s: packages must be a character vector of R package names (character(0) for base R only)", c$id))
  # The analysis dataset: explicit, or the last dataset the case touched.
  ds <- if (!is.null(res$dataset)) res$dataset else utils::tail(used, 1)
  if (length(ds) != 1L || !(ds %in% used)) stop(sprintf("case %s: analysis dataset must be one of the datasets it read", c$id))
  doc <- list(
    schema = 1L, id = c$id, area = c$area, method = c$method,
    reference = res$reference, call = res$call,
    dataset = ds,
    datasets = I(unique(used)),
    r_version = R.version.string,
    packages = as.list(setNames(vapply(pk, function(p) as.character(utils::packageVersion(p)), ""), pk)),
    values = res$values
  )
  json <- jsonlite::toJSON(doc, digits = I(17), auto_unbox = TRUE, na = "null", pretty = TRUE, null = "null")
  writeLines(json, file.path(out_dir, paste0(c$id, ".json")))
  invisible(doc)
}

# Numeric comparison of two case documents (used by --check). Returns a
# character vector of differences (empty = identical within rel 1e-12). Each
# value must also keep its kind (scalar, unnamed array or {names, values}), its
# names and the position of every null (R NA).
compare_case_files <- function(a_path, b_path, rel = 1e-12) {
  a <- jsonlite::fromJSON(a_path, simplifyVector = FALSE)
  b <- jsonlite::fromJSON(b_path, simplifyVector = FALSE)
  diffs <- character(0)
  ka <- names(a$values); kb <- names(b$values)
  if (!setequal(ka, kb)) diffs <- c(diffs, sprintf("value keys differ: %s vs %s", paste(ka, collapse = ","), paste(kb, collapse = ",")))
  kind <- function(v) if (!is.list(v)) "scalar" else if (is.null(names(v))) "array" else "named"
  # fromJSON reads null as NULL, which unlist() would drop: map it to NA in place.
  num <- function(v) {
    vals <- if (kind(v) == "named") v$values else v
    if (is.null(vals)) return(NA_real_)
    vapply(vals, function(z) if (is.null(z)) NA_real_ else as.numeric(z), 0)
  }
  nms <- function(v) vapply(v$names, function(z) if (is.null(z)) NA_character_ else as.character(z), "")
  for (k in intersect(ka, kb)) {
    va <- a$values[[k]]; vb <- b$values[[k]]
    if (kind(va) != kind(vb)) { diffs <- c(diffs, sprintf("%s: kind differs: %s vs %s", k, kind(va), kind(vb))); next }
    if (kind(va) == "named" && !identical(nms(va), nms(vb)))
      diffs <- c(diffs, sprintf("%s: names differ: %s vs %s", k, paste(nms(va), collapse = ","), paste(nms(vb), collapse = ",")))
    x <- num(va); y <- num(vb)
    if (length(x) != length(y)) { diffs <- c(diffs, sprintf("%s: length %d vs %d", k, length(x), length(y))); next }
    bad <- which(xor(is.na(x), is.na(y)) | (!is.na(x) & !is.na(y) & abs(x - y) > rel * pmax(abs(x), abs(y), 1e-300)))
    if (length(bad)) diffs <- c(diffs, sprintf("%s[%s]: committed %s, recomputed %s", k, paste(bad, collapse = ","),
                                              paste(format(x[bad], digits = 17), collapse = ","), paste(format(y[bad], digits = 17), collapse = ",")))
  }
  diffs
}
