# validation/r/run.R -- compute or check the R reference cases.
#   Rscript validation/r/run.R --check [ids...]   exit 1 if any recomputed case differs from the committed file
#   Rscript validation/r/run.R --write [ids...]   write validation/expected/<id>.json and validation/datasets/generated/*.csv
#   Rscript validation/r/run.R --list
args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[1] else "--check"
ids <- args[-1]
here <- local({ f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)); dirname(normalizePath(f)) })
source(file.path(here, "harness.R"))
for (f in sort(list.files(file.path(here, "cases"), pattern = "\\.R$", full.names = TRUE))) source(f)
root <- normalizePath(file.path(here, ".."))   # validation/ (run.R lives in validation/r/)
sel <- if (length(ids)) ids else names(.harness$cases)
unknown <- setdiff(sel, names(.harness$cases))
if (length(unknown)) stop("unknown case ids: ", paste(unknown, collapse = ", "))

if (mode == "--list") { for (id in sel) cat(id, "\t", .harness$cases[[id]]$method, "\n"); quit(status = 0) }
if (!mode %in% c("--check", "--write")) stop("mode must be --check, --write or --list")

if (mode == "--write") {
  out_dir <- file.path(root, "expected"); gen_dir <- file.path(root, "datasets", "generated")
} else {
  tmp <- tempfile("p2a-r-check-"); out_dir <- file.path(tmp, "expected"); gen_dir <- file.path(tmp, "generated")
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE); dir.create(gen_dir, recursive = TRUE, showWarnings = FALSE)

failures <- character(0)
for (id in sel) {
  ok <- tryCatch({ run_case(.harness$cases[[id]], out_dir, gen_dir, root); TRUE },
                 error = function(e) { failures <<- c(failures, sprintf("%s: R error: %s", id, conditionMessage(e))); FALSE })
  if (!ok || mode == "--write") { cat(if (ok) "wrote " else "FAILED ", id, "\n", sep = ""); next }
  committed <- file.path(root, "expected", paste0(id, ".json"))
  if (!file.exists(committed)) { failures <- c(failures, sprintf("%s: no committed file %s", id, committed)); next }
  d <- compare_case_files(committed, file.path(out_dir, paste0(id, ".json")))
  if (length(d)) failures <- c(failures, paste0(id, ": ", d)) else cat("ok     ", id, "\n", sep = "")
}
if (mode == "--check" && !length(ids)) {
  orphans <- setdiff(sub("\\.json$", "", list.files(file.path(root, "expected"), pattern = "\\.json$")), names(.harness$cases))
  if (length(orphans)) failures <- c(failures, sprintf("committed case %s has no definition in validation/r/cases/", orphans))
  stale <- setdiff(list.files(file.path(root, "datasets", "generated"), pattern = "\\.csv$"), list.files(gen_dir, pattern = "\\.csv$"))
  if (length(stale)) failures <- c(failures, sprintf("generated dataset %s is not produced by any case", stale))
}
if (mode == "--check") {
  for (f in list.files(gen_dir, pattern = "\\.csv$")) {
    committed <- file.path(root, "datasets", "generated", f)
    if (!file.exists(committed)) failures <- c(failures, sprintf("generated dataset %s is not committed", f))
    else if (!identical(readLines(committed), readLines(file.path(gen_dir, f)))) failures <- c(failures, sprintf("generated dataset %s differs from the committed file", f))
  }
}
if (length(failures)) { cat("\nR reference check FAILED:\n", paste0("  ", failures, collapse = "\n"), "\n", sep = ""); quit(status = 1) }
cat(sprintf("\nR reference %s: %d case(s) %s.\n", sub("--", "", mode), length(sel), if (mode == "--check") "match the committed files" else "written"))
