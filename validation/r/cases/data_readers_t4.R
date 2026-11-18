# validation/r/cases/data_readers_t4.R -- haven (ReadStat) reads of the Stata format, value-label and
# strL fixtures (T4: DATA-74, DATA-75), without zapping formats and labels. The fixtures are written
# by validation/r/fixtures/data_readers_t4.R and listed in data_readers_t4_manifest.csv, this case's
# analysis dataset. Per fixture <f> (file stem):
#   <f>__id, <f>__mo          the numeric values (NA -> null)
#   <f>__dt                   the %td column as days since 1970-01-01
#   <f>__tc                   the %tc column as milliseconds since 1970-01-01 00:00 UTC, rounded
#                             (haven's double seconds print 951782400001 as ...400000.99988)
#   <f>__gl, <f>__gd          the value-labelled codes
#   <f>__gl_text, <f>__gd_text   as.character(as_factor(x)) of the labelled columns: named, names =
#                             the text ("" for NA), values = 1 when NA, else 0
#   <f>__sl                   the strL column: named, names = the strings, values = UTF-8 bytes
#   <f>__formats              named: names = format.stata per column, values = the column's position
#   <f>__labels_gl, <f>__labels_gd   named: names = the labels, values = their codes
suppressPackageStartupMessages(library(haven))

.readers_t4_dir <- function(ctx) {
  file.path(get("data_root", envir = environment(ctx$dataset)), "datasets", "files")
}

case("data_readers_formats_values", "data/readers",
     "Stata .dta 117-119 (LSF and MSF) display formats (%td, %tc, %tm), value labels and strL text as read by haven",
     function(ctx) {
  m <- ctx$dataset("datasets/files/data_readers_t4_manifest.csv")
  dir <- .readers_t4_dir(ctx)
  vals <- list()
  for (i in seq_len(nrow(m))) {
    f <- file.path(dir, m$file[i]); stopifnot(file.exists(f), m$format[i] == "dta")
    enc <- m$encoding[i]
    d <- read_dta(f, encoding = if (is.na(enc) || enc == "") NULL else enc)
    p <- sub("\\.dta$", "", m$file[i])
    stopifnot(identical(names(d), c("id", "dt", "tc", "gl", "gd", "mo", "sl")))
    put <- function(key, v) vals[[paste0(p, "__", key)]] <<- v
    put("id", vec(as.numeric(d$id)))
    put("mo", vec(as.numeric(zap_formats(d$mo))))
    stopifnot(inherits(d$dt, "Date"), inherits(d$tc, "POSIXct"))
    put("dt", vec(as.numeric(d$dt)))
    put("tc", vec(round(as.numeric(d$tc) * 1000)))
    for (nm in c("gl", "gd")) {
      x <- d[[nm]]; stopifnot(is.labelled(x))
      put(nm, vec(as.numeric(zap_labels(zap_formats(x)))))
      txt <- as.character(as_factor(x))
      put(paste0(nm, "_text"), named(setNames(as.numeric(is.na(txt)), enc2utf8(ifelse(is.na(txt), "", txt)))))
      lab <- attr(x, "labels")
      put(paste0("labels_", nm), named(setNames(as.numeric(lab), enc2utf8(names(lab)))))
    }
    sl <- as.vector(zap_formats(d$sl)); stopifnot(is.character(sl), !anyNA(sl), all(validUTF8(sl)))
    put("sl", named(setNames(as.numeric(nchar(sl, "bytes")), enc2utf8(sl))))
    fmt <- vapply(d, function(x) attr(x, "format.stata"), "")
    put("formats", named(setNames(as.numeric(seq_along(fmt)), unname(fmt))))
  }
  stopifnot(identical(vals$dta119_formats_msf__tc, vals$dta119_formats__tc),
            identical(vals$dta117_formats__sl, vals$dta118_formats__sl),
            vals$dta118_formats__sl$values[2] == 3000)
  list(values = vals,
       reference = "haven::read_dta(file, encoding = <manifest encoding>) (ReadStat bundled with haven), formats and labels kept",
       call = "for (f in manifest$file) read_dta(f, encoding = enc); as_factor() for the labelled columns",
       packages = "haven")
})
