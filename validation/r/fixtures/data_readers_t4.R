# validation/r/fixtures/data_readers_t4.R -- writes the Stata fixtures of the format, value-label and
# strL decoding (DATA-74, DATA-75) and the one-byte-row fixture of the cell-check guard test into
# validation/datasets/files/.
#
#   nice -n 19 Rscript validation/r/fixtures/data_readers_t4.R
#
# Not part of `run.R --check`, for the reason validation/r/fixtures/data_readers.R gives (haven
# stamps the write time into every file). The case validation/r/cases/data_readers_t4.R reads the
# committed files back with haven and records the values; after a rewrite, run
# `Rscript validation/r/run.R --write data_readers_formats_values` and commit both.
#
# dta11{7,8,9}_formats.dta are written by haven; dta119_formats_msf.dta is dta119_formats.dta
# converted to big-endian here, and haven must read it exactly as its source (stopifnot below).
suppressPackageStartupMessages(library(haven))
stopifnot(packageVersion("haven") >= "2.5.0")

here <- local({ f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)); dirname(normalizePath(f)) })
out_dir <- normalizePath(file.path(here, "..", "..", "datasets", "files"), mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fx <- function(name) file.path(out_dir, name)

rd_u <- function(raw, pos, n, big = FALSE) {
  b <- as.integer(raw[pos:(pos + n - 1L)]); if (big) b <- rev(b)
  sum(b * 256^(seq_len(n) - 1L))
}
find_bytes <- function(raw, pat, from = 1L) {
  pat <- charToRaw(pat); n <- length(pat)
  for (i in from:(length(raw) - n + 1L)) if (raw[i] == pat[1] && all(raw[i:(i + n - 1L)] == pat)) return(i)
  stop("pattern not found: ", rawToChar(pat))
}

# ---------------------------------------------------------------------------------------------
# The formats frame: a date (%td), a datetime (%tc), two value-labelled columns (integer and
# double codes), a monthly date (%tm, kept numeric by the reader) and a strL column (strl_threshold
# = 10 makes every string column longer than 10 bytes a strL; haven refuses NA in a strL write).
df <- data.frame(id = 1:5)
df$dt <- as.Date(c("1960-01-01", "2024-02-29", NA, "1899-12-31", "2262-04-11"))
df$tc <- .POSIXct(c(1705322096789, -315619200000, NA, -315619201000, 951782400001) / 1000, tz = "UTC")
df$gl <- labelled(c(1L, 2L, 3L, NA, 1L), c(low = 1L, high = 2L))
df$gd <- labelled(c(10, 20, 10, 30, NA), c(ten = 10, twenty = 20, "über" = 30))
df$mo <- structure(c(0, 1, 779, NA, -1), format.stata = "%tm")
df$sl <- c("short", strrep("x", 3000), "", "short", "été")

write_dta(df, fx("dta117_formats.dta"), version = 13, strl_threshold = 10)
write_dta(df, fx("dta118_formats.dta"), version = 14, strl_threshold = 10)
write_dta(df, fx("dta119_formats.dta"), version = 15, strl_threshold = 10)

# dta119_formats_msf.dta: dta119_formats.dta in big-endian (MSF). Every integer and float of the
# header, map, type list, sort list, data rows (strL references per field: v u16, o 6 bytes),
# <strls> (GSO v u32, o u64, length u32) and <value_labels> (len, n, txtlen, off[], val[] as i32)
# is byte-swapped; text is unchanged.
local({
  raw <- readBin(fx("dta119_formats.dta"), "raw", file.size(fx("dta119_formats.dta")))
  stopifnot(rawToChar(raw[29:31]) == "119", rawToChar(raw[53:55]) == "LSF")
  swap <- function(raw, pos, n) { raw[pos:(pos + n - 1)] <- rev(raw[pos:(pos + n - 1)]); raw }
  pk <- find_bytes(raw, "<K>") + 3L; K <- rd_u(raw, pk, 4L)
  pn <- find_bytes(raw, "<N>") + 3L; N <- rd_u(raw, pn, 8L)
  pm <- find_bytes(raw, "<map>") + 5L
  map <- vapply(0:13, function(i) rd_u(raw, pm + 8L * i, 8L), 0)
  vt <- map[3] + 1 + nchar("<variable_types>")
  types <- vapply(0:(K - 1), function(i) rd_u(raw, vt + 2 * i, 2L), 0)
  width <- unname(ifelse(types <= 2045, types,
                         c(`65530` = 1, `65529` = 2, `65528` = 4, `65527` = 4, `65526` = 8, `32768` = 8)[as.character(types)]))
  data <- map[10] + 1 + nchar("<data>")
  raw[53:55] <- charToRaw("MSF")
  raw <- swap(raw, pk, 4L); raw <- swap(raw, pn, 8L)
  pl <- find_bytes(raw, "<label>") + 7L; raw <- swap(raw, pl, 2L)
  for (i in 0:13) raw <- swap(raw, pm + 8L * i, 8L)
  for (i in 0:(K - 1)) raw <- swap(raw, vt + 2 * i, 2L)
  sl <- map[5] + 1 + nchar("<sortlist>")
  for (i in 0:K) raw <- swap(raw, sl + 4 * i, 4L)
  rs <- sum(width); off <- cumsum(c(0, width))
  for (r in 0:(N - 1)) for (j in seq_len(K)) {
    at <- data + r * rs + off[j]
    if (types[j] == 32768) { raw <- swap(raw, at, 2L); raw <- swap(raw, at + 2, 6L) }
    else if (types[j] > 2045 && width[j] > 1) raw <- swap(raw, at, width[j])
  }
  # <strls>: GSO blocks up to </strls>.
  p <- map[11] + 1 + nchar("<strls>"); end <- map[12] + 1 - nchar("</strls>")
  while (p < end) {
    stopifnot(rawToChar(raw[p:(p + 2)]) == "GSO")
    len <- rd_u(raw, p + 16, 4L)
    raw <- swap(raw, p + 3, 4L); raw <- swap(raw, p + 7, 8L); raw <- swap(raw, p + 16, 4L)
    p <- p + 20 + len
  }
  # <value_labels>: <lbl> len i32, labname 129, 3 padding, then the table of len bytes.
  p <- map[12] + 1 + nchar("<value_labels>"); end <- map[13] + 1 - nchar("</value_labels>")
  while (p < end) {
    stopifnot(rawToChar(raw[p:(p + 4)]) == "<lbl>")
    len <- rd_u(raw, p + 5, 4L); raw <- swap(raw, p + 5, 4L)
    t <- p + 5 + 4 + 129 + 3
    n <- rd_u(raw, t, 4L)
    for (i in 0:(1 + 2 * n)) raw <- swap(raw, t + 4 * i, 4L)
    p <- t + len + nchar("</lbl>")
  }
  writeBin(raw, fx("dta119_formats_msf.dta"))
})

local({
  same <- function(a, b) {
    a <- as.data.frame(a); b <- as.data.frame(b)
    identical(names(a), names(b)) && all(vapply(names(a), function(nm) identical(a[[nm]], b[[nm]]), TRUE))
  }
  ref <- read_dta(fx("dta118_formats.dta"))
  stopifnot(same(read_dta(fx("dta117_formats.dta"), encoding = "UTF-8"), ref),
            same(read_dta(fx("dta119_formats.dta")), ref),
            same(read_dta(fx("dta119_formats_msf.dta")), ref),
            identical(as.vector(zap_formats(ref$sl)), df$sl))
})

# dta118_str1.dta: one str1 column of three rows, a 1-byte row. The cell-check guard test declares
# many rows of it, which needs far less memory than the 57-byte rows of dta118_haven.dta.
write_dta(data.frame(s1 = c("a", "b", "c")), fx("dta118_str1.dta"), version = 14)
local({
  raw <- readBin(fx("dta118_str1.dta"), "raw", file.size(fx("dta118_str1.dta")))
  stopifnot(rawToChar(raw[68:70]) == "<K>", rawToChar(raw[77:79]) == "<N>", rawToChar(raw[159:163]) == "<map>",
            identical(as.vector(zap_formats(read_dta(fx("dta118_str1.dta"))$s1)), c("a", "b", "c")))
})
