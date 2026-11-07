# validation/r/cases/data_readers.R -- haven (ReadStat) reads of the committed Stata and SAS reader
# fixtures (T3-WP51: DATA-50, DATA-73; DATA-18, DATA-19, DATA-20, DATA-21, DATA-22, DATA-47, DATA-48,
# DATA-51). The fixtures are binary files in validation/datasets/files/, written by
# validation/r/fixtures/data_readers.R and listed in data_readers_manifest.csv there, which is this
# case's analysis dataset. Per fixture <f> (file stem) the values are:
#   <f>__shape            c(rows, columns)
#   <f>__columns          named: column names in file order -> 1 numeric, 0 text
#   <f>__<column>         numeric column: the values (NA and every tagged NA -> null);
#                         text column: named, names = the strings haven returns, values = their
#                         length in UTF-8 bytes (0 for "")
# and, for the corrupted copies the Rust guard tests make, what ReadStat does with the same bytes:
#   guard__<what>         the number of rows ReadStat reads, or -1 when it refuses the file.
suppressPackageStartupMessages(library(haven))

.readers_dir <- function(ctx) {
  # validation/: the directory ctx$dataset() resolves against (harness.R, run_case(data_root)).
  file.path(get("data_root", envir = environment(ctx$dataset)), "datasets", "files")
}
.readers_values <- function(prefix, d) {
  d <- as.data.frame(zap_formats(zap_labels(d)), stringsAsFactors = FALSE)
  v <- list()
  v[[paste0(prefix, "__shape")]] <- vec(c(nrow(d), ncol(d)))
  v[[paste0(prefix, "__columns")]] <- named(setNames(as.numeric(vapply(d, is.numeric, TRUE)), names(d)))
  for (nm in names(d)) {
    x <- d[[nm]]
    v[[paste0(prefix, "__", nm)]] <- if (is.numeric(x)) vec(as.numeric(x)) else {
      stopifnot(is.character(x), !anyNA(x), all(validUTF8(x)))
      named(setNames(as.numeric(nchar(x, "bytes")), enc2utf8(x)))
    }
  }
  v
}
# A copy of fixture `file` with each patch (0-based offset `at`, then the bytes) written in, read
# by `reader`: the number of rows, or -1 when ReadStat refuses it.
.patched_rows <- function(dir, file, patches, reader) {
  r <- readBin(file.path(dir, file), "raw", file.size(file.path(dir, file)))
  for (p in patches) r[p$at + seq_along(p$bytes)] <- as.raw(p$bytes)
  tmp <- tempfile(fileext = sub(".*(\\.[a-z0-9]+)$", "\\1", file)); on.exit(unlink(tmp)); writeBin(r, tmp)
  out <- -1
  utils::capture.output(out <- tryCatch(nrow(reader(tmp)), error = function(e) -1))   # ReadStat prints its diagnostics
  out
}
.le <- function(v, n) { b <- integer(n); for (k in seq_len(n)) { b[k] <- v %% 256; v <- v %/% 256 }; b }

case("data_readers_values", "data/readers",
     "Stata .dta 117-119 (LSF, MSF, Windows-1252, Stata-written, byte/int/long/float/double) and SAS .sas7bdat (32/64-bit, header pad, short numerics, big-endian, Windows-1252, RLE, RDC) as read by haven",
     function(ctx) {
  m <- ctx$dataset("datasets/files/data_readers_manifest.csv")
  dir <- .readers_dir(ctx)
  vals <- list()
  for (i in seq_len(nrow(m))) {
    f <- file.path(dir, m$file[i]); stopifnot(file.exists(f))
    enc <- m$encoding[i]                                   # "" (or NA, with T2's na.strings): the file's own
    d <- if (m$format[i] == "dta") read_dta(f, encoding = if (is.na(enc) || enc == "") NULL else enc) else read_sas(f)
    vals <- c(vals, .readers_values(sub("\\.(dta|sas7bdat)$", "", m$file[i]), d))
  }

  # Layout facts the Rust tests rely on (0-based offsets; the Rust test_*_fixture_offsets tests check
  # the same bytes). sas64_haven: page 0 at 8192 (row size subheader at +3288, column attributes at
  # +2912, column names at +3084), page 1 at 12288; sas32_iris: one mix page at 65536 (row size at
  # +65056, column attributes at +64456); dta118_haven: <K> value at 70, <N> value at 79.
  rd <- function(file) readBin(file.path(dir, file), "raw", file.size(file.path(dir, file)))
  h <- rd("sas64_haven.sas7bdat"); ir <- rd("sas32_iris.sas7bdat"); dt <- rd("dta118_haven.dta")
  stopifnot(all(h[8192 + 3288 + 1:4] == as.raw(0xF7)), identical(h[8192 + 2912 + 1:4], as.raw(c(0xFC, 0xFF, 0xFF, 0xFF))),
            all(h[8192 + 3084 + 1:4] == as.raw(0xFF)), identical(h[12288 + 32 + 1:2], as.raw(c(0x00, 0x01))),
            all(ir[65536 + 65056 + 1:4] == as.raw(0xF7)), identical(ir[65536 + 64456 + 1:4], as.raw(c(0xFC, 0xFF, 0xFF, 0xFF))),
            identical(ir[65536 + 16 + 1:2], as.raw(c(0x00, 0x02))),
            rawToChar(dt[68:70]) == "<K>", rawToChar(dt[77:79]) == "<N>")

  # What ReadStat does with the corrupted copies of the Rust guard tests.
  rs <- function(file, at, bytes) .patched_rows(dir, file, list(list(at = at, bytes = bytes)), read_sas)
  vals$guard__sas_page_type_0x0280 <- rs("sas32_iris.sas7bdat", 65536 + 16, .le(0x0280, 2))     # a mix page
  vals$guard__sas_page_type_0x4000 <- rs("sas64_haven.sas7bdat", 8192 + 32, .le(0x4000, 2))     # a metadata page
  vals$guard__sas_page_type_0x0300 <- rs("sas64_haven.sas7bdat", 12288 + 32, .le(0x0300, 2))    # unknown
  vals$guard__sas_row_count_59 <- rs("sas64_haven.sas7bdat", 8192 + 3288 + 48, .le(59, 8))
  vals$guard__sas_row_count_61 <- rs("sas64_haven.sas7bdat", 8192 + 3288 + 48, .le(61, 8))
  vals$guard__sas_type_code_3 <- rs("sas64_haven.sas7bdat", 8192 + 2912 + 16 + 14, 3)
  vals$guard__sas_char_only_row_count_2 <- local({                                   # every column text, 2 rows
    r <- rd("sas32_iris.sas7bdat"); for (j in 0:4) r[65536 + 64456 + 12 + 12 * j + 10 + 1] <- as.raw(2)
    r[65536 + 65056 + 24 + 1:4] <- as.raw(.le(2, 4))
    tmp <- tempfile(fileext = ".sas7bdat"); on.exit(unlink(tmp)); writeBin(r, tmp)
    d <- read_sas(tmp); stopifnot(all(vapply(d, is.character, TRUE))); nrow(d)
  })
  vals$guard__dta_k0_n2p40 <- .patched_rows(dir, "dta118_haven.dta",                 # K = 0, N = 2^40
                                             list(list(at = 70, bytes = c(0, 0)), list(at = 79, bytes = .le(2^40, 8))), read_dta)
  vals$guard__dta_k0_n3 <- .patched_rows(dir, "dta118_haven.dta",                     # K = 0, N = 3
                                         list(list(at = 70, bytes = c(0, 0)), list(at = 79, bytes = .le(3, 8))), read_dta)
  # The encoding byte (70) of sas64_wlatin1 (62, Windows-1252; its first grp cell "Z\u00fcrich" at
  # 12376): ReadStat refuses the five bytes Windows-1252 leaves unassigned, reads code 204 as
  # Windows-1252 too, and code 29 as ISO-8859-1 (0x80 -> U+0080).
  vals$guard__sas_wlatin1_byte_0x81 <- rs("sas64_wlatin1.sas7bdat", 12376, 0x81)
  vals$guard__sas_wlatin1_code_204 <- rs("sas64_wlatin1.sas7bdat", 70, 204)
  local({
    r <- rd("sas64_wlatin1.sas7bdat"); r[70 + 1] <- as.raw(29)
    tmp <- tempfile(fileext = ".sas7bdat"); on.exit(unlink(tmp)); writeBin(r, tmp)
    g <- read_sas(tmp)$grp; stopifnot(all(validUTF8(g)))
    vals$sas64_wlatin1_code29__grp <<- named(setNames(as.numeric(nchar(g, "bytes")), enc2utf8(g)))
  })
  for (k in names(vals)[startsWith(names(vals), "guard__")]) vals[[k]] <- as.numeric(vals[[k]])
  stopifnot(vals$guard__sas_page_type_0x0280 == 150, vals$guard__sas_page_type_0x4000 == 60,
            vals$guard__sas_page_type_0x0300 == -1, vals$guard__sas_row_count_59 == 59,
            vals$guard__sas_row_count_61 == -1, vals$guard__sas_type_code_3 == -1,
            vals$guard__sas_char_only_row_count_2 == 2, vals$guard__dta_k0_n2p40 == -1, vals$guard__dta_k0_n3 == -1,
            vals$guard__sas_wlatin1_byte_0x81 == -1, vals$guard__sas_wlatin1_code_204 == 60,
            utf8ToInt(substr(vals$sas64_wlatin1_code29__grp$names[5], 1, 1)) == 0x80)

  # Properties the fixtures exist to exercise (the reference is haven; these guard the fixtures).
  stopifnot(identical(vals$dta117_latin1__ss$names, vals$dta118_haven__ss$names),
            identical(vals$dta119_msf__xd, vals$dta119_haven__xd),
            vals$dta118_types__vf[2] == 1.7014117331926443e38,                # float max, not missing
            vals$dta118_types__vd[2] == 8.9884656743115785e307,               # double max, not missing
            all(is.na(vals$dta118_types__vb[3:4])),
            identical(vals$sas64_rle__income, vals$sas64_haven__income),
            identical(vals$sas64_rdc__code$names, vals$sas64_haven__code$names),
            identical(vals$sas64_wlatin1__grp$names, vals$sas64_haven__grp$names),
            identical(vals$sas32_nopad_rle__Sepal_Length, vals$sas32_iris__Sepal_Length),
            identical(vals$sas64_short_be__income, vals$sas64_short__income),
            vals$sas64_haven__ratio[11] == 1e308,                                       # a valid SAS value
            vals$sas64_short__ratio[12] != vals$sas64_haven__ratio[12])                # -0.1 truncated to 6 bytes
  list(values = vals,
       reference = "haven::read_dta(file, encoding = <manifest encoding>) and haven::read_sas(file) (ReadStat bundled with haven)",
       call = "for (f in manifest$file) read_dta(f, encoding = enc) | read_sas(f); guard__*: nrow(read_sas|read_dta(<patched copy>)) or -1",
       packages = "haven")
})
