# validation/r/fixtures/data_readers.R -- writes the binary fixtures of the Stata and SAS readers
# (T3-WP51: DATA-50; DATA-18..22, DATA-47, DATA-48, DATA-51, DATA-73) into validation/datasets/files/.
#
#   Rscript validation/r/fixtures/data_readers.R
#
# Not part of `run.R --check`: haven stamps the write time into every .dta and .sas7bdat it writes,
# so a rewrite is never byte-identical. The committed binaries are canonical; the case file
# validation/r/cases/data_readers.R reads them back with haven (ReadStat) and records the values.
# Run this script only to regenerate the fixtures on purpose, then rerun
# `Rscript validation/r/run.R --write data_readers_values` and commit both.
#
# Every fixture is (a) written by haven (ReadStat), (b) copied from haven's examples (iris.sas7bdat
# written by SAS 9.4, iris.dta written by Stata 14), or (c) a documented byte-level transformation of
# (a) or (b). Each transformed file is read back with haven and must give exactly the data of its
# source (stopifnot below), so ReadStat -- not the p2a reader -- certifies its layout. The script
# needs haven >= 2.5.0 and nothing else; it takes about 5 seconds.
suppressPackageStartupMessages(library(haven))
stopifnot(packageVersion("haven") >= "2.5.0")

here <- local({ f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)); dirname(normalizePath(f)) })
out_dir <- normalizePath(file.path(here, "..", "..", "datasets", "files"), mustWork = FALSE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
fx <- function(name) file.path(out_dir, name)

# ---------------------------------------------------------------------------------------------
# Little-endian / big-endian integer helpers on raw vectors (1-based positions, as R indexes).
rd_u <- function(raw, pos, n, big = FALSE) {
  b <- as.integer(raw[pos:(pos + n - 1L)]); if (big) b <- rev(b)
  sum(b * 256^(seq_len(n) - 1L))
}
wr_u <- function(raw, pos, n, value, big = FALSE) {
  stopifnot(value >= 0, value < 256^n, value == floor(value))
  b <- integer(n); v <- value
  for (k in seq_len(n)) { b[k] <- v %% 256; v <- v %/% 256 }
  if (big) b <- rev(b)
  raw[pos:(pos + n - 1L)] <- as.raw(b); raw
}
find_bytes <- function(raw, pat, from = 1L) {
  pat <- charToRaw(pat); n <- length(pat)
  for (i in from:(length(raw) - n + 1L)) if (raw[i] == pat[1] && all(raw[i:(i + n - 1L)] == pat)) return(i)
  stop("pattern not found: ", rawToChar(pat))
}
same_data <- function(a, b) {
  a <- as.data.frame(zap_formats(zap_labels(a))); b <- as.data.frame(zap_formats(zap_labels(b)))
  for (nm in names(a)) attributes(a[[nm]]) <- NULL
  for (nm in names(b)) attributes(b[[nm]]) <- NULL
  identical(a, b)
}

# =============================================================================================
# Stata (.dta releases 117, 118, 119)
# =============================================================================================
# haven 2.5.5 refuses one-character names for version 13 (its regex needs two characters), so
# every name has at least two.
dta_base <- data.frame(
  xd = c(1.5, -2.25, NA, 1e300, 3.141592653589793, -0.1),
  ii = c(1L, NA, -3L, 2147483620L, -2147483647L, 7L),      # Stata long range is -2147483647..2147483620
  lg = c(TRUE, NA, FALSE, TRUE, FALSE, TRUE),              # haven writes logicals as long
  ss = c("a", "", "Z\u00fcrich", "bc d", "\u20ac5", "z"),   # str7: "Z\u00fcrich" is 7 UTF-8 bytes
  sw = c("first row text", "x", "", "a much longer string value", "mid", "end"),
  stringsAsFactors = FALSE)
dta_base$tg <- c(2.5, tagged_na("a"), 0, tagged_na("z"), -1e-5, 42)   # Stata .a and .z

write_dta(dta_base, fx("dta117_haven.dta"), version = 13)   # release 117
write_dta(dta_base, fx("dta118_haven.dta"), version = 14)   # release 118
write_dta(dta_base, fx("dta119_haven.dta"), version = 15)   # release 119

# Section offsets from <map> (14 u64: stata_dta, map, variable_types, varnames, sortlist, formats,
# value_label_names, variable_labels, characteristics, data, strls, value_labels, </stata_dta>, eof).
dta_parse <- function(raw) {
  rel <- as.integer(rawToChar(raw[29:31])); big <- rawToChar(raw[53:55]) == "MSF"
  kw <- if (rel == 119L) 4L else 2L; nw <- if (rel == 117L) 4L else 8L
  pk <- find_bytes(raw, "<K>") + 3L; K <- rd_u(raw, pk, kw, big)
  pn <- find_bytes(raw, "<N>") + 3L; N <- rd_u(raw, pn, nw, big)
  pm <- find_bytes(raw, "<map>") + 5L
  map <- vapply(0:13, function(i) rd_u(raw, pm + 8L * i, 8L, big), 0)
  vt <- map[3] + 1 + nchar("<variable_types>")
  types <- vapply(0:(K - 1), function(i) rd_u(raw, vt + 2 * i, 2L, big), 0)
  width <- ifelse(types <= 2045, types, c(`65530` = 1, `65529` = 2, `65528` = 4, `65527` = 4, `65526` = 8)[as.character(types)])
  list(rel = rel, big = big, K = K, N = N, pk = pk, pn = pn, kw = kw, nw = nw, pm = pm, map = map,
       types = types, width = unname(width), data = map[10] + 1 + nchar("<data>"))
}

# dta117_latin1.dta: Stata 13 itself stores str# text in the platform 8-bit encoding
# (Windows-1252), while haven writes UTF-8 bytes into every release. Rewrite the two non-ASCII
# cells of `ss` in Windows-1252 (same fixed width, NUL padded): "Z\u00fcrich" -> 5A FC 72 69 63 68,
# "\u20ac5" -> 80 35. haven reads the result with encoding = "windows-1252".
local({
  raw <- readBin(fx("dta117_haven.dta"), "raw", file.size(fx("dta117_haven.dta")))
  p <- dta_parse(raw); stopifnot(p$rel == 117L, p$K == 6, p$N == 6)
  col <- which(names(dta_base) == "ss"); w <- p$width[col]; rs <- sum(p$width)
  put <- function(raw, row, bytes) {
    at <- p$data + (row - 1) * rs + sum(p$width[seq_len(col - 1)])
    raw[at:(at + w - 1)] <- c(bytes, as.raw(rep(0L, w - length(bytes)))); raw
  }
  raw <- put(raw, 3, as.raw(c(0x5A, 0xFC, 0x72, 0x69, 0x63, 0x68)))
  raw <- put(raw, 5, as.raw(c(0x80, 0x35)))
  writeBin(raw, fx("dta117_latin1.dta"))
})

# dta119_msf.dta: the release-119 file converted to big-endian (MSF). Every integer and float of
# the header, map, type list, sort list and data is byte-swapped; text is unchanged. haven writes
# no value labels, characteristics or strLs for this frame (asserted), so nothing else needs swapping.
local({
  raw <- readBin(fx("dta119_haven.dta"), "raw", file.size(fx("dta119_haven.dta")))
  p <- dta_parse(raw); stopifnot(p$rel == 119L, !p$big)
  sect <- function(i) rawToChar(raw[(p$map[i] + 1):(p$map[i + 1])])
  stopifnot(sect(9) == "<characteristics></characteristics>", sect(11) == "<strls></strls>",
            sect(12) == "<value_labels></value_labels>")
  swap <- function(raw, pos, n) { raw[pos:(pos + n - 1)] <- rev(raw[pos:(pos + n - 1)]); raw }
  raw[53:55] <- charToRaw("MSF")
  raw <- swap(raw, p$pk, p$kw); raw <- swap(raw, p$pn, p$nw)
  pl <- find_bytes(raw, "<label>") + 7L; raw <- swap(raw, pl, 2L)          # label length (u16)
  for (i in 0:13) raw <- swap(raw, p$pm + 8L * i, 8L)
  vt <- p$map[3] + 1 + nchar("<variable_types>")
  for (i in 0:(p$K - 1)) raw <- swap(raw, vt + 2 * i, 2L)
  sl <- p$map[5] + 1 + nchar("<sortlist>")
  for (i in 0:p$K) raw <- swap(raw, sl + 4 * i, 4L)                           # K + 1 u32 for 119
  rs <- sum(p$width); off <- cumsum(c(0, p$width))
  for (r in 0:(p$N - 1)) for (j in seq_len(p$K)) if (p$types[j] > 2045 && p$width[j] > 1)
    raw <- swap(raw, p$data + r * rs + off[j], p$width[j])
  writeBin(raw, fx("dta119_msf.dta"))
})

local({
  ref <- read_dta(fx("dta118_haven.dta"))
  stopifnot(same_data(read_dta(fx("dta117_haven.dta"), encoding = "UTF-8"), ref),
            same_data(read_dta(fx("dta117_latin1.dta"), encoding = "windows-1252"), ref),
            same_data(read_dta(fx("dta119_haven.dta")), ref),
            same_data(read_dta(fx("dta119_msf.dta")), ref))
})

# dta118_stata_iris.dta: haven's example iris.dta, written by Stata 14 itself (release 118, 4 float
# columns with variable labels, one str10): the one Stata fixture no R code produced.
invisible(file.copy(system.file("examples", "iris.dta", package = "haven"), fx("dta118_stata_iris.dta"), overwrite = TRUE))
stopifnot(rawToChar(readBin(fx("dta118_stata_iris.dta"), "raw", 40)[29:31]) == "118")

# dta118_types.dta: every numeric storage type at its valid extremes, with the system missing value
# and one extended missing value (.a, .b, .c, .z, .y) per column. haven writes every R double as a
# Stata double, so the file is written with doubles and then narrowed: the type code and the cells of
# vb, vi, vl and vf are rewritten as byte, int, long and float (every value is exactly representable).
# Stata's missing codes map by their index k (0 = ., 1 = .a, ..., 26 = .z): double
# 0x7FE0000000000000 + k 2^40 -> byte 101 + k, int 32741 + k, long 2147483621 + k, float
# 0x7F000000 + k 2^11. The <data> section shrinks, so the map slots after it (strls, value_labels,
# </stata_dta>, eof) move down by the same number of bytes.
dta_narrow <- function(raw, to) {             # to: named list column -> "byte" | "int" | "long" | "float"
  p <- dta_parse(raw); stopifnot(!p$big, p$rel >= 118L)
  nm_at <- p$map[4] + 1 + nchar("<varnames>")
  vnames <- vapply(0:(p$K - 1), function(i) { b <- raw[nm_at + 129 * i + 0:128]; rawToChar(b[seq_len(which(b == as.raw(0))[1] - 1)]) }, "")
  code <- c(byte = 65530, int = 65529, long = 65528, float = 65527)
  tgt <- rep(NA_character_, p$K)
  for (nm in names(to)) { j <- match(nm, vnames); stopifnot(!is.na(j), p$types[j] == 65526); tgt[j] <- to[[nm]] }
  vt <- p$map[3] + 1 + nchar("<variable_types>")
  for (j in which(!is.na(tgt))) raw <- wr_u(raw, vt + 2 * (j - 1), 2, code[[tgt[j]]])
  off <- cumsum(c(0, p$width)); rs <- sum(p$width)
  rows <- lapply(0:(p$N - 1), function(r) {
    out <- raw(0)
    for (j in seq_len(p$K)) {
      cell <- raw[p$data + r * rs + off[j] + 0:(p$width[j] - 1)]
      if (!is.na(tgt[j])) {
        v <- readBin(cell, "double", size = 8, endian = "little")
        hi <- rd_u(cell, 5, 4)                                      # high 32 bits of the double
        k <- if (hi >= 0x7FE00000 && hi < 0x80000000) (hi - 0x7FE00000) / 256 else NA   # 2^40 is 2^8 there
        cell <- switch(tgt[j],
          byte  = if (is.na(k)) { stopifnot(v >= -127, v <= 100, v == round(v)); as.raw(v %% 256) } else as.raw(101 + k),
          int   = wr_u(raw(2), 1, 2, if (is.na(k)) { stopifnot(v >= -32767, v <= 32740, v == round(v)); v %% 65536 } else 32741 + k),
          long  = wr_u(raw(4), 1, 4, if (is.na(k)) { stopifnot(v >= -2147483647, v <= 2147483620, v == round(v)); v %% 2^32 } else 2147483621 + k),
          float = if (is.na(k)) { f <- writeBin(v, raw(), size = 4, endian = "little"); stopifnot(readBin(f, "double", size = 4, endian = "little") == v); f }
                  else wr_u(raw(4), 1, 4, 0x7F000000 + k * 2048))
      }
      out <- c(out, cell)
    }
    out
  })
  data <- unlist(rows); old_len <- p$N * rs; delta <- old_len - length(data)
  out <- c(raw[1:(p$data - 1)], data, raw[(p$data + old_len):length(raw)])
  for (slot in 11:14) out <- wr_u(out, p$pm + 8L * (slot - 1), 8, p$map[slot] - delta)
  out
}
f32 <- function(x) readBin(writeBin(x, raw(), size = 4), "double", size = 4)   # nearest float
dta_types <- data.frame(
  vb = c(-127, 100, NA, NA, 5, -1),                                            # byte: -127..100
  vi = c(-32767, 32740, NA, NA, -2, 300),                                      # int: -32767..32740
  vl = c(-2147483647, 2147483620, NA, NA, -1, 70000),                          # long
  vf = c(-1.7014117331926443e38, 1.7014117331926443e38, NA, NA, -0.25, f32(3e-38)),     # float max 0x7EFFFFFF
  vd = c(-8.9884656743115785e307, 8.9884656743115785e307, NA, NA, 2.5, -0.1))  # double max 0x7FDFFFFFFFFFFFFF
dta_types$vb[4] <- tagged_na("a"); dta_types$vi[4] <- tagged_na("b"); dta_types$vl[4] <- tagged_na("c")
dta_types$vf[4] <- tagged_na("z"); dta_types$vd[4] <- tagged_na("y")
local({
  tmp <- tempfile(fileext = ".dta"); on.exit(unlink(tmp))
  write_dta(dta_types, tmp, version = 14)
  writeBin(dta_narrow(readBin(tmp, "raw", file.size(tmp)), list(vb = "byte", vi = "int", vl = "long", vf = "float")),
           fx("dta118_types.dta"))
  a <- read_dta(tmp); b <- read_dta(fx("dta118_types.dta"))
  stopifnot(identical(dta_parse(readBin(fx("dta118_types.dta"), "raw", file.size(fx("dta118_types.dta"))))$types,
                      c(65530, 65529, 65528, 65527, 65526)))
  for (nm in names(a)) stopifnot(identical(as.numeric(a[[nm]]), as.numeric(b[[nm]])), identical(na_tag(a[[nm]]), na_tag(b[[nm]])))
})

# =============================================================================================
# SAS (.sas7bdat)
# =============================================================================================
# Layout per ReadStat (readstat_sas.c, readstat_sas7bdat_read.c) and Shotwell's sas7bdat notes:
# byte 32 == 0x33 -> 64-bit integers; byte 35 == 0x33 -> 4 pad bytes at offset 164; byte 37 == 1 ->
# little-endian; header_length / page_size / page_count at 196/200/204 + pad. Page header: type at
# B, block count at B + 2, subheader count at B + 4 (B = 32 | 16), pointers from B + 8, each
# (offset, length: int, compression: u8, type: u8) of 3 * int bytes.
sas_layout <- function(raw) {
  stopifnot(raw[38] == as.raw(0x01))                               # sources are little-endian
  u64 <- raw[33] == as.raw(0x33); pad <- if (raw[36] == as.raw(0x33)) 4L else 0L
  il <- if (u64) 8L else 4L
  B <- if (u64) 32L else 16L
  list(u64 = u64, pad = pad, il = il, hl = rd_u(raw, 197 + pad, 4), ps = rd_u(raw, 201 + pad, 4),
       pc = rd_u(raw, 205 + pad, il), B = B, phs = B + 8L, sps = 3L * il)
}
sas_page <- function(raw, L, i) {             # i: 0-based page index
  pb <- L$hl + i * L$ps + 1                    # 1-based position of the page's first byte
  shc <- rd_u(raw, pb + L$B + 4, 2)
  ptr <- lapply(seq_len(shc) - 1, function(k) {
    q <- pb + L$phs + k * L$sps
    off <- rd_u(raw, q, L$il); len <- rd_u(raw, q + L$il, L$il)
    list(q = q, off = off, len = len, comp = as.integer(raw[q + 2 * L$il]), type = as.integer(raw[q + 2 * L$il + 1]),
         sig = if (len > 0) rd_u(raw, pb + off, 4) else NA)
  })
  list(pb = pb, type = rd_u(raw, pb + L$B, 2), bc = rd_u(raw, pb + L$B + 2, 2), shc = shc, ptr = ptr)
}
SIG_ROW_SIZE <- 0xF7F7F7F7; SIG_COL_TEXT <- 0xFFFFFFFD; SIG_COL_ATTR <- 0xFFFFFFFC
sas_find <- function(raw, L, page, sig) Filter(function(p) p$comp == 0 && !is.na(p$sig) && p$sig == sig, page$ptr)
sas_row_length <- function(raw, L, page) { p <- sas_find(raw, L, page, SIG_ROW_SIZE)[[1]]; rd_u(raw, page$pb + p$off + 5 * L$il, L$il) }
sas_row_count  <- function(raw, L, page) { p <- sas_find(raw, L, page, SIG_ROW_SIZE)[[1]]; rd_u(raw, page$pb + p$off + 6 * L$il, L$il) }
# Raw rows of the uncompressed data (type 0x100) and mix (0x200) pages, in file order.
sas_rows <- function(raw, L) {
  meta <- sas_page(raw, L, 0); rl <- sas_row_length(raw, L, meta); rc <- sas_row_count(raw, L, meta)
  rows <- list()
  for (i in seq_len(L$pc) - 1) {
    pg <- sas_page(raw, L, i); kind <- bitwAnd(pg$type, 0x0F00)
    if (kind == 0x100) { start <- pg$pb + L$phs; n <- pg$bc }
    else if (kind == 0x200) { start <- pg$pb + L$phs + pg$shc * L$sps; if ((start - pg$pb) %% 8 == 4) start <- start + 4; n <- pg$bc - pg$shc }
    else next
    for (r in seq_len(n) - 1) if (length(rows) < rc) rows[[length(rows) + 1]] <- raw[(start + r * rl):(start + (r + 1) * rl - 1)]
  }
  stopifnot(length(rows) == rc)
  rows
}

# --- transformation 1: move the 4-byte header pad in or out (byte 35) --------------------------
# ReadStat reads 164 bytes, skips `pad` bytes, then the dates, header_length, page_size and
# page_count. Inserting (removing) 4 zero bytes at offset 164 and removing (appending) 4 zero bytes
# at the end of the header block keeps header_length and every page offset unchanged.
sas_set_pad <- function(raw, pad) {
  L <- sas_layout(raw); if (L$pad == pad) return(raw)
  hl <- L$hl; tail <- raw[(hl - 3):hl]; stopifnot(all(tail == as.raw(0)))
  if (pad == 4L) { out <- c(raw[1:164], as.raw(rep(0L, 4)), raw[165:(hl - 4)], raw[(hl + 1):length(raw)]); out[36] <- as.raw(0x33) }
  else { stopifnot(all(raw[165:168] == as.raw(0))); out <- c(raw[1:164], raw[169:hl], as.raw(rep(0L, 4)), raw[(hl + 1):length(raw)]); out[36] <- as.raw(0x22) }
  stopifnot(length(out) == length(raw)); out
}

# --- transformation 2: numeric columns stored with fewer than 8 bytes (SAS LENGTH 3..7) --------
# A SAS numeric of length w keeps the w high-order bytes of the IEEE double; little-endian files
# store bytes 9-w..8 of the 8-byte value. Column offsets and the row length are recomputed; rows stay
# on their pages (they only shrink).
sas_shorten <- function(raw, widths) {          # widths: new width per column (attribute order)
  L <- sas_layout(raw); meta <- sas_page(raw, L, 0); rows <- sas_rows(raw, L)
  a <- sas_find(raw, L, meta, SIG_COL_ATTR)[[1]]; sh <- meta$pb + a$off
  nc <- (a$len - 2 * L$il - 12) %/% (L$il + 8); stopifnot(nc == length(widths))
  q <- sh + L$il + 8 + (seq_len(nc) - 1) * (L$il + 8)
  off <- vapply(q, function(x) rd_u(raw, x, L$il), 0); wid <- vapply(q, function(x) rd_u(raw, x + L$il, 4), 0)
  typ <- vapply(q, function(x) as.integer(raw[x + L$il + 6]), 0L)
  stopifnot(all(widths[typ == 2] == wid[typ == 2]), all(widths[typ == 1] >= 3 & widths[typ == 1] <= 8), all(wid[typ == 1] == 8))
  ord <- order(off); new_off <- numeric(nc); new_off[ord] <- cumsum(c(0, widths[ord]))[seq_len(nc)]
  rl <- sum(widths)
  for (j in seq_len(nc)) { raw <- wr_u(raw, q[j], L$il, new_off[j]); raw <- wr_u(raw, q[j] + L$il, 4, widths[j]) }
  rs <- sas_find(raw, L, meta, SIG_ROW_SIZE)[[1]]; raw <- wr_u(raw, meta$pb + rs$off + 5 * L$il, L$il, rl)
  new_rows <- lapply(rows, function(r) {
    o <- raw(rl)
    for (j in seq_len(nc)) {
      cell <- r[(off[j] + 1):(off[j] + wid[j])]
      if (typ[j] == 1) cell <- cell[(9 - widths[j]):8]                    # high-order bytes (LE)
      o[(new_off[j] + 1):(new_off[j] + widths[j])] <- cell
    }
    o
  })
  k <- 0
  for (i in seq_len(L$pc) - 1) {
    pg <- sas_page(raw, L, i); if (bitwAnd(pg$type, 0x0F00) != 0x100) next
    start <- pg$pb + L$phs; raw[start:(pg$pb + L$ps - 1)] <- as.raw(0)
    for (r in seq_len(pg$bc) - 1) { k <- k + 1; raw[(start + r * rl):(start + (r + 1) * rl - 1)] <- new_rows[[k]] }
  }
  stopifnot(k == length(rows)); raw
}

# --- transformation 3: big-endian byte order (byte 37 = 0x00) ---------------------------------
# Every integer field ReadStat reads and every numeric cell is byte-reversed; text is unchanged. A
# truncated numeric keeps its high-order bytes, so its stored bytes are simply reversed too. The
# source must be uncompressed with all metadata on page 0 (the haven-written layout).
sas_to_big_endian <- function(raw) {
  L <- sas_layout(raw); meta <- sas_page(raw, L, 0); rl <- sas_row_length(raw, L, meta)
  swap <- function(raw, pos, n) { raw[pos:(pos + n - 1)] <- rev(raw[pos:(pos + n - 1)]); raw }
  attrs <- NULL
  for (p in meta$ptr) if (p$len > 0) {
    sh <- meta$pb + p$off
    if (p$sig == SIG_COL_ATTR) {
      nc <- (p$len - 2 * L$il - 12) %/% (L$il + 8); q <- sh + L$il + 8 + (seq_len(nc) - 1) * (L$il + 8)
      attrs <- data.frame(off = vapply(q, function(x) rd_u(raw, x, L$il), 0), wid = vapply(q, function(x) rd_u(raw, x + L$il, 4), 0),
                          typ = vapply(q, function(x) as.integer(raw[x + L$il + 6]), 0L))
      raw <- swap(raw, sh + L$il, 2)
      for (x in q) { raw <- swap(raw, x, L$il); raw <- swap(raw, x + L$il, 4); raw <- swap(raw, x + L$il + 4, 2) }
    } else if (p$sig == 0xFFFFFFFF) {                                           # column names
      raw <- swap(raw, sh + L$il, 2)
      for (i in seq_len((p$len - 2 * L$il - 12) %/% 8) - 1) for (f in 0:2) raw <- swap(raw, sh + L$il + 8 + 8 * i + 2 * f, 2)
    } else if (p$sig == SIG_COL_TEXT) {
      raw <- swap(raw, sh + L$il, 2)
    } else if (p$sig == SIG_ROW_SIZE || p$sig == 0xF6F6F6F6) {                  # row size, column size: int fields
      for (k in seq(L$il, p$len - L$il, by = L$il)) raw <- swap(raw, sh + k, L$il)
    }
    # Signature: an 8-byte integer when its high half is all ones (ReadStat reads the second word of a
    # big-endian 64-bit signature that starts FF FF FF FF); F7F7F7F7 / F6F6F6F6 stay as written.
    if (!(p$sig %in% c(SIG_ROW_SIZE, 0xF6F6F6F6))) raw <- swap(raw, sh, L$il)
  }
  for (i in seq_len(L$pc) - 1) {
    pg <- sas_page(raw, L, i)
    if (bitwAnd(pg$type, 0x0F00) == 0x100) for (r in seq_len(pg$bc) - 1) for (j in which(attrs$typ == 1))
      raw <- swap(raw, pg$pb + L$phs + r * rl + attrs$off[j], attrs$wid[j])
    for (k in seq_along(pg$ptr)) { q <- pg$ptr[[k]]$q; raw <- swap(raw, q, L$il); raw <- swap(raw, q + L$il, L$il) }
    for (f in 0:2) raw <- swap(raw, pg$pb + L$B + 2 * f, 2)
  }
  for (pos in c(165, 173) + L$pad) raw <- swap(raw, pos, 8)                     # creation / modification time
  raw <- swap(raw, 197 + L$pad, 4); raw <- swap(raw, 201 + L$pad, 4); raw <- swap(raw, 205 + L$pad, L$il)
  raw[38] <- as.raw(0x00)
  raw
}

# --- transformation 3b: Windows-1252 text (byte 70 = 62, SAS "wlatin1") -------------------------
# SAS on Windows writes text in the session encoding, usually wlatin1 (code 62 at byte 70; every
# SAS-written fixture and public file checked here has it). haven writes UTF-8 (code 20). Every
# text cell of the data pages is re-encoded to Windows-1252 and padded with blanks to its width.
sas_recode_1252 <- function(raw) {
  L <- sas_layout(raw); meta <- sas_page(raw, L, 0); rl <- sas_row_length(raw, L, meta)
  a <- sas_find(raw, L, meta, SIG_COL_ATTR)[[1]]; sh <- meta$pb + a$off
  nc <- (a$len - 2 * L$il - 12) %/% (L$il + 8); q <- sh + L$il + 8 + (seq_len(nc) - 1) * (L$il + 8)
  off <- vapply(q, function(x) rd_u(raw, x, L$il), 0); wid <- vapply(q, function(x) rd_u(raw, x + L$il, 4), 0)
  typ <- vapply(q, function(x) as.integer(raw[x + L$il + 6]), 0L)
  for (i in seq_len(L$pc) - 1) {
    pg <- sas_page(raw, L, i); if (bitwAnd(pg$type, 0x0F00) != 0x100) next
    for (r in seq_len(pg$bc) - 1) for (j in which(typ == 2)) {
      at <- pg$pb + L$phs + r * rl + off[j]; cell <- raw[at:(at + wid[j] - 1)]
      keep <- which(!(cell %in% as.raw(c(0x00, 0x20)))); if (!length(keep)) next
      txt <- rawToChar(cell[1:max(keep)]); Encoding(txt) <- "UTF-8"
      b <- iconv(txt, "UTF-8", "WINDOWS-1252", toRaw = TRUE)[[1]]; stopifnot(!is.null(b), length(b) <= wid[j])
      raw[at:(at + wid[j] - 1)] <- c(b, as.raw(rep(0x20, wid[j] - length(b))))
    }
  }
  raw[71] <- as.raw(62)
  raw
}

# --- the two compression methods, as written by SAS (COMPRESS=CHAR | BINARY) ------------------
# SAS RLE ("SASYZCRL"): a control byte = command (high nibble) | length (low nibble); the commands
# and their lengths are those ReadStat's sas_rle_decompress decodes (readstat_sas_rle.c):
#   0 copy next+64+256*l bytes | 2 copy l+96 | 4 insert byte b (n+18+256*l) | 5,6,7 insert '@',' ',0
#   (n+17+256*l) | 8,9,10,11 copy l+1, l+17, l+33, l+49 | 12 insert byte b (l+3) | 13,14,15 insert
#   '@',' ',0 (l+2) | 1 copy n+64+256*l+4096 (literals of 4160 bytes or more: sas64_rle_wide).
#   `k` rotates between equivalent encodings so that the fixture rows use every command above; the
#   script checks that they do.
rle_used <- integer(0)
rle_compress <- function(x, k) {
  out <- raw(0); n <- length(x); i <- 1L; lit <- raw(0)
  emit <- function(...) out <<- c(out, as.raw(c(...)))
  use <- function(cmd) rle_used <<- union(rle_used, cmd)
  flush <- function() {
    while (length(lit) > 0) {
      L <- length(lit); k <<- k + 1L
      if (L >= 4160) {                                           # 1: copy n + 64 + 256 l + 4096
        take <- min(L, 8255); m <- take - 4160; emit(0x10 + m %/% 256, m %% 256); use(1)
        out <<- c(out, lit[seq_len(take)]); lit <<- lit[-seq_len(take)]; next
      }
      if (L >= 96 && L <= 111 && k %% 2 == 1) { emit(0x20 + (L - 96)); out <<- c(out, lit); use(2); lit <<- raw(0); next }
      take <- if (L >= 64 && !(L == 64 && k %% 2 == 1)) min(L, 64 + 4095) else min(L, 64)
      if (take >= 64 && !(take == 64 && k %% 2 == 1)) { m <- take - 64; emit(0x00 + m %/% 256, m %% 256); use(0) }
      else if (take >= 49) { emit(0xB0 + take - 49); use(11) }
      else if (take >= 33) { emit(0xA0 + take - 33); use(10) }
      else if (take >= 17) { emit(0x90 + take - 17); use(9) }
      else { emit(0x80 + take - 1); use(8) }
      out <<- c(out, lit[seq_len(take)]); lit <<- lit[-seq_len(take)]
    }
  }
  while (i <= n) {
    b <- x[i]; r <- 1L
    while (i + r <= n && x[i + r] == b) r <- r + 1L
    sp <- match(as.integer(b), c(0x40, 0x20, 0x00))            # '@', blank, zero -> commands 5/13, 6/14, 7/15
    if (!(r >= 3L || (!is.na(sp) && r >= 2L))) { lit <- c(lit, b); i <- i + 1L; next }
    flush(); k <- k + 1L
    r <- min(r, 4113L); generic <- is.na(sp) || (r >= 3L && k %% 3 == 1)
    if (!generic) {
      if (r > 4112L) r <- 4112L
      if (r <= 17L && !(r == 17L && k %% 2 == 0)) { emit((12 + sp) * 16 + r - 2); use(12 + sp) }
      else { m <- r - 17L; emit((4 + sp) * 16 + m %/% 256, m %% 256); use(4 + sp) }
    } else {
      if (r <= 18L && !(r == 18L && k %% 2 == 0)) { emit(0xC0 + r - 3, as.integer(b)); use(12) }
      else { m <- r - 18L; emit(0x40 + m %/% 256, m %% 256, as.integer(b)); use(4) }
    }
    i <- i + r
  }
  flush(); out
}
# SAS RDC ("SASYZCR2", Ross Data Compression): groups of 16 items behind a big-endian control word
# (bit set = command). Commands (ReadStat sas7bdat_parse_subheader_rdc): marker <= 0x0F: insert
# marker+3 copies of the next byte; 0x1n: insert n+19+16*next copies of the byte after; 0x2n: copy
# 16+third bytes from back offset n+3+16*next; 0xcn, c = 3..15: copy c bytes from back offset
# n+3+16*next. ReadStat refuses overlapping copies, so the copy length never exceeds the offset.
rdc_used <- character(0)
rdc_compress <- function(x) {
  n <- length(x); i <- 1L; items <- list(); is_cmd <- logical(0)
  add <- function(bytes, cmd, what) { items[[length(items) + 1L]] <<- as.raw(bytes); is_cmd <<- c(is_cmd, cmd); rdc_used <<- union(rdc_used, what) }
  while (i <= n) {
    b <- x[i]; r <- 1L; while (i + r <= n && x[i + r] == b && r < 4114L) r <- r + 1L
    if (r >= 19L) { m <- r - 19L; add(c(0x10 + m %% 16, m %/% 16, as.integer(b)), TRUE, "long_rle"); i <- i + r; next }
    if (r >= 3L) { add(c(r - 3L, as.integer(b)), TRUE, "short_rle"); i <- i + r; next }
    best <- 0L; bofs <- 0L
    for (ofs in 3:min(4098L, i - 1L)) {
      if (ofs > i - 1L) break
      len <- 0L; maxlen <- min(271L, ofs, n - i + 1L)
      while (len < maxlen && x[i + len] == x[i - ofs + len]) len <- len + 1L
      if (len > best) { best <- len; bofs <- ofs }
    }
    if (best >= 16L) { o <- bofs - 3L; add(c(0x20 + o %% 16, o %/% 16, best - 16L), TRUE, "long_pattern"); i <- i + best; next }
    if (best >= 3L) { o <- bofs - 3L; add(c(best * 16 + o %% 16, o %/% 16), TRUE, "short_pattern"); i <- i + best; next }
    add(as.integer(b), FALSE, "literal"); i <- i + 1L
  }
  out <- raw(0)
  for (g in seq(1L, length(items), by = 16L)) {
    idx <- g:min(g + 15L, length(items)); ctrl <- sum(2^(15 - (seq_along(idx) - 1)) * is_cmd[idx])
    out <- c(out, as.raw(c(ctrl %/% 256, ctrl %% 256)), unlist(items[idx]))
  }
  out
}

# --- transformation 4: compressed rows ----------------------------------------------------------
# SAS stores the rows of a compressed file as subheaders (pointer compression 4, type 1) on
# metadata pages, packed from the end of the page; each page also ends with one truncated
# pointer (compression 1), which readers skip. The compression literal goes into the first column
# text subheader at text offset 12, and the row-size subheader's text reference at length - 118
# points at it (as in SAS-written files). Page 0 keeps the metadata (a mix page becomes a metadata
# page); the uncompressed data pages are dropped.
# With comp_page = TRUE a page of type 0x9000 follows page 0, as in SAS-written compressed files
# (its counts and pointer area are left as junk that a reader must not parse: readers skip 0x9000).
# Rows listed in raw_rows are stored uncompressed (pointer compression 0, type 1), as SAS stores a row
# that compression would not shorten.
sas_compress <- function(raw, method, row_page_type = 0L, page_size = NULL, header_length = NULL, comp_page = FALSE,
                         raw_rows = integer(0)) {
  L <- sas_layout(raw); meta <- sas_page(raw, L, 0); rows <- sas_rows(raw, L); rl <- length(rows[[1]])
  ps <- if (is.null(page_size)) L$ps else page_size; hl <- if (is.null(header_length)) L$hl else header_length
  lit <- charToRaw(c(RLE = "SASYZCRL", RDC = "SASYZCR2")[[method]])
  t <- sas_find(raw, L, meta, SIG_COL_TEXT)[[1]]; at <- meta$pb + t$off + L$il + 12
  stopifnot(all(raw[at:(at + 7)] %in% as.raw(c(0x00, 0x20)))); raw[at:(at + 7)] <- lit
  rs <- sas_find(raw, L, meta, SIG_ROW_SIZE)[[1]]; ref <- meta$pb + rs$off + rs$len - 118
  stopifnot(all(raw[ref:(ref + 5)] == as.raw(0)))
  raw <- wr_u(raw, ref, 2, 0); raw <- wr_u(raw, ref + 2, 2, 12); raw <- wr_u(raw, ref + 4, 2, 8)
  page0 <- raw[meta$pb:(meta$pb + L$ps - 1)]
  if (bitwAnd(meta$type, 0x0F00) == 0x200) {                         # mix -> meta: drop the rows
    page0 <- wr_u(page0, L$B + 1, 2, 0); page0 <- wr_u(page0, L$B + 3, 2, meta$shc)
    lo <- min(vapply(meta$ptr, function(p) if (p$len > 0) p$off else Inf, 0))
    page0[(L$phs + meta$shc * L$sps + 1):lo] <- as.raw(0)
  }
  if (ps != L$ps) {          # smaller pages: move page 0's subheaders (packed at its end) down
    tab <- L$phs + meta$shc * L$sps; delta <- L$ps - ps
    lo <- min(vapply(meta$ptr, function(p) if (p$len > 0) p$off else Inf, 0)); stopifnot(lo - delta >= tab)
    new0 <- raw(ps); new0[1:tab] <- page0[1:tab]; new0[(lo - delta + 1):ps] <- page0[(lo + 1):L$ps]
    for (k in seq_along(meta$ptr)) if (meta$ptr[[k]]$len > 0)
      new0 <- wr_u(new0, L$phs + (k - 1) * L$sps + 1, L$il, meta$ptr[[k]]$off - delta)
    new0 <- wr_u(new0, rs$off - delta + 13 * L$il + 1, L$il, ps)   # the row-size subheader's page size
    page0 <- new0
  }
  comp <- lapply(seq_along(rows), function(r) {
    if (r %in% raw_rows) return(rows[[r]])
    z <- if (method == "RLE") rle_compress(rows[[r]], r) else rdc_compress(rows[[r]])
    stopifnot(length(z) < rl); z
  })
  pages <- list(); r <- 1L
  while (r <= length(comp)) {
    pg <- raw(ps); end <- ps; ptrs <- list()
    while (r <= length(comp)) {
      z <- comp[[r]]; need <- L$phs + (length(ptrs) + 2) * L$sps   # this row + the truncated pointer
      if (end - length(z) < need + 8) break
      end <- end - length(z); pg[(end + 1):(end + length(z))] <- z
      ptrs[[length(ptrs) + 1]] <- c(end, length(z), if (r %in% raw_rows) 0 else 4, 1); r <- r + 1L
    }
    stopifnot(length(ptrs) > 0)
    tend <- L$phs + (length(ptrs) + 1) * L$sps                     # truncated pointer: SAS leaves
    junk <- 8L; pg[(tend + 1):(tend + junk)] <- as.raw(0xFF)       # a partial row behind it
    ptrs[[length(ptrs) + 1]] <- c(tend, junk, 1, 1)
    pg <- wr_u(pg, L$B + 1, 2, row_page_type); pg <- wr_u(pg, L$B + 3, 2, length(ptrs)); pg <- wr_u(pg, L$B + 5, 2, length(ptrs))
    for (k in seq_along(ptrs)) {
      q <- L$phs + (k - 1) * L$sps + 1; p <- ptrs[[k]]
      pg <- wr_u(pg, q, L$il, p[1]); pg <- wr_u(pg, q + L$il, L$il, p[2]); pg[q + 2 * L$il] <- as.raw(p[3]); pg[q + 2 * L$il + 1] <- as.raw(p[4])
    }
    pages[[length(pages) + 1]] <- pg
  }
  if (comp_page) {
    np <- 2L + length(pages)
    cp <- raw(ps); cp <- wr_u(cp, L$B + 1, 2, 0x9000); cp <- wr_u(cp, L$B + 3, 2, np); cp <- wr_u(cp, L$B + 5, 2, np - 1L)
    cp[(L$phs + 1):(L$phs + (np - 1L) * L$sps)] <- as.raw(0xFF)       # pointers that would point outside the page
    pages <- c(list(cp), pages)
  }
  if (hl < L$hl) stopifnot(all(raw[(hl + 1):L$hl] == as.raw(0)))
  hdr <- c(raw[1:min(hl, L$hl)], raw(max(0, hl - L$hl)))
  hdr <- wr_u(hdr, 197 + L$pad, 4, hl); hdr <- wr_u(hdr, 201 + L$pad, 4, ps); hdr <- wr_u(hdr, 205 + L$pad, L$il, 1 + length(pages))
  c(hdr, page0, unlist(pages))
}

# --- the SAS data -----------------------------------------------------------------------------
set.seed(20261022)
n <- 60
sas_base <- data.frame(
  id = as.numeric(1:n),
  income = round(rlnorm(n, 10, 0.6), 2),
  ratio = round(rnorm(n), 6),
  z1 = 0, z2 = 0, z3 = 0,
  grp = sample(c("A", "B", "Z\u00fcrich", "", "C d", "\u20ac5"), n, replace = TRUE),   # "\u20ac" is 0x80 in Windows-1252
  note = sample(c("", "@@ two ats", "@@@@@@@@@@@@@@@@@@@@ twenty ats", "x  y", "wide                      gap", "dashes ----------------------",
                  "zzz", "a", "1234567890123456789012345678901234567890"), n, replace = TRUE),
  stringsAsFactors = FALSE)
sas_base$income[c(5, 17, 40)] <- NA
sas_base$ratio[9] <- NA; sas_base$ratio[11] <- 1e308; sas_base$ratio[12] <- -0.1; sas_base$ratio[13] <- 0
sas_base$z1[seq(2, n, 7)] <- c(1, -1, 0.5, 2, 1e-3, 4, 7, 8, 9)[seq_along(seq(2, n, 7))]
sas_base$z2[seq(3, n, 11)] <- 2.5
alnum <- c(letters, LETTERS, 0:9)
sas_base$code <- vapply(seq_len(n), function(i) if (i %% 4 == 0) paste(sample(alnum, 100 + i %% 11, TRUE), collapse = "") else
  if (i %% 4 == 1) paste(sample(alnum, 70, TRUE), collapse = "") else if (i %% 4 == 2) "" else "short code", "")
sas_base$code[1] <- paste(sample(alnum, 110, TRUE), collapse = "")    # column width 110
# Row 7: a 40-byte literal between two blank runs, so the RLE rows use command 0xA (copy 33..48).
sas_base$note[7] <- "a"; sas_base$code[7] <- "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"

suppressWarnings(write_sas(sas_base, fx("sas64_haven.sas7bdat")))          # ReadStat writer: 64-bit, LE, no pad
invisible(file.copy(system.file("examples", "iris.sas7bdat", package = "haven"), fx("sas32_iris.sas7bdat"), overwrite = TRUE))

rd <- function(name) readBin(fx(name), "raw", file.size(fx(name)))
wr <- function(raw, name) writeBin(raw, fx(name))
wr(sas_set_pad(rd("sas64_haven.sas7bdat"), 4L), "sas64_pad.sas7bdat")
chr_width <- vapply(sas_base[c("grp", "note", "code")], function(x) max(nchar(x, "bytes")), 0)   # 7, 40, 110
wr(sas_shorten(rd("sas64_haven.sas7bdat"), c(3, 4, 6, 7, 5, 8, chr_width)), "sas64_short.sas7bdat")
wr(sas_to_big_endian(rd("sas64_short.sas7bdat")), "sas64_short_be.sas7bdat")
wr(sas_recode_1252(rd("sas64_haven.sas7bdat")), "sas64_wlatin1.sas7bdat")
wr(sas_compress(rd("sas64_haven.sas7bdat"), "RLE", comp_page = TRUE), "sas64_rle.sas7bdat")
wr(sas_compress(rd("sas64_haven.sas7bdat"), "RDC", row_page_type = 0x4000, raw_rows = c(1L, 30L)), "sas64_rdc.sas7bdat")
wr(sas_compress(sas_set_pad(rd("sas32_iris.sas7bdat"), 0L), "RLE", page_size = 4096, header_length = 1024, comp_page = TRUE), "sas32_nopad_rle.sas7bdat")
wr(sas_compress(rd("sas32_iris.sas7bdat"), "RDC", page_size = 4096, header_length = 1024), "sas32_rdc.sas7bdat")
# A row wider than 4160 bytes of literal text (haven writes it on 8192-byte pages, one row each).
sas_wide <- data.frame(id = c(1, 2, 3), txt = vapply(0:2, function(i) paste(alnum[(seq_len(4300) + 7 * i - 1) %% 62 + 1], collapse = ""), ""),
                       stringsAsFactors = FALSE)
local({
  tmp <- tempfile(fileext = ".sas7bdat"); on.exit(unlink(tmp))
  suppressWarnings(write_sas(sas_wide, tmp))
  wr(sas_compress(readBin(tmp, "raw", file.size(tmp)), "RLE"), "sas64_rle_wide.sas7bdat")
  stopifnot(same_data(read_sas(fx("sas64_rle_wide.sas7bdat")), sas_wide))
})
stopifnot(setequal(rle_used, c(0, 1, 2, 4:15)),
          setequal(rdc_used, c("literal", "short_rle", "long_rle", "short_pattern", "long_pattern")))

local({
  h <- read_sas(fx("sas64_haven.sas7bdat")); i <- read_sas(fx("sas32_iris.sas7bdat"))
  stopifnot(same_data(h, sas_base),
            same_data(read_sas(fx("sas64_pad.sas7bdat")), h),
            same_data(read_sas(fx("sas64_rle.sas7bdat")), h),
            same_data(read_sas(fx("sas64_rdc.sas7bdat")), h),
            same_data(read_sas(fx("sas32_nopad_rle.sas7bdat")), i),
            same_data(read_sas(fx("sas32_rdc.sas7bdat")), i),
            nrow(read_sas(fx("sas64_short.sas7bdat"))) == n,
            same_data(read_sas(fx("sas64_short_be.sas7bdat")), read_sas(fx("sas64_short.sas7bdat"))),
            same_data(read_sas(fx("sas64_wlatin1.sas7bdat")), h),
            any(grepl("\u20ac", h$grp)), any(grepl("\u00fc", h$grp)))
})

# The manifest is the analysis dataset of the case `data_readers_values` (validation/r/cases/data_readers.R):
# one row per fixture, with the haven call that reads it.
manifest <- data.frame(
  file = c("dta117_haven.dta", "dta117_latin1.dta", "dta118_haven.dta", "dta119_haven.dta", "dta119_msf.dta",
           "dta118_stata_iris.dta", "dta118_types.dta",
           "sas64_haven.sas7bdat", "sas64_pad.sas7bdat", "sas64_short.sas7bdat", "sas64_short_be.sas7bdat", "sas64_wlatin1.sas7bdat",
           "sas64_rle.sas7bdat", "sas64_rle_wide.sas7bdat", "sas64_rdc.sas7bdat",
           "sas32_iris.sas7bdat", "sas32_nopad_rle.sas7bdat", "sas32_rdc.sas7bdat"),
  format = c(rep("dta", 7), rep("sas7bdat", 11)),
  encoding = c("UTF-8", "windows-1252", "", "", "", "", "", rep("", 11)),
  origin = c("haven::write_dta(version = 13)", "dta117_haven.dta, ss cells rewritten in Windows-1252",
             "haven::write_dta(version = 14)", "haven::write_dta(version = 15)", "dta119_haven.dta converted to MSF",
             "haven examples/iris.dta (written by Stata 14)", "haven::write_dta(version = 14), vb/vi/vl/vf narrowed to byte/int/long/float",
             "haven::write_sas", "sas64_haven.sas7bdat, header pad inserted", "sas64_haven.sas7bdat, numeric widths 3,4,6,7,5,8",
             "sas64_short.sas7bdat converted to big-endian", "sas64_haven.sas7bdat, text re-encoded in Windows-1252 (encoding 62)",
             "sas64_haven.sas7bdat, rows RLE-compressed (page type 0), 0x9000 page",
             "haven::write_sas of 3 rows of 4300 bytes of text, RLE-compressed (command 0x1)", "sas64_haven.sas7bdat, rows RDC-compressed (page type 0x4000), rows 1 and 30 stored uncompressed",
             "haven examples/iris.sas7bdat (written by SAS 9.4)", "sas32_iris.sas7bdat, pad removed, 1024/4096 pages, RLE, 0x9000 page",
             "sas32_iris.sas7bdat, 1024/4096 pages, RDC"),
  stringsAsFactors = FALSE)
stopifnot(all(file.exists(fx(manifest$file))))
utils::write.csv(manifest, fx("data_readers_manifest.csv"), row.names = FALSE)
cat("wrote", nrow(manifest), "fixtures and the manifest to", out_dir, "\n")
