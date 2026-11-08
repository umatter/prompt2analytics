# Binary reader fixtures (`validation/datasets/files/`)

Small Stata (`.dta`) and SAS (`.sas7bdat`) files that the readers in
`crates/p2a-core/src/data/stata.rs` and `crates/p2a-core/src/data/sas.rs` are tested on. Every value
the tests assert is what `haven::read_dta` / `haven::read_sas` (ReadStat) read from the same file:
the R case `data_readers_values` (`validation/r/cases/data_readers.R`) records it in
`validation/expected/data_readers_values.json`. Method page: [`validation/data/readers.md`](../../data/readers.md).

## Where the files come from

`validation/r/fixtures/data_readers.R` writes all of them:

```
Rscript validation/r/fixtures/data_readers.R                # rewrites every file below (about 5 s)
Rscript validation/r/run.R --write data_readers_values      # then the case JSON
```

The script is **not** part of `run.R --check`. haven stamps the write time into every file it writes,
so a rewrite is never byte-identical (only the timestamp bytes change; the case JSON stays identical,
which `run.R --check` confirms). The committed binaries are canonical: regenerate them only on
purpose, and commit the files and the JSON together. Never edit a fixture by hand. Each file is:

- written by haven (ReadStat's writer), or
- copied from haven's `inst/examples` (`iris.sas7bdat` written by SAS 9.4, `iris.dta` written by
  Stata 14), or
- a byte-level transformation of one of those, documented in the script. haven must read every
  transformed file exactly as its source (the script stops otherwise), so ReadStat, not the p2a
  reader, certifies the layout.

| File | Origin | Exercises |
|---|---|---|
| `dta117_haven.dta` | `write_dta(version = 13)` | release 117: 4-byte `<N>`, 1-byte label length, UTF-8 text |
| `dta117_latin1.dta` | `dta117_haven.dta`, `ss` cells in Windows-1252 | Stata 13 platform-encoded `str#` |
| `dta118_haven.dta` | `write_dta(version = 14)` | release 118: `<map>` slots, doubles, longs, tagged missing |
| `dta118_stata_iris.dta` | haven `examples/iris.dta` (Stata 14) | a Stata-written file: floats, variable labels |
| `dta118_types.dta` | `write_dta(version = 14)`, columns narrowed | byte, int, long, float, double at their extremes; `.`, `.a`-`.z` |
| `dta119_haven.dta` | `write_dta(version = 15)` | release 119: 4-byte `<K>` |
| `dta119_msf.dta` | `dta119_haven.dta` in big-endian | MSF byte order |
| `sas64_haven.sas7bdat` | `write_sas` | 64-bit, no header pad, UTF-8 (encoding 20), 1e308 |
| `sas64_pad.sas7bdat` | `sas64_haven.sas7bdat`, header pad inserted | byte 35 = 0x33 on a 64-bit file |
| `sas64_short.sas7bdat` | `sas64_haven.sas7bdat`, numeric widths 3, 4, 6, 7, 5, 8 | truncated doubles |
| `sas64_short_be.sas7bdat` | `sas64_short.sas7bdat` in big-endian | truncated doubles, big-endian |
| `sas64_wlatin1.sas7bdat` | `sas64_haven.sas7bdat`, text in Windows-1252 (encoding 62) | SAS's usual encoding |
| `sas64_rle.sas7bdat` | `sas64_haven.sas7bdat`, rows RLE-compressed, a 0x9000 page | `SASYZCRL`, every RLE command but 0x1 |
| `sas64_rle_wide.sas7bdat` | `write_sas` of 4300-byte rows, RLE-compressed | RLE command 0x1 |
| `sas64_rdc.sas7bdat` | `sas64_haven.sas7bdat`, rows RDC-compressed on 0x4000 pages, 2 rows stored plain | `SASYZCR2` |
| `sas32_iris.sas7bdat` | haven `examples/iris.sas7bdat` (SAS 9.4) | 32-bit with the pad, a mix page, encoding 62 |
| `sas32_nopad_rle.sas7bdat` | `sas32_iris.sas7bdat`, pad removed, RLE | 32-bit RLE |
| `sas32_rdc.sas7bdat` | `sas32_iris.sas7bdat`, RDC | 32-bit RDC |

`data_readers_manifest.csv` lists the files with the `encoding` argument the case passes to
`read_dta` (empty: the file's own) and their origin; it is the case's analysis dataset.

The tests patch fixed byte offsets of `sas64_haven.sas7bdat`, `sas32_iris.sas7bdat`,
`sas64_wlatin1.sas7bdat`, `sas64_rle.sas7bdat`, `sas64_rdc.sas7bdat`, `dta118_haven.dta` and
`dta119_haven.dta`; `test_sas_fixture_offsets` and `test_stata_fixture_offsets` fail if a regenerated
file moves them.

## Licence of the two copied files

`dta118_stata_iris.dta` and `sas32_iris.sas7bdat` are `inst/examples/iris.dta` and
`inst/examples/iris.sas7bdat` of the R package haven 2.5.5, distributed under the MIT licence:

> Copyright (c) 2023 haven authors
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
> associated documentation files (the "Software"), to deal in the Software without restriction,
> including without limitation the rights to use, copy, modify, merge, publish, distribute,
> sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
> NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
> OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
> CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The iris measurements themselves (Anderson 1935, Fisher 1936) are public domain. Every other file
here is generated by the script above from simulated data.
