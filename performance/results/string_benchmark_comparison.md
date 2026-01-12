# String Operations Benchmark: Rust/Polars vs R (stringi/stringr)

**Date:** 2026-01-12
**Test Sizes:** 10,000 and 100,000 rows
**Iterations:** 100 per benchmark

## Summary Table

All times in milliseconds (ms). Best performer highlighted.

### n = 10,000 rows

| Operation | Rust/Polars | stringi | stringr | base R | Fastest |
|-----------|-------------|---------|---------|--------|---------|
| str_length | **0.32** | 0.11 | 0.11 | 0.72 | **stringi** (3x faster) |
| str_concat | 1.39 | **1.21** | 1.24 | 1.68 | **stringi** (15% faster) |
| str_substring | 2.58 | 1.01 | 1.02 | **0.74** | **base R** (3.5x faster) |
| str_split | 5.31 | **5.23** | 5.27 | 7.11 | **stringi** (similar) |
| regex_replace | **2.93** | 9.40 | 9.57 | 3.36 | **Rust** (14% faster) |
| regex_extract | **2.10** | 5.25 | 5.36 | 95.02 | **Rust** (2.5x faster) |
| regex_count | **5.70** | 12.82 | 12.74 | 41.32 | **Rust** (2.2x faster) |

### n = 100,000 rows

| Operation | Rust/Polars | stringi | stringr | base R | Fastest |
|-----------|-------------|---------|---------|--------|---------|
| str_length | 3.59 | **2.22** | 2.19 | 10.36 | **stringi/stringr** (1.6x faster) |
| str_concat | 14.14 | **12.32** | 12.40 | 16.62 | **stringi** (15% faster) |
| str_substring | 25.79 | 18.25 | 18.22 | **13.47** | **base R** (1.9x faster) |
| str_split | 59.49 | **87.08** | 87.52 | 102.22 | **Rust** (32% faster) |
| regex_replace | **24.60** | 105.56 | 105.69 | 40.53 | **Rust** (1.6x faster) |
| regex_extract | **19.49** | 47.51 | 47.34 | 1178.22 | **Rust** (2.4x faster) |
| regex_count | **52.65** | 121.18 | 121.43 | 777.18 | **Rust** (2.3x faster) |

## Key Findings

### Where Rust/Polars Excels

1. **Regex Operations** - Rust consistently outperforms R's stringi/stringr:
   - `regex_replace`: 1.5-4x faster than stringi
   - `regex_extract`: 2.4x faster than stringi
   - `regex_count`: 2.2x faster than stringi

2. **String Split (at scale)** - At 100K rows, Rust is 32% faster than stringi

### Where R (stringi/stringr) Excels

1. **Basic String Operations**:
   - `str_length`: stringi is 1.6-3x faster (highly optimized C backend)
   - `str_concat`: stringi is 15% faster
   - `str_substring`: base R's `substr()` is fastest (3.5x faster at 10K)

2. **Simple Pattern Matching**: For literal string operations, R's implementation is competitive

## Analysis

### Regex Operations

The biggest performance win for Rust/Polars is in regex operations. The Rust `regex` crate uses a highly optimized DFA/NFA hybrid approach that outperforms R's PCRE-based regex engine, especially for:
- Pattern extraction with capture groups
- Pattern counting across large datasets
- Replacement operations

### Simple String Operations

R's stringi package has extremely optimized implementations for basic operations (length, concat, substring). These operations in stringi are implemented in ICU (International Components for Unicode) with decades of optimization.

### Scaling Characteristics

At larger dataset sizes (100K+), Rust's columnar operations in Polars show better scaling characteristics due to:
- Cache-friendly memory layout
- SIMD vectorization
- Zero-copy string views where possible

## Raw Data

### Rust/Polars Results (median times)

| Operation | 10K | 100K |
|-----------|-----|------|
| str_length | 0.32 ms | 3.59 ms |
| str_concat | 1.39 ms | 14.14 ms |
| str_substring | 2.58 ms | 25.79 ms |
| str_split | 5.31 ms | 59.49 ms |
| regex_replace | 2.93 ms | 24.60 ms |
| regex_extract | 2.10 ms | 19.49 ms |
| regex_count | 5.70 ms | 52.65 ms |

### R Results (median times)

| Operation | stringi | stringr | base R |
|-----------|---------|---------|--------|
| **10K rows** |
| trim | 1.85 ms | 2.10 ms | 7.39 ms |
| to_lowercase | 7.82 ms | 7.89 ms | 6.89 ms |
| to_uppercase | 7.65 ms | 7.42 ms | 6.69 ms |
| replace_literal | 0.74 ms | 0.80 ms | 0.64 ms |
| regex_replace | 9.40 ms | 9.57 ms | 3.36 ms |
| regex_extract | 5.25 ms | 5.36 ms | 95.02 ms |
| regex_count | 12.82 ms | 12.74 ms | 41.32 ms |
| split | 5.23 ms | 5.27 ms | 7.11 ms |
| concat | 1.21 ms | 1.24 ms | 1.68 ms |
| length | 0.11 ms | 0.11 ms | 0.72 ms |
| substring | 1.01 ms | 1.02 ms | 0.74 ms |
| **100K rows** |
| trim | 31.82 ms | 32.12 ms | 94.88 ms |
| to_lowercase | 84.35 ms | 84.53 ms | 73.63 ms |
| to_uppercase | 84.33 ms | 83.98 ms | 73.61 ms |
| replace_literal | 13.68 ms | 13.88 ms | 11.71 ms |
| regex_replace | 105.56 ms | 105.69 ms | 40.53 ms |
| regex_extract | 47.51 ms | 47.34 ms | 1178.22 ms |
| regex_count | 121.18 ms | 121.43 ms | 777.18 ms |
| split | 87.08 ms | 87.52 ms | 102.22 ms |
| concat | 12.32 ms | 12.40 ms | 16.62 ms |
| length | 2.22 ms | 2.19 ms | 10.36 ms |
| substring | 18.25 ms | 18.22 ms | 13.47 ms |

## Conclusion

Rust/Polars provides significant performance advantages for regex-heavy workloads (2-4x faster). For simple string manipulations, R's stringi package remains extremely competitive and is often faster for basic operations like length and concatenation.

**Recommendation:** Use p2a-core's string operations when:
- Working with regex patterns (extraction, replacement, counting)
- Processing very large datasets (100K+ rows)
- Combining string operations with other data transformations

Use R stringi when:
- Performing simple string manipulations
- Working with complex Unicode text processing
- Rapid prototyping of string operations
