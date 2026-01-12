#!/usr/bin/env Rscript
# String Operations Benchmarks for Cross-Language Comparison
# Compares R's stringi and stringr against p2a Rust/Polars implementation

library(microbenchmark)
library(stringi)
library(stringr)
library(data.table)

# Ensure reproducibility
set.seed(42)

# =============================================================================
# DATA GENERATION
# =============================================================================

# Generate string data matching Rust benchmarks
generate_string_data <- function(n) {
  # Text with leading/trailing whitespace (for trim tests)
  text <- sapply(1:n, function(i) {
    spaces_before <- paste(rep(" ", sample(0:4, 1)), collapse = "")
    spaces_after <- paste(rep(" ", sample(0:4, 1)), collapse = "")
    word_len <- sample(5:19, 1)
    word <- paste(sample(letters, word_len, replace = TRUE), collapse = "")
    paste0(spaces_before, word, spaces_after)
  })

  # Mixed case text (for case conversion tests)
  mixed_case <- sapply(1:n, function(i) {
    len <- sample(10:49, 1)
    chars <- sapply(1:len, function(j) {
      if (runif(1) < 0.5) {
        sample(LETTERS, 1)
      } else {
        sample(letters, 1)
      }
    })
    paste(chars, collapse = "")
  })

  # Email-like strings (for regex tests)
  domains <- c("gmail.com", "yahoo.com", "outlook.com", "example.org", "test.net")
  email <- paste0("user", 1:n, "@", sample(domains, n, replace = TRUE))

  # Phone-like strings (for regex extract tests)
  phone <- sapply(1:n, function(i) {
    area <- sample(100:999, 1)
    prefix <- sample(100:999, 1)
    line <- sample(1000:9999, 1)
    if (runif(1) < 0.5) {
      sprintf("(%d) %d-%d", area, prefix, line)
    } else {
      sprintf("%d-%d-%d", area, prefix, line)
    }
  })

  # Code-like strings with patterns (for regex count tests)
  code <- sapply(1:n, function(i) {
    n_vars <- sample(1:9, 1)
    vars <- sapply(1:n_vars, function(j) {
      var_len <- sample(3:7, 1)
      var <- paste(sample(letters, var_len, replace = TRUE), collapse = "")
      sprintf("let %s = %d;", var, sample(0:99, 1))
    })
    paste(vars, collapse = " ")
  })

  # Delimited strings (for split tests)
  delimited <- sapply(1:n, function(i) {
    n_parts <- sample(2:7, 1)
    parts <- sapply(1:n_parts, function(j) {
      len <- sample(3:9, 1)
      paste(sample(letters, len, replace = TRUE), collapse = "")
    })
    paste(parts, collapse = ",")
  })

  # First and last name columns (for concat tests)
  first_names <- c("John", "Jane", "Bob", "Alice", "Charlie", "Diana", "Eve", "Frank")
  last_names <- c("Smith", "Johnson", "Williams", "Brown", "Jones", "Davis", "Miller")

  data.table(
    id = 1:n,
    text = text,
    mixed_case = mixed_case,
    email = email,
    phone = phone,
    code = code,
    delimited = delimited,
    first_name = sample(first_names, n, replace = TRUE),
    last_name = sample(last_names, n, replace = TRUE)
  )
}

# =============================================================================
# TRIM BENCHMARKS
# =============================================================================

benchmark_trim <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking trim with n=%d\n", n))
    dt <- generate_string_data(n)

    bm <- microbenchmark(
      stringi = stri_trim_both(dt$text),
      stringr = str_trim(dt$text),
      base_r = trimws(dt$text),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("trim_", n)]] <- summary(bm)
  }

  results
}

# =============================================================================
# CASE CONVERSION BENCHMARKS
# =============================================================================

benchmark_case_conversion <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking case conversion with n=%d\n", n))
    dt <- generate_string_data(n)

    # To lowercase
    bm_lower <- microbenchmark(
      stringi = stri_trans_tolower(dt$mixed_case),
      stringr = str_to_lower(dt$mixed_case),
      base_r = tolower(dt$mixed_case),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("to_lowercase_", n)]] <- summary(bm_lower)

    # To uppercase
    bm_upper <- microbenchmark(
      stringi = stri_trans_toupper(dt$mixed_case),
      stringr = str_to_upper(dt$mixed_case),
      base_r = toupper(dt$mixed_case),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("to_uppercase_", n)]] <- summary(bm_upper)
  }

  results
}

# =============================================================================
# REPLACE BENCHMARKS
# =============================================================================

benchmark_replace <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking replace with n=%d\n", n))
    dt <- generate_string_data(n)

    # Literal replacement
    bm_literal <- microbenchmark(
      stringi = stri_replace_all_fixed(dt$email, "gmail.com", "google.com"),
      stringr = str_replace_all(dt$email, fixed("gmail.com"), "google.com"),
      base_r = gsub("gmail.com", "google.com", dt$email, fixed = TRUE),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("replace_literal_", n)]] <- summary(bm_literal)

    # Regex replacement
    bm_regex <- microbenchmark(
      stringi = stri_replace_first_regex(dt$email, "@\\w+\\.com", "@replaced.com"),
      stringr = str_replace(dt$email, "@\\w+\\.com", "@replaced.com"),
      base_r = sub("@\\w+\\.com", "@replaced.com", dt$email, perl = TRUE),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("regex_replace_", n)]] <- summary(bm_regex)
  }

  results
}

# =============================================================================
# REGEX EXTRACT BENCHMARKS
# =============================================================================

benchmark_regex_extract <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking regex extract with n=%d\n", n))
    dt <- generate_string_data(n)

    # Extract area code from phone numbers
    bm_extract <- microbenchmark(
      stringi = stri_match_first_regex(dt$phone, "\\(?(\\d{3})\\)?")[, 2],
      stringr = str_match(dt$phone, "\\(?(\\d{3})\\)?")[, 2],
      base_r = {
        m <- regmatches(dt$phone, regexec("\\(?(\\d{3})\\)?", dt$phone))
        sapply(m, function(x) if (length(x) > 1) x[2] else NA)
      },
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("regex_extract_", n)]] <- summary(bm_extract)
  }

  results
}

# =============================================================================
# REGEX COUNT BENCHMARKS
# =============================================================================

benchmark_regex_count <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking regex count with n=%d\n", n))
    dt <- generate_string_data(n)

    # Count "let" keywords in code
    bm_count <- microbenchmark(
      stringi = stri_count_regex(dt$code, "let\\s+\\w+"),
      stringr = str_count(dt$code, "let\\s+\\w+"),
      base_r = sapply(gregexpr("let\\s+\\w+", dt$code, perl = TRUE), function(x) sum(x > 0)),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("regex_count_", n)]] <- summary(bm_count)
  }

  results
}

# =============================================================================
# STRING SPLIT BENCHMARKS
# =============================================================================

benchmark_split <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking split with n=%d\n", n))
    dt <- generate_string_data(n)

    bm_split <- microbenchmark(
      stringi = stri_split_fixed(dt$delimited, ",", n = 4),
      stringr = str_split(dt$delimited, fixed(","), n = 4),
      base_r = strsplit(dt$delimited, ",", fixed = TRUE),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("split_", n)]] <- summary(bm_split)
  }

  results
}

# =============================================================================
# STRING CONCAT BENCHMARKS
# =============================================================================

benchmark_concat <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking concat with n=%d\n", n))
    dt <- generate_string_data(n)

    bm_concat <- microbenchmark(
      stringi = stri_c(dt$first_name, dt$last_name, sep = " "),
      stringr = str_c(dt$first_name, dt$last_name, sep = " "),
      base_r = paste(dt$first_name, dt$last_name),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("concat_", n)]] <- summary(bm_concat)
  }

  results
}

# =============================================================================
# STRING LENGTH BENCHMARKS
# =============================================================================

benchmark_length <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking length with n=%d\n", n))
    dt <- generate_string_data(n)

    bm_length <- microbenchmark(
      stringi = stri_length(dt$text),
      stringr = str_length(dt$text),
      base_r = nchar(dt$text),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("length_", n)]] <- summary(bm_length)
  }

  results
}

# =============================================================================
# STRING SUBSTRING BENCHMARKS
# =============================================================================

benchmark_substring <- function() {
  results <- list()

  for (n in c(10000, 100000)) {
    cat(sprintf("Benchmarking substring with n=%d\n", n))
    dt <- generate_string_data(n)

    # Extract first 10 characters
    bm_substr <- microbenchmark(
      stringi = stri_sub(dt$mixed_case, 1, 10),
      stringr = str_sub(dt$mixed_case, 1, 10),
      base_r = substr(dt$mixed_case, 1, 10),
      times = 100,
      unit = "microseconds"
    )

    results[[paste0("substring_", n)]] <- summary(bm_substr)
  }

  results
}

# =============================================================================
# RUN ALL BENCHMARKS
# =============================================================================

cat("=== Trim Benchmarks ===\n")
trim_results <- benchmark_trim()

cat("\n=== Case Conversion Benchmarks ===\n")
case_results <- benchmark_case_conversion()

cat("\n=== Replace Benchmarks ===\n")
replace_results <- benchmark_replace()

cat("\n=== Regex Extract Benchmarks ===\n")
extract_results <- benchmark_regex_extract()

cat("\n=== Regex Count Benchmarks ===\n")
count_results <- benchmark_regex_count()

cat("\n=== Split Benchmarks ===\n")
split_results <- benchmark_split()

cat("\n=== Concat Benchmarks ===\n")
concat_results <- benchmark_concat()

cat("\n=== Length Benchmarks ===\n")
length_results <- benchmark_length()

cat("\n=== Substring Benchmarks ===\n")
substring_results <- benchmark_substring()

# =============================================================================
# SAVE RESULTS
# =============================================================================

save_results <- function(results, filename) {
  df <- do.call(rbind, lapply(names(results), function(name) {
    r <- results[[name]]
    data.frame(
      method = name,
      expr = as.character(r$expr),
      mean_us = r$mean,
      median_us = r$median,
      min_us = r$min,
      max_us = r$max,
      n_eval = r$neval
    )
  }))

  write.csv(df, filename, row.names = FALSE)
  cat(sprintf("Results saved to %s\n", filename))
}

# Create results directory if needed
dir.create("results", showWarnings = FALSE)

save_results(trim_results, "results/string_trim.csv")
save_results(case_results, "results/string_case.csv")
save_results(replace_results, "results/string_replace.csv")
save_results(extract_results, "results/string_extract.csv")
save_results(count_results, "results/string_count.csv")
save_results(split_results, "results/string_split.csv")
save_results(concat_results, "results/string_concat.csv")
save_results(length_results, "results/string_length.csv")
save_results(substring_results, "results/string_substring.csv")

# =============================================================================
# PRINT SUMMARY
# =============================================================================

cat("\n=== Summary ===\n")

cat("\nTrim results:\n")
print(do.call(rbind, trim_results))

cat("\nCase conversion results:\n")
print(do.call(rbind, case_results))

cat("\nReplace results:\n")
print(do.call(rbind, replace_results))

cat("\nRegex extract results:\n")
print(do.call(rbind, extract_results))

cat("\nRegex count results:\n")
print(do.call(rbind, count_results))

cat("\nSplit results:\n")
print(do.call(rbind, split_results))

cat("\nConcat results:\n")
print(do.call(rbind, concat_results))

cat("\nLength results:\n")
print(do.call(rbind, length_results))

cat("\nSubstring results:\n")
print(do.call(rbind, substring_results))
