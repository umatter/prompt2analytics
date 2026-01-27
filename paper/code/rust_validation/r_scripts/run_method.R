#!/usr/bin/env Rscript
# run_method.R - Generic R method runner that outputs JSON
# Usage: Rscript run_method.R --method ols --data datasets/longley.csv --output results/r_ols.json

suppressPackageStartupMessages({
  library(optparse)
  library(jsonlite)
})

# Parse command line arguments
option_list <- list(
  make_option(c("-m", "--method"), type = "character", help = "Method name (e.g., ols, panel_fe)"),
  make_option(c("-d", "--data"), type = "character", help = "Path to input CSV file"),
  make_option(c("-o", "--output"), type = "character", help = "Path to output JSON file"),
  make_option(c("-y", "--dep_var"), type = "character", default = NULL, help = "Dependent variable name"),
  make_option(c("-x", "--indep_vars"), type = "character", default = NULL, help = "Independent variables (comma-separated)"),
  make_option(c("-e", "--entity"), type = "character", default = NULL, help = "Entity variable for panel data"),
  make_option(c("-t", "--time"), type = "character", default = NULL, help = "Time variable for panel data"),
  make_option(c("-c", "--cluster"), type = "character", default = NULL, help = "Cluster variable"),
  make_option(c("-i", "--instrument"), type = "character", default = NULL, help = "Instrumental variable(s)"),
  make_option(c("-k", "--k"), type = "integer", default = 3, help = "Number of clusters (for kmeans)"),
  make_option(c("-n", "--n_components"), type = "integer", default = NULL, help = "Number of components (for PCA)"),
  make_option(c("-p", "--arima_order"), type = "character", default = "1,1,1", help = "ARIMA order p,d,q"),
  make_option(c("-r", "--robust"), type = "character", default = NULL, help = "Robust SE type (hc0, hc1, etc.)"),
  make_option(c("-s", "--seed"), type = "integer", default = 42, help = "Random seed")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$method) || is.null(opt$data)) {
  stop("--method and --data are required")
}

# Set seed for reproducibility
set.seed(opt$seed)

# Load the data
data <- read.csv(opt$data)
n <- nrow(data)

# Parse independent variables
parse_vars <- function(vars_string) {
  if (is.null(vars_string)) return(NULL)
  trimws(strsplit(vars_string, ",")[[1]])
}

indep_vars <- parse_vars(opt$indep_vars)

# Find script directory (works with Rscript and source())
get_script_dir <- function() {
  # Try commandArgs first (for Rscript)
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  # Fallback: current working directory + r_scripts
  if (file.exists("r_scripts/methods")) {
    return("r_scripts")
  }
  # Last resort
  return(".")
}

script_dir <- get_script_dir()
method_script <- file.path(script_dir, "methods", paste0(opt$method, ".R"))
if (!file.exists(method_script)) {
  stop(paste("Method script not found:", method_script))
}
source(method_script)

# Run the method (each method script defines run_method())
result <- run_method(
  data = data,
  dep_var = opt$dep_var,
  indep_vars = indep_vars,
  entity_var = opt$entity,
  time_var = opt$time,
  cluster_var = opt$cluster,
  instrument_vars = parse_vars(opt$instrument),
  k = opt$k,
  n_components = opt$n_components,
  arima_order = as.integer(strsplit(opt$arima_order, ",")[[1]]),
  robust = opt$robust,
  seed = opt$seed
)

# Build output structure
output <- list(
  method = opt$method,
  dataset = basename(opt$data),
  n = n,
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  results = result
)

# Write JSON output
if (!is.null(opt$output)) {
  dir.create(dirname(opt$output), showWarnings = FALSE, recursive = TRUE)
  write_json(output, opt$output, pretty = TRUE, auto_unbox = TRUE, digits = 10)
  cat("Results written to:", opt$output, "\n")
} else {
  cat(toJSON(output, pretty = TRUE, auto_unbox = TRUE, digits = 10))
}
