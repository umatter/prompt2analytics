# requirements.R
# R package dependencies for paper exhibit generation
#
# Usage: Rscript requirements.R
# Or:    source("requirements.R")

# Required packages with minimum versions
required_packages <- list(

# Core tidyverse
  tidyverse   = "2.0.0",
  ggplot2     = "3.4.0",
  dplyr       = "1.1.0",
  tidyr       = "1.3.0",
  purrr       = "1.0.0",
  readr       = "2.1.0",
  stringr     = "1.5.0",

# Data formats
  jsonlite    = "1.8.0",

# Tables
  xtable      = "1.8-4",

# Visualization
  patchwork   = "1.1.0",
  scales      = "1.2.0",
  viridis     = "0.6.0",

# Benchmarking (for rust_validation)
  bench       = "1.1.0",
  optparse    = "1.7.0",

# Econometrics (for rust_validation R scripts)
  sandwich    = "3.0-0",
  lmtest      = "0.9-40",
  plm         = "2.6-0",
  fixest      = "0.11.0",
  AER         = "1.2-10",
  MASS        = "7.3-58",
  forecast    = "8.21"
)

# Function to check and install packages
install_if_missing <- function(pkg, min_version = NULL) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing: ", pkg)
    install.packages(pkg)
  } else if (!is.null(min_version)) {
    installed_version <- as.character(packageVersion(pkg))
    if (compareVersion(installed_version, min_version) < 0) {
      message("Upgrading ", pkg, " from ", installed_version, " to >= ", min_version)
      install.packages(pkg)
    }
  }
}

# Install/update all packages
message("Checking R package dependencies for paper...")
message("")

for (pkg in names(required_packages)) {
  install_if_missing(pkg, required_packages[[pkg]])
}

message("")
message("All packages installed. Versions:")
message("")

for (pkg in names(required_packages)) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    ver <- as.character(packageVersion(pkg))
    status <- if (compareVersion(ver, required_packages[[pkg]]) >= 0) "OK" else "OUTDATED"
    message(sprintf("  %-12s %s (%s)", pkg, ver, status))
  } else {
    message(sprintf("  %-12s MISSING", pkg))
  }
}

message("")
message("Done.")
