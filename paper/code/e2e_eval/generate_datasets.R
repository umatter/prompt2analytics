#!/usr/bin/env Rscript
# generate_datasets.R
# Generate 6 synthetic evaluation datasets for e2e evaluation of prompt2analytics
# Each dataset has a known DGP so tool outputs can be verified against ground truth.

set.seed(42)

script_dir <- tryCatch(
  dirname(sys.frame(1)$ofile),
  error = function(e) getwd()
)
outdir <- file.path(script_dir, "datasets")
if (!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

cat("Output directory:", outdir, "\n\n")

# ---------------------------------------------------------------------------
# Dataset 1: eval_cross_section.csv  (n=500)
# ---------------------------------------------------------------------------
cat("Generating Dataset 1: eval_cross_section.csv ...\n")

n1 <- 500
education  <- sample(8:22, n1, replace = TRUE)
experience <- sample(0:40, n1, replace = TRUE)
department <- sample(c("Engineering", "Marketing", "Finance", "Operations", "HR"),
                     n1, replace = TRUE)
gender     <- sample(0:1, n1, replace = TRUE)
salary     <- 25000 + 3000 * education + 1500 * experience +
              500 * education * experience + rnorm(n1, 0, 8000)

df1 <- data.frame(salary, education, experience, department, gender)
write.csv(df1, file.path(outdir, "eval_cross_section.csv"), row.names = FALSE)
cat("  rows:", nrow(df1), " cols:", ncol(df1), "\n")

# ---------------------------------------------------------------------------
# Dataset 2: eval_panel.csv  (N=50, T=10)
# ---------------------------------------------------------------------------
cat("Generating Dataset 2: eval_panel.csv ...\n")

N2 <- 50; T2 <- 10
firm_ids <- 1:N2
years    <- 2010:2019

firm_fe   <- rnorm(N2, 0, 100)
year_fe   <- rnorm(T2, 0, 30)
sectors   <- sample(c("Manufacturing", "Tech", "Finance", "Retail"), N2, replace = TRUE)

panel <- expand.grid(firm_id = firm_ids, year = years)
panel <- panel[order(panel$firm_id, panel$year), ]

panel$value   <- rnorm(nrow(panel), 1000, 300)
panel$capital <- rnorm(nrow(panel), 500, 150)
panel$sector  <- sectors[panel$firm_id]

panel$investment <- 100 +
  0.1  * panel$value +
  0.05 * panel$capital +
  firm_fe[panel$firm_id] +
  year_fe[panel$year - 2009] +
  rnorm(nrow(panel), 0, 50)

write.csv(panel, file.path(outdir, "eval_panel.csv"), row.names = FALSE)
cat("  rows:", nrow(panel), " cols:", ncol(panel), "\n")

# ---------------------------------------------------------------------------
# Dataset 3: eval_timeseries.csv  (T=200)
# ---------------------------------------------------------------------------
cat("Generating Dataset 3: eval_timeseries.csv ...\n")

T3 <- 200
dates <- seq.Date(as.Date("2005-01-01"), by = "month", length.out = T3)

# gdp_growth: AR(2) + trend + seasonal
gdp <- numeric(T3)
eps_gdp <- rnorm(T3, 0, 1)
gdp[1] <- eps_gdp[1]
gdp[2] <- 0.6 * gdp[1] + eps_gdp[2]
for (t in 3:T3) {
  gdp[t] <- 0.6 * gdp[t-1] - 0.2 * gdp[t-2] + eps_gdp[t]
}
gdp <- gdp + 0.01 * (1:T3) + 0.5 * sin(2 * pi * (1:T3) / 12)

# unemployment: correlated with gdp (rho=-0.5) + AR(1) phi=0.8
eps_u_raw <- rnorm(T3)
eps_u     <- -0.5 * eps_gdp + sqrt(1 - 0.5^2) * eps_u_raw
unemp     <- numeric(T3)
unemp[1]  <- eps_u[1]
for (t in 2:T3) {
  unemp[t] <- 0.8 * unemp[t-1] + eps_u[t]
}

# inflation: correlated with unemployment (rho=0.3) + AR(1) phi=0.5
eps_i_raw <- rnorm(T3)
eps_i     <- 0.3 * eps_u + sqrt(1 - 0.3^2) * eps_i_raw
inflation <- numeric(T3)
inflation[1] <- eps_i[1]
for (t in 2:T3) {
  inflation[t] <- 0.5 * inflation[t-1] + eps_i[t]
}

df3 <- data.frame(date = dates, gdp_growth = gdp,
                  unemployment = unemp, inflation = inflation)
write.csv(df3, file.path(outdir, "eval_timeseries.csv"), row.names = FALSE)
cat("  rows:", nrow(df3), " cols:", ncol(df3), "\n")

# ---------------------------------------------------------------------------
# Dataset 4: eval_treatment.csv  (n=1000 units x 2 periods = 2000 rows)
# ---------------------------------------------------------------------------
cat("Generating Dataset 4: eval_treatment.csv ...\n")

n4 <- 1000
x1 <- rnorm(n4)
x2 <- rnorm(n4)
x3 <- rnorm(n4)  # noise / potential instrument

propensity <- plogis(0.5 * x1 - 0.3 * x2)
treatment  <- rbinom(n4, 1, propensity)

# Expand to 2 periods (period 0 and 1)
unit_id <- rep(1:n4, each = 2)
period  <- rep(0:1, times = n4)
x1_exp  <- rep(x1, each = 2)
x2_exp  <- rep(x2, each = 2)
x3_exp  <- rep(x3, each = 2)
trt_exp <- rep(treatment, each = 2)

# outcome: treatment effect = 0 in period 0, = 3 in period 1
outcome <- 5 + 3 * trt_exp * period + 2 * x1_exp + 1 * x2_exp + rnorm(n4 * 2, 0, 2)

df4 <- data.frame(unit_id = unit_id, period = period,
                  treatment = trt_exp, x1 = x1_exp, x2 = x2_exp, x3 = x3_exp,
                  outcome = outcome)
write.csv(df4, file.path(outdir, "eval_treatment.csv"), row.names = FALSE)
cat("  rows:", nrow(df4), " cols:", ncol(df4), "\n")

# ---------------------------------------------------------------------------
# Dataset 5: eval_messy.csv  (n=300, messy version of cross-section DGP)
# ---------------------------------------------------------------------------
cat("Generating Dataset 5: eval_messy.csv ...\n")

n5 <- 300
edu5  <- sample(8:22, n5, replace = TRUE)
exp5  <- sample(0:40, n5, replace = TRUE)
dept5 <- sample(c("Engineering", "Marketing", "Finance", "Operations", "HR"),
                n5, replace = TRUE)
gen5  <- sample(0:1, n5, replace = TRUE)
sal5  <- 25000 + 3000 * edu5 + 1500 * exp5 + 500 * edu5 * exp5 + rnorm(n5, 0, 8000)

# Generate dates with mixed formats
base_dates <- seq.Date(as.Date("2023-01-01"), by = "day", length.out = n5)
fmt_choice <- sample(c("iso", "us"), n5, replace = TRUE)
var_06 <- ifelse(fmt_choice == "iso",
                 format(base_dates, "%Y-%m-%d"),
                 format(base_dates, "%m/%d/%Y"))

# Build data.frame with obfuscated column names
df5 <- data.frame(
  var_01 = sal5,
  var_02 = as.character(edu5),  # will inject "N/A"
  var_03 = exp5,
  var_04 = dept5,
  var_05 = gen5,
  var_06 = var_06,
  stringsAsFactors = FALSE
)

# Inject ~5% missing as -99 in var_01
idx_salary <- sample(n5, round(0.05 * n5))
df5$var_01[idx_salary] <- -99

# Inject ~5% "N/A" in var_02 (making it mixed type)
idx_edu <- sample(n5, round(0.05 * n5))
df5$var_02[idx_edu] <- "N/A"

write.csv(df5, file.path(outdir, "eval_messy.csv"), row.names = FALSE)
cat("  rows:", nrow(df5), " cols:", ncol(df5), "\n")

# ---------------------------------------------------------------------------
# Dataset 6: eval_survey.csv  (n=400)
# ---------------------------------------------------------------------------
cat("Generating Dataset 6: eval_survey.csv ...\n")

n6 <- 400
age    <- sample(18:75, n6, replace = TRUE)
income <- rlnorm(n6, meanlog = 10.5, sdlog = 0.7)
region <- sample(c("North", "South", "East", "West"), n6, replace = TRUE)

# satisfaction: ordered 1-5 via latent variable
latent_sat <- 0.5 + 0.02 * age + 0.00003 * income + rnorm(n6)
thresholds <- c(-1, 0, 1, 2)
satisfaction <- as.integer(cut(latent_sat, breaks = c(-Inf, thresholds, Inf), labels = 1:5))

# purchased: binary logit
prob_purchase <- plogis(-1 + 0.01 * age + 0.00002 * income)
purchased     <- rbinom(n6, 1, prob_purchase)

# visits: Poisson count
lambda_visits <- exp(0.5 + 0.01 * age + 0.000005 * income)
visits        <- rpois(n6, lambda_visits)

df6 <- data.frame(age, income, region, satisfaction, purchased, visits)
write.csv(df6, file.path(outdir, "eval_survey.csv"), row.names = FALSE)
cat("  rows:", nrow(df6), " cols:", ncol(df6), "\n")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
cat("\n=== Summary ===\n")
files <- list.files(outdir, pattern = "\\.csv$", full.names = TRUE)
for (f in files) {
  d <- read.csv(f)
  cat(sprintf("  %-30s  %4d rows x %2d cols\n", basename(f), nrow(d), ncol(d)))
}
cat("\nAll datasets written to:", outdir, "\n")
