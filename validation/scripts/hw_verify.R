#!/usr/bin/env Rscript
# Verify Holt-Winters values against Rust implementation

y <- c(112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
       115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140)

cat("=== R HoltWinters with fixed parameters ===\n")
m <- HoltWinters(ts(y, frequency=12), alpha=0.2, beta=0.1, gamma=0.3, seasonal="additive")

cat("\nSSE:", m$SSE, "\n")
cat("Coefficients:\n")
print(m$coefficients)

cat("\nFitted values (xhat column):\n")
print(m$fitted[, "xhat"])

cat("\nFirst period mean:", mean(y[1:12]), "\n")
cat("Second period mean:", mean(y[13:24]), "\n")

cat("\n=== Initialization values ===\n")
# R's internal initialization for comparison
l_start <- mean(y[1:12])
b_start <- (mean(y[13:24]) - mean(y[1:12])) / 12
cat("l_start:", l_start, "\n")
cat("b_start:", b_start, "\n")

# Initial seasonal indices
s_init <- numeric(12)
for (j in 1:12) {
    dev1 <- y[j] - mean(y[1:12])
    dev2 <- y[j + 12] - mean(y[13:24])
    s_init[j] <- (dev1 + dev2) / 2
}
# Normalize to sum to zero
s_init <- s_init - mean(s_init)
cat("Initial seasonal indices:\n")
for (i in 1:12) {
    cat(sprintf("  s[%d] = %.4f\n", i-1, s_init[i]))
}

cat("\n=== Summary ===\n")
cat("Expected SSE:", 244.53, "(from Rust trace)\n")
cat("R's SSE:", m$SSE, "\n")
cat("Match:", abs(m$SSE - 244.53) < 1, "\n")
