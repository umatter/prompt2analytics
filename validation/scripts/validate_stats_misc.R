# R validation script for Phase 10: Stats & Misc methods
# Generates expected values for comparison with Rust implementations

# Power Analysis (power.t.test, power.prop.test, power.anova.test)
cat("=== Power Analysis ===\n")

# power.t.test: solve for power
result <- power.t.test(n = 30, delta = 0.5, sd = 1, sig.level = 0.05, type = "two.sample")
cat(sprintf("power.t.test (n=30, delta=0.5, sd=1): power = %.6f\n", result$power))

# power.t.test: solve for n
result <- power.t.test(delta = 0.5, sd = 1, sig.level = 0.05, power = 0.8, type = "two.sample")
cat(sprintf("power.t.test (power=0.8, delta=0.5): n = %.1f\n", result$n))

# power.prop.test: solve for power
result <- power.prop.test(n = 100, p1 = 0.3, p2 = 0.5, sig.level = 0.05)
cat(sprintf("power.prop.test (n=100, p1=0.3, p2=0.5): power = %.6f\n", result$power))

# power.prop.test: solve for n
result <- power.prop.test(p1 = 0.3, p2 = 0.5, sig.level = 0.05, power = 0.8)
cat(sprintf("power.prop.test (power=0.8, p1=0.3, p2=0.5): n = %.1f\n", result$n))

# power.anova.test: solve for power
result <- power.anova.test(groups = 3, n = 20, between.var = 0.25, within.var = 1, sig.level = 0.05)
cat(sprintf("power.anova.test (groups=3, n=20, between=0.25, within=1): power = %.6f\n", result$power))

# power.anova.test: solve for n
result <- power.anova.test(groups = 3, between.var = 0.25, within.var = 1, sig.level = 0.05, power = 0.8)
cat(sprintf("power.anova.test (power=0.8): n = %.1f\n", result$n))

# Correlation tests (cor.test)
cat("\n=== Correlation Tests ===\n")

# Pearson
set.seed(42)
x <- c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
y <- c(1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0)

result <- cor.test(x, y, method = "pearson")
cat(sprintf("Pearson cor: %.6f, t = %.6f, p = %.6f\n", result$estimate, result$statistic, result$p.value))
cat(sprintf("  conf.int: [%.6f, %.6f]\n", result$conf.int[1], result$conf.int[2]))

# Spearman
result <- cor.test(x, y, method = "spearman")
cat(sprintf("Spearman rho: %.6f, S = %.1f, p = %.6f\n", result$estimate, result$statistic, result$p.value))

# Kendall
result <- cor.test(x, y, method = "kendall")
cat(sprintf("Kendall tau: %.6f, z = %.6f, p = %.6f\n", result$estimate, result$statistic, result$p.value))

# Robust statistics
cat("\n=== Robust Statistics ===\n")

# fivenum
data <- c(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
result <- fivenum(data)
cat(sprintf("fivenum: min=%.1f, q1=%.1f, median=%.1f, q3=%.1f, max=%.1f\n",
            result[1], result[2], result[3], result[4], result[5]))

# IQR
result <- IQR(data)
cat(sprintf("IQR: %.6f\n", result))

# MAD
result <- mad(data)
cat(sprintf("MAD: %.6f\n", result))

# MAD with constant = 1 (no scaling)
result <- mad(data, constant = 1)
cat(sprintf("MAD (constant=1): %.6f\n", result))

# ECDF
e <- ecdf(c(1.0, 2.0, 3.0, 4.0, 5.0))
cat(sprintf("ECDF values: e(0)=%.2f, e(1)=%.2f, e(3)=%.2f, e(5)=%.2f, e(6)=%.2f\n",
            e(0), e(1), e(3), e(5), e(6)))

# density (kernel density estimation)
data <- c(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
d <- density(data, bw = 0.5, kernel = "gaussian", n = 10, from = 0, to = 6)
cat(sprintf("density: bw=%.6f, n=%d\n", d$bw, length(d$x)))
cat(sprintf("  x[1]=%.2f, y[1]=%.6f\n", d$x[1], d$y[1]))
cat(sprintf("  x[5]=%.2f, y[5]=%.6f\n", d$x[5], d$y[5]))

# Weighted statistics
cat("\n=== Weighted Statistics ===\n")

# weighted.mean
x <- c(1.0, 2.0, 3.0, 4.0, 5.0)
w <- c(1.0, 2.0, 3.0, 2.0, 1.0)
result <- weighted.mean(x, w)
cat(sprintf("weighted.mean: %.6f\n", result))

# Equal weights should equal regular mean
w_equal <- c(1.0, 1.0, 1.0, 1.0, 1.0)
result <- weighted.mean(x, w_equal)
cat(sprintf("weighted.mean (equal): %.6f (should equal %.6f)\n", result, mean(x)))

# cov.wt
data <- matrix(c(1, 2, 3, 4, 5, 4, 5, 6, 7, 8), nrow = 5, ncol = 2)
result <- cov.wt(data)
cat(sprintf("cov.wt center: [%.6f, %.6f]\n", result$center[1], result$center[2]))
cat(sprintf("cov.wt cov[1,1]=%.6f, cov[2,2]=%.6f, cov[1,2]=%.6f\n",
            result$cov[1,1], result$cov[2,2], result$cov[1,2]))

# cov.wt with weights
w <- c(1, 2, 3, 2, 1)
result <- cov.wt(data, wt = w)
cat(sprintf("cov.wt (weighted) center: [%.6f, %.6f]\n", result$center[1], result$center[2]))

# cov.wt ML method
result <- cov.wt(data, method = "ML")
cat(sprintf("cov.wt ML cov[1,1]=%.6f\n", result$cov[1,1]))

# Isotonic regression
cat("\n=== Isotonic Regression ===\n")

y <- c(1.0, 0.0, 4.0, 3.0, 3.0, 5.0, 4.0, 2.0, 0.0)
result <- isoreg(y)
cat(sprintf("isoreg n=%d\n", length(y)))
cat(sprintf("  yf: [%.4f", result$yf[1]))
for (i in 2:length(result$yf)) {
  cat(sprintf(", %.4f", result$yf[i]))
}
cat("]\n")

# Already monotone data
y2 <- c(1.0, 2.0, 3.0, 4.0, 5.0)
result2 <- isoreg(y2)
cat(sprintf("isoreg monotone: yf == y? %s\n", all.equal(result2$yf, y2)))

# Strictly decreasing (should pool to single block)
y3 <- c(5.0, 4.0, 3.0, 2.0, 1.0)
result3 <- isoreg(y3)
cat(sprintf("isoreg decreasing: yf = [%.4f", result3$yf[1]))
for (i in 2:length(result3$yf)) {
  cat(sprintf(", %.4f", result3$yf[i]))
}
cat("]\n")
cat(sprintf("  mean(y3) = %.4f\n", mean(y3)))

# Constrained optimization
cat("\n=== Constrained Optimization ===\n")

# Minimize x^2 + y^2 subject to x + y >= 1
# Optimal is at x = y = 0.5, with value = 0.5
f <- function(theta) theta[1]^2 + theta[2]^2
grad <- function(theta) c(2 * theta[1], 2 * theta[2])

ui <- matrix(c(1, 1), nrow = 1)  # x + y >= 1
ci <- 1
theta0 <- c(1, 1)

result <- constrOptim(theta0, f, grad, ui, ci)
cat(sprintf("constrOptim (quadratic): par = [%.6f, %.6f]\n", result$par[1], result$par[2]))
cat(sprintf("  value = %.6f (optimal = 0.5)\n", result$value))
cat(sprintf("  constraint: x + y = %.6f >= 1\n", sum(result$par)))

# Multiple constraints: minimize -x - y subject to x >= 0, y >= 0, x + y <= 1
f2 <- function(theta) -theta[1] - theta[2]
grad2 <- function(theta) c(-1, -1)

ui2 <- matrix(c(1, 0, 0, 1, -1, -1), nrow = 3, byrow = TRUE)
ci2 <- c(0, 0, -1)
theta0_2 <- c(0.3, 0.3)

result2 <- constrOptim(theta0_2, f2, grad2, ui2, ci2)
cat(sprintf("constrOptim (linear, 3 constraints): par = [%.6f, %.6f]\n", result2$par[1], result2$par[2]))
cat(sprintf("  value = %.6f (optimal = -1)\n", result2$value))

cat("\n=== Summary ===\n")
cat("Validation data generated successfully.\n")
