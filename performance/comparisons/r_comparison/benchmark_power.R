#!/usr/bin/env Rscript
# power.t.test, power.prop.test, power.anova.test R Benchmarks

library(microbenchmark)

cat("=== Power Analysis R Benchmarks ===\n")

# power.t.test
cat("\n--- power.t.test ---\n")
bm_t <- microbenchmark(
  power.t.test(n = 30, delta = 0.5, sd = 1, sig.level = 0.05),
  power.t.test(delta = 0.5, sd = 1, sig.level = 0.05, power = 0.8),
  times = 1000,
  unit = "microseconds"
)
cat(sprintf("  Compute power:  %.2f us (median)\n", median(bm_t$time[1:1000]) / 1000))
cat(sprintf("  Solve for n:    %.2f us (median)\n", median(bm_t$time[1001:2000]) / 1000))

# power.prop.test
cat("\n--- power.prop.test ---\n")
bm_prop <- microbenchmark(
  power.prop.test(n = 100, p1 = 0.5, p2 = 0.6, sig.level = 0.05),
  power.prop.test(p1 = 0.5, p2 = 0.6, sig.level = 0.05, power = 0.8),
  times = 1000,
  unit = "microseconds"
)
cat(sprintf("  Compute power:  %.2f us (median)\n", median(bm_prop$time[1:1000]) / 1000))
cat(sprintf("  Solve for n:    %.2f us (median)\n", median(bm_prop$time[1001:2000]) / 1000))

# power.anova.test
cat("\n--- power.anova.test ---\n")
bm_anova <- microbenchmark(
  power.anova.test(groups = 4, n = 20, between.var = 1, within.var = 3),
  power.anova.test(groups = 4, between.var = 1, within.var = 3, power = 0.8),
  times = 1000,
  unit = "microseconds"
)
cat(sprintf("  Compute power:  %.2f us (median)\n", median(bm_anova$time[1:1000]) / 1000))
cat(sprintf("  Solve for n:    %.2f us (median)\n", median(bm_anova$time[1001:2000]) / 1000))

# Validation
cat("\n=== Validation ===\n")
result_t <- power.t.test(n = 30, delta = 0.5, sd = 1, sig.level = 0.05)
cat(sprintf("power.t.test: power = %.6f\n", result_t$power))

result_prop <- power.prop.test(n = 100, p1 = 0.5, p2 = 0.6, sig.level = 0.05)
cat(sprintf("power.prop.test: power = %.6f\n", result_prop$power))

result_anova <- power.anova.test(groups = 4, n = 20, between.var = 1, within.var = 3)
cat(sprintf("power.anova.test: power = %.6f\n", result_anova$power))
