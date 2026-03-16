#!/usr/bin/env Rscript
# power.t.test, power.prop.test, power.anova.test R Benchmarks

library(microbenchmark)

cat("=== Power Analysis R Benchmarks ===\n")

# power.t.test
cat("\n--- power.t.test ---\n")
for (n in c(100, 1000, 10000)) {
  bm_t <- microbenchmark(
    power.t.test(n = n, delta = 0.5, sd = 1, sig.level = 0.05),
    times = 1000,
    unit = "microseconds"
  )
  cat(sprintf("  n=%d compute power:  %.2f us (median)\n", n, median(bm_t$time) / 1000))
}
bm_t_solve <- microbenchmark(
  power.t.test(delta = 0.5, sd = 1, sig.level = 0.05, power = 0.8),
  times = 1000,
  unit = "microseconds"
)
cat(sprintf("  Solve for n:    %.2f us (median)\n", median(bm_t_solve$time) / 1000))

# power.prop.test
cat("\n--- power.prop.test ---\n")
for (n in c(100, 1000, 10000)) {
  bm_prop <- microbenchmark(
    power.prop.test(n = n, p1 = 0.5, p2 = 0.6, sig.level = 0.05),
    times = 1000,
    unit = "microseconds"
  )
  cat(sprintf("  n=%d compute power:  %.2f us (median)\n", n, median(bm_prop$time) / 1000))
}
bm_prop_solve <- microbenchmark(
  power.prop.test(p1 = 0.5, p2 = 0.6, sig.level = 0.05, power = 0.8),
  times = 1000,
  unit = "microseconds"
)
cat(sprintf("  Solve for n:    %.2f us (median)\n", median(bm_prop_solve$time) / 1000))

# power.anova.test
cat("\n--- power.anova.test ---\n")
for (n in c(100, 1000, 10000)) {
  bm_anova <- microbenchmark(
    power.anova.test(groups = 4, n = n, between.var = 1, within.var = 3),
    times = 1000,
    unit = "microseconds"
  )
  cat(sprintf("  n=%d compute power:  %.2f us (median)\n", n, median(bm_anova$time) / 1000))
}
bm_anova_solve <- microbenchmark(
  power.anova.test(groups = 4, between.var = 1, within.var = 3, power = 0.8),
  times = 1000,
  unit = "microseconds"
)
cat(sprintf("  Solve for n:    %.2f us (median)\n", median(bm_anova_solve$time) / 1000))

# Validation
cat("\n=== Validation ===\n")
result_t <- power.t.test(n = 100, delta = 0.5, sd = 1, sig.level = 0.05)
cat(sprintf("power.t.test: power = %.6f\n", result_t$power))

result_prop <- power.prop.test(n = 100, p1 = 0.5, p2 = 0.6, sig.level = 0.05)
cat(sprintf("power.prop.test: power = %.6f\n", result_prop$power))

result_anova <- power.anova.test(groups = 4, n = 100, between.var = 1, within.var = 3)
cat(sprintf("power.anova.test: power = %.6f\n", result_anova$power))
