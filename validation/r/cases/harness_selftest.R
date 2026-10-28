# validation/r/cases/harness_selftest.R -- end-to-end check of the harness itself
# (R -> JSON -> Rust). Area "harness" is exempt from the methods registry.
# crates/p2a-validation/tests/cases.rs recomputes these values from the CSV.

case("harness_selftest", "harness",
     "Harness self-test: Grunfeld inv mean and variance, named and unnamed vectors, NA as null",
     function(ctx) {
       d <- ctx$dataset("datasets/grunfeld.csv")
       stopifnot(nrow(d) == 200L, identical(names(d), c("firm", "year", "inv", "value", "capital")))
       list(
         values = list(
           n          = nrow(d),
           inv_mean   = mean(d$inv),
           inv_var    = var(d$inv),
           means      = named(c(value = mean(d$value), capital = mean(d$capital))),
           inv_max    = named(c(inv = max(d$inv))),
           inv_first3 = vec(d$inv[1:3]),
           with_na    = vec(c(sum(d$inv), NA))
         ),
         reference = "base R mean(), var(), max(), sum()",
         call = "mean(d$inv); var(d$inv); c(value = mean(d$value), capital = mean(d$capital)); c(inv = max(d$inv)); d$inv[1:3]; c(sum(d$inv), NA)",
         packages = "jsonlite"
       )
     })
