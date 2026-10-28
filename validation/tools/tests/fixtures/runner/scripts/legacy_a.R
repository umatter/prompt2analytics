# Fixture legacy script for validation/tools/tests/test_run_validation.sh (run by the
# fake Rscript only): writes its reference values relative to the working directory, as
# the scripts in validation/scripts/ do.
write.csv(data.frame(term = "x", estimate = 0.5), "validation/expected/legacy_a.csv", row.names = FALSE)
