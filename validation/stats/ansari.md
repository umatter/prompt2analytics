# Validation: Ansari-Bradley Test of Scale (ansari.test)

## Method Overview

The Ansari-Bradley test compares the dispersion of two samples. The pooled sample is ranked
(`stats::ranks::midranks`, R's `rank()`), each observation is scored by its distance from the
ends of the ordering, `min(r, N + 1 - r)`, and the statistic AB is the sum of the scores of the
first sample. p2a reports a normal approximation and, for small samples without ties, an exact
distribution.

## Reference

| Package | Language | Function | Version | Case |
|---------|----------|----------|---------|------|
| stats | R | `ansari.test` | R 4.3.3 | `stat_rank_two_sample`, `stat_rank_small_groups` |
