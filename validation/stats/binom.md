# Validation: Exact Binomial Test (binom.test)

## Method Overview

`binom_test(x, n, p, alternative, conf_level)` tests H0: the success probability is `p` from `x`
successes in `n` trials. One-sided p-values are binomial tails, the two-sided p-value sums the
probabilities of the outcomes no more likely than the one observed, and the confidence interval
is Clopper-Pearson (Beta quantiles).

## Reference

| Package | Language | Function | Version | Case |
|---------|----------|----------|---------|------|
| stats | R | `binom.test` | R 4.3.3 | `stat_rank_binom` |
