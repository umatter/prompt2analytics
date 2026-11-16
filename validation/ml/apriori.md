# Association rules (`ml::apriori`: `apriori`, `eclat`)

## Method

`apriori(transactions, config)` finds the frequent itemsets level by level (candidate join and prune, one pass
over the transactions per level); `eclat(transactions, config)` finds the same itemsets depth first over sorted
transaction-id lists. Both generate the rules with one shared generator and return the same itemsets and rules in
the same order.

## Conventions (arules)

- **Frequent itemset:** support = count / n at least `min_support` (the minimum count is the smallest count whose
  support reaches `min_support`, so an itemset exactly on the boundary is kept).
- **Rules:** for every frequent itemset of at least `max(1, min_length)` items, one rule per item on the
  right-hand side, the other items on the left. A frequent single item gives the rule `{} => {x}` (left-hand side
  support 1, lift 1), as arules' default `minlen = 1`; `min_length = 2` leaves out these rules and the single items.
  Confidence filters the rules (`min_confidence`). arules has no lift filter; `min_lift` is p2a's own and 0 (off)
  by default.
- **Measures:** support, confidence = supp(X ∪ Y) / supp(X), lift = confidence / supp(Y), coverage = supp(X),
  leverage = supp(X ∪ Y) - supp(X) supp(Y), conviction = (1 - supp(Y)) / (1 - confidence) (inf at confidence 1),
  count.
- **Order:** itemsets by length, then items; rules by lift (descending), then left-hand side, then right-hand side.
- **Defaults** differ from arules: `min_support` 0.01 and `min_confidence` 0.5 (arules: 0.1 and 0.8);
  `max_length` 10 as arules.
- **Refused:** `min_support` outside (0, 1] or NaN, `min_confidence` outside [0, 1], `min_lift` negative or not
  finite, `max_length` 0, `min_length` outside 1 to `max_length`, no transactions.
- **Budget:** no `maxtime`; the search and the rule generation stop at the call's budget (`Cancelled`).

## R reference

`arules::apriori` and `arules::eclat`, `arules::interestMeasure` (leverage, conviction), arules 1.7-10, R 4.3.3.
Case file: `validation/r/cases/ml_apriori_t4.R`. Registry: `validation/methods/ml_apriori_t4.json`.

**Status: R-pinned** (see the registry for the quantities).

## Cases, values and tests

| Case | Data | Decisive values |
|---|---|---|
| `ml_apriori_baskets` | 100 simulated baskets over a-h (`apriori_baskets.csv`, seed 20261115); h in exactly 7 | support 0.07, confidence 0.5, minlen 1, maxlen 10: 37 frequent itemsets ({h}, {f,g}, {b,g} on the boundary at 0.07), 50 rules, 3 with an empty left-hand side |

R-pinned tests: `test_validate_apriori_baskets`, `test_validate_eclat_baskets` (`apriori.rs`). Tolerance
(`validation/README.md`, coefficient row): n = 100 1e-8 relative; counts exact. Conviction is pinned where it is
finite (the harness refuses Inf); rules with confidence 1 are checked for `inf` in the test.

Property tests: `apriori_keeps_itemsets_exactly_at_min_support` (every k / n for n = 100 and 50),
`eclat_reports_each_itemset_once` (against apriori and a brute-force count of all 255 itemsets),
`apriori_refuses_invalid_settings`, `apriori_lists_itemsets_in_a_fixed_order`, `rules_have_one_item_on_the_right`,
`rules_match_a_brute_force_enumeration`, `rule_generator_handles_a_40_item_itemset`, `apriori_stops_at_the_budget`.

## Timing

Preliminary (lane timing, 2026-11-16), protocol as `roc_auc.md`: release builds, base `f80b0131` (v0.5.0), one
core, 9 interleaved processes per side, each the median of 2,001 / 401 / 41 calls, in µs. n baskets over 20
items, each item present with probability 0.25; `min_support` 0.05, `min_confidence` 0.5, `min_lift` 0 on both
sides (the base default 1.0 would drop rules this tree keeps).

| Function | n | base (µs) | this tree (µs) | ratio |
|---|---:|---:|---:|---:|
| `apriori` | 100 | 9,130 | 934.5 | 0.10 |
|  | 1,000 | 89,388 | 5,956 | 0.07 |
|  | 10,000 | 1,049,158 | 63,754 | 0.06 |
| `eclat` | 100 | 1,380 | 613.7 | 0.44 |
|  | 1,000 | 9,189 | 3,999 | 0.44 |
|  | 10,000 | 97,849 | 43,089 | 0.44 |
