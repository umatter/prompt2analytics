# validation/r/cases/ml_apriori_t4.R -- T4-WP23 association-rule references (ML-117, ML-118, ML-119).
# arules 1.7-10 (the last release for R < 4.5).
#  * ml_apriori_baskets -- 100 simulated baskets over 8 items (0/1 matrix, dataset apriori_baskets), item h in
#    exactly 7 baskets so that min support 0.07 sits on the boundary (ML-117: support >= minsupp keeps h).
#    arules::apriori(target = "frequent itemsets") and arules::eclat at support 0.07, maxlen 10: every frequent
#    itemset's support and count. arules::apriori(target = "rules") at support 0.07, confidence 0.5, minlen 1,
#    maxlen 10: one-item right-hand sides, empty left-hand sides included (ML-118); support, confidence, lift,
#    coverage, count, leverage and conviction (arules::interestMeasure) per rule, keyed "{lhs} => {rhs}" with the
#    items of each side in sorted order.
suppressPackageStartupMessages(library(arules))

.wp23_key <- function(labels) {
  # arules prints itemsets as "{a,b}"; p2a keys are "{a,b}" with items sorted, the same text
  vapply(labels, function(s) {
    inner <- gsub("^\\{|\\}$", "", s)
    if (!nzchar(inner)) return("{}")
    paste0("{", paste(sort(strsplit(inner, ",", fixed = TRUE)[[1]]), collapse = ","), "}")
  }, "")
}

case("ml_apriori_baskets", "ml/association", "arules apriori and eclat itemsets and apriori rules, 100 baskets, support on the boundary", function(ctx) {
  set.seed(20261115)
  n <- 100
  items <- letters[1:8]
  p <- c(0.55, 0.45, 0.35, 0.30, 0.25, 0.20, 0.15, 0)
  M <- sapply(p, function(pi) as.integer(runif(n) < pi))
  M[, 2] <- ifelse(M[, 1] == 1 & runif(n) < 0.5, 1L, M[, 2])      # b follows a
  M[, 3] <- ifelse(M[, 1] == 1 & M[, 2] == 1 & runif(n) < 0.4, 1L, M[, 3])
  M[sample.int(n, 7), 8] <- 1L                                      # h in exactly 7 baskets
  colnames(M) <- items
  d <- ctx$gen_dataset("apriori_baskets", data.frame(id = seq_len(n), M))
  stopifnot(sum(d$h) == 7)
  tr <- as(as.matrix(d[, items]) == 1, "transactions")
  fi <- apriori(tr, parameter = list(support = 0.07, maxlen = 10, minlen = 1, target = "frequent itemsets"),
                control = list(verbose = FALSE))
  ec <- eclat(tr, parameter = list(support = 0.07, maxlen = 10, minlen = 1), control = list(verbose = FALSE))
  stopifnot(length(fi) == length(ec), "{h}" %in% .wp23_key(labels(fi)))
  ru <- apriori(tr, parameter = list(support = 0.07, confidence = 0.5, minlen = 1, maxlen = 10, target = "rules"),
                control = list(verbose = FALSE))
  stopifnot(all(size(rhs(ru)) == 1))
  q <- quality(ru)
  key <- paste(.wp23_key(labels(lhs(ru))), "=>", .wp23_key(labels(rhs(ru))))
  im <- interestMeasure(ru, c("leverage", "conviction"), transactions = tr)
  nm <- function(x, k) { names(x) <- k; named(x) }
  fk <- .wp23_key(labels(fi)); ek <- .wp23_key(labels(ec))
  conv <- im$conviction
  v <- list(
    n_itemsets = length(fi), n_rules = length(ru), n_rules_empty_lhs = sum(size(lhs(ru)) == 0),
    itemset_support = nm(quality(fi)$support, fk), itemset_count = nm(quality(fi)$count, fk),
    eclat_support = nm(quality(ec)$support, ek),
    rule_support = nm(q$support, key), rule_confidence = nm(q$confidence, key), rule_lift = nm(q$lift, key),
    rule_coverage = nm(q$coverage, key), rule_count = nm(q$count, key),
    rule_leverage = nm(im$leverage, key),
    rule_conviction_finite = nm(conv[is.finite(conv)], key[is.finite(conv)])   # the harness refuses Inf
  )
  list(values = v, reference = paste0("arules::apriori and arules::eclat ", packageVersion("arules"), ", interestMeasure"),
       call = "apriori(tr, parameter = list(support = 0.07, confidence = 0.5, minlen = 1, maxlen = 10, target = 'rules'))",
       packages = "arules")
})
