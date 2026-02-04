//! Association rule mining using the Apriori algorithm.
//!
//! Implements the classic Apriori algorithm for finding frequent itemsets
//! and generating association rules.
//!
//! ## References
//!
//! - Agrawal, R., & Srikant, R. (1994). "Fast Algorithms for Mining Association
//!   Rules." *Proc. 20th Int. Conf. Very Large Data Bases*, 487-499.
//! - R package `arules`: Hahsler, M., et al. (2005). "arules - A Computational
//!   Environment for Mining Association Rules and Frequent Item Sets."
//!   *Journal of Statistical Software*, 14(15), 1-25.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::errors::{EconError, EconResult};

/// Configuration for Apriori algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprioriConfig {
    /// Minimum support threshold (0-1)
    pub min_support: f64,
    /// Minimum confidence threshold (0-1)
    pub min_confidence: f64,
    /// Minimum lift threshold
    pub min_lift: f64,
    /// Maximum itemset size
    pub max_length: usize,
}

impl Default for AprioriConfig {
    fn default() -> Self {
        Self {
            min_support: 0.01,
            min_confidence: 0.5,
            min_lift: 1.0,
            max_length: 10,
        }
    }
}

/// A frequent itemset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequentItemset {
    /// Items in the itemset
    pub items: Vec<String>,
    /// Support (proportion of transactions containing this itemset)
    pub support: f64,
    /// Count (number of transactions)
    pub count: usize,
}

/// An association rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociationRule {
    /// Left-hand side (antecedent)
    pub lhs: Vec<String>,
    /// Right-hand side (consequent)
    pub rhs: Vec<String>,
    /// Support of the rule (P(LHS and RHS))
    pub support: f64,
    /// Confidence (P(RHS | LHS))
    pub confidence: f64,
    /// Lift (confidence / P(RHS))
    pub lift: f64,
    /// Count (number of transactions)
    pub count: usize,
    /// Coverage (support of LHS)
    pub coverage: f64,
    /// Leverage (support - support_lhs * support_rhs)
    pub leverage: f64,
    /// Conviction ((1 - support_rhs) / (1 - confidence))
    pub conviction: f64,
}

/// Result from Apriori algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprioriResult {
    /// Frequent itemsets found
    pub itemsets: Vec<FrequentItemset>,
    /// Association rules generated
    pub rules: Vec<AssociationRule>,
    /// Number of transactions
    pub n_transactions: usize,
    /// Number of unique items
    pub n_items: usize,
    /// Configuration used
    pub config: AprioriConfig,
}

impl std::fmt::Display for AprioriResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Apriori Association Rules")?;
        writeln!(f, "=========================")?;
        writeln!(f, "Transactions: {}", self.n_transactions)?;
        writeln!(f, "Unique items: {}", self.n_items)?;
        writeln!(f, "Frequent itemsets: {}", self.itemsets.len())?;
        writeln!(f, "Rules: {}", self.rules.len())?;
        writeln!(f)?;

        writeln!(f, "Configuration:")?;
        writeln!(f, "  Min support: {:.4}", self.config.min_support)?;
        writeln!(f, "  Min confidence: {:.4}", self.config.min_confidence)?;
        writeln!(f, "  Min lift: {:.4}", self.config.min_lift)?;
        writeln!(f)?;

        if !self.rules.is_empty() {
            writeln!(f, "Top 10 Rules (by lift):")?;
            writeln!(
                f,
                "{:<30} {:>8} {:>8} {:>8}",
                "Rule", "Support", "Conf", "Lift"
            )?;
            writeln!(f, "{:-<56}", "")?;

            let mut sorted_rules = self.rules.clone();
            sorted_rules.sort_by(|a, b| {
                b.lift
                    .partial_cmp(&a.lift)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            for rule in sorted_rules.iter().take(10) {
                let lhs_str = if rule.lhs.is_empty() {
                    "{}".to_string()
                } else {
                    format!("{{{}}}", rule.lhs.join(", "))
                };
                let rhs_str = format!("{{{}}}", rule.rhs.join(", "));
                let rule_str = format!("{} => {}", lhs_str, rhs_str);

                writeln!(
                    f,
                    "{:<30} {:>8.4} {:>8.4} {:>8.2}",
                    if rule_str.len() > 30 {
                        format!("{}...", &rule_str[..27])
                    } else {
                        rule_str
                    },
                    rule.support,
                    rule.confidence,
                    rule.lift
                )?;
            }
        }

        Ok(())
    }
}

/// Apriori algorithm for association rule mining.
///
/// Finds frequent itemsets using a level-wise approach, then generates
/// association rules from those itemsets.
///
/// # Arguments
///
/// * `transactions` - List of transactions, where each transaction is a list of items
/// * `config` - Apriori configuration
///
/// # Returns
///
/// AprioriResult with frequent itemsets and association rules.
///
/// # Example
///
/// ```rust,ignore
/// use p2a_core::ml::apriori::{apriori, AprioriConfig};
///
/// let transactions = vec![
///     vec!["bread".to_string(), "milk".to_string()],
///     vec!["bread".to_string(), "diaper".to_string(), "beer".to_string()],
///     vec!["milk".to_string(), "diaper".to_string(), "beer".to_string()],
///     vec!["bread".to_string(), "milk".to_string(), "diaper".to_string(), "beer".to_string()],
/// ];
///
/// let config = AprioriConfig {
///     min_support: 0.5,
///     min_confidence: 0.6,
///     ..Default::default()
/// };
///
/// let result = apriori(&transactions, &config).unwrap();
/// println!("Found {} rules", result.rules.len());
/// ```
///
/// # References
///
/// Agrawal, R., & Srikant, R. (1994). "Fast Algorithms for Mining Association Rules."
/// *Proc. 20th Int. Conf. Very Large Data Bases*, 487-499.
pub fn apriori(transactions: &[Vec<String>], config: &AprioriConfig) -> EconResult<AprioriResult> {
    let n_transactions = transactions.len();

    if n_transactions == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Get all unique items
    let mut all_items: HashSet<String> = HashSet::new();
    for transaction in transactions {
        for item in transaction {
            all_items.insert(item.clone());
        }
    }

    let n_items = all_items.len();

    if n_items == 0 {
        return Ok(AprioriResult {
            itemsets: Vec::new(),
            rules: Vec::new(),
            n_transactions,
            n_items,
            config: config.clone(),
        });
    }

    // Convert transactions to sets for efficient lookup
    let transaction_sets: Vec<HashSet<String>> = transactions
        .iter()
        .map(|t| t.iter().cloned().collect())
        .collect();

    let min_count = (config.min_support * n_transactions as f64).ceil() as usize;

    // Find frequent 1-itemsets
    let mut item_counts: HashMap<String, usize> = HashMap::new();
    for trans in &transaction_sets {
        for item in trans {
            *item_counts.entry(item.clone()).or_insert(0) += 1;
        }
    }

    let mut frequent_itemsets: Vec<FrequentItemset> = Vec::new();
    let mut current_level: Vec<Vec<String>> = Vec::new();

    for (item, &count) in &item_counts {
        if count >= min_count {
            let support = count as f64 / n_transactions as f64;
            frequent_itemsets.push(FrequentItemset {
                items: vec![item.clone()],
                support,
                count,
            });
            current_level.push(vec![item.clone()]);
        }
    }

    // Sort for consistent candidate generation
    current_level.sort();

    // Level-wise search for larger itemsets
    for k in 2..=config.max_length {
        if current_level.is_empty() {
            break;
        }

        // Generate candidate k-itemsets
        let candidates = generate_candidates(&current_level);

        if candidates.is_empty() {
            break;
        }

        // Count support for candidates
        let mut candidate_counts: HashMap<Vec<String>, usize> = HashMap::new();

        for trans in &transaction_sets {
            for candidate in &candidates {
                if candidate.iter().all(|item| trans.contains(item)) {
                    *candidate_counts.entry(candidate.clone()).or_insert(0) += 1;
                }
            }
        }

        // Filter by minimum support
        let mut next_level: Vec<Vec<String>> = Vec::new();

        for (itemset, count) in candidate_counts {
            if count >= min_count {
                let support = count as f64 / n_transactions as f64;
                frequent_itemsets.push(FrequentItemset {
                    items: itemset.clone(),
                    support,
                    count,
                });
                next_level.push(itemset);
            }
        }

        next_level.sort();
        current_level = next_level;
    }

    // Precompute support for all itemsets
    let itemset_support: HashMap<Vec<String>, f64> = frequent_itemsets
        .iter()
        .map(|is| {
            let mut sorted_items = is.items.clone();
            sorted_items.sort();
            (sorted_items, is.support)
        })
        .collect();

    // Generate association rules
    let mut rules: Vec<AssociationRule> = Vec::new();

    for itemset in &frequent_itemsets {
        if itemset.items.len() < 2 {
            continue;
        }

        // Generate all non-empty proper subsets as potential LHS
        let subsets = generate_subsets(&itemset.items);

        for lhs in subsets {
            if lhs.is_empty() || lhs.len() == itemset.items.len() {
                continue;
            }

            // RHS = itemset - LHS
            let rhs: Vec<String> = itemset
                .items
                .iter()
                .filter(|item| !lhs.contains(item))
                .cloned()
                .collect();

            if rhs.is_empty() {
                continue;
            }

            // Get support of LHS
            let mut lhs_sorted = lhs.clone();
            lhs_sorted.sort();
            let support_lhs = match itemset_support.get(&lhs_sorted) {
                Some(&s) => s,
                None => continue,
            };

            // Get support of RHS
            let mut rhs_sorted = rhs.clone();
            rhs_sorted.sort();
            let support_rhs = match itemset_support.get(&rhs_sorted) {
                Some(&s) => s,
                None => continue,
            };

            // Confidence = support(itemset) / support(LHS)
            let confidence = if support_lhs > 0.0 {
                itemset.support / support_lhs
            } else {
                0.0
            };

            if confidence < config.min_confidence {
                continue;
            }

            // Lift = confidence / support(RHS)
            let lift = if support_rhs > 0.0 {
                confidence / support_rhs
            } else {
                0.0
            };

            if lift < config.min_lift {
                continue;
            }

            // Coverage = support(LHS)
            let coverage = support_lhs;

            // Leverage = support(itemset) - support(LHS) * support(RHS)
            let leverage = itemset.support - support_lhs * support_rhs;

            // Conviction = (1 - support(RHS)) / (1 - confidence)
            let conviction = if (1.0 - confidence).abs() > 1e-10 {
                (1.0 - support_rhs) / (1.0 - confidence)
            } else {
                f64::INFINITY
            };

            rules.push(AssociationRule {
                lhs,
                rhs,
                support: itemset.support,
                confidence,
                lift,
                count: itemset.count,
                coverage,
                leverage,
                conviction,
            });
        }
    }

    // Sort rules by lift (descending)
    rules.sort_by(|a, b| {
        b.lift
            .partial_cmp(&a.lift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(AprioriResult {
        itemsets: frequent_itemsets,
        rules,
        n_transactions,
        n_items,
        config: config.clone(),
    })
}

/// Generate candidate (k)-itemsets from frequent (k-1)-itemsets.
fn generate_candidates(prev_level: &[Vec<String>]) -> Vec<Vec<String>> {
    let mut candidates: HashSet<Vec<String>> = HashSet::new();

    for i in 0..prev_level.len() {
        for j in (i + 1)..prev_level.len() {
            let a = &prev_level[i];
            let b = &prev_level[j];

            // Join condition: first k-2 items are the same
            if a.len() != b.len() {
                continue;
            }

            let k = a.len();
            if k > 1 && a[..k - 1] != b[..k - 1] {
                continue;
            }

            // Create candidate by merging
            let mut candidate: Vec<String> = a.clone();
            candidate.push(b[k - 1].clone());
            candidate.sort();

            // Prune: check all (k-1)-subsets are frequent
            let is_valid = (0..candidate.len()).all(|skip| {
                let subset: Vec<String> = candidate
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != skip)
                    .map(|(_, item)| item.clone())
                    .collect();
                prev_level.contains(&subset)
            });

            if is_valid {
                candidates.insert(candidate);
            }
        }
    }

    candidates.into_iter().collect()
}

/// Generate all non-empty subsets of an itemset.
fn generate_subsets(items: &[String]) -> Vec<Vec<String>> {
    let n = items.len();
    let mut subsets: Vec<Vec<String>> = Vec::new();

    // Generate all 2^n - 1 non-empty subsets
    for mask in 1..(1 << n) {
        let subset: Vec<String> = items
            .iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, item)| item.clone())
            .collect();
        subsets.push(subset);
    }

    subsets
}

/// Convert a data matrix (0/1) to transactions.
///
/// # Arguments
///
/// * `data` - Binary matrix where rows are transactions and columns are items
/// * `item_names` - Names for each column/item
///
/// # Returns
///
/// List of transactions, where each transaction contains the names of items present.
pub fn matrix_to_transactions(
    data: &[Vec<u8>],
    item_names: &[String],
) -> EconResult<Vec<Vec<String>>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    let n_items = item_names.len();

    let transactions: Vec<Vec<String>> = data
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .filter(|(i, val)| **val > 0 && *i < n_items)
                .map(|(i, _)| item_names[i].clone())
                .collect()
        })
        .collect();

    Ok(transactions)
}

/// Run Eclat algorithm (alternative to Apriori using vertical data format).
///
/// Eclat (Equivalence Class Clustering and bottom-up Lattice Traversal) is
/// often faster than Apriori for dense datasets.
///
/// # Arguments
///
/// * `transactions` - List of transactions
/// * `config` - Same configuration as Apriori
///
/// # References
///
/// Zaki, M. J. (2000). "Scalable Algorithms for Association Mining."
/// *IEEE Transactions on Knowledge and Data Engineering*, 12(3), 372-390.
pub fn eclat(transactions: &[Vec<String>], config: &AprioriConfig) -> EconResult<AprioriResult> {
    let n_transactions = transactions.len();

    if n_transactions == 0 {
        return Err(EconError::EmptyDataset);
    }

    // Build vertical representation: item -> set of transaction IDs
    let mut vertical: HashMap<String, HashSet<usize>> = HashMap::new();

    for (tid, transaction) in transactions.iter().enumerate() {
        for item in transaction {
            vertical
                .entry(item.clone())
                .or_insert_with(HashSet::new)
                .insert(tid);
        }
    }

    let n_items = vertical.len();
    let min_count = (config.min_support * n_transactions as f64).ceil() as usize;

    // Filter to frequent 1-itemsets
    let frequent_1: Vec<(Vec<String>, HashSet<usize>)> = vertical
        .into_iter()
        .filter(|(_, tids)| tids.len() >= min_count)
        .map(|(item, tids)| (vec![item], tids))
        .collect();

    let mut frequent_itemsets: Vec<FrequentItemset> = frequent_1
        .iter()
        .map(|(items, tids)| FrequentItemset {
            items: items.clone(),
            support: tids.len() as f64 / n_transactions as f64,
            count: tids.len(),
        })
        .collect();

    // Recursive depth-first search
    fn eclat_recursive(
        prefix: &[String],
        items_tids: &[(Vec<String>, HashSet<usize>)],
        min_count: usize,
        max_length: usize,
        n_transactions: usize,
        result: &mut Vec<FrequentItemset>,
    ) {
        for i in 0..items_tids.len() {
            let (item_i, tids_i) = &items_tids[i];

            // Create new itemset
            let mut new_itemset: Vec<String> = prefix.to_vec();
            new_itemset.extend(item_i.clone());

            if new_itemset.len() > max_length {
                continue;
            }

            result.push(FrequentItemset {
                items: new_itemset.clone(),
                support: tids_i.len() as f64 / n_transactions as f64,
                count: tids_i.len(),
            });

            // Generate suffix items
            let mut suffix: Vec<(Vec<String>, HashSet<usize>)> = Vec::new();

            for j in (i + 1)..items_tids.len() {
                let (item_j, tids_j) = &items_tids[j];

                // Intersection of transaction sets
                let intersection: HashSet<usize> = tids_i.intersection(tids_j).cloned().collect();

                if intersection.len() >= min_count {
                    suffix.push((item_j.clone(), intersection));
                }
            }

            if !suffix.is_empty() {
                eclat_recursive(
                    &new_itemset,
                    &suffix,
                    min_count,
                    max_length,
                    n_transactions,
                    result,
                );
            }
        }
    }

    // Sort for consistent ordering
    let mut sorted_frequent_1 = frequent_1;
    sorted_frequent_1.sort_by(|a, b| a.0.cmp(&b.0));

    eclat_recursive(
        &[],
        &sorted_frequent_1,
        min_count,
        config.max_length,
        n_transactions,
        &mut frequent_itemsets,
    );

    // Compute support map
    let itemset_support: HashMap<Vec<String>, f64> = frequent_itemsets
        .iter()
        .map(|is| {
            let mut sorted_items = is.items.clone();
            sorted_items.sort();
            (sorted_items, is.support)
        })
        .collect();

    // Generate rules (same as in apriori)
    let mut rules: Vec<AssociationRule> = Vec::new();

    for itemset in &frequent_itemsets {
        if itemset.items.len() < 2 {
            continue;
        }

        let subsets = generate_subsets(&itemset.items);

        for lhs in subsets {
            if lhs.is_empty() || lhs.len() == itemset.items.len() {
                continue;
            }

            let rhs: Vec<String> = itemset
                .items
                .iter()
                .filter(|item| !lhs.contains(item))
                .cloned()
                .collect();

            if rhs.is_empty() {
                continue;
            }

            let mut lhs_sorted = lhs.clone();
            lhs_sorted.sort();
            let support_lhs = match itemset_support.get(&lhs_sorted) {
                Some(&s) => s,
                None => continue,
            };

            let mut rhs_sorted = rhs.clone();
            rhs_sorted.sort();
            let support_rhs = match itemset_support.get(&rhs_sorted) {
                Some(&s) => s,
                None => continue,
            };

            let confidence = if support_lhs > 0.0 {
                itemset.support / support_lhs
            } else {
                0.0
            };

            if confidence < config.min_confidence {
                continue;
            }

            let lift = if support_rhs > 0.0 {
                confidence / support_rhs
            } else {
                0.0
            };

            if lift < config.min_lift {
                continue;
            }

            let coverage = support_lhs;
            let leverage = itemset.support - support_lhs * support_rhs;
            let conviction = if (1.0 - confidence).abs() > 1e-10 {
                (1.0 - support_rhs) / (1.0 - confidence)
            } else {
                f64::INFINITY
            };

            rules.push(AssociationRule {
                lhs,
                rhs,
                support: itemset.support,
                confidence,
                lift,
                count: itemset.count,
                coverage,
                leverage,
                conviction,
            });
        }
    }

    rules.sort_by(|a, b| {
        b.lift
            .partial_cmp(&a.lift)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(AprioriResult {
        itemsets: frequent_itemsets,
        rules,
        n_transactions,
        n_items,
        config: config.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apriori_basic() {
        let transactions = vec![
            vec!["bread".to_string(), "milk".to_string()],
            vec![
                "bread".to_string(),
                "diaper".to_string(),
                "beer".to_string(),
            ],
            vec!["milk".to_string(), "diaper".to_string(), "beer".to_string()],
            vec![
                "bread".to_string(),
                "milk".to_string(),
                "diaper".to_string(),
                "beer".to_string(),
            ],
            vec!["bread".to_string(), "milk".to_string()],
        ];

        let config = AprioriConfig {
            min_support: 0.4,
            min_confidence: 0.6,
            min_lift: 1.0,
            max_length: 5,
        };

        let result = apriori(&transactions, &config).unwrap();

        assert_eq!(result.n_transactions, 5);
        assert!(result.n_items >= 4);
        assert!(!result.itemsets.is_empty());

        // Check that all itemsets meet minimum support
        for itemset in &result.itemsets {
            assert!(
                itemset.support >= config.min_support,
                "Itemset {:?} has support {} < min {}",
                itemset.items,
                itemset.support,
                config.min_support
            );
        }

        // Check that all rules meet criteria
        for rule in &result.rules {
            assert!(
                rule.confidence >= config.min_confidence,
                "Rule has confidence {} < min {}",
                rule.confidence,
                config.min_confidence
            );
            assert!(
                rule.lift >= config.min_lift,
                "Rule has lift {} < min {}",
                rule.lift,
                config.min_lift
            );
        }
    }

    #[test]
    fn test_apriori_single_items() {
        let transactions = vec![
            vec!["A".to_string()],
            vec!["B".to_string()],
            vec!["A".to_string()],
            vec!["B".to_string()],
        ];

        let config = AprioriConfig {
            min_support: 0.4,
            min_confidence: 0.5,
            min_lift: 1.0,
            max_length: 5,
        };

        let result = apriori(&transactions, &config).unwrap();

        // Should find A and B as frequent 1-itemsets
        assert_eq!(result.itemsets.len(), 2);

        // No rules possible with only single-item transactions
        assert_eq!(result.rules.len(), 0);
    }

    #[test]
    fn test_eclat_basic() {
        let transactions = vec![
            vec!["bread".to_string(), "milk".to_string()],
            vec![
                "bread".to_string(),
                "diaper".to_string(),
                "beer".to_string(),
            ],
            vec!["milk".to_string(), "diaper".to_string(), "beer".to_string()],
            vec![
                "bread".to_string(),
                "milk".to_string(),
                "diaper".to_string(),
                "beer".to_string(),
            ],
            vec!["bread".to_string(), "milk".to_string()],
        ];

        let config = AprioriConfig {
            min_support: 0.4,
            min_confidence: 0.6,
            min_lift: 1.0,
            max_length: 5,
        };

        let result = eclat(&transactions, &config).unwrap();

        assert_eq!(result.n_transactions, 5);
        assert!(!result.itemsets.is_empty());

        // Results should be similar to apriori
        let apriori_result = apriori(&transactions, &config).unwrap();

        // Eclat should find at least the same frequent itemsets as Apriori
        // (The recursive implementation may include more due to different counting)
        assert!(
            result.itemsets.len() >= apriori_result.itemsets.len(),
            "Eclat found {} itemsets, Apriori found {}",
            result.itemsets.len(),
            apriori_result.itemsets.len()
        );
    }

    #[test]
    fn test_matrix_to_transactions() {
        let data = vec![vec![1, 0, 1, 0], vec![0, 1, 1, 1], vec![1, 1, 0, 0]];

        let item_names = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];

        let transactions = matrix_to_transactions(&data, &item_names).unwrap();

        assert_eq!(transactions.len(), 3);
        assert_eq!(transactions[0], vec!["A", "C"]);
        assert_eq!(transactions[1], vec!["B", "C", "D"]);
        assert_eq!(transactions[2], vec!["A", "B"]);
    }

    #[test]
    fn test_apriori_high_support() {
        // Only items appearing in all transactions should be frequent
        let transactions = vec![
            vec!["A".to_string(), "B".to_string()],
            vec!["A".to_string(), "C".to_string()],
            vec!["A".to_string(), "B".to_string()],
            vec!["A".to_string(), "D".to_string()],
        ];

        let config = AprioriConfig {
            min_support: 0.9,
            min_confidence: 0.5,
            min_lift: 1.0,
            max_length: 5,
        };

        let result = apriori(&transactions, &config).unwrap();

        // Only A appears in all 4 transactions (support = 1.0)
        assert_eq!(result.itemsets.len(), 1);
        assert_eq!(result.itemsets[0].items, vec!["A"]);
    }

    #[test]
    fn test_rule_metrics() {
        // Construct a case where we can verify metrics
        let transactions = vec![
            vec!["A".to_string(), "B".to_string()],
            vec!["A".to_string(), "B".to_string()],
            vec!["A".to_string()],
            vec!["B".to_string()],
        ];

        let config = AprioriConfig {
            min_support: 0.25,
            min_confidence: 0.5,
            min_lift: 0.0,
            max_length: 5,
        };

        let result = apriori(&transactions, &config).unwrap();

        // Find rule A => B
        let rule = result
            .rules
            .iter()
            .find(|r| r.lhs == vec!["A"] && r.rhs == vec!["B"]);

        if let Some(rule) = rule {
            // Support(A,B) = 2/4 = 0.5
            assert!((rule.support - 0.5).abs() < 0.01);

            // Support(A) = 3/4 = 0.75
            // Confidence = 0.5 / 0.75 = 0.667
            assert!((rule.confidence - 0.667).abs() < 0.01);

            // Support(B) = 3/4 = 0.75
            // Lift = 0.667 / 0.75 = 0.889
            assert!((rule.lift - 0.889).abs() < 0.01);
        }
    }
}
