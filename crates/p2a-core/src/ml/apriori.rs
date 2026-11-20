//! Association rule mining using the Apriori algorithm.
//!
//! Implements the classic Apriori algorithm (and Eclat) for finding frequent itemsets
//! and generating association rules.
//!
//! Conventions follow R's `arules`: an itemset is frequent when its support (count / n) is at least
//! `min_support`; each rule has a single item on the right-hand side, and every frequent single item gives
//! a rule with an empty left-hand side (`minlen = 1`; set `min_length = 2` to leave them out); confidence
//! filters the rules and lift does not (`min_lift` is p2a's own filter, off by default). The default
//! support and confidence are p2a's, 0.01 and 0.5 (arules: 0.1 and 0.8). There is no `maxtime`: the call
//! stops at its budget instead.
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
use crate::resource::{checkpoint, checkpoint_stride, try_vec, try_with_capacity};

/// Configuration for Apriori algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprioriConfig {
    /// Minimum support, in (0, 1] (arules `supp`): an itemset is frequent when count / n >= min_support
    pub min_support: f64,
    /// Minimum confidence, in [0, 1] (arules `conf`)
    pub min_confidence: f64,
    /// Minimum lift, finite and >= 0; arules has no lift filter, so the default 0 keeps every rule
    pub min_lift: f64,
    /// Maximum itemset size, at least 1 (arules `maxlen`)
    pub max_length: usize,
    /// Minimum itemset size of the reported itemsets and of the rules (left plus right side), 1 to
    /// `max_length` (arules `minlen`): 1 reports single items and their empty-LHS rules, 2 leaves them out
    #[serde(default = "default_min_length")]
    pub min_length: usize,
}

fn default_min_length() -> usize {
    1
}

impl Default for AprioriConfig {
    fn default() -> Self {
        Self {
            min_support: 0.01,
            min_confidence: 0.5,
            min_lift: 0.0,
            max_length: 10,
            min_length: 1,
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
    /// Association rules generated, sorted by lift (descending), then lhs, then rhs; `Display`
    /// prints the first 10 as the top rules
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

            // the rules are already in lift order
            for rule in self.rules.iter().take(10) {
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
                    p2a_text::ellipsize(&rule_str, 30),
                    rule.support,
                    rule.confidence,
                    rule.lift
                )?;
            }
        }

        Ok(())
    }
}

/// Check the configuration: `min_support` finite in (0, 1], `min_confidence` in [0, 1], `min_lift` finite and
/// non-negative, `max_length` at least 1, `min_length` from 1 to `max_length`.
fn check_config(config: &AprioriConfig) -> EconResult<()> {
    let bad = |message: String| Err(EconError::InvalidSpecification { message });
    if !(config.min_support.is_finite() && config.min_support > 0.0 && config.min_support <= 1.0) {
        return bad(format!(
            "min_support must be in (0, 1], got {}",
            config.min_support
        ));
    }
    if !(0.0..=1.0).contains(&config.min_confidence) {
        return bad(format!(
            "min_confidence must be in [0, 1], got {}",
            config.min_confidence
        ));
    }
    if !(config.min_lift.is_finite() && config.min_lift >= 0.0) {
        return bad(format!(
            "min_lift must be finite and at least 0, got {}",
            config.min_lift
        ));
    }
    if config.max_length < 1 {
        return bad(format!(
            "max_length must be at least 1, got {}",
            config.max_length
        ));
    }
    if config.min_length < 1 || config.min_length > config.max_length {
        return bad(format!(
            "min_length must be between 1 and max_length ({}), got {}",
            config.max_length, config.min_length
        ));
    }
    Ok(())
}

/// The smallest count `c` whose support `c / n` (as reported) is at least `min_support`: an itemset is frequent
/// exactly when its reported support is `>= min_support`. `ceil(min_support * n)` in floating point asked one
/// more for supports such as 0.07 over 100 transactions.
fn min_count(min_support: f64, n: usize) -> usize {
    let nf = n as f64;
    let mut c = ((min_support * nf).ceil().max(0.0) as usize).min(n);
    while c > 0 && (c - 1) as f64 / nf >= min_support {
        c -= 1;
    }
    while c < n && (c as f64 / nf) < min_support {
        c += 1;
    }
    c.max(1)
}

/// Transactions as sorted, deduplicated item ids; ids follow the items' sorted order, so itemsets of ids
/// sort as itemsets of names.
struct Encoded {
    names: Vec<String>,
    transactions: Vec<Vec<u32>>,
}

fn encode(transactions: &[Vec<String>]) -> EconResult<Encoded> {
    if transactions.len() > u32::MAX as usize {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "apriori/eclat take at most {} transactions, got {}",
                u32::MAX,
                transactions.len()
            ),
        });
    }
    let mut names: Vec<String> = transactions.iter().flatten().cloned().collect();
    names.sort_unstable();
    names.dedup();
    if names.len() > u32::MAX as usize {
        return Err(EconError::InvalidSpecification {
            message: format!(
                "apriori/eclat take at most {} distinct items, got {}",
                u32::MAX,
                names.len()
            ),
        });
    }
    let index: HashMap<&str, u32> = names
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i as u32))
        .collect();
    let mut encoded = try_with_capacity(transactions.len(), "apriori transactions")?;
    for t in transactions {
        let mut ids: Vec<u32> = t.iter().map(|s| index[s.as_str()]).collect();
        ids.sort_unstable();
        ids.dedup();
        encoded.push(ids);
    }
    Ok(Encoded {
        names,
        transactions: encoded,
    })
}

/// Frequent itemsets of ids with their counts, sorted by length, then items.
type IdItemsets = Vec<(Vec<u32>, usize)>;

fn to_itemsets(found: IdItemsets, names: &[String], n: usize) -> Vec<FrequentItemset> {
    found
        .into_iter()
        .map(|(ids, count)| FrequentItemset {
            items: ids.iter().map(|&i| names[i as usize].clone()).collect(),
            support: count as f64 / n as f64,
            count,
        })
        .collect()
}

fn sort_itemsets(found: &mut IdItemsets) {
    found.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0)));
}

/// Apriori's level-wise search over encoded transactions.
fn apriori_itemsets(enc: &Encoded, min_count: usize, max_length: usize) -> EconResult<IdItemsets> {
    let n_items = enc.names.len();
    let mut counts = try_vec(n_items, 0usize, "apriori item counts")?;
    for t in &enc.transactions {
        for &i in t {
            counts[i as usize] += 1;
        }
    }
    let mut found: IdItemsets = Vec::new();
    let mut level: Vec<Vec<u32>> = Vec::new();
    for (i, &c) in counts.iter().enumerate() {
        if c >= min_count {
            found.push((vec![i as u32], c));
            level.push(vec![i as u32]);
        }
    }
    let mut member = try_vec(n_items, false, "apriori transaction mask")?;
    for _k in 2..=max_length {
        checkpoint()?;
        if level.is_empty() {
            break;
        }
        let candidates = generate_candidates(&level)?;
        if candidates.is_empty() {
            break;
        }
        let mut cand_counts = try_vec(candidates.len(), 0usize, "apriori candidate counts")?;
        let stride = checkpoint_stride(candidates.len());
        for (row, t) in enc.transactions.iter().enumerate() {
            if row % stride == 0 {
                checkpoint()?;
            }
            if t.len() < candidates[0].len() {
                continue;
            }
            for &i in t {
                member[i as usize] = true;
            }
            for (c, cand) in cand_counts.iter_mut().zip(&candidates) {
                if cand.iter().all(|&i| member[i as usize]) {
                    *c += 1;
                }
            }
            for &i in t {
                member[i as usize] = false;
            }
        }
        level = Vec::new();
        for (cand, c) in candidates.into_iter().zip(cand_counts) {
            if c >= min_count {
                found.push((cand.clone(), c));
                level.push(cand);
            }
        }
    }
    sort_itemsets(&mut found);
    Ok(found)
}

/// Candidate k-itemsets from the sorted frequent (k-1)-itemsets: join two that share their first k-2 items,
/// keep the join when every (k-1)-subset is frequent. Sorted output.
fn generate_candidates(prev_level: &[Vec<u32>]) -> EconResult<Vec<Vec<u32>>> {
    let frequent: HashSet<&[u32]> = prev_level.iter().map(|v| v.as_slice()).collect();
    let mut candidates = Vec::new();
    for (i, a) in prev_level.iter().enumerate() {
        checkpoint()?;
        let k = a.len();
        for b in &prev_level[i + 1..] {
            // prev_level is sorted, so the itemsets sharing a's prefix follow it
            if a[..k - 1] != b[..k - 1] {
                break;
            }
            let mut candidate = a.clone();
            candidate.push(b[k - 1]);
            let mut subset = Vec::with_capacity(k);
            let valid = (0..candidate.len()).all(|skip| {
                subset.clear();
                subset.extend(
                    candidate
                        .iter()
                        .enumerate()
                        .filter(|&(idx, _)| idx != skip)
                        .map(|(_, &item)| item),
                );
                frequent.contains(subset.as_slice())
            });
            if valid {
                push_fallibly(&mut candidates, candidate, "apriori candidate itemsets")?;
            }
        }
    }
    Ok(candidates)
}

/// `v.push(x)`, growing the buffer (doubling, as `Vec`) through the per-allocation ceiling.
fn push_fallibly<T>(v: &mut Vec<T>, x: T, what: &str) -> EconResult<()> {
    if v.len() == v.capacity() {
        let cap = v.capacity().saturating_mul(2).max(4);
        crate::resource::check_alloc::<T>(cap, what)?;
        v.try_reserve_exact(cap - v.len())
            .map_err(|_| crate::resource::reserve_failed(what, cap, std::mem::size_of::<T>()))?;
    }
    v.push(x);
    Ok(())
}

/// Eclat's depth-first search over sorted tid lists.
fn eclat_itemsets(enc: &Encoded, min_count: usize, max_length: usize) -> EconResult<IdItemsets> {
    let n_items = enc.names.len();
    // one tid list per item, each allocated at its exact length
    let mut counts = try_vec(n_items, 0usize, "eclat item counts")?;
    for t in &enc.transactions {
        for &i in t {
            counts[i as usize] += 1;
        }
    }
    let mut tids: Vec<Vec<u32>> = try_with_capacity(n_items, "eclat tid-list table")?;
    for &c in &counts {
        tids.push(try_with_capacity(c, "eclat tid list")?);
    }
    for (tid, t) in enc.transactions.iter().enumerate() {
        for &i in t {
            tids[i as usize].push(tid as u32);
        }
    }
    let frequent_1: Vec<(u32, Vec<u32>)> = tids
        .into_iter()
        .enumerate()
        .filter(|(_, t)| t.len() >= min_count)
        .map(|(i, t)| (i as u32, t))
        .collect();
    let mut found: IdItemsets = Vec::new();
    eclat_recursive(&[], &frequent_1, min_count, max_length, &mut found)?;
    sort_itemsets(&mut found);
    Ok(found)
}

fn intersect(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len().min(b.len()));
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out
}

/// Each prefix + item is reported once; the recursion extends it with the later items.
fn eclat_recursive(
    prefix: &[u32],
    items_tids: &[(u32, Vec<u32>)],
    min_count: usize,
    max_length: usize,
    result: &mut IdItemsets,
) -> EconResult<()> {
    for (i, (item_i, tids_i)) in items_tids.iter().enumerate() {
        checkpoint()?;
        let mut itemset = prefix.to_vec();
        itemset.push(*item_i);
        result.push((itemset.clone(), tids_i.len()));
        if itemset.len() >= max_length {
            continue;
        }
        let mut suffix: Vec<(u32, Vec<u32>)> = Vec::new();
        for (item_j, tids_j) in &items_tids[i + 1..] {
            let both = intersect(tids_i, tids_j);
            if both.len() >= min_count {
                suffix.push((*item_j, both));
            }
        }
        if !suffix.is_empty() {
            eclat_recursive(&itemset, &suffix, min_count, max_length, result)?;
        }
    }
    Ok(())
}

/// Shared driver of [`apriori`] and [`eclat`].
fn mine(
    transactions: &[Vec<String>],
    config: &AprioriConfig,
    search: fn(&Encoded, usize, usize) -> EconResult<IdItemsets>,
) -> EconResult<AprioriResult> {
    check_config(config)?;
    let n_transactions = transactions.len();
    if n_transactions == 0 {
        return Err(EconError::EmptyDataset);
    }
    let enc = encode(transactions)?;
    let n_items = enc.names.len();
    let found = if n_items == 0 {
        Vec::new()
    } else {
        search(
            &enc,
            min_count(config.min_support, n_transactions),
            config.max_length,
        )?
    };
    let mut itemsets = to_itemsets(found, &enc.names, n_transactions);
    let rules = generate_rules(&itemsets, config)?;
    itemsets.retain(|s| s.items.len() >= config.min_length);
    Ok(AprioriResult {
        itemsets,
        rules,
        n_transactions,
        n_items,
        config: config.clone(),
    })
}

/// Apriori algorithm for association rule mining.
///
/// Finds frequent itemsets using a level-wise approach, then generates
/// association rules from those itemsets. An itemset is frequent when its support (count / n) is at least
/// `min_support`. Itemsets are listed by length, then items.
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
/// # Errors
///
/// `InvalidSpecification` for a setting out of range (see [`AprioriConfig`]), `EmptyDataset` for no
/// transactions, `ResourceLimit` when a buffer exceeds the allocation ceiling, `Cancelled` when the call's
/// budget is cancelled or its deadline passes.
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
    mine(transactions, config, apriori_itemsets)
}

/// Association rules as arules generates them: for every frequent itemset of at least
/// `max(1, min_length)` items, one rule per item on the right-hand side with the other items on the left
/// (empty for a single item, whose support is 1). The itemsets must be all frequent itemsets, so that every
/// left-hand side's support is known (downward closure); a rule whose left-hand side is missing is skipped.
/// Rules are sorted by lift (descending), then left-hand side, then right-hand side.
fn generate_rules(
    frequent_itemsets: &[FrequentItemset],
    config: &AprioriConfig,
) -> EconResult<Vec<AssociationRule>> {
    let itemset_support: HashMap<&[String], f64> = frequent_itemsets
        .iter()
        .map(|is| (is.items.as_slice(), is.support))
        .collect();
    let min_k = config.min_length.max(1);
    let mut rules: Vec<AssociationRule> = Vec::new();
    let mut lhs: Vec<String> = Vec::new();
    for (idx, itemset) in frequent_itemsets.iter().enumerate() {
        if idx % 256 == 0 {
            checkpoint()?;
        }
        let k = itemset.items.len();
        if k < min_k {
            continue;
        }
        for r in 0..k {
            lhs.clear();
            lhs.extend(
                itemset
                    .items
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != r)
                    .map(|(_, item)| item.clone()),
            );
            let support_lhs = if lhs.is_empty() {
                1.0
            } else {
                match itemset_support.get(lhs.as_slice()) {
                    Some(&s) => s,
                    None => continue,
                }
            };
            let rhs = std::slice::from_ref(&itemset.items[r]);
            let Some(&support_rhs) = itemset_support.get(rhs) else {
                continue;
            };
            let confidence = itemset.support / support_lhs;
            if confidence < config.min_confidence {
                continue;
            }
            let lift = confidence / support_rhs;
            if lift < config.min_lift {
                continue;
            }
            let conviction = if (1.0 - confidence).abs() > 1e-10 {
                (1.0 - support_rhs) / (1.0 - confidence)
            } else {
                f64::INFINITY
            };
            rules.push(AssociationRule {
                lhs: lhs.clone(),
                rhs: rhs.to_vec(),
                support: itemset.support,
                confidence,
                lift,
                count: itemset.count,
                coverage: support_lhs,
                leverage: itemset.support - support_lhs * support_rhs,
                conviction,
            });
        }
    }
    rules.sort_by(|a, b| {
        b.lift
            .total_cmp(&a.lift)
            .then_with(|| a.lhs.cmp(&b.lhs))
            .then_with(|| a.rhs.cmp(&b.rhs))
    });
    Ok(rules)
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
/// often faster than Apriori for dense datasets. Same itemsets, rules, order and errors as [`apriori`].
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
    mine(transactions, config, eclat_itemsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The data-sized buffers of both searches are allocated through the per-allocation ceiling:
    /// eclat's per-item tid-list table and apriori's candidate list.
    #[test]
    fn test_itemset_searches_allocate_fallibly() {
        use crate::resource::with_alloc_ceiling;
        // 10 transactions of 1,000 distinct items each: 10,000 items, a 240,000-byte tid table
        let wide: Vec<Vec<String>> = (0..10)
            .map(|t| (0..1000).map(|i| format!("i{}", t * 1000 + i)).collect())
            .collect();
        let mut config = AprioriConfig::default();
        config.min_support = 0.1;
        config.max_length = 1;
        let e = with_alloc_ceiling(Some(200_000), || eclat(&wide, &config)).unwrap_err();
        assert!(
            matches!(&e, EconError::ResourceLimit(m) if m.contains("eclat")),
            "{e}"
        );
        assert!(eclat(&wide, &config).is_ok());
        // 150 items in both transactions: 11,175 candidate pairs, 268,200 bytes of candidates and
        // 89,400 bytes of counts
        let dense: Vec<Vec<String>> = (0..2)
            .map(|_| (0..150).map(|i| format!("i{i:03}")).collect())
            .collect();
        config.min_support = 0.5;
        config.max_length = 2;
        config.min_confidence = 1.0;
        config.min_lift = f64::MAX; // no rules: the itemset search alone
        let e = with_alloc_ceiling(Some(200_000), || apriori(&dense, &config)).unwrap_err();
        assert!(
            matches!(&e, EconError::ResourceLimit(m) if m.contains("apriori candidate")),
            "{e}"
        );
    }

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
            ..Default::default()
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
            ..Default::default()
        };

        let result = apriori(&transactions, &config).unwrap();

        // Should find A and B as frequent 1-itemsets
        assert_eq!(result.itemsets.len(), 2);

        // Only the empty-LHS rules {} => {A} and {} => {B} (support 0.5 >= confidence 0.5), as arules
        assert_eq!(result.rules.len(), 2);
        assert!(
            result
                .rules
                .iter()
                .all(|r| r.lhs.is_empty() && r.rhs.len() == 1)
        );
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
            ..Default::default()
        };

        let result = eclat(&transactions, &config).unwrap();

        assert_eq!(result.n_transactions, 5);
        assert!(!result.itemsets.is_empty());

        // Results should be similar to apriori
        let apriori_result = apriori(&transactions, &config).unwrap();

        // Eclat finds exactly Apriori's itemsets (it reported every single item twice before)
        assert_eq!(
            itemset_keys(&result.itemsets),
            itemset_keys(&apriori_result.itemsets)
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
            ..Default::default()
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
            ..Default::default()
        };

        let result = apriori(&transactions, &config).unwrap();

        // Find rule A => B
        let rule = result
            .rules
            .iter()
            .find(|r| r.lhs == vec!["A"] && r.rhs == vec!["B"]);

        let rule = rule.expect("A => B must be found: support 0.5 >= 0.25, confidence 2/3 >= 0.5");
        // Support(A,B) = 2/4
        assert_eq!(rule.support, 0.5);
        // Confidence = support(A,B) / support(A) = 0.5 / 0.75
        assert!((rule.confidence - 2.0 / 3.0).abs() < 1e-15);
        // Lift = confidence / support(B) = (2/3) / 0.75
        assert!((rule.lift - 8.0 / 9.0).abs() < 1e-15);
    }

    /// ML-62: the rule string is 49 bytes and 'ü' spans bytes 26..28, so &rule_str[..27] panicked in Display.
    #[test]
    fn test_display_non_ascii_rule_is_cut_on_char_boundaries() {
        let (v, m) = (
            "Vollkornbrot geschnitten überbacken".to_string(),
            "Milch".to_string(),
        );
        let tx = vec![
            vec![v.clone(), m.clone()],
            vec![v.clone(), m.clone()],
            vec![v.clone(), m.clone()],
            vec![m.clone()],
            vec!["Käse".to_string()],
        ];
        let r = apriori(&tx, &AprioriConfig::default()).unwrap();
        // {V}=>{M} conf 1.0 and {M}=>{V} conf 0.75, both lift 1.25; {} => {V} conf 0.6 and {} => {M} conf
        // 0.8, lift 1 ({} => {Käse} has conf 0.2)
        assert_eq!(r.rules.len(), 4);
        let out = r.to_string();
        assert!(out.contains("{Vollkornbrot geschnitten ü... "), "{out}");
        assert!(out.contains("{Milch} => {Vollkornbrot ge... "), "{out}"); // ASCII cut, as before
    }

    fn tx(rows: &[&[&str]]) -> Vec<Vec<String>> {
        rows.iter()
            .map(|r| r.iter().map(|s| s.to_string()).collect())
            .collect()
    }

    fn itemset_keys(sets: &[FrequentItemset]) -> Vec<(Vec<String>, usize, f64)> {
        sets.iter()
            .map(|s| (s.items.clone(), s.count, s.support))
            .collect()
    }

    fn config(min_support: f64) -> AprioriConfig {
        AprioriConfig {
            min_support,
            ..Default::default()
        }
    }

    /// Seeded baskets over items a-h with item-specific densities.
    pub(super) fn seeded_baskets(seed: u64, n: usize) -> Vec<Vec<String>> {
        use rand::{Rng, SeedableRng, rngs::StdRng};
        let mut rng = StdRng::seed_from_u64(seed);
        let p = [0.55, 0.6, 0.5, 0.35, 0.25, 0.3, 0.12, 0.07];
        (0..n)
            .map(|_| {
                let mut t: Vec<String> = Vec::new();
                for (j, &pj) in p.iter().enumerate() {
                    // correlate b with a and d with c
                    let pj = match j {
                        1 if t.iter().any(|x| x == "a") => 0.8,
                        3 if t.iter().any(|x| x == "c") => 0.55,
                        _ => pj,
                    };
                    if rng.gen_bool(pj) {
                        t.push(((b'a' + j as u8) as char).to_string());
                    }
                }
                t
            })
            .collect()
    }

    /// Every itemset over a-h (255) counted directly.
    pub(super) fn brute_force_itemsets(
        baskets: &[Vec<String>],
        min_support: f64,
    ) -> Vec<(Vec<String>, usize, f64)> {
        let n = baskets.len();
        let mut out = Vec::new();
        for mask in 1u32..256 {
            let items: Vec<String> = (0..8)
                .filter(|j| mask & (1 << j) != 0)
                .map(|j| ((b'a' + j as u8) as char).to_string())
                .collect();
            let count = baskets
                .iter()
                .filter(|t| items.iter().all(|i| t.contains(i)))
                .count();
            let support = count as f64 / n as f64;
            if support >= min_support {
                out.push((items, count, support));
            }
        }
        out.sort_by(|a, b| a.0.len().cmp(&b.0.len()).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// ML-117: an itemset whose support equals min_support is kept. Before, ceil(min_support * n) in floating
    /// point dropped A (7 of 100) at 0.07, and every k in {7, 14, 28, 55, 56} (n = 100), {7, 14, 28} (n = 50).
    #[test]
    fn apriori_keeps_itemsets_exactly_at_min_support() {
        let mut baskets = vec![vec!["A".to_string()]; 7];
        baskets.extend(vec![vec!["B".to_string()]; 93]);
        for f in [apriori, eclat] {
            let r = f(&baskets, &config(0.07)).unwrap();
            assert_eq!(
                itemset_keys(&r.itemsets),
                vec![
                    (vec!["A".to_string()], 7, 0.07),
                    (vec!["B".to_string()], 93, 0.93)
                ]
            );
        }
        let mut dropped = Vec::new();
        for n in [100usize, 50] {
            for k in 1..n {
                let mut baskets = vec![vec!["A".to_string()]; k];
                baskets.extend(vec![vec!["B".to_string()]; n - k]);
                let s = k as f64 / n as f64;
                for f in [apriori, eclat] {
                    let r = f(&baskets, &config(s)).unwrap();
                    if !r.itemsets.iter().any(|i| i.items == ["A"]) {
                        dropped.push((n, k));
                    }
                }
            }
        }
        assert!(dropped.is_empty(), "dropped at the boundary: {dropped:?}");
    }

    /// ML-119: eclat reported every frequent single item twice (7 itemsets here, 4 now), and it finds exactly
    /// the itemsets of apriori and of a brute-force count.
    #[test]
    fn eclat_reports_each_itemset_once() {
        let mut baskets = vec![vec!["A".to_string(), "B".to_string()]; 3];
        baskets.extend(vec![vec!["C".to_string()]; 7]);
        let r = eclat(&baskets, &config(0.2)).unwrap();
        let items: Vec<Vec<String>> = r.itemsets.iter().map(|i| i.items.clone()).collect();
        assert_eq!(items, vec![vec!["A"], vec!["B"], vec!["C"], vec!["A", "B"]]);
        let baskets = seeded_baskets(20261115, 200);
        for s in [0.02, 0.07, 0.15] {
            let truth = brute_force_itemsets(&baskets, s);
            let a = apriori(&baskets, &config(s)).unwrap();
            let e = eclat(&baskets, &config(s)).unwrap();
            assert_eq!(itemset_keys(&a.itemsets), truth, "apriori at {s}");
            assert_eq!(itemset_keys(&e.itemsets), truth, "eclat at {s}");
        }
    }

    /// Settings out of range are refused, naming the field and the bound.
    #[test]
    fn apriori_refuses_invalid_settings() {
        let baskets = tx(&[&["a", "b"], &["a"]]);
        let cases: Vec<(AprioriConfig, &str)> = vec![
            (config(f64::NAN), "min_support"),
            (config(0.0), "min_support"),
            (config(-0.1), "min_support"),
            (config(1.5), "min_support"),
            (
                AprioriConfig {
                    min_confidence: f64::NAN,
                    ..Default::default()
                },
                "min_confidence",
            ),
            (
                AprioriConfig {
                    min_confidence: -0.1,
                    ..Default::default()
                },
                "min_confidence",
            ),
            (
                AprioriConfig {
                    min_confidence: 1.1,
                    ..Default::default()
                },
                "min_confidence",
            ),
            (
                AprioriConfig {
                    min_lift: f64::NAN,
                    ..Default::default()
                },
                "min_lift",
            ),
            (
                AprioriConfig {
                    min_lift: -1.0,
                    ..Default::default()
                },
                "min_lift",
            ),
            (
                AprioriConfig {
                    max_length: 0,
                    ..Default::default()
                },
                "max_length",
            ),
        ];
        for (c, field) in cases {
            for f in [apriori, eclat] {
                match f(&baskets, &c) {
                    Err(EconError::InvalidSpecification { message }) => {
                        assert!(message.contains(field), "{message}");
                        assert!(message.contains("got"), "{message}");
                    }
                    other => panic!("{field}: {other:?}"),
                }
            }
        }
    }

    /// Itemsets and rules come out in a fixed order (by length, then items), whatever the basket order.
    #[test]
    fn apriori_lists_itemsets_in_a_fixed_order() {
        let baskets = seeded_baskets(5, 150);
        let mut reversed = baskets.clone();
        reversed.reverse();
        for item_order in [false, true] {
            let b2: Vec<Vec<String>> = if item_order {
                reversed
                    .iter()
                    .map(|t| t.iter().rev().cloned().collect())
                    .collect()
            } else {
                reversed.clone()
            };
            for f in [apriori, eclat] {
                let a = f(&baskets, &config(0.05)).unwrap();
                let b = f(&baskets, &config(0.05)).unwrap();
                let c = f(&b2, &config(0.05)).unwrap();
                assert_eq!(itemset_keys(&a.itemsets), itemset_keys(&b.itemsets));
                assert_eq!(itemset_keys(&a.itemsets), itemset_keys(&c.itemsets));
                assert!(
                    a.itemsets
                        .windows(2)
                        .all(|w| (w[0].items.len(), &w[0].items) < (w[1].items.len(), &w[1].items))
                );
            }
        }
    }

    fn rule_key(r: &AssociationRule) -> String {
        format!("{{{}}} => {{{}}}", r.lhs.join(","), r.rhs.join(","))
    }

    /// ML-118: every rule has one item on the right, and single items give empty-LHS rules. Before: 12 rules
    /// of which 3 had a two-item right-hand side, and none with an empty left-hand side.
    #[test]
    fn rules_have_one_item_on_the_right() {
        let baskets = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]; 4];
        let c = AprioriConfig {
            min_support: 0.01,
            min_confidence: 0.5,
            ..Default::default()
        };
        for f in [apriori, eclat] {
            let r = f(&baskets, &c).unwrap();
            assert_eq!(r.rules.len(), 12);
            assert!(r.rules.iter().all(|x| x.rhs.len() == 1));
            let by_lhs = |n: usize| r.rules.iter().filter(|x| x.lhs.len() == n).count();
            assert_eq!((by_lhs(0), by_lhs(1), by_lhs(2)), (3, 6, 3));
            for x in &r.rules {
                assert_eq!(
                    (x.confidence, x.lift, x.conviction),
                    (1.0, 1.0, f64::INFINITY)
                );
            }
        }
    }

    /// Every rule with a one-item right-hand side from a brute-force count, against apriori's and eclat's
    /// rules; `min_length = 2` removes exactly the empty-LHS rules and the single items.
    #[test]
    fn rules_match_a_brute_force_enumeration() {
        let baskets = seeded_baskets(20261115, 200);
        let n = baskets.len() as f64;
        let s = 0.05;
        let truth = brute_force_itemsets(&baskets, s);
        let support = |items: &[String]| -> f64 {
            if items.is_empty() {
                return 1.0;
            }
            baskets
                .iter()
                .filter(|t| items.iter().all(|i| t.contains(i)))
                .count() as f64
                / n
        };
        let c = AprioriConfig {
            min_support: s,
            min_confidence: 0.3,
            ..Default::default()
        };
        let mut expected: Vec<(String, [f64; 6], usize)> = Vec::new();
        for (items, count, sup) in &truth {
            for r in 0..items.len() {
                let lhs: Vec<String> = items
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != r)
                    .map(|(_, x)| x.clone())
                    .collect();
                let rhs = vec![items[r].clone()];
                let (sl, sr) = (support(&lhs), support(&rhs));
                let conf = sup / sl;
                if conf < c.min_confidence {
                    continue;
                }
                let conviction = if (1.0 - conf).abs() > 1e-10 {
                    (1.0 - sr) / (1.0 - conf)
                } else {
                    f64::INFINITY
                };
                expected.push((
                    format!("{{{}}} => {{{}}}", lhs.join(","), rhs.join(",")),
                    [*sup, conf, conf / sr, sl, sup - sl * sr, conviction],
                    *count,
                ));
            }
        }
        expected.sort_by(|a, b| a.0.cmp(&b.0));
        for f in [apriori, eclat] {
            let r = f(&baskets, &c).unwrap();
            let mut got: Vec<(String, [f64; 6], usize)> = r
                .rules
                .iter()
                .map(|x| {
                    (
                        rule_key(x),
                        [
                            x.support,
                            x.confidence,
                            x.lift,
                            x.coverage,
                            x.leverage,
                            x.conviction,
                        ],
                        x.count,
                    )
                })
                .collect();
            got.sort_by(|a, b| a.0.cmp(&b.0));
            assert_eq!(got.len(), expected.len());
            for (g, e) in got.iter().zip(&expected) {
                assert_eq!((&g.0, g.2), (&e.0, e.2));
                for (u, v) in g.1.iter().zip(&e.1) {
                    assert!(
                        u == v || (u - v).abs() <= 1e-12 * v.abs(),
                        "{}: {u} {v}",
                        g.0
                    );
                }
            }
            let c2 = AprioriConfig {
                min_length: 2,
                ..c.clone()
            };
            let r2 = f(&baskets, &c2).unwrap();
            let mut kept: Vec<String> = r
                .rules
                .iter()
                .filter(|x| !x.lhs.is_empty())
                .map(rule_key)
                .collect();
            let mut got2: Vec<String> = r2.rules.iter().map(rule_key).collect();
            kept.sort();
            got2.sort();
            assert_eq!(got2, kept);
            assert!(r.rules.iter().any(|x| x.lhs.is_empty()));
            let multi: Vec<Vec<String>> = r
                .itemsets
                .iter()
                .filter(|i| i.items.len() >= 2)
                .map(|i| i.items.clone())
                .collect();
            let got_sets: Vec<Vec<String>> = r2.itemsets.iter().map(|i| i.items.clone()).collect();
            assert_eq!(got_sets, multi);
        }
    }

    /// HEAD's subset enumeration used an i32 mask over 2^k subsets: empty at 31 items, overflow at 32.
    #[test]
    fn rule_generator_handles_a_40_item_itemset() {
        let items: Vec<String> = (0..40).map(|i| format!("i{i:02}")).collect();
        let set = |v: Vec<String>| FrequentItemset {
            items: v,
            support: 1.0,
            count: 10,
        };
        let mut sets = vec![set(items.clone())];
        for r in 0..40 {
            sets.push(set(items
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != r)
                .map(|(_, x)| x.clone())
                .collect()));
            sets.push(set(vec![items[r].clone()]));
        }
        let rules = generate_rules(&sets, &AprioriConfig::default()).unwrap();
        let by_lhs = |n: usize| rules.iter().filter(|x| x.lhs.len() == n).count();
        assert_eq!((by_lhs(39), by_lhs(38), by_lhs(0)), (40, 0, 40));
        assert_eq!(rules.len(), 80);
    }

    /// A cancelled budget stops both searches.
    #[test]
    fn apriori_stops_at_the_budget() {
        use crate::resource::{Budget, CancelToken, with_budget};
        let baskets: Vec<Vec<String>> = (0..300)
            .map(|t| {
                (0..20)
                    .filter(|j| (t * 7 + j * 3) % 5 != 0)
                    .map(|j| format!("x{j}"))
                    .collect()
            })
            .collect();
        let token = CancelToken::new();
        token.cancel();
        for f in [apriori, eclat] {
            let r = with_budget(Budget::new(None, token.clone()), || {
                f(&baskets, &config(0.05))
            });
            assert!(
                matches!(r, Err(EconError::Cancelled(_))),
                "{:?}",
                r.map(|x| x.itemsets.len())
            );
        }
    }

    /// The `ml_apriori_baskets` dataset (0/1 columns a-h) as transactions.
    fn case_baskets(ds: &crate::data::Dataset) -> Vec<Vec<String>> {
        use crate::data::columns::{ColumnRequest, NaPolicy};
        let names = ["a", "b", "c", "d", "e", "f", "g", "h"];
        let c = ColumnRequest::new(ds.df())
            .numerics(&names)
            .na(NaPolicy::Fail)
            .extract()
            .unwrap();
        let n = c.numeric("a").len();
        let data: Vec<Vec<u8>> = (0..n)
            .map(|i| names.iter().map(|k| c.numeric(k)[i] as u8).collect())
            .collect();
        let items: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        matrix_to_transactions(&data, &items).unwrap()
    }

    fn case_config() -> AprioriConfig {
        let mut c = AprioriConfig::default();
        c.min_support = 0.07;
        c.min_confidence = 0.5;
        c.min_length = 1;
        c.max_length = 10;
        c.min_lift = 0.0;
        c
    }

    fn set_key(items: &[String]) -> String {
        format!("{{{}}}", items.join(","))
    }

    fn check_itemsets(
        c: &mut p2a_validation::Check<'_>,
        key: &str,
        r: &AprioriResult,
        counts: bool,
    ) {
        use p2a_validation::Tol;
        let names: Vec<String> = r.itemsets.iter().map(|i| set_key(&i.items)).collect();
        let support: Vec<f64> = r.itemsets.iter().map(|i| i.support).collect();
        c.named(key, &names, &support, Tol::coef(100));
        if counts {
            let count: Vec<f64> = r.itemsets.iter().map(|i| i.count as f64).collect();
            c.named("itemset_count", &names, &count, Tol::exact())
                .scalar("n_itemsets", r.itemsets.len() as f64, Tol::exact());
        }
    }

    /// ML-117, ML-118: arules' frequent itemsets (h, {f,g} and {b,g} exactly on support 0.07) and rules (one
    /// item on the right, three with an empty left-hand side). Before: the boundary itemsets were dropped and
    /// rules had multi-item right-hand sides.
    #[test]
    fn test_validate_apriori_baskets() {
        use p2a_validation::Tol;
        let (case, ds) = crate::test_support::load_case("ml_apriori_baskets");
        let r = apriori(&case_baskets(&ds), &case_config()).unwrap();
        let mut c = p2a_validation::Check::new(&case);
        check_itemsets(&mut c, "itemset_support", &r, true);
        let keys: Vec<String> = r
            .rules
            .iter()
            .map(|x| format!("{} => {}", set_key(&x.lhs), set_key(&x.rhs)))
            .collect();
        let col = |f: fn(&AssociationRule) -> f64| -> Vec<f64> { r.rules.iter().map(f).collect() };
        let tol = Tol::coef(100);
        c.named("rule_support", &keys, &col(|x| x.support), tol)
            .named("rule_confidence", &keys, &col(|x| x.confidence), tol)
            .named("rule_lift", &keys, &col(|x| x.lift), tol)
            .named("rule_coverage", &keys, &col(|x| x.coverage), tol)
            .named("rule_count", &keys, &col(|x| x.count as f64), Tol::exact())
            .named("rule_leverage", &keys, &col(|x| x.leverage), tol)
            .scalar("n_rules", r.rules.len() as f64, Tol::exact())
            .scalar(
                "n_rules_empty_lhs",
                r.rules.iter().filter(|x| x.lhs.is_empty()).count() as f64,
                Tol::exact(),
            );
        // conviction: R pins the finite values; a rule with confidence 1 has conviction inf
        let finite: Vec<usize> = (0..keys.len())
            .filter(|&i| r.rules[i].conviction.is_finite())
            .collect();
        let fk: Vec<String> = finite.iter().map(|&i| keys[i].clone()).collect();
        let fv: Vec<f64> = finite.iter().map(|&i| r.rules[i].conviction).collect();
        c.named("rule_conviction_finite", &fk, &fv, tol).finish();
        for x in &r.rules {
            assert_eq!(x.conviction.is_infinite(), x.confidence == 1.0);
        }
    }

    /// ML-119: eclat's itemsets are arules::eclat's, each once.
    #[test]
    fn test_validate_eclat_baskets() {
        let (case, ds) = crate::test_support::load_case("ml_apriori_baskets");
        let r = eclat(&case_baskets(&ds), &case_config()).unwrap();
        let mut c = p2a_validation::Check::new(&case);
        check_itemsets(&mut c, "eclat_support", &r, false);
        c.scalar(
            "n_itemsets",
            r.itemsets.len() as f64,
            p2a_validation::Tol::exact(),
        )
        .finish();
        let a = apriori(&case_baskets(&ds), &case_config()).unwrap();
        assert_eq!(itemset_keys(&r.itemsets), itemset_keys(&a.itemsets));
    }

    /// R5: rules are listed by non-increasing lift; lift is finite by construction (confidence / support_rhs,
    /// support_rhs >= 1/n, else 0.0), so total_cmp orders exactly as the old NaN-tolerant comparison did.
    #[test]
    fn test_rules_sorted_by_lift_descending() {
        let tx: Vec<Vec<String>> = vec![
            vec!["bread", "milk"],
            vec!["bread", "diaper", "beer"],
            vec!["milk", "diaper", "beer"],
            vec!["bread", "milk", "diaper", "beer"],
            vec!["bread", "milk"],
        ]
        .into_iter()
        .map(|t| t.into_iter().map(String::from).collect())
        .collect();
        let c = AprioriConfig {
            min_support: 0.4,
            min_confidence: 0.6,
            min_lift: 0.0,
            ..Default::default()
        };
        for r in [apriori(&tx, &c).unwrap(), eclat(&tx, &c).unwrap()] {
            assert!(r.rules.len() >= 2);
            assert!(r.rules.windows(2).all(|w| w[0].lift >= w[1].lift));
        }
    }
}
