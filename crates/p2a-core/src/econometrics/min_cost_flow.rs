//! Minimum-cost network flow by the network simplex method, used for optimal full matching as
//! `optmatch::fullmatch` (Hansen and Klopfer 2006; Bertsekas 1998, ch. 5). The algorithm is
//! LEMON's `NetworkSimplex` (Kiraly and Kovacs 2012): block-search pivoting and Cunningham's
//! strongly feasible spanning trees, on integer costs and capacities.
//!
//! The problem: given a directed network with arc capacities `0 <= flow <= capacity`, integer
//! arc costs and node supplies (positive = supply, negative = demand, summing to zero), find
//! the flow that meets every supply exactly at minimum total cost, or report that none exists.
//!
//! The solver starts from LEMON's artificial spanning tree (one artificial arc per node to or
//! from an extra root, the arcs into demand nodes priced at a big-M cost), chooses entering
//! arcs by block search (blocks of `max(ceil(sqrt(m)), 10)` arcs) and leaving arcs by
//! Cunningham's rule, which keeps every tree strongly feasible and so never cycles on
//! degenerate problems (Cunningham 1976; Ahuja, Magnanti and Orlin 1993, ch. 11). Costs, node
//! potentials and reduced costs are `i128`, so nothing in the pivots overflows for the
//! documented bounds; the reported total cost, a sum of up to `2^114` per arc, is computed
//! with checked arithmetic and refused (`Computation`) in the extreme case it would not fit.
//! The problem is infeasible when an artificial arc still carries flow at the optimum.
//!
//! Pinned to `rlemon::MinCostFlow(algorithm = "NetworkSimplex")` (rlemon 0.2.1, LEMON 1.3) and
//! cross-checked with `lpSolve` on the seeded networks of the R case `mcf_networks`
//! (`validation/r/cases/causal_matching.R`).
//!
//! References:
//! - Ahuja, R.K., Magnanti, T.L. & Orlin, J.B. (1993). *Network Flows*. Prentice Hall, ch. 11.
//! - Cunningham, W.H. (1976). A network simplex method. *Mathematical Programming*, 11, 105-116.
//! - Kiraly, Z. & Kovacs, P. (2012). Efficient implementations of minimum-cost flow
//!   algorithms. *Acta Universitatis Sapientiae, Informatica*, 4(1), 67-118.
//! - Hansen, B.B. & Klopfer, S.O. (2006). Optimal full matching and related designs via
//!   network flows. *Journal of Computational and Graphical Statistics*, 15(3), 609-627.

use serde::{Deserialize, Serialize};

use crate::errors::{EconError, EconResult};
use crate::resource;

/// Largest accepted `|cost|` of an arc.
pub const MAX_ARC_COST: i64 = 1 << 52;
/// Largest accepted arc capacity and `|supply|` of a node.
pub const MAX_ARC_CAPACITY: i64 = 1 << 62;

/// A directed network with integer capacities, costs and node supplies.
#[derive(Debug, Clone, Default)]
pub struct Network {
    tails: Vec<usize>,
    heads: Vec<usize>,
    capacities: Vec<i64>,
    costs: Vec<i64>,
    supplies: Vec<i64>,
}

fn spec(message: String) -> EconError {
    EconError::InvalidSpecification { message }
}

impl Network {
    /// A network of `nodes` nodes with zero supplies and no arcs; `ResourceLimit` when the
    /// node array exceeds the allocation ceiling.
    pub fn new(nodes: usize) -> EconResult<Network> {
        Network::with_arc_capacity(nodes, 0)
    }

    /// As [`Network::new`], with room for `arcs` arcs reserved fallibly.
    pub fn with_arc_capacity(nodes: usize, arcs: usize) -> EconResult<Network> {
        const WHAT: &str = "min-cost-flow network arcs";
        Ok(Network {
            tails: resource::try_with_capacity(arcs, WHAT)?,
            heads: resource::try_with_capacity(arcs, WHAT)?,
            capacities: resource::try_with_capacity(arcs, WHAT)?,
            costs: resource::try_with_capacity(arcs, WHAT)?,
            supplies: resource::try_vec(nodes, 0, "min-cost-flow network nodes")?,
        })
    }

    /// Add the arc `tail -> head` with `0 <= flow <= capacity` and unit cost `cost`; returns
    /// its index. `InvalidSpecification` naming `tail` or `head` (not a node), `capacity`
    /// (negative or above [`MAX_ARC_CAPACITY`]) or `cost` (`|cost|` above [`MAX_ARC_COST`]).
    pub fn add_arc(
        &mut self,
        tail: usize,
        head: usize,
        capacity: i64,
        cost: i64,
    ) -> EconResult<usize> {
        let nodes = self.supplies.len();
        if tail >= nodes {
            return Err(spec(format!(
                "arc tail {tail} is not a node (the network has {nodes})"
            )));
        }
        if head >= nodes {
            return Err(spec(format!(
                "arc head {head} is not a node (the network has {nodes})"
            )));
        }
        if !(0..=MAX_ARC_CAPACITY).contains(&capacity) {
            return Err(spec(format!(
                "arc capacity must lie in [0, {MAX_ARC_CAPACITY}], got {capacity}"
            )));
        }
        if cost.unsigned_abs() > MAX_ARC_COST as u64 {
            return Err(spec(format!(
                "arc cost must lie in [-{MAX_ARC_COST}, {MAX_ARC_COST}], got {cost}"
            )));
        }
        self.tails.push(tail);
        self.heads.push(head);
        self.capacities.push(capacity);
        self.costs.push(cost);
        Ok(self.tails.len() - 1)
    }

    /// Set the supply of `node`: positive = supply, negative = demand. `InvalidSpecification`
    /// naming `node` (not a node) or `supply` (`|supply|` above [`MAX_ARC_CAPACITY`]).
    pub fn set_supply(&mut self, node: usize, supply: i64) -> EconResult<()> {
        let nodes = self.supplies.len();
        if node >= nodes {
            return Err(spec(format!(
                "node {node} is not a node (the network has {nodes})"
            )));
        }
        if supply.unsigned_abs() > MAX_ARC_CAPACITY as u64 {
            return Err(spec(format!(
                "supply must lie in [-{MAX_ARC_CAPACITY}, {MAX_ARC_CAPACITY}], got {supply}"
            )));
        }
        self.supplies[node] = supply;
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.supplies.len()
    }

    /// Number of arcs.
    pub fn arc_count(&self) -> usize {
        self.tails.len()
    }

    /// `(tail, head, capacity, cost)` of an arc.
    pub fn arc(&self, arc: usize) -> (usize, usize, i64, i64) {
        (
            self.tails[arc],
            self.heads[arc],
            self.capacities[arc],
            self.costs[arc],
        )
    }

    /// Supply of a node.
    pub fn supply(&self, node: usize) -> i64 {
        self.supplies[node]
    }
}

/// Outcome of [`network_simplex`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowStatus {
    /// A feasible flow of minimum cost was found.
    Optimal,
    /// No flow meets the supplies within the capacities.
    Infeasible,
}

/// A minimum-cost flow.
#[derive(Debug, Clone)]
pub struct FlowSolution {
    /// Optimal or infeasible.
    pub status: FlowStatus,
    /// Flow on each arc of the network (all 0 when Infeasible).
    pub flows: Vec<i64>,
    /// `sum flow * cost` (0 when Infeasible).
    pub total_cost: i128,
    /// Node potentials at the optimum: the reduced cost `cost + pi[tail] - pi[head]` is `>= 0`
    /// on arcs at 0, `<= 0` on arcs at capacity and 0 in between (empty when Infeasible).
    pub potentials: Vec<i128>,
    /// Number of simplex pivots.
    pub pivots: usize,
}

const STATE_UPPER: i8 = -1;
const STATE_TREE: i8 = 0;
const STATE_LOWER: i8 = 1;
/// Capacity of the artificial arcs (and the "unbounded" marker of a residual).
const INF: i128 = i128::MAX / 4;

/// Solve the minimum-cost flow problem of `network` by the network simplex method.
///
/// `InvalidSpecification` when the supplies do not sum to zero; `Cancelled` or `ResourceLimit`
/// from the call's budget and allocation ceiling; `Internal` if the pivot guard
/// (`50 (arcs + nodes + 1)` pivots) is ever reached, which turns a defect into an error
/// instead of a hang.
pub fn network_simplex(network: &Network) -> EconResult<FlowSolution> {
    let n = network.node_count();
    let m = network.arc_count();
    let supply_sum: i128 = network.supplies.iter().map(|&s| s as i128).sum();
    if supply_sum != 0 {
        return Err(spec(format!(
            "supplies must sum to zero (sum = {supply_sum})"
        )));
    }
    let all = m
        .checked_add(n)
        .ok_or_else(|| EconError::ResourceLimit("network simplex: arc count overflows".into()))?;
    let root = n;
    const ARCS: &str = "network simplex arc arrays";
    const NODES: &str = "network simplex node arrays";
    let mut source = resource::try_vec(all, 0usize, ARCS)?;
    let mut target = resource::try_vec(all, 0usize, ARCS)?;
    let mut cap = resource::try_vec(all, 0i128, ARCS)?;
    let mut cost = resource::try_vec(all, 0i128, ARCS)?;
    let mut flow = resource::try_vec(all, 0i128, ARCS)?;
    let mut state = resource::try_vec(all, STATE_LOWER, ARCS)?;
    for e in 0..m {
        source[e] = network.tails[e];
        target[e] = network.heads[e];
        cap[e] = network.capacities[e] as i128;
        cost[e] = network.costs[e] as i128;
    }
    let max_cost = network
        .costs
        .iter()
        .map(|c| c.unsigned_abs() as i128)
        .max()
        .unwrap_or(0);
    let art_cost = (max_cost + 1) * (n as i128 + 1);

    let mut parent = resource::try_vec(n + 1, usize::MAX, NODES)?;
    let mut pred = resource::try_vec(n + 1, usize::MAX, NODES)?;
    let mut up = resource::try_vec(n + 1, false, NODES)?;
    let mut depth = resource::try_vec(n + 1, 0usize, NODES)?;
    let mut pi = resource::try_vec(n + 1, 0i128, NODES)?;
    let mut pos = resource::try_vec(n + 1, 0usize, NODES)?;
    let mut children: Vec<Vec<usize>> = resource::try_vec(n + 1, Vec::new(), NODES)?;
    children[root] = resource::try_with_capacity(n, NODES)?;
    for u in 0..n {
        let e = m + u;
        let s = network.supplies[u] as i128;
        cap[e] = INF;
        state[e] = STATE_TREE;
        if s >= 0 {
            source[e] = u;
            target[e] = root;
            flow[e] = s;
            cost[e] = 0;
            up[u] = true;
            pi[u] = 0;
        } else {
            source[e] = root;
            target[e] = u;
            flow[e] = -s;
            cost[e] = art_cost;
            up[u] = false;
            pi[u] = art_cost;
        }
        parent[u] = root;
        pred[u] = e;
        depth[u] = 1;
        pos[u] = children[root].len();
        children[root].push(u);
    }

    let block = ((m as f64).sqrt().ceil() as usize).max(10);
    let guard = 50usize.saturating_mul(m.saturating_add(n).saturating_add(1));
    let mut next_arc = 0usize;
    let mut pivots = 0usize;
    let mut stack: Vec<usize> = Vec::new();
    loop {
        if pivots.is_multiple_of(256) {
            resource::checkpoint()?;
        }
        // Entering arc: block search over the real arcs.
        let reduced = |e: usize, pi: &[i128]| -> i128 {
            state[e] as i128 * (cost[e] + pi[source[e]] - pi[target[e]])
        };
        let mut min = 0i128;
        let mut in_arc = usize::MAX;
        let mut found = false;
        if m > 0 {
            let mut cnt = block;
            let mut e = next_arc;
            for _ in 0..m {
                let c = reduced(e, &pi);
                if c < min {
                    min = c;
                    in_arc = e;
                }
                cnt -= 1;
                if cnt == 0 {
                    if min < 0 {
                        next_arc = e;
                        found = true;
                        break;
                    }
                    cnt = block;
                }
                e += 1;
                if e == m {
                    e = 0;
                }
            }
            if !found && min < 0 {
                found = true;
            }
        }
        if !found {
            break;
        }
        if pivots >= guard {
            return Err(EconError::Internal(format!(
                "network simplex: pivot guard reached after {pivots} pivots"
            )));
        }
        pivots += 1;

        // Join node.
        let (mut u, mut v) = (source[in_arc], target[in_arc]);
        while u != v {
            if depth[u] > depth[v] {
                u = parent[u];
            } else if depth[v] > depth[u] {
                v = parent[v];
            } else {
                u = parent[u];
                v = parent[v];
            }
        }
        let join = u;

        // Leaving arc (Cunningham's rule).
        let (first, second) = if state[in_arc] == STATE_LOWER {
            (source[in_arc], target[in_arc])
        } else {
            (target[in_arc], source[in_arc])
        };
        let residual = |a: usize, flow: &[i128]| if cap[a] >= INF { INF } else { cap[a] - flow[a] };
        let mut delta = cap[in_arc];
        let mut result = 0u8;
        let mut u_out = usize::MAX;
        let mut w = first;
        while w != join {
            let a = pred[w];
            let d = if up[w] { flow[a] } else { residual(a, &flow) };
            if d < delta {
                delta = d;
                u_out = w;
                result = 1;
            }
            w = parent[w];
        }
        let mut w = second;
        while w != join {
            let a = pred[w];
            let d = if up[w] { residual(a, &flow) } else { flow[a] };
            if d <= delta {
                delta = d;
                u_out = w;
                result = 2;
            }
            w = parent[w];
        }
        if delta >= INF {
            return Err(EconError::Internal(
                "network simplex: unbounded cycle".into(),
            ));
        }

        // Augment along the cycle.
        if delta > 0 {
            let val = state[in_arc] as i128 * delta;
            flow[in_arc] += val;
            let mut w = source[in_arc];
            while w != join {
                let a = pred[w];
                flow[a] -= if up[w] { val } else { -val };
                w = parent[w];
            }
            let mut w = target[in_arc];
            while w != join {
                let a = pred[w];
                flow[a] += if up[w] { val } else { -val };
                w = parent[w];
            }
        }
        if result == 0 {
            state[in_arc] = -state[in_arc];
            continue;
        }
        state[in_arc] = STATE_TREE;
        let out_arc = pred[u_out];
        state[out_arc] = if flow[out_arc] == 0 {
            STATE_LOWER
        } else {
            STATE_UPPER
        };

        // Re-hang the subtree of u_out under the entering arc.
        let (u_in, v_in) = if result == 1 {
            (first, second)
        } else {
            (second, first)
        };
        let mut u = u_in;
        let mut p_new = v_in;
        let mut a_new = in_arc;
        let mut up_new = source[in_arc] == u_in;
        loop {
            let (p_old, a_old, up_old) = (parent[u], pred[u], up[u]);
            // move u from children[p_old] to children[p_new]
            let i = pos[u];
            children[p_old].swap_remove(i);
            if i < children[p_old].len() {
                let moved = children[p_old][i];
                pos[moved] = i;
            }
            pos[u] = children[p_new].len();
            children[p_new].push(u);
            parent[u] = p_new;
            pred[u] = a_new;
            up[u] = up_new;
            if u == u_out {
                break;
            }
            p_new = u;
            a_new = a_old;
            up_new = !up_old;
            u = p_old;
        }
        // Depths and potentials of the re-hung subtree.
        stack.clear();
        stack.push(u_in);
        while let Some(x) = stack.pop() {
            let p = parent[x];
            depth[x] = depth[p] + 1;
            let c = cost[pred[x]];
            pi[x] = if up[x] { pi[p] - c } else { pi[p] + c };
            stack.extend_from_slice(&children[x]);
        }
    }

    if (m..m + n).any(|e| flow[e] != 0) {
        return Ok(FlowSolution {
            status: FlowStatus::Infeasible,
            flows: vec![0; m],
            total_cost: 0,
            potentials: Vec::new(),
            pivots,
        });
    }
    let flows: Vec<i64> = flow[..m].iter().map(|&f| f as i64).collect();
    let total_cost = (0..m)
        .try_fold(0i128, |acc, e| {
            flow[e]
                .checked_mul(cost[e])
                .and_then(|c| acc.checked_add(c))
        })
        .ok_or_else(|| {
            EconError::Computation(
                "network simplex: the total cost of the optimal flow overflows i128".into(),
            )
        })?;
    pi.truncate(n);
    Ok(FlowSolution {
        status: FlowStatus::Optimal,
        flows,
        total_cost,
        potentials: pi,
        pivots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::design::DesignMatrix;
    use p2a_validation::{Check, Tol};

    fn networks(ds: &crate::data::Dataset) -> Vec<Network> {
        let col = |c: &str| DesignMatrix::extract_column(ds.df(), c).unwrap().to_vec();
        let (net, kind, a, b, cap, cost) = (
            col("network"),
            col("kind"),
            col("a"),
            col("b"),
            col("cap"),
            col("cost"),
        );
        (1..=5)
            .map(|k| {
                let rows: Vec<usize> = (0..net.len()).filter(|&i| net[i] == k as f64).collect();
                let mut g = Network::new(rows.iter().filter(|&&i| kind[i] == 2.0).count()).unwrap();
                for &i in &rows {
                    if kind[i] == 1.0 {
                        g.add_arc(a[i] as usize, b[i] as usize, cap[i] as i64, cost[i] as i64)
                            .unwrap();
                    } else {
                        g.set_supply(a[i] as usize, cap[i] as i64).unwrap();
                    }
                }
                g
            })
            .collect()
    }
    /// Bounds, conservation and complementary slackness of an Optimal solution.
    fn assert_optimal(g: &Network, s: &FlowSolution) {
        let mut net_out = vec![0i64; g.node_count()];
        for e in 0..g.arc_count() {
            let (t, h, cap, cost) = g.arc(e);
            let f = s.flows[e];
            assert!(
                (0..=cap).contains(&f),
                "arc {e}: flow {f} outside [0, {cap}]"
            );
            net_out[t] += f;
            net_out[h] -= f;
            let rc = cost as i128 + s.potentials[t] - s.potentials[h];
            assert!(
                !(f == 0 && rc < 0) && !(f == cap && rc > 0) && !(f > 0 && f < cap && rc != 0),
                "arc {e}: rc {rc}"
            );
        }
        for v in 0..g.node_count() {
            assert_eq!(net_out[v], g.supply(v), "node {v}");
        }
        let cost: i128 = (0..g.arc_count())
            .map(|e| s.flows[e] as i128 * g.arc(e).3 as i128)
            .sum();
        assert_eq!(cost, s.total_cost);
    }

    /// rlemon::MinCostFlow(algorithm = "NetworkSimplex") on the five seeded networks of
    /// validation/r/cases/causal_matching.R (lpSolve gives the same optimal costs).
    #[test]
    fn test_validate_mcf_networks() {
        let (case, ds) = crate::test_support::load_case("mcf_networks");
        let mut check = Check::new(&case);
        for (k, g) in networks(&ds).iter().enumerate() {
            let s = network_simplex(g).unwrap();
            let k = k + 1;
            check.scalar(
                &format!("net{k}_feasible"),
                f64::from(u8::from(s.status == FlowStatus::Optimal)),
                Tol::exact(),
            );
            if s.status == FlowStatus::Optimal {
                check.scalar(&format!("net{k}_cost"), s.total_cost as f64, Tol::exact());
                assert_optimal(g, &s);
            } else {
                assert!(s.flows.iter().all(|&f| f == 0) && s.total_cost == 0);
            }
        }
        check.finish();
    }

    /// Hand-computed transportation problem: sources 0 (3 units) and 1 (2), sinks 2 (2) and 3 (3), costs
    /// 0->2 4, 0->3 6, 1->2 5, 1->3 3. With x02 = a: x03 = 3 - a, x12 = 2 - a, x13 = a, cost 28 - 4a on
    /// [0, 2], so a = 2 and the optimum is 20.
    #[test]
    fn test_network_simplex_transportation_by_hand() {
        let mut g = Network::new(4).unwrap();
        for (t, h, c) in [(0, 2, 4), (0, 3, 6), (1, 2, 5), (1, 3, 3)] {
            g.add_arc(t, h, 10, c).unwrap();
        }
        for (v, s) in [(0, 3), (1, 2), (2, -2), (3, -3)] {
            g.set_supply(v, s).unwrap();
        }
        let s = network_simplex(&g).unwrap();
        assert_eq!(
            (s.status, s.total_cost, s.flows.clone()),
            (FlowStatus::Optimal, 20, vec![2, 1, 0, 2])
        );
        assert_optimal(&g, &s);
    }

    /// 6 x 6 assignment with costs ((7 i + 13 j) mod 11) - 3 (negative costs included): the minimum over all
    /// 720 permutations is -15, attained only by (0, 2, 4, 1, 3, 5).
    #[test]
    fn test_network_simplex_assignment_matches_brute_force() {
        let c = |i: usize, j: usize| ((7 * i + 13 * j) % 11) as i64 - 3;
        let mut g = Network::new(12).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                g.add_arc(i, 6 + j, 1, c(i, j)).unwrap();
            }
            g.set_supply(i, 1).unwrap();
            g.set_supply(6 + i, -1).unwrap();
        }
        let s = network_simplex(&g).unwrap();
        assert_eq!(s.total_cost, -15);
        let perm: Vec<usize> = (0..6)
            .map(|i| (0..6).find(|&j| s.flows[6 * i + j] == 1).unwrap())
            .collect();
        assert_eq!(perm, vec![0, 2, 4, 1, 3, 5]);
    }

    /// Fully degenerate: a 30 x 30 assignment with every cost 5 (Cunningham's rule must not cycle).
    #[test]
    fn test_network_simplex_degenerate_assignment_terminates() {
        let mut g = Network::new(60).unwrap();
        for i in 0..30 {
            for j in 0..30 {
                g.add_arc(i, 30 + j, 1, 5).unwrap();
            }
            g.set_supply(i, 1).unwrap();
            g.set_supply(30 + i, -1).unwrap();
        }
        let s = network_simplex(&g).unwrap();
        assert_eq!((s.status, s.total_cost), (FlowStatus::Optimal, 150));
        assert_optimal(&g, &s);
    }

    #[test]
    fn test_network_simplex_infeasible() {
        let mut g = Network::new(2).unwrap();
        g.add_arc(0, 1, 3, 1).unwrap();
        g.set_supply(0, 5).unwrap();
        g.set_supply(1, -5).unwrap();
        let s = network_simplex(&g).unwrap();
        assert_eq!(
            (s.status, s.flows.clone(), s.total_cost),
            (FlowStatus::Infeasible, vec![0], 0)
        );
        assert_eq!(
            network_simplex(&{
                let mut h = Network::new(1).unwrap();
                h.set_supply(0, 0).unwrap();
                h
            })
            .unwrap()
            .status,
            FlowStatus::Optimal
        );
    }

    #[test]
    fn test_network_rejects_invalid_input() {
        let spec = |e: EconError| match e {
            EconError::InvalidSpecification { message } => message,
            o => panic!("{o:?}"),
        };
        let mut g = Network::new(2).unwrap();
        assert!(spec(g.add_arc(2, 0, 1, 1).unwrap_err()).contains("tail"));
        assert!(spec(g.add_arc(0, 5, 1, 1).unwrap_err()).contains("head"));
        assert!(spec(g.add_arc(0, 1, -1, 1).unwrap_err()).contains("capacity"));
        assert!(spec(g.add_arc(0, 1, 1, MAX_ARC_COST + 1).unwrap_err()).contains("cost"));
        assert!(spec(g.set_supply(3, 1).unwrap_err()).contains("node"));
        g.set_supply(0, 2).unwrap();
        assert!(spec(network_simplex(&g).unwrap_err()).contains("sum"));
    }

    #[test]
    fn test_network_simplex_stops_when_cancelled() {
        use crate::resource::{Budget, CancelToken, with_budget};
        let mut g = Network::new(60).unwrap();
        for i in 0..30 {
            for j in 0..30 {
                g.add_arc(i, 30 + j, 1, ((i * j) % 7) as i64).unwrap();
            }
            g.set_supply(i, 1).unwrap();
            g.set_supply(30 + i, -1).unwrap();
        }
        let token = CancelToken::new();
        token.cancel();
        let r = with_budget(Budget::new(None, token), || network_simplex(&g));
        assert!(matches!(r, Err(EconError::Cancelled(_))));
    }

    /// At the documented maxima each arc's flow * cost is 2^114, so 2^13 + 8 such arcs sum
    /// past i128: an error, not a debug panic or a release wrap-around.
    #[test]
    fn test_network_simplex_total_cost_overflow_is_an_error() {
        let pairs = (1usize << 13) + 8;
        let mut g = Network::with_arc_capacity(2 * pairs, pairs).unwrap();
        for i in 0..pairs {
            g.add_arc(2 * i, 2 * i + 1, MAX_ARC_CAPACITY, MAX_ARC_COST)
                .unwrap();
            g.set_supply(2 * i, MAX_ARC_CAPACITY).unwrap();
            g.set_supply(2 * i + 1, -MAX_ARC_CAPACITY).unwrap();
        }
        let r = network_simplex(&g);
        assert!(
            matches!(r, Err(EconError::Computation(ref m)) if m.contains("overflows")),
            "{r:?}"
        );
        // a single arc at the maxima still reports its exact cost
        let mut g = Network::new(2).unwrap();
        g.add_arc(0, 1, MAX_ARC_CAPACITY, MAX_ARC_COST).unwrap();
        g.set_supply(0, MAX_ARC_CAPACITY).unwrap();
        g.set_supply(1, -MAX_ARC_CAPACITY).unwrap();
        assert_eq!(network_simplex(&g).unwrap().total_cost, 1i128 << 114);
    }
}
