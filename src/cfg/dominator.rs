//! Cooper-Harvey-Kennedy Dominator & Post-Dominator Trees.
//!
//! Provides linear-time dominator tree construction, dominance testing,
//! post-dominator join resolution, and natural loop extraction.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::cfg::ir::{BlockId, ControlFlowGraph};

/// Dominator Tree computed using the Cooper-Harvey-Kennedy iterative algorithm.
#[derive(Clone, Debug, Default)]
pub struct DominatorTree {
    pub rpo_order: Vec<BlockId>,
    pub rpo_index: HashMap<BlockId, usize>,
    pub idom: HashMap<BlockId, BlockId>,
}

impl DominatorTree {
    /// Computes immediate dominators (idom) for a CFG.
    pub fn compute<'arena>(cfg: &ControlFlowGraph<'arena>) -> Self {
        if cfg.blocks.is_empty() {
            return Self::default();
        }

        let entry = cfg.entry;
        let rpo_order = Self::compute_rpo(cfg, entry);
        let mut rpo_index = HashMap::with_capacity(rpo_order.len());
        for (idx, &bid) in rpo_order.iter().enumerate() {
            rpo_index.insert(bid, idx);
        }

        let mut idom: HashMap<BlockId, BlockId> = HashMap::with_capacity(rpo_order.len());
        idom.insert(entry, entry);

        let mut changed = true;
        while changed {
            changed = false;

            // Iterate over blocks in RPO order, skipping entry at index 0
            for &b in &rpo_order[1..] {
                let Some(block) = cfg.blocks.get(&b) else {
                    continue;
                };

                // Filter predecessors that already have a computed idom
                let mut processed_preds = block
                    .predecessors
                    .iter()
                    .copied()
                    .filter(|p| idom.contains_key(p));

                if let Some(first_p) = processed_preds.next() {
                    let mut new_idom = first_p;
                    for p in processed_preds {
                        new_idom = Self::intersect(p, new_idom, &idom, &rpo_index);
                    }

                    if idom.get(&b) != Some(&new_idom) {
                        idom.insert(b, new_idom);
                        changed = true;
                    }
                }
            }
        }

        Self {
            rpo_order,
            rpo_index,
            idom,
        }
    }

    /// Two-finger Lowest Common Ancestor (LCA) intersection in the dominator tree.
    pub fn intersect(
        mut b1: BlockId,
        mut b2: BlockId,
        idom: &HashMap<BlockId, BlockId>,
        rpo_index: &HashMap<BlockId, usize>,
    ) -> BlockId {
        while b1 != b2 {
            while rpo_index.get(&b1).copied().unwrap_or(0) > rpo_index.get(&b2).copied().unwrap_or(0) {
                if let Some(&parent) = idom.get(&b1) {
                    if parent == b1 {
                        break;
                    }
                    b1 = parent;
                } else {
                    break;
                }
            }
            while rpo_index.get(&b2).copied().unwrap_or(0) > rpo_index.get(&b1).copied().unwrap_or(0) {
                if let Some(&parent) = idom.get(&b2) {
                    if parent == b2 {
                        break;
                    }
                    b2 = parent;
                } else {
                    break;
                }
            }
        }
        b1
    }

    /// Tests whether block `a` dominates block `b` (`a dom b`).
    pub fn dominates(&self, a: BlockId, mut b: BlockId) -> bool {
        loop {
            if b == a {
                return true;
            }
            match self.idom.get(&b) {
                Some(&parent) if parent != b => b = parent,
                _ => return false,
            }
        }
    }

    /// Returns Reverse Post-Order traversal of the CFG.
    fn compute_rpo<'arena>(cfg: &ControlFlowGraph<'arena>, entry: BlockId) -> Vec<BlockId> {
        let mut visited = HashSet::new();
        let mut post_order = Vec::new();

        fn dfs<'arena>(
            bid: BlockId,
            cfg: &ControlFlowGraph<'arena>,
            visited: &mut HashSet<BlockId>,
            post_order: &mut Vec<BlockId>,
        ) {
            if !visited.insert(bid) {
                return;
            }
            if let Some(block) = cfg.blocks.get(&bid) {
                for &succ in &block.successors {
                    dfs(succ, cfg, visited, post_order);
                }
            }
            post_order.push(bid);
        }

        dfs(entry, cfg, &mut visited, &mut post_order);
        post_order.reverse();
        post_order
    }
}

/// Post-Dominator Tree computed on the reversed CFG.
#[derive(Clone, Debug, Default)]
pub struct PostDominatorTree {
    pub ipdom: HashMap<BlockId, BlockId>,
}

impl PostDominatorTree {
    /// Computes immediate post-dominators by running CHK on inverted control-flow edges.
    pub fn compute<'arena>(cfg: &ControlFlowGraph<'arena>) -> Self {
        if cfg.blocks.is_empty() {
            return Self::default();
        }

        let synthetic_exit = BlockId::new(u32::MAX - 1);
        let mut r_succs: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        let mut r_preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();

        // Reverse edges
        for (&bid, block) in &cfg.blocks {
            for &succ in &block.successors {
                r_succs.entry(succ).or_default().push(bid);
                r_preds.entry(bid).or_default().push(succ);
            }
            // Terminal blocks (no successors) connect from synthetic exit in reversed graph
            if block.successors.is_empty() {
                r_succs.entry(synthetic_exit).or_default().push(bid);
                r_preds.entry(bid).or_default().push(synthetic_exit);
            }
        }

        // Compute RPO on reversed graph starting from synthetic_exit
        let mut visited = HashSet::new();
        let mut post_order = Vec::new();

        fn dfs_rev(
            bid: BlockId,
            r_succs: &HashMap<BlockId, Vec<BlockId>>,
            visited: &mut HashSet<BlockId>,
            post_order: &mut Vec<BlockId>,
        ) {
            if !visited.insert(bid) {
                return;
            }
            if let Some(succs) = r_succs.get(&bid) {
                for &succ in succs {
                    dfs_rev(succ, r_succs, visited, post_order);
                }
            }
            post_order.push(bid);
        }

        dfs_rev(synthetic_exit, &r_succs, &mut visited, &mut post_order);
        post_order.reverse();

        let mut rpo_index = HashMap::with_capacity(post_order.len());
        for (idx, &bid) in post_order.iter().enumerate() {
            rpo_index.insert(bid, idx);
        }

        let mut idom_rev: HashMap<BlockId, BlockId> = HashMap::with_capacity(post_order.len());
        idom_rev.insert(synthetic_exit, synthetic_exit);

        let mut changed = true;
        while changed {
            changed = false;
            for &b in &post_order[1..] {
                let preds = r_preds.get(&b).cloned().unwrap_or_default();
                let mut processed = preds.into_iter().filter(|p| idom_rev.contains_key(p));

                if let Some(first_p) = processed.next() {
                    let mut new_idom = first_p;
                    for p in processed {
                        new_idom = DominatorTree::intersect(p, new_idom, &idom_rev, &rpo_index);
                    }
                    if idom_rev.get(&b) != Some(&new_idom) {
                        idom_rev.insert(b, new_idom);
                        changed = true;
                    }
                }
            }
        }

        // Filter out synthetic exit
        let mut ipdom = HashMap::new();
        for (&b, &pdom) in &idom_rev {
            if b != synthetic_exit && pdom != synthetic_exit && pdom != b {
                ipdom.insert(b, pdom);
            }
        }

        Self { ipdom }
    }
}

/// Natural loop representation.
#[derive(Clone, Debug)]
pub struct NaturalLoop {
    pub header: BlockId,
    pub latches: Vec<BlockId>,
    pub body: HashSet<BlockId>,
    pub exits: Vec<(BlockId, BlockId)>,
}

pub struct LoopExtractor;

impl LoopExtractor {
    /// Extracts all natural loops in the CFG using dominator back-edge analysis.
    pub fn extract_loops<'arena>(
        cfg: &ControlFlowGraph<'arena>,
        dom: &DominatorTree,
    ) -> Vec<NaturalLoop> {
        let mut header_to_latches: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();

        // 1. Identify back-edges: (tail, head) where head dominates tail
        for (&tail, block) in &cfg.blocks {
            for &head in &block.successors {
                if dom.dominates(head, tail) {
                    header_to_latches.entry(head).or_default().push(tail);
                }
            }
        }

        let mut loops = Vec::new();

        // 2. Extract loop bodies and exits in deterministic header order
        for (header, mut latches) in header_to_latches {
            latches.sort_by_key(|b| b.0);
            let mut body = HashSet::new();
            body.insert(header);

            let mut stack = latches.clone();
            while let Some(node) = stack.pop() {
                if body.insert(node) {
                    if let Some(block) = cfg.blocks.get(&node) {
                        for &pred in &block.predecessors {
                            if !body.contains(&pred) {
                                stack.push(pred);
                            }
                        }
                    }
                }
            }

            // Identify exits in strictly deterministic node/successor order
            let mut body_nodes: Vec<BlockId> = body.iter().copied().collect();
            body_nodes.sort_by_key(|b| b.0);

            let mut exits = Vec::new();
            for node in body_nodes {
                if let Some(block) = cfg.blocks.get(&node) {
                    for &succ in &block.successors {
                        if !body.contains(&succ) {
                            exits.push((node, succ));
                        }
                    }
                }
            }
            exits.sort_by_key(|&(n, s)| (n.0, s.0));

            loops.push(NaturalLoop {
                header,
                latches,
                body,
                exits,
            });
        }

        // Sort loops innermost first (smallest body first), tie-breaking by header ID
        loops.sort_by(|a, b| {
            a.body
                .len()
                .cmp(&b.body.len())
                .then_with(|| a.header.0.cmp(&b.header.0))
        });
        loops
    }
}
