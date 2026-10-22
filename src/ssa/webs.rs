//! Disjoint-Set Union-Find (DSU) Variable Web Coalescence.
//!
//! Unifies reaching definitions and uses into variable equivalence classes (webs),
//! resolves open upvalue aliasing, computes live interval spans, and isolates loop variables.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;

use crate::cfg::ir::{BlockId, ControlFlowGraph, Terminator};
use super::dataflow::{DefKey, EventKind, ReachingDefsResult};
use super::{DefSite, ProgramPoint, SsaContext, UseSite, WebId};

/// Generic Disjoint-Set Union-Find with path compression and rank optimization.
#[derive(Clone, Debug)]
pub struct UnionFind<T: Eq + Hash + Clone> {
    parent: HashMap<T, T>,
    rank: HashMap<T, u32>,
}

impl<T: Eq + Hash + Clone> Default for UnionFind<T> {
    fn default() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }
}

impl<T: Eq + Hash + Clone> UnionFind<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Finds the canonical representative of element `x`, applying two-pass path compression.
    pub fn find(&mut self, x: &T) -> T {
        if !self.parent.contains_key(x) {
            self.parent.insert(x.clone(), x.clone());
            self.rank.insert(x.clone(), 0);
            return x.clone();
        }

        let mut root = x.clone();
        while let Some(p) = self.parent.get(&root) {
            if p == &root {
                break;
            }
            root = p.clone();
        }

        let mut curr = x.clone();
        while curr != root {
            if let Some(next) = self.parent.get(&curr).cloned() {
                self.parent.insert(curr, root.clone());
                curr = next;
            } else {
                break;
            }
        }

        root
    }

    /// Unifies the sets containing elements `a` and `b`.
    pub fn union(&mut self, a: &T, b: &T) {
        let root_a = self.find(a);
        let root_b = self.find(b);
        if root_a != root_b {
            let rank_a = self.rank.get(&root_a).copied().unwrap_or(0);
            let rank_b = self.rank.get(&root_b).copied().unwrap_or(0);

            if rank_a < rank_b {
                self.parent.insert(root_a, root_b);
            } else if rank_a > rank_b {
                self.parent.insert(root_b, root_a);
            } else {
                self.parent.insert(root_b, root_a.clone());
                self.rank.insert(root_a, rank_a + 1);
            }
        }
    }
}

/// Coalesces reaching definitions and use sites into high-level variable webs.
pub fn coalesce_variable_webs<'arena>(
    cfg: &ControlFlowGraph<'arena>,
    dataflow: &ReachingDefsResult,
    num_params: u8,
) -> SsaContext {
    let mut uf = UnionFind::<DefSite>::new();
    let mut use_web_raw: HashMap<UseSite, Option<DefSite>> = HashMap::new();
    let mut def_web_raw: HashMap<DefSite, DefSite> = HashMap::new();
    let mut captured_regs: HashSet<u16> = HashSet::new();

    // Register initial parameters
    for p in 0..num_params as u16 {
        let d = DefSite::param(p);
        uf.find(&d);
        def_web_raw.insert(d, d);
    }

    // Re-simulate events per block to execute Union-Find coalescing
    let mut sorted_bids: Vec<BlockId> = dataflow.events.keys().copied().collect();
    sorted_bids.sort_by_key(|b| b.0);

    for bid in sorted_bids {
        let ev_list = &dataflow.events[&bid];
        let mut cur = dataflow.in_states.get(&bid).cloned().unwrap_or_default();

        for ev in ev_list {
            let point = ev.point;
            let reg = ev.reg;

            match ev.kind {
                EventKind::Use => {
                    let site = match point {
                        ProgramPoint::Stmt(b, idx) => UseSite::inst(b, idx, reg),
                        ProgramPoint::Terminator(b) => UseSite::term(b, reg),
                        ProgramPoint::Param => UseSite::inst(BlockId::ENTRY, 0, reg),
                    };

                    let ds = cur.get(&DefKey::Reg(reg));
                    if let Some(defs) = ds {
                        if !defs.is_empty() {
                            let mut iter = defs.iter();
                            let first = *iter.next().unwrap();
                            for &other in iter {
                                uf.union(&first, &other);
                            }
                            use_web_raw.insert(site, Some(first));
                        } else {
                            use_web_raw.insert(site, None);
                        }
                    } else {
                        use_web_raw.insert(site, None);
                    }
                }
                EventKind::Capture => {
                    captured_regs.insert(reg);
                    let reg_key = DefKey::Reg(reg);
                    let open_key = DefKey::OpenUpvalue(reg);

                    if !cur.contains_key(&reg_key) || cur[&reg_key].is_empty() {
                        let syn = DefSite::synthetic_capture(point, reg);
                        uf.find(&syn);
                        let mut s = BTreeSet::new();
                        s.insert(syn);
                        cur.insert(reg_key, s);
                        def_web_raw.insert(syn, syn);
                    }

                    let defs: Vec<DefSite> = cur.get(&reg_key).cloned().unwrap_or_default().into_iter().collect();
                    if !defs.is_empty() {
                        for other in &defs[1..] {
                            uf.union(&defs[0], other);
                        }
                    }

                    cur.entry(open_key).or_default().extend(defs);
                }
                EventKind::Def => {
                    let d = match point {
                        ProgramPoint::Param => DefSite::param(reg),
                        ProgramPoint::Stmt(b, idx) => DefSite::inst(b, idx, reg),
                        ProgramPoint::Terminator(b) => DefSite::term(b, reg),
                    };
                    uf.find(&d);

                    // Unify with all active open upvalue definitions
                    if let Some(open_defs) = cur.get(&DefKey::OpenUpvalue(reg)) {
                        for o in open_defs {
                            uf.union(o, &d);
                        }
                    }

                    let mut s = BTreeSet::new();
                    s.insert(d);
                    cur.insert(DefKey::Reg(reg), s);
                    def_web_raw.insert(d, d);
                }
                EventKind::Close => {
                    cur.remove(&DefKey::OpenUpvalue(reg));
                }
            }
        }
    }

    // Canonicalization into SsaContext
    let mut ssa_ctx = SsaContext::new(cfg.proto_id);
    ssa_ctx.captured_regs = captured_regs;

    let mut rep_to_web: HashMap<DefSite, WebId> = HashMap::new();

    // Process all definitions
    let mut sorted_defs: Vec<DefSite> = def_web_raw.keys().copied().collect();
    sorted_defs.sort();

    for def_site in sorted_defs {
        let canonical_root = uf.find(&def_site);
        let web_id = *rep_to_web.entry(canonical_root).or_insert_with(|| {
            ssa_ctx.alloc_web(def_site.reg)
        });

        let is_upval = ssa_ctx.captured_regs.contains(&def_site.reg);
        ssa_ctx.def_to_web.insert(def_site, web_id);
        let web = ssa_ctx.get_web_mut(web_id).unwrap();
        web.defs.push(def_site);
        if is_upval {
            web.is_upvalue = true;
        }
    }

    // Process all uses
    let mut sorted_uses: Vec<(UseSite, Option<DefSite>)> = use_web_raw.into_iter().collect();
    sorted_uses.sort_by_key(|(u, _)| *u);

    for (use_site, maybe_def) in sorted_uses {
        if let Some(def_site) = maybe_def {
            let canonical_root = uf.find(&def_site);
            let web_id = *rep_to_web.entry(canonical_root).or_insert_with(|| {
                ssa_ctx.alloc_web(use_site.reg)
            });

            let is_upval = ssa_ctx.captured_regs.contains(&use_site.reg);
            ssa_ctx.use_to_web.insert(use_site, Some(web_id));
            let web = ssa_ctx.get_web_mut(web_id).unwrap();
            web.uses.push(use_site);
            if is_upval {
                web.is_upvalue = true;
            }
        } else {
            ssa_ctx.use_to_web.insert(use_site, None);
        }
    }

    // Identify loop variable webs
    let mut sorted_loop_bids: Vec<BlockId> = cfg.blocks.keys().copied().collect();
    sorted_loop_bids.sort_by_key(|b| b.0);

    for bid in sorted_loop_bids {
        let block = &cfg.blocks[&bid];
        match &block.terminator {
            Terminator::ForLoop { var, .. } => {
                let d = DefSite::term(bid, *var);
                if let Some(&w) = ssa_ctx.def_to_web.get(&d) {
                    ssa_ctx.loop_webs.insert(w);
                    if let Some(web) = ssa_ctx.get_web_mut(w) {
                        web.is_loop_var = true;
                    }
                }
            }
            Terminator::ForInLoop { vars, .. } => {
                for &v in vars {
                    let d = DefSite::term(bid, v);
                    if let Some(&w) = ssa_ctx.def_to_web.get(&d) {
                        ssa_ctx.loop_webs.insert(w);
                        if let Some(web) = ssa_ctx.get_web_mut(w) {
                            web.is_loop_var = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Compute Live Intervals based on block pc_ranges
    for web in &mut ssa_ctx.webs {
        let mut min_pc = usize::MAX;
        let mut max_pc = 0;

        for d in &web.defs {
            if let Some(bid) = d.point.block_id() {
                if let Some(block) = cfg.blocks.get(&bid) {
                    min_pc = min_pc.min(block.pc_range.0);
                    max_pc = max_pc.max(block.pc_range.1);
                }
            }
        }
        for u in &web.uses {
            if let Some(bid) = u.point.block_id() {
                if let Some(block) = cfg.blocks.get(&bid) {
                    min_pc = min_pc.min(block.pc_range.0);
                    max_pc = max_pc.max(block.pc_range.1);
                }
            }
        }

        if min_pc <= max_pc {
            web.live_interval = (min_pc, max_pc);
        } else {
            web.live_interval = (0, 0);
        }
    }

    ssa_ctx
}
