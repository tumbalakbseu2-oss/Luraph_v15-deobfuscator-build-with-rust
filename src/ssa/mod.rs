//! SSA Variable Web Coalescence, Reaching Definitions, and Dead Store Elimination.

pub mod dataflow;
pub mod webs;

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::cfg::ir::{BlockId, ControlFlowGraph, IrExpr, IrInst};

pub use dataflow::{
    extract_block_events, run_reaching_definitions, BlockEvent, DefKey, EventKind, ReachingDefsResult,
};
pub use webs::{coalesce_variable_webs, UnionFind};

/// Strongly typed unique identifier for an SSA variable web.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct WebId(pub u32);

impl WebId {
    pub const INVALID: Self = Self(u32::MAX);

    #[inline]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self != Self::INVALID
    }
}

impl fmt::Debug for WebId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::INVALID {
            write!(f, "WebId(INVALID)")
        } else {
            write!(f, "Web_{}", self.0)
        }
    }
}

impl fmt::Display for WebId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::INVALID {
            write!(f, "w_invalid")
        } else {
            write!(f, "w_{}", self.0)
        }
    }
}

/// Identifies the discrete program point where an event occurs.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ProgramPoint {
    /// Function entry parameter arrival.
    Param,
    /// Inside block `block`, at instruction index `stmt_idx`.
    Stmt(BlockId, usize),
    /// At the terminator of block `block`.
    Terminator(BlockId),
}

impl ProgramPoint {
    #[inline]
    pub fn block_id(self) -> Option<BlockId> {
        match self {
            Self::Param => Some(BlockId::ENTRY),
            Self::Stmt(bid, _) | Self::Terminator(bid) => Some(bid),
        }
    }
}

/// Concrete definition site of a virtual register.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DefSite {
    /// The location where the definition occurs.
    pub point: ProgramPoint,
    /// The virtual register slot being defined (0..255).
    pub reg: u16,
    /// True if this definition was synthetically created when an uninitialized
    /// register was captured by an inner closure.
    pub is_synthetic_capture: bool,
}

impl DefSite {
    #[inline]
    pub fn param(reg: u16) -> Self {
        Self {
            point: ProgramPoint::Param,
            reg,
            is_synthetic_capture: false,
        }
    }

    #[inline]
    pub fn inst(block: BlockId, stmt_idx: usize, reg: u16) -> Self {
        Self {
            point: ProgramPoint::Stmt(block, stmt_idx),
            reg,
            is_synthetic_capture: false,
        }
    }

    #[inline]
    pub fn term(block: BlockId, reg: u16) -> Self {
        Self {
            point: ProgramPoint::Terminator(block),
            reg,
            is_synthetic_capture: false,
        }
    }

    #[inline]
    pub fn synthetic_capture(point: ProgramPoint, reg: u16) -> Self {
        Self {
            point,
            reg,
            is_synthetic_capture: true,
        }
    }
}

/// Concrete use site of a virtual register.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct UseSite {
    /// The location where the use occurs.
    pub point: ProgramPoint,
    /// The virtual register slot being read (0..255).
    pub reg: u16,
}

impl UseSite {
    #[inline]
    pub fn inst(block: BlockId, stmt_idx: usize, reg: u16) -> Self {
        Self {
            point: ProgramPoint::Stmt(block, stmt_idx),
            reg,
        }
    }

    #[inline]
    pub fn term(block: BlockId, reg: u16) -> Self {
        Self {
            point: ProgramPoint::Terminator(block),
            reg,
        }
    }
}

/// An equivalence class of definition and use sites coalesced into a single
/// high-level variable identity.
#[derive(Clone, Debug)]
pub struct VariableWeb {
    /// Unique web identifier.
    pub id: WebId,
    /// Original virtual register slot associated with this web.
    pub original_reg: u16,
    /// Reconstructed or assigned variable name (e.g. "r1", "players", "v0").
    pub assigned_name: String,
    /// Bytecode instruction span [start_pc, end_pc] where this variable is live.
    pub live_interval: (usize, usize),
    /// True if this variable is captured as an upvalue by any child closure.
    pub is_upvalue: bool,
    /// True if this variable is a loop iteration counter/variable (for/forin header).
    pub is_loop_var: bool,
    /// All definition sites belonging to this web.
    pub defs: Vec<DefSite>,
    /// All use sites belonging to this web.
    pub uses: Vec<UseSite>,
    /// Inferred semantic hint (if available).
    pub semantic_hint: Option<crate::namer::SemanticHint>,
}

impl VariableWeb {
    pub fn new(id: WebId, original_reg: u16) -> Self {
        Self {
            id,
            original_reg,
            assigned_name: String::new(),
            live_interval: (usize::MAX, 0),
            is_upvalue: false,
            is_loop_var: false,
            defs: Vec::new(),
            uses: Vec::new(),
            semantic_hint: None,
        }
    }

    /// Extends the live interval span with a bytecode program counter.
    #[inline]
    pub fn record_pc(&mut self, pc: usize) {
        if pc < self.live_interval.0 {
            self.live_interval.0 = pc;
        }
        if pc > self.live_interval.1 {
            self.live_interval.1 = pc;
        }
    }
}

/// Complete SSA Web Coalescence context for a single prototype CFG.
#[derive(Clone, Debug, Default)]
pub struct SsaContext {
    /// Prototype ID from which this context was derived.
    pub proto_id: usize,
    /// All coalesced variable webs indexed by WebId.
    pub webs: Vec<VariableWeb>,
    /// Mapping from each DefSite to its canonical WebId.
    pub def_to_web: HashMap<DefSite, WebId>,
    /// Mapping from each UseSite to its canonical WebId (or None if uninitialized/nil).
    pub use_to_web: HashMap<UseSite, Option<WebId>>,
    /// Set of registers that are captured across closure boundaries.
    pub captured_regs: HashSet<u16>,
    /// Web IDs corresponding to numeric/generic loop iterator variables.
    pub loop_webs: HashSet<WebId>,
}

impl SsaContext {
    pub fn new(proto_id: usize) -> Self {
        Self {
            proto_id,
            webs: Vec::new(),
            def_to_web: HashMap::new(),
            use_to_web: HashMap::new(),
            captured_regs: HashSet::new(),
            loop_webs: HashSet::new(),
        }
    }

    /// Allocates a new VariableWeb and returns its unique WebId.
    pub fn alloc_web(&mut self, reg: u16) -> WebId {
        let id = WebId::new(self.webs.len() as u32);
        self.webs.push(VariableWeb::new(id, reg));
        id
    }

    #[inline]
    pub fn get_web(&self, id: WebId) -> Option<&VariableWeb> {
        self.webs.get(id.index())
    }

    #[inline]
    pub fn get_web_mut(&mut self, id: WebId) -> Option<&mut VariableWeb> {
        self.webs.get_mut(id.index())
    }
}

/// Analyzes reaching definitions and coalesces variable webs for a CFG.
pub fn analyze_ssa<'arena>(cfg: &ControlFlowGraph<'arena>, num_params: u8) -> SsaContext {
    let dataflow = dataflow::run_reaching_definitions(cfg, num_params);
    webs::coalesce_variable_webs(cfg, &dataflow, num_params)
}

/// Evaluates whether an expression is side-effect free.
pub fn is_pure_expr<'arena>(expr: &IrExpr<'arena>) -> bool {
    match expr {
        IrExpr::Nil
        | IrExpr::Boolean(_)
        | IrExpr::Number(_)
        | IrExpr::String(_)
        | IrExpr::Register(_)
        | IrExpr::Upvalue(_)
        | IrExpr::Global(_)
        | IrExpr::Constant(_)
        | IrExpr::Vararg => true,
        IrExpr::Unary { expr, .. } => is_pure_expr(expr),
        IrExpr::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        IrExpr::TableGet { table, key } => is_pure_expr(table) && is_pure_expr(key),
        IrExpr::IfExp { cond, then_expr, else_expr } => {
            is_pure_expr(cond) && is_pure_expr(then_expr) && is_pure_expr(else_expr)
        }
        IrExpr::NewTable { .. } => true,
        IrExpr::Closure { upvalues, .. } => upvalues.is_empty(),
        _ => false,
    }
}

/// Eliminates dead stores to virtual registers that are never used and not captured.
pub fn eliminate_dead_stores<'arena>(
    cfg: &mut ControlFlowGraph<'arena>,
    ssa: &SsaContext,
) -> usize {
    let mut eliminated = 0;

    for (&bid, block) in &mut cfg.blocks {
        for (i, inst) in block.instructions.iter_mut().enumerate() {
            if let IrInst::Assign { dest, src } = inst {
                let site = DefSite::inst(bid, i, *dest);
                if let Some(&web_id) = ssa.def_to_web.get(&site) {
                    if let Some(web) = ssa.get_web(web_id) {
                        if web.uses.is_empty()
                            && !ssa.captured_regs.contains(dest)
                            && !ssa.loop_webs.contains(&web_id)
                            && is_pure_expr(src)
                        {
                            *inst = IrInst::Nop;
                            eliminated += 1;
                        }
                    }
                }
            }
        }
    }

    eliminated
}
