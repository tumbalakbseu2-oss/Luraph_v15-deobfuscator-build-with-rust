//! Sparse Conditional Constant Propagation (SCCP) and Abstract Interpreter.
//!
//! Evaluates abstract values over a 4-state lattice (Top, Constant, KnownTruth, Bottom),
//! evaluates Luraph affine state variable updates, resolves opaque predicates,
//! and eliminates dead control-flow branches and trap stubs.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::cfg::ir::{
    BinOp, BlockId, ControlFlowGraph, IrExpr, IrInst, Literal, Terminator, UnOp,
};

/// 4-State Abstract Value Lattice for dataflow analysis.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum LatticeValue {
    /// ⊤ (Top): Uninitialized / unreached. Identity element for meet.
    #[default]
    Top,

    /// Constant: Exact runtime value is statically known.
    Constant(Literal),

    /// KnownTruth: Exact value is dynamic, but truthiness is statically invariant.
    KnownTruth(bool),

    /// ⊥ (Bottom): Overdefined, dynamic, or conflicting values.
    Bottom,
}

impl LatticeValue {
    /// Returns Some(true) if guaranteed truthy, Some(false) if guaranteed falsy, None if dynamic (⊤ / ⊥).
    #[inline]
    pub fn truthiness(&self) -> Option<bool> {
        match self {
            LatticeValue::Constant(lit) => Some(lit.is_truthy()),
            LatticeValue::KnownTruth(b) => Some(*b),
            LatticeValue::Top | LatticeValue::Bottom => None,
        }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            LatticeValue::Constant(lit) => lit.as_i64(),
            _ => None,
        }
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            LatticeValue::Constant(lit) => lit.as_f64(),
            _ => None,
        }
    }

    /// Computes the greatest lower bound (meet) of two lattice elements.
    pub fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            // Identity: Top meet x = x
            (LatticeValue::Top, x) | (x, LatticeValue::Top) => x.clone(),

            // Absorbing element: Bottom meet x = Bottom
            (LatticeValue::Bottom, _) | (_, LatticeValue::Bottom) => LatticeValue::Bottom,

            // Two constants
            (LatticeValue::Constant(c1), LatticeValue::Constant(c2)) => {
                if c1 == c2 {
                    LatticeValue::Constant(c1.clone())
                } else if c1.is_truthy() == c2.is_truthy() {
                    LatticeValue::KnownTruth(c1.is_truthy())
                } else {
                    LatticeValue::Bottom
                }
            }

            // Constant and KnownTruth
            (LatticeValue::Constant(c), LatticeValue::KnownTruth(t))
            | (LatticeValue::KnownTruth(t), LatticeValue::Constant(c)) => {
                if c.is_truthy() == *t {
                    LatticeValue::KnownTruth(*t)
                } else {
                    LatticeValue::Bottom
                }
            }

            // Two KnownTruth values
            (LatticeValue::KnownTruth(t1), LatticeValue::KnownTruth(t2)) => {
                if t1 == t2 {
                    LatticeValue::KnownTruth(*t1)
                } else {
                    LatticeValue::Bottom
                }
            }
        }
    }
}

/// Abstract state vector tracking registers and the affine dispatcher state variable.
#[derive(Clone, Debug, Default)]
pub struct VmState {
    pub registers: HashMap<u16, LatticeValue>,
    pub affine_state: LatticeValue,
}

impl VmState {
    pub fn new() -> Self {
        Self {
            registers: HashMap::new(),
            affine_state: LatticeValue::Top,
        }
    }

    #[inline]
    pub fn get_reg(&self, reg: u16) -> LatticeValue {
        self.registers.get(&reg).cloned().unwrap_or(LatticeValue::Top)
    }

    #[inline]
    pub fn set_reg(&mut self, reg: u16, val: LatticeValue) {
        self.registers.insert(reg, val);
    }
}

/// Abstract Interpreter evaluating IR expressions under the current lattice state.
pub struct AbstractInterpreter<'a> {
    pub state: &'a VmState,
}

impl<'a> AbstractInterpreter<'a> {
    pub fn new(state: &'a VmState) -> Self {
        Self { state }
    }

    /// Evaluates an IrExpr into a LatticeValue.
    pub fn eval(&self, expr: &IrExpr) -> LatticeValue {
        match expr {
            IrExpr::Nil => LatticeValue::Constant(Literal::Nil),
            IrExpr::Boolean(b) => LatticeValue::Constant(Literal::Bool(*b)),
            IrExpr::Number(n) => {
                if n.fract() == 0.0 && *n >= (i64::MIN as f64) && *n <= (i64::MAX as f64) {
                    LatticeValue::Constant(Literal::Int(*n as i64))
                } else {
                    LatticeValue::Constant(Literal::Float(*n))
                }
            }
            IrExpr::String(s) => LatticeValue::Constant(Literal::String(Arc::from(*s))),
            IrExpr::Register(r) => self.state.get_reg(*r),
            IrExpr::Unary { op, expr } => {
                let val = self.eval(expr);
                self.eval_unary(*op, &val)
            }
            IrExpr::Binary { op, left, right } => {
                let v_left = self.eval(left);
                let v_right = self.eval(right);
                self.eval_binary(*op, &v_left, &v_right)
            }
            // Objects/Tables/Closures have invariant truthiness = true in Luau
            IrExpr::NewTable { .. } | IrExpr::Closure { .. } => LatticeValue::KnownTruth(true),
            IrExpr::IfExp {
                cond,
                then_expr,
                else_expr,
            } => match self.eval(cond).truthiness() {
                Some(true) => self.eval(then_expr),
                Some(false) => self.eval(else_expr),
                None => {
                    let vt = self.eval(then_expr);
                    let ve = self.eval(else_expr);
                    vt.meet(&ve)
                }
            },
            _ => LatticeValue::Bottom,
        }
    }

    fn eval_unary(&self, op: UnOp, val: &LatticeValue) -> LatticeValue {
        if matches!(val, LatticeValue::Top) {
            return LatticeValue::Top;
        }

        match op {
            UnOp::Not => match val.truthiness() {
                Some(b) => LatticeValue::Constant(Literal::Bool(!b)),
                None => LatticeValue::Bottom,
            },
            UnOp::Neg => {
                if let Some(i) = val.as_i64() {
                    LatticeValue::Constant(Literal::Int(i.wrapping_neg()))
                } else if let Some(f) = val.as_f64() {
                    LatticeValue::Constant(Literal::Float(-f))
                } else {
                    LatticeValue::Bottom
                }
            }
            UnOp::Len => match val {
                LatticeValue::Constant(Literal::String(s)) => {
                    LatticeValue::Constant(Literal::Int(s.len() as i64))
                }
                _ => LatticeValue::Bottom,
            },
            UnOp::BitNot => {
                if let Some(i) = val.as_i64() {
                    let u = (i as u32) ^ 0xFFFFFFFF;
                    LatticeValue::Constant(Literal::Int(u as i64))
                } else {
                    LatticeValue::Bottom
                }
            }
        }
    }

    pub fn eval_binary(&self, op: BinOp, left: &LatticeValue, right: &LatticeValue) -> LatticeValue {
        // If either operand is Top, result is Top
        if matches!(left, LatticeValue::Top) || matches!(right, LatticeValue::Top) {
            return LatticeValue::Top;
        }

        // Short-circuiting logical operations
        if op == BinOp::And {
            return match left.truthiness() {
                Some(true) => right.clone(),
                Some(false) => left.clone(),
                None => LatticeValue::Bottom,
            };
        }
        if op == BinOp::Or {
            return match left.truthiness() {
                Some(true) => left.clone(),
                Some(false) => right.clone(),
                None => LatticeValue::Bottom,
            };
        }

        // Relational comparisons
        if matches!(
            op,
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge
        ) {
            return self.eval_comparison(op, left, right);
        }

        // Standard arithmetic and bitwise requires concrete constants
        let (LatticeValue::Constant(c1), LatticeValue::Constant(c2)) = (left, right) else {
            return LatticeValue::Bottom;
        };

        match op {
            BinOp::Add => self.eval_arith(c1, c2, |a, b| a + b, |a, b| a.wrapping_add(b)),
            BinOp::Sub => self.eval_arith(c1, c2, |a, b| a - b, |a, b| a.wrapping_sub(b)),
            BinOp::Mul => self.eval_arith(c1, c2, |a, b| a * b, |a, b| a.wrapping_mul(b)),
            BinOp::Div => {
                if let (Some(a), Some(b)) = (c1.as_f64(), c2.as_f64()) {
                    if b == 0.0 {
                        LatticeValue::Constant(Literal::Float(a / b)) // Lua inf/-inf
                    } else {
                        LatticeValue::Constant(Literal::Float(a / b))
                    }
                } else {
                    LatticeValue::Bottom
                }
            }
            BinOp::FloorDiv => {
                if let (Some(a), Some(b)) = (c1.as_f64(), c2.as_f64()) {
                    if b == 0.0 {
                        LatticeValue::Bottom
                    } else {
                        LatticeValue::Constant(Literal::Int((a / b).floor() as i64))
                    }
                } else {
                    LatticeValue::Bottom
                }
            }
            BinOp::Mod => {
                // Exact Luau Euclidean modulo: a - b * floor(a / b)
                if let (Some(i1), Some(i2)) = (c1.as_i64(), c2.as_i64()) {
                    if i2 == 0 {
                        LatticeValue::Bottom
                    } else {
                        let rem = i1.rem_euclid(i2);
                        LatticeValue::Constant(Literal::Int(rem))
                    }
                } else if let (Some(a), Some(b)) = (c1.as_f64(), c2.as_f64()) {
                    if b == 0.0 {
                        LatticeValue::Bottom
                    } else {
                        let m = a - b * (a / b).floor();
                        LatticeValue::Constant(Literal::Float(m))
                    }
                } else {
                    LatticeValue::Bottom
                }
            }
            BinOp::BitAnd => self.eval_bit32(c1, c2, |a, b| a & b),
            BinOp::BitOr => self.eval_bit32(c1, c2, |a, b| a | b),
            BinOp::BitXor => self.eval_bit32(c1, c2, |a, b| a ^ b),
            BinOp::ShiftLeft => self.eval_bit32(c1, c2, |a, b| if b >= 32 { 0 } else { a << b }),
            BinOp::ShiftRight => self.eval_bit32(c1, c2, |a, b| if b >= 32 { 0 } else { a >> b }),
            BinOp::Concat => {
                if let (Some(s1), Some(s2)) = (c1.as_str(), c2.as_str()) {
                    let mut s = String::with_capacity(s1.len() + s2.len());
                    s.push_str(s1);
                    s.push_str(s2);
                    LatticeValue::Constant(Literal::String(Arc::from(s.as_str())))
                } else {
                    LatticeValue::Bottom
                }
            }
            _ => LatticeValue::Bottom,
        }
    }

    fn eval_arith<F, I>(&self, c1: &Literal, c2: &Literal, float_op: F, int_op: I) -> LatticeValue
    where
        F: Fn(f64, f64) -> f64,
        I: Fn(i64, i64) -> i64,
    {
        if let (Some(i1), Some(i2)) = (c1.as_i64(), c2.as_i64()) {
            LatticeValue::Constant(Literal::Int(int_op(i1, i2)))
        } else if let (Some(f1), Some(f2)) = (c1.as_f64(), c2.as_f64()) {
            LatticeValue::Constant(Literal::Float(float_op(f1, f2)))
        } else {
            LatticeValue::Bottom
        }
    }

    fn eval_bit32<F>(&self, c1: &Literal, c2: &Literal, op: F) -> LatticeValue
    where
        F: Fn(u32, u32) -> u32,
    {
        let u1 = c1.as_i64().map(|x| x as u32);
        let u2 = c2.as_i64().map(|x| x as u32);
        if let (Some(a), Some(b)) = (u1, u2) {
            LatticeValue::Constant(Literal::Int(op(a, b) as i64))
        } else {
            LatticeValue::Bottom
        }
    }

    fn eval_comparison(&self, op: BinOp, left: &LatticeValue, right: &LatticeValue) -> LatticeValue {
        if let (LatticeValue::Constant(c1), LatticeValue::Constant(c2)) = (left, right) {
            // Direct Literal::Nil comparison evaluation:
            // nil == nil -> true; nil != nil -> false; nil == <other> -> false; nil != <other> -> true
            if matches!(c1, Literal::Nil) || matches!(c2, Literal::Nil) {
                let res = match op {
                    BinOp::Eq => c1 == c2,
                    BinOp::Ne => c1 != c2,
                    _ => return LatticeValue::Bottom,
                };
                return LatticeValue::Constant(Literal::Bool(res));
            }

            let res = match op {
                BinOp::Eq => c1 == c2,
                BinOp::Ne => c1 != c2,
                BinOp::Lt => match (c1.as_f64(), c2.as_f64()) {
                    (Some(a), Some(b)) => a < b,
                    _ => return LatticeValue::Bottom,
                },
                BinOp::Le => match (c1.as_f64(), c2.as_f64()) {
                    (Some(a), Some(b)) => a <= b,
                    _ => return LatticeValue::Bottom,
                },
                BinOp::Gt => match (c1.as_f64(), c2.as_f64()) {
                    (Some(a), Some(b)) => a > b,
                    _ => return LatticeValue::Bottom,
                },
                BinOp::Ge => match (c1.as_f64(), c2.as_f64()) {
                    (Some(a), Some(b)) => a >= b,
                    _ => return LatticeValue::Bottom,
                },
                _ => return LatticeValue::Bottom,
            };
            return LatticeValue::Constant(Literal::Bool(res));
        }

        // Truthiness comparison refinement
        if matches!(op, BinOp::Eq | BinOp::Ne) {
            if let (Some(t1), Some(t2)) = (left.truthiness(), right.truthiness()) {
                if t1 != t2 {
                    // Guaranteed distinct truthiness -> can never be equal
                    return LatticeValue::Constant(Literal::Bool(op == BinOp::Ne));
                }
            }
        }

        LatticeValue::Bottom
    }
}

/// Primary Sparse Conditional Constant Propagation engine.
pub struct SccpEngine<'a, 'arena> {
    pub cfg: &'a mut ControlFlowGraph<'arena>,
    pub executable_edges: HashSet<(BlockId, BlockId)>,
    pub executed_blocks: HashSet<BlockId>,
    pub cfg_worklist: VecDeque<(BlockId, BlockId)>,
    pub ssa_worklist: VecDeque<u16>,
    pub lat_vals: HashMap<u16, LatticeValue>,
    pub state_val: LatticeValue,
    pub use_sites: HashMap<u16, Vec<(BlockId, usize)>>,
}

impl<'a, 'arena> SccpEngine<'a, 'arena> {
    pub fn new(cfg: &'a mut ControlFlowGraph<'arena>) -> Self {
        Self::with_params(cfg, 30)
    }

    pub fn with_params(cfg: &'a mut ControlFlowGraph<'arena>, num_params: u16) -> Self {
        let mut engine = Self {
            cfg,
            executable_edges: HashSet::new(),
            executed_blocks: HashSet::new(),
            cfg_worklist: VecDeque::new(),
            ssa_worklist: VecDeque::new(),
            lat_vals: HashMap::new(),
            state_val: LatticeValue::Top,
            use_sites: HashMap::new(),
        };
        engine.build_use_sites();
        engine.seed_unassigned_registers(num_params);
        engine
    }

    fn collect_used_registers(expr: &IrExpr, out: &mut HashSet<u16>) {
        match expr {
            IrExpr::Register(r) => {
                out.insert(*r);
            }
            IrExpr::Unary { expr, .. } => Self::collect_used_registers(expr, out),
            IrExpr::Binary { left, right, .. } => {
                Self::collect_used_registers(left, out);
                Self::collect_used_registers(right, out);
            }
            IrExpr::TableGet { table, key } => {
                Self::collect_used_registers(table, out);
                Self::collect_used_registers(key, out);
            }
            IrExpr::Call { callee, args } => {
                Self::collect_used_registers(callee, out);
                for a in *args {
                    Self::collect_used_registers(a, out);
                }
            }
            IrExpr::IfExp { cond, then_expr, else_expr } => {
                Self::collect_used_registers(cond, out);
                Self::collect_used_registers(then_expr, out);
                Self::collect_used_registers(else_expr, out);
            }
            IrExpr::Closure { upvalues, .. } => {
                for u in *upvalues {
                    Self::collect_used_registers(u, out);
                }
            }
            IrExpr::Phi(entries) => {
                for (_, e) in *entries {
                    Self::collect_used_registers(e, out);
                }
            }
            _ => {}
        }
    }

    /// Seeds registers that are used but NEVER defined anywhere in the function CFG
    /// with `LatticeValue::Constant(Literal::Nil)`.
    fn seed_unassigned_registers(&mut self, num_params: u16) {
        let mut def_regs = HashSet::new();
        let mut used_regs = HashSet::new();

        for block in self.cfg.blocks.values() {
            for inst in &block.instructions {
                match inst {
                    IrInst::Assign { dest, src } => {
                        def_regs.insert(*dest);
                        Self::collect_used_registers(src, &mut used_regs);
                    }
                    IrInst::CallStmt { callee, args, ret_dest, .. } => {
                        if let Some(dest) = ret_dest {
                            def_regs.insert(*dest);
                        }
                        Self::collect_used_registers(callee, &mut used_regs);
                        for a in *args {
                            Self::collect_used_registers(a, &mut used_regs);
                        }
                    }
                    IrInst::TableSet { table, key, value } => {
                        Self::collect_used_registers(table, &mut used_regs);
                        Self::collect_used_registers(key, &mut used_regs);
                        Self::collect_used_registers(value, &mut used_regs);
                    }
                    IrInst::SetUpvalue { src, .. } | IrInst::SetGlobal { src, .. } => {
                        Self::collect_used_registers(src, &mut used_regs);
                    }
                    IrInst::SetList { table, values, .. } => {
                        Self::collect_used_registers(table, &mut used_regs);
                        for v in *values {
                            Self::collect_used_registers(v, &mut used_regs);
                        }
                    }
                    _ => {}
                }
            }
            match &block.terminator {
                Terminator::Branch { cond, .. } => {
                    Self::collect_used_registers(cond, &mut used_regs);
                }
                Terminator::Return(vals) => {
                    for v in *vals {
                        Self::collect_used_registers(v, &mut used_regs);
                    }
                }
                Terminator::ForLoop { var, start, limit, step, .. } => {
                    def_regs.insert(*var);
                    Self::collect_used_registers(start, &mut used_regs);
                    Self::collect_used_registers(limit, &mut used_regs);
                    if let Some(s) = step {
                        Self::collect_used_registers(s, &mut used_regs);
                    }
                }
                Terminator::ForInLoop { vars, iterators, .. } => {
                    for v in vars {
                        def_regs.insert(*v);
                    }
                    for it in iterators {
                        Self::collect_used_registers(it, &mut used_regs);
                    }
                }
                _ => {}
            }
        }

        for &reg in self.use_sites.keys() {
            used_regs.insert(reg);
        }

        // Registers >= num_params that are read in used_regs but have 0 definitions
        // are uninitialized Luau registers whose static runtime value is guaranteed nil.
        for reg in used_regs {
            if reg >= num_params && !def_regs.contains(&reg) {
                self.lat_vals.insert(reg, LatticeValue::Constant(Literal::Nil));
            }
        }
    }

    /// Indexes all register use sites across basic blocks for SSA propagation.
    fn build_use_sites(&mut self) {
        for (&bid, block) in &self.cfg.blocks {
            for (idx, inst) in block.instructions.iter().enumerate() {
                Self::collect_expr_uses(inst, bid, idx, &mut self.use_sites);
            }
        }
    }

    fn collect_expr_uses(
        inst: &IrInst,
        bid: BlockId,
        idx: usize,
        use_sites: &mut HashMap<u16, Vec<(BlockId, usize)>>,
    ) {
        fn walk(expr: &IrExpr, bid: BlockId, idx: usize, sites: &mut HashMap<u16, Vec<(BlockId, usize)>>) {
            match expr {
                IrExpr::Register(r) => sites.entry(*r).or_default().push((bid, idx)),
                IrExpr::Unary { expr, .. } => walk(expr, bid, idx, sites),
                IrExpr::Binary { left, right, .. } => {
                    walk(left, bid, idx, sites);
                    walk(right, bid, idx, sites);
                }
                IrExpr::TableGet { table, key } => {
                    walk(table, bid, idx, sites);
                    walk(key, bid, idx, sites);
                }
                IrExpr::Call { callee, args } => {
                    walk(callee, bid, idx, sites);
                    for a in *args {
                        walk(a, bid, idx, sites);
                    }
                }
                IrExpr::IfExp { cond, then_expr, else_expr } => {
                    walk(cond, bid, idx, sites);
                    walk(then_expr, bid, idx, sites);
                    walk(else_expr, bid, idx, sites);
                }
                _ => {}
            }
        }

        match inst {
            IrInst::Assign { src, .. } => walk(src, bid, idx, use_sites),
            IrInst::TableSet { table, key, value } => {
                walk(table, bid, idx, use_sites);
                walk(key, bid, idx, use_sites);
                walk(value, bid, idx, use_sites);
            }
            IrInst::SetUpvalue { src, .. } => walk(src, bid, idx, use_sites),
            IrInst::SetGlobal { src, .. } => walk(src, bid, idx, use_sites),
            IrInst::CallStmt { callee, args, .. } => {
                walk(callee, bid, idx, use_sites);
                for a in *args {
                    walk(a, bid, idx, use_sites);
                }
            }
            _ => {}
        }
    }

    /// Runs SCCP analysis to fixpoint and prunes dead branches.
    pub fn run(&mut self) {
        let entry = self.cfg.entry;
        self.cfg_worklist.push_back((BlockId::INVALID, entry));

        while !self.cfg_worklist.is_empty() || !self.ssa_worklist.is_empty() {
            // Process CFG Edge Worklist
            if let Some((u, v)) = self.cfg_worklist.pop_front() {
                if !self.executable_edges.contains(&(u, v)) {
                    self.executable_edges.insert((u, v));
                    let first_visit = self.executed_blocks.insert(v);

                    if first_visit {
                        self.evaluate_block(v);
                    }
                }
            }
            // Process SSA Worklist
            else if let Some(reg) = self.ssa_worklist.pop_front() {
                let uses = self.use_sites.get(&reg).cloned().unwrap_or_default();
                for (bid, inst_idx) in uses {
                    if self.executed_blocks.contains(&bid) {
                        self.re_evaluate_inst(bid, inst_idx);
                    }
                }
            }
        }

        // Post-SCCP CFG Rewrite Pass
        self.rewrite_branches();
    }

    fn evaluate_block(&mut self, bid: BlockId) {
        let Some(block) = self.cfg.blocks.get(&bid) else {
            return;
        };

        let mut vm_state = VmState::new();
        vm_state.registers = self.lat_vals.clone();
        vm_state.affine_state = self.state_val.clone();

        let interp = AbstractInterpreter::new(&vm_state);

        // Evaluate linear instructions
        for inst in &block.instructions {
            if let IrInst::Assign { dest, src } = inst {
                let val = interp.eval(src);
                let old = self.lat_vals.get(dest).cloned().unwrap_or(LatticeValue::Top);
                let meet_val = old.meet(&val);
                if meet_val != old {
                    self.lat_vals.insert(*dest, meet_val);
                    self.ssa_worklist.push_back(*dest);
                }
            }
        }

        // Evaluate terminator
        match &block.terminator {
            Terminator::Goto(target) => {
                if target.is_valid() {
                    self.cfg_worklist.push_back((bid, *target));
                }
            }
            Terminator::Branch {
                cond,
                true_target,
                false_target,
            } => {
                let cond_val = interp.eval(cond);
                match cond_val.truthiness() {
                    Some(true) => {
                        if true_target.is_valid() {
                            self.cfg_worklist.push_back((bid, *true_target));
                        }
                    }
                    Some(false) => {
                        if false_target.is_valid() {
                            self.cfg_worklist.push_back((bid, *false_target));
                        }
                    }
                    None => {
                        if true_target.is_valid() {
                            self.cfg_worklist.push_back((bid, *true_target));
                        }
                        if false_target.is_valid() {
                            self.cfg_worklist.push_back((bid, *false_target));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn re_evaluate_inst(&mut self, bid: BlockId, inst_idx: usize) {
        let Some(block) = self.cfg.blocks.get(&bid) else {
            return;
        };
        let Some(inst) = block.instructions.get(inst_idx) else {
            return;
        };

        if let IrInst::Assign { dest, src } = inst {
            let mut vm_state = VmState::new();
            vm_state.registers = self.lat_vals.clone();
            vm_state.affine_state = self.state_val.clone();
            let interp = AbstractInterpreter::new(&vm_state);

            let val = interp.eval(src);
            let old = self.lat_vals.get(dest).cloned().unwrap_or(LatticeValue::Top);
            let meet_val = old.meet(&val);
            if meet_val != old {
                self.lat_vals.insert(*dest, meet_val);
                self.ssa_worklist.push_back(*dest);
            }
        }
    }

    /// Rewrites conditional branches with constant predicates into unconditional Goto jumps.
    fn rewrite_branches(&mut self) {
        let mut vm_state = VmState::new();
        vm_state.registers = self.lat_vals.clone();
        vm_state.affine_state = self.state_val.clone();
        let interp = AbstractInterpreter::new(&vm_state);

        for block in self.cfg.blocks.values_mut() {
            if let Terminator::Branch {
                cond,
                true_target,
                false_target,
            } = &block.terminator
            {
                let cond_val = interp.eval(cond);
                match cond_val.truthiness() {
                    Some(true) => {
                        block.terminator = Terminator::Goto(*true_target);
                    }
                    Some(false) => {
                        block.terminator = Terminator::Goto(*false_target);
                    }
                    None => {}
                }
            }
        }

        self.cfg.recompute_successors();
        crate::cfg::builder::prune_unreachable(self.cfg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bumpalo::Bump;
    use crate::cfg::ir::BasicBlock;

    #[test]
    fn test_sccp_unassigned_registers_opaque_ne() {
        let bump = Bump::new();
        let mut cfg = ControlFlowGraph::new(0, BlockId::ENTRY);

        // Block 0:
        // if (r1075 ~= r2225) then Block 1 (Trap) else Block 2 (Safe)
        // Neither r1075 nor r2225 is defined in the CFG!
        let b0 = BasicBlock::new(
            BlockId(0),
            (1, 1),
            Terminator::Branch {
                cond: IrExpr::Binary {
                    op: BinOp::Ne,
                    left: Box::leak(Box::new(IrExpr::Register(1075))),
                    right: Box::leak(Box::new(IrExpr::Register(2225))),
                },
                true_target: BlockId(1),
                false_target: BlockId(2),
            },
        );

        // Block 1: Trap
        let b1 = BasicBlock::new(BlockId(1), (2, 2), Terminator::Crash);

        // Block 2: Safe Return
        let ret_vals: &[IrExpr] = bump.alloc_slice_clone(&[]);
        let b2 = BasicBlock::new(BlockId(2), (3, 3), Terminator::Return(ret_vals));

        cfg.add_block(b0);
        cfg.add_block(b1);
        cfg.add_block(b2);
        cfg.recompute_successors();

        let mut sccp = SccpEngine::new(&mut cfg);
        sccp.run();

        // Branch must be rewritten to Goto(BlockId(2)) because nil ~= nil is false
        let b0_after = &sccp.cfg.blocks[&BlockId::ENTRY];
        assert_eq!(b0_after.terminator, Terminator::Goto(BlockId(2)));
        assert!(!sccp.cfg.blocks.contains_key(&BlockId(1)), "Trap block 1 must be pruned");
    }
}

