//! Intermediate Representation (IR) and Control Flow Graph data structures.
//!
//! Provides arena-compatible expression, statement, terminator, and block definitions
//! for the Luraph v15 deobfuscation and structuring pipeline.

use std::fmt;
use std::sync::Arc;
use indexmap::IndexMap;

/// Strongly typed identifier for a basic block within a CFG.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BlockId(pub u32);

impl BlockId {
    /// Canonical entry block identifier.
    pub const ENTRY: Self = Self(0);
    /// Sentinel indicating an invalid or unassigned target.
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

impl fmt::Debug for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::INVALID {
            write!(f, "BlockId(INVALID)")
        } else {
            write!(f, "BB_{}", self.0)
        }
    }
}

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Self::INVALID {
            write!(f, "INVALID")
        } else {
            write!(f, "bb_{}", self.0)
        }
    }
}

/// Control-flow edge classification.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum EdgeType {
    /// Unconditional direct jump (goto / jmp).
    Unconditional,
    /// Conditional branch taken edge (true condition).
    ConditionalTrue,
    /// Conditional branch fallthrough / not-taken edge (false condition).
    ConditionalFalse,
    /// Natural linear fallthrough into adjacent basic block.
    Fallthrough,
    /// Return / termination edge to function exit.
    Return,
    /// Hostile crash / anti-tamper abort edge (LPH_CRASH).
    Crash,
}

/// Directed edge between basic blocks.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Edge {
    pub from: BlockId,
    pub to: BlockId,
    pub kind: EdgeType,
}

impl Edge {
    pub fn new(from: BlockId, to: BlockId, kind: EdgeType) -> Self {
        Self { from, to, kind }
    }
}

/// Unary operators in Luau / Luraph VM.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum UnOp {
    Neg,    // -
    Not,    // not
    Len,    // #
    BitNot, // ~
}

impl UnOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            UnOp::Neg => "-",
            UnOp::Not => "not ",
            UnOp::Len => "#",
            UnOp::BitNot => "bit32.bnot",
        }
    }
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Binary operators in Luau / Luraph VM.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum BinOp {
    Add,        // +
    Sub,        // -
    Mul,        // *
    Div,        // /
    FloorDiv,   // //
    Mod,        // %
    Pow,        // ^
    Concat,     // ..
    Eq,         // ==
    Ne,         // ~=
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    And,        // and
    Or,         // or
    BitAnd,     // & / bit32.band
    BitOr,      // | / bit32.bor
    BitXor,     // ~ / bit32.bxor
    ShiftLeft,  // << / bit32.lshift
    ShiftRight, // >> / bit32.rshift
}

impl BinOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::FloorDiv => "//",
            BinOp::Mod => "%",
            BinOp::Pow => "^",
            BinOp::Concat => "..",
            BinOp::Eq => "==",
            BinOp::Ne => "~=",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::BitXor => "~",
            BinOp::ShiftLeft => "<<",
            BinOp::ShiftRight => ">>",
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Concrete constant literal recognized by the abstract interpreter and IR.
#[derive(Clone, Debug, PartialEq)]
pub enum Literal {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(Arc<str>),
}

impl Literal {
    /// Evaluates Lua truthiness: nil and false are falsy; everything else is truthy.
    #[inline]
    pub fn is_truthy(&self) -> bool {
        match self {
            Literal::Nil => false,
            Literal::Bool(b) => *b,
            Literal::Int(_) | Literal::Float(_) | Literal::String(_) => true,
        }
    }

    #[inline]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Literal::Int(i) => Some(*i),
            Literal::Float(f) if f.fract() == 0.0 && f.is_finite() => Some(*f as i64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Literal::Int(i) => Some(*i as f64),
            Literal::Float(f) => Some(*f),
            _ => None,
        }
    }

    #[inline]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Literal::String(s) => Some(s.as_ref()),
            _ => None,
        }
    }
}

/// Rich Intermediate Representation Expression.
#[derive(Clone, Debug, PartialEq)]
pub enum IrExpr<'arena> {
    Nil,
    Boolean(bool),
    Number(f64),
    String(&'arena str),
    /// Virtual register R(A).
    Register(u16),
    /// Upvalue slot UP(A).
    Upvalue(u16),
    /// Global environment variable reference.
    Global(&'arena str),
    /// Reference to constant table index.
    Constant(usize),
    /// Unary operation.
    Unary {
        op: UnOp,
        expr: &'arena IrExpr<'arena>,
    },
    /// Binary operation.
    Binary {
        op: BinOp,
        left: &'arena IrExpr<'arena>,
        right: &'arena IrExpr<'arena>,
    },
    /// Table indexing: table[key].
    TableGet {
        table: &'arena IrExpr<'arena>,
        key: &'arena IrExpr<'arena>,
    },
    /// Function call expression: callee(args...).
    Call {
        callee: &'arena IrExpr<'arena>,
        args: &'arena [IrExpr<'arena>],
    },
    /// Table constructor { [k] = v, ... }.
    NewTable {
        array_size: u32,
        hash_size: u32,
    },
    /// Subprototype closure creation.
    Closure {
        proto_id: usize,
        upvalues: &'arena [IrExpr<'arena>],
    },
    /// Vararg expression (...).
    Vararg,
    /// Ternary if-expression (if c then a else b).
    IfExp {
        cond: &'arena IrExpr<'arena>,
        then_expr: &'arena IrExpr<'arena>,
        else_expr: &'arena IrExpr<'arena>,
    },
    /// SSA Phi node (used during SSA transformation).
    Phi(&'arena [(BlockId, IrExpr<'arena>)]),
}

/// Non-branching linear instructions within a basic block.
#[derive(Clone, Debug, PartialEq)]
pub enum IrInst<'arena> {
    /// Assignment to virtual register: R(dest) = src.
    Assign {
        dest: u16,
        src: IrExpr<'arena>,
    },
    /// Table write: table[key] = value.
    TableSet {
        table: IrExpr<'arena>,
        key: IrExpr<'arena>,
        value: IrExpr<'arena>,
    },
    /// Write to upvalue slot: UP(idx) = src.
    SetUpvalue {
        upvalue_idx: u16,
        src: IrExpr<'arena>,
    },
    /// Write to global: _G[name] = src.
    SetGlobal {
        name: &'arena str,
        src: IrExpr<'arena>,
    },
    /// Standalone function call statement (discards return or assigns to multret).
    CallStmt {
        callee: IrExpr<'arena>,
        args: &'arena [IrExpr<'arena>],
        ret_dest: Option<u16>,
        ret_count: i32,
    },
    /// Multret table constructor tail initialization: table[start..] = values.
    SetList {
        table: IrExpr<'arena>,
        start_idx: u32,
        values: &'arena [IrExpr<'arena>],
    },
    /// Closes active upvalues at and above register slot.
    CloseUpvalue(u16),
    /// Diagnostic comment or preserved side-effect annotation.
    Comment(&'arena str),
    /// No-operation (placeholder).
    Nop,
}

/// Control-flow terminator ending a basic block.
#[derive(Clone, Debug, PartialEq)]
pub enum Terminator<'arena> {
    /// Unconditional transfer of control to target block.
    Goto(BlockId),
    /// Conditional branch based on predicate expression.
    Branch {
        cond: IrExpr<'arena>,
        true_target: BlockId,
        false_target: BlockId,
    },
    /// Function return returning values to caller.
    Return(&'arena [IrExpr<'arena>]),
    /// High-level numeric for-loop header terminator.
    ForLoop {
        var: u16,
        start: IrExpr<'arena>,
        limit: IrExpr<'arena>,
        step: Option<IrExpr<'arena>>,
        body: BlockId,
        exit: BlockId,
    },
    /// High-level generic for-in loop header terminator.
    ForInLoop {
        vars: Vec<u16>,
        iterators: Vec<IrExpr<'arena>>,
        body: BlockId,
        exit: BlockId,
    },
    /// Explicit anti-tamper trap or VM crash (LPH_CRASH).
    Crash,
    /// Unreachable / dead-end block.
    Unreachable,
}

/// A maximal linear sequence of instructions ending in a single terminator.
#[derive(Clone, Debug)]
pub struct BasicBlock<'arena> {
    /// Unique identifier for this block.
    pub id: BlockId,
    /// Bytecode instruction range [start_pc, end_pc] corresponding to this block.
    pub pc_range: (usize, usize),
    /// Sequence of non-branching instructions.
    pub instructions: Vec<IrInst<'arena>>,
    /// Block control-flow terminator.
    pub terminator: Terminator<'arena>,
    /// Predecessor basic block IDs.
    pub predecessors: Vec<BlockId>,
    /// Successor basic block IDs.
    pub successors: Vec<BlockId>,
}

impl<'arena> BasicBlock<'arena> {
    pub fn new(id: BlockId, pc_range: (usize, usize), terminator: Terminator<'arena>) -> Self {
        Self {
            id,
            pc_range,
            instructions: Vec::new(),
            terminator,
            predecessors: Vec::new(),
            successors: Vec::new(),
        }
    }

    /// Returns true if this block has no active instructions and performs an unconditional goto.
    #[inline]
    pub fn is_empty_trampoline(&self) -> bool {
        let has_real_insts = self
            .instructions
            .iter()
            .any(|inst| !matches!(inst, IrInst::Nop | IrInst::Comment(_)));
        !has_real_insts && matches!(self.terminator, Terminator::Goto(_))
    }
}

/// Directed graph of basic blocks for a single prototype.
#[derive(Clone, Debug)]
pub struct ControlFlowGraph<'arena> {
    /// Prototype ID from which this CFG was lifted.
    pub proto_id: usize,
    /// Entry block of the function.
    pub entry: BlockId,
    /// Blocks indexed by BlockId preserving order.
    pub blocks: IndexMap<BlockId, BasicBlock<'arena>>,
}

impl<'arena> ControlFlowGraph<'arena> {
    pub fn new(proto_id: usize, entry: BlockId) -> Self {
        Self {
            proto_id,
            entry,
            blocks: IndexMap::new(),
        }
    }

    pub fn add_block(&mut self, block: BasicBlock<'arena>) {
        self.blocks.insert(block.id, block);
    }

    pub fn get_block(&self, id: BlockId) -> Option<&BasicBlock<'arena>> {
        self.blocks.get(&id)
    }

    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock<'arena>> {
        self.blocks.get_mut(&id)
    }

    /// Recomputes predecessor lists for all blocks from their successor links.
    pub fn recompute_predecessors(&mut self) {
        for block in self.blocks.values_mut() {
            block.predecessors.clear();
        }

        let edges: Vec<(BlockId, BlockId)> = self
            .blocks
            .iter()
            .flat_map(|(&from, block)| block.successors.iter().map(move |&to| (from, to)))
            .collect();

        for (from, to) in edges {
            if let Some(to_block) = self.blocks.get_mut(&to) {
                if !to_block.predecessors.contains(&from) {
                    to_block.predecessors.push(from);
                }
            }
        }
    }

    /// Recomputes successor lists for all blocks based on their terminators.
    pub fn recompute_successors(&mut self) {
        for block in self.blocks.values_mut() {
            block.successors.clear();
            match &block.terminator {
                Terminator::Goto(target) => {
                    if target.is_valid() {
                        block.successors.push(*target);
                    }
                }
                Terminator::Branch {
                    true_target,
                    false_target,
                    ..
                } => {
                    if true_target.is_valid() {
                        block.successors.push(*true_target);
                    }
                    if false_target.is_valid() && *false_target != *true_target {
                        block.successors.push(*false_target);
                    }
                }
                Terminator::ForLoop { body, exit, .. } | Terminator::ForInLoop { body, exit, .. } => {
                    if body.is_valid() {
                        block.successors.push(*body);
                    }
                    if exit.is_valid() && *exit != *body {
                        block.successors.push(*exit);
                    }
                }
                Terminator::Return(_) | Terminator::Crash | Terminator::Unreachable => {}
            }
        }
        self.recompute_predecessors();
    }
}

/// Loop construct kinds.
#[derive(Clone, Debug, PartialEq)]
pub enum LoopKind<'arena> {
    While(IrExpr<'arena>),
    Repeat(IrExpr<'arena>),
    Numeric {
        var: u16,
        start: IrExpr<'arena>,
        limit: IrExpr<'arena>,
        step: Option<IrExpr<'arena>>,
    },
    Generic {
        vars: Vec<u16>,
        iterators: Vec<IrExpr<'arena>>,
    },
}

/// High-level recovered structured control flow region.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuredRegion<'arena> {
    /// A single basic block identified by BlockId.
    Block(BlockId),
    /// Conditional branching: if cond then ... else ... end.
    If {
        cond: IrExpr<'arena>,
        then_branch: Box<StructuredRegion<'arena>>,
        else_branch: Option<Box<StructuredRegion<'arena>>>,
    },
    /// High-level loop construct.
    Loop {
        kind: LoopKind<'arena>,
        body: Box<StructuredRegion<'arena>>,
    },
    /// Linear sequence of structured regions.
    Seq(Vec<StructuredRegion<'arena>>),
    /// Break out of enclosing loop.
    Break,
    /// Continue to next iteration of enclosing loop.
    Continue,
    /// Return statement.
    Return(Vec<IrExpr<'arena>>),
    /// Anti-tamper crash abort.
    Crash,
    /// Diagnostic error message.
    Error(String),
}
