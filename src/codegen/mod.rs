//! Luau Abstract Syntax Tree statement and expression definitions.
//!
//! Provides arena-compatible and owned AST nodes for structured Luau statements,
//! expressions, and lexical scopes with automated formatted source emission.

pub mod emitter;
pub mod idioms;

pub use emitter::{is_valid_ident, EmitterConfig, IndentStyle, LuauEmitter};
pub use idioms::IdiomPipeline;

/// Individual entry within a Luau table constructor literal: `{ ... }`.
#[derive(Clone, Debug, PartialEq)]
pub enum LuauTableEntry<'arena> {
    /// Sequential array item without explicit key: `{ val }`
    Array(LuauExpr<'arena>),
    /// Alternative/alias for sequential item: `{ val }`
    Value(LuauExpr<'arena>),
    /// Record field with valid string identifier: `{ key = val }`
    Record(&'arena str, LuauExpr<'arena>),
    /// General hashed key expression: `{ [key] = val }`
    General(LuauExpr<'arena>, LuauExpr<'arena>),
}

/// Binary operators supported in Luau expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Or,
    And,
    Equal,
    NotEqual,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Concat,
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
}

impl BinaryOp {
    /// Returns the standard Luau operator precedence level (1..=8).
    /// Higher values bind tighter.
    pub const fn precedence(self) -> u8 {
        match self {
            BinaryOp::Or => 1,
            BinaryOp::And => 2,
            BinaryOp::Equal
            | BinaryOp::NotEqual
            | BinaryOp::Less
            | BinaryOp::LessEq
            | BinaryOp::Greater
            | BinaryOp::GreaterEq => 3,
            BinaryOp::Concat => 4,
            BinaryOp::Add | BinaryOp::Sub => 5,
            BinaryOp::Mul | BinaryOp::Div | BinaryOp::FloorDiv | BinaryOp::Mod => 6,
            BinaryOp::BitAnd | BinaryOp::BitOr | BinaryOp::BitXor | BinaryOp::ShiftLeft | BinaryOp::ShiftRight => 6,
            BinaryOp::Pow => 8, // Binds tighter than unary operators (precedence 7)
        }
    }

    /// Whether this operator is right-associative (`..` and `^`).
    pub const fn is_right_associative(self) -> bool {
        matches!(self, BinaryOp::Concat | BinaryOp::Pow)
    }

    /// Canonical Luau operator string.
    pub const fn as_str(self) -> &'static str {
        match self {
            BinaryOp::Or => "or",
            BinaryOp::And => "and",
            BinaryOp::Equal => "==",
            BinaryOp::NotEqual => "~=",
            BinaryOp::Less => "<",
            BinaryOp::LessEq => "<=",
            BinaryOp::Greater => ">",
            BinaryOp::GreaterEq => ">=",
            BinaryOp::Concat => "..",
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::FloorDiv => "//",
            BinaryOp::Mod => "%",
            BinaryOp::Pow => "^",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "~",
            BinaryOp::ShiftLeft => "<<",
            BinaryOp::ShiftRight => ">>",
        }
    }

    /// Attempts to convert a BinaryOp to a Luau CompoundOp (if supported).
    pub const fn to_compound(self) -> Option<CompoundOp> {
        match self {
            BinaryOp::Add => Some(CompoundOp::Add),
            BinaryOp::Sub => Some(CompoundOp::Sub),
            BinaryOp::Mul => Some(CompoundOp::Mul),
            BinaryOp::Div => Some(CompoundOp::Div),
            BinaryOp::FloorDiv => Some(CompoundOp::FloorDiv),
            BinaryOp::Mod => Some(CompoundOp::Mod),
            BinaryOp::Pow => Some(CompoundOp::Pow),
            BinaryOp::Concat => Some(CompoundOp::Concat),
            _ => None,
        }
    }
}

impl From<&'static str> for BinaryOp {
    fn from(s: &'static str) -> Self {
        match s.trim() {
            "or" => BinaryOp::Or,
            "and" => BinaryOp::And,
            "==" => BinaryOp::Equal,
            "~=" => BinaryOp::NotEqual,
            "<" => BinaryOp::Less,
            "<=" => BinaryOp::LessEq,
            ">" => BinaryOp::Greater,
            ">=" => BinaryOp::GreaterEq,
            ".." => BinaryOp::Concat,
            "+" => BinaryOp::Add,
            "-" => BinaryOp::Sub,
            "*" => BinaryOp::Mul,
            "/" => BinaryOp::Div,
            "//" => BinaryOp::FloorDiv,
            "%" => BinaryOp::Mod,
            "^" => BinaryOp::Pow,
            "&" => BinaryOp::BitAnd,
            "|" => BinaryOp::BitOr,
            "~" => BinaryOp::BitXor,
            "<<" => BinaryOp::ShiftLeft,
            ">>" => BinaryOp::ShiftRight,
            _ => BinaryOp::Add,
        }
    }
}

impl From<crate::cfg::BinOp> for BinaryOp {
    fn from(op: crate::cfg::BinOp) -> Self {
        match op {
            crate::cfg::BinOp::Add => BinaryOp::Add,
            crate::cfg::BinOp::Sub => BinaryOp::Sub,
            crate::cfg::BinOp::Mul => BinaryOp::Mul,
            crate::cfg::BinOp::Div => BinaryOp::Div,
            crate::cfg::BinOp::FloorDiv => BinaryOp::FloorDiv,
            crate::cfg::BinOp::Mod => BinaryOp::Mod,
            crate::cfg::BinOp::Pow => BinaryOp::Pow,
            crate::cfg::BinOp::Concat => BinaryOp::Concat,
            crate::cfg::BinOp::Eq => BinaryOp::Equal,
            crate::cfg::BinOp::Ne => BinaryOp::NotEqual,
            crate::cfg::BinOp::Lt => BinaryOp::Less,
            crate::cfg::BinOp::Le => BinaryOp::LessEq,
            crate::cfg::BinOp::Gt => BinaryOp::Greater,
            crate::cfg::BinOp::Ge => BinaryOp::GreaterEq,
            crate::cfg::BinOp::And => BinaryOp::And,
            crate::cfg::BinOp::Or => BinaryOp::Or,
            crate::cfg::BinOp::BitAnd => BinaryOp::BitAnd,
            crate::cfg::BinOp::BitOr => BinaryOp::BitOr,
            crate::cfg::BinOp::BitXor => BinaryOp::BitXor,
            crate::cfg::BinOp::ShiftLeft => BinaryOp::ShiftLeft,
            crate::cfg::BinOp::ShiftRight => BinaryOp::ShiftRight,
        }
    }
}

/// Unary operators supported in Luau expressions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Not,
    Negate,
    Len,
    BitNot,
}

impl UnaryOp {
    /// Unary operator precedence in Luau (Level 7).
    pub const fn precedence(self) -> u8 {
        7
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            UnaryOp::Not => "not ",
            UnaryOp::Negate => "-",
            UnaryOp::Len => "#",
            UnaryOp::BitNot => "~",
        }
    }
}

impl From<&'static str> for UnaryOp {
    fn from(s: &'static str) -> Self {
        match s.trim() {
            "not" | "not " => UnaryOp::Not,
            "-" => UnaryOp::Negate,
            "#" => UnaryOp::Len,
            "~" | "bit32.bnot" => UnaryOp::BitNot,
            _ => UnaryOp::Not,
        }
    }
}

impl From<crate::cfg::UnOp> for UnaryOp {
    fn from(op: crate::cfg::UnOp) -> Self {
        match op {
            crate::cfg::UnOp::Neg => UnaryOp::Negate,
            crate::cfg::UnOp::Not => UnaryOp::Not,
            crate::cfg::UnOp::Len => UnaryOp::Len,
            crate::cfg::UnOp::BitNot => UnaryOp::BitNot,
        }
    }
}

/// Compound assignment operators in Luau.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CompoundOp {
    Add,      // '+='
    Sub,      // '-='
    Mul,      // '*='
    Div,      // '/='
    FloorDiv, // '//='
    Mod,      // '%='
    Pow,      // '^='
    Concat,   // '..='
}

impl CompoundOp {
    pub const fn as_str(self) -> &'static str {
        match self {
            CompoundOp::Add => "+=",
            CompoundOp::Sub => "-=",
            CompoundOp::Mul => "*=",
            CompoundOp::Div => "/=",
            CompoundOp::FloorDiv => "//=",
            CompoundOp::Mod => "%=",
            CompoundOp::Pow => "^=",
            CompoundOp::Concat => "..=",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LuauExpr<'arena> {
    Nil,
    Boolean(bool),
    Number(f64),
    String(&'arena str),
    Identifier(&'arena str),
    /// Table constructor with typed entries
    Table(Vec<LuauTableEntry<'arena>>),
    /// Unary operation
    Unary {
        op: UnaryOp,
        expr: Box<LuauExpr<'arena>>,
    },
    /// Binary operation
    Binary {
        op: BinaryOp,
        left: Box<LuauExpr<'arena>>,
        right: Box<LuauExpr<'arena>>,
    },
    /// Table index access: `table[key]` or `table.field`
    Index {
        table: Box<LuauExpr<'arena>>,
        key: Box<LuauExpr<'arena>>,
    },
    /// Regular function call: `func(a, b, c)`
    Call {
        func: Box<LuauExpr<'arena>>,
        args: Vec<LuauExpr<'arena>>,
    },
    /// Method call syntax sugar: `target:method(a, b)`
    MethodCall {
        target: Box<LuauExpr<'arena>>,
        method: &'arena str,
        args: Vec<LuauExpr<'arena>>,
    },
    /// Anonymous function literal: `function(a, b) ... end`
    Function {
        params: Vec<&'arena str>,
        is_vararg: bool,
        body: Vec<LuauStat<'arena>>,
    },
    /// Ternary if-expression in Luau: `if cond then true_expr else false_expr`
    IfExp {
        cond: Box<LuauExpr<'arena>>,
        then_expr: Box<LuauExpr<'arena>>,
        else_expr: Box<LuauExpr<'arena>>,
    },
    Vararg,
    Raw(&'arena str),
}

impl<'arena> LuauExpr<'arena> {
    /// Formats this expression into Luau source code using the default emitter configuration.
    pub fn to_code(&self) -> String {
        let emitter = LuauEmitter::new(EmitterConfig::default());
        emitter.format_expr(self, 0)
    }
}

/// An `elseif` branch within a flattened If statement ladder.
#[derive(Clone, Debug, PartialEq)]
pub struct ElseIfBranch<'arena> {
    pub cond: LuauExpr<'arena>,
    pub body: Vec<LuauStat<'arena>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LuauStat<'arena> {
    /// Bare local variable declaration: `local a, b, c`
    LocalDecl(Vec<&'arena str>),
    /// Initialized local variable declaration: `local a, b = v1, v2`
    LocalAssign {
        names: Vec<&'arena str>,
        values: Vec<LuauExpr<'arena>>,
    },
    /// Local function definition: `local function name(a, b) ... end`
    LocalFunctionDef {
        name: &'arena str,
        params: Vec<&'arena str>,
        is_vararg: bool,
        body: Vec<LuauStat<'arena>>,
    },
    /// Global or top-level function definition: `function name(a, b) ... end`
    FunctionDef {
        name: &'arena str,
        params: Vec<&'arena str>,
        is_vararg: bool,
        body: Vec<LuauStat<'arena>>,
    },
    /// Plain multi-target assignment statement: `a, b = v1, v2`
    Assign {
        targets: Vec<LuauExpr<'arena>>,
        values: Vec<LuauExpr<'arena>>,
    },
    /// Luau compound assignment: `target += value`
    CompoundAssign {
        target: LuauExpr<'arena>,
        op: CompoundOp,
        value: LuauExpr<'arena>,
    },
    /// Standalone expression / function call statement: `print(x)` or `obj:Method(a)`
    Call(LuauExpr<'arena>),
    /// Function return: `return a, b`
    Return(Vec<LuauExpr<'arena>>),
    /// Conditional branch with optional elseif ladder and else block:
    /// `if cond then ... elseif cond2 then ... else ... end`
    If {
        cond: LuauExpr<'arena>,
        then_branch: Vec<LuauStat<'arena>>,
        elseif_branches: Vec<ElseIfBranch<'arena>>,
        else_branch: Option<Vec<LuauStat<'arena>>>,
    },
    /// While loop: `while cond do ... end`
    While {
        cond: LuauExpr<'arena>,
        body: Vec<LuauStat<'arena>>,
    },
    /// Repeat loop: `repeat ... until cond`
    Repeat {
        cond: LuauExpr<'arena>,
        body: Vec<LuauStat<'arena>>,
    },
    /// Numeric for loop: `for var = start, limit, step do ... end`
    NumericFor {
        var: &'arena str,
        start: LuauExpr<'arena>,
        limit: LuauExpr<'arena>,
        step: Option<LuauExpr<'arena>>,
        body: Vec<LuauStat<'arena>>,
    },
    /// Generic for loop: `for k, v in iter do ... end`
    GenericFor {
        vars: Vec<&'arena str>,
        iterators: Vec<LuauExpr<'arena>>,
        body: Vec<LuauStat<'arena>>,
    },
    /// Scope-isolated block resetting VM register allocation: `do ... end`
    DoBlock(Vec<LuauStat<'arena>>),
    /// Loop break: `break`
    Break,
    /// Loop continue: `continue`
    Continue,
    /// Comment or diagnostic note: `-- note`
    Comment(&'arena str),
    /// Raw unparsed statement
    Raw(&'arena str),
}

impl<'arena> LuauStat<'arena> {
    /// Formats and appends this statement to `out` with indentation depth `indent`.
    pub fn emit(&self, indent: usize, out: &mut String) {
        let mut emitter = LuauEmitter::new(EmitterConfig::default());
        emitter.current_indent = indent;
        emitter.emit_stat(self);
        out.push_str(&emitter.buffer);
    }
}

#[derive(Clone, Debug, Default)]
pub struct LuauModule<'arena> {
    pub body: Vec<LuauStat<'arena>>,
}

impl<'arena> LuauModule<'arena> {
    pub fn new() -> Self {
        Self { body: Vec::new() }
    }

    pub fn to_code(&self, indent: usize) -> String {
        let mut emitter = LuauEmitter::new(EmitterConfig::default());
        emitter.current_indent = indent;
        for st in &self.body {
            emitter.emit_stat(st);
        }
        emitter.buffer
    }
}
