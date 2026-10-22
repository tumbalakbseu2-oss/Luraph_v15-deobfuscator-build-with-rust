//! Bytecode instruction, upvalue, constant pool, and prototype representations.

use std::sync::Arc;

/// Decoded VM Instruction.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Instruction {
    pub pc: usize,
    pub opcode: u8,
    pub a: u16,
    pub b: i32,
    pub c: i32,
    pub raw: [u32; 4],
}

impl Instruction {
    pub const fn new(pc: usize, opcode: u8, a: u16, b: i32, c: i32, raw: [u32; 4]) -> Self {
        Self {
            pc,
            opcode,
            a,
            b,
            c,
            raw,
        }
    }

    /// Original raw opcode word before unhashing.
    #[inline]
    pub fn raw_opcode(&self) -> u32 {
        self.raw[2]
    }

    /// Original unshifted operand B word.
    #[inline]
    pub fn raw_b(&self) -> i32 {
        self.raw[3] as i32
    }

    /// Original unshifted operand C word.
    #[inline]
    pub fn raw_c(&self) -> i32 {
        self.raw[0] as i32
    }
}

/// Luraph Operand Decoding Mode (3-bit tag).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandKind {
    RegisterOrImmediate, // Mode 0 or 4
    ConstantIndex,       // Mode 1: index into constant table
    ChildProtoIndex,     // Mode 3: index into subprototypes
    RelativeJumpBack,    // Mode 5: target = PC - value
    RelativeJumpForward, // Mode 6: target = PC + value
    Unknown(u8),
}

impl OperandKind {
    pub fn from_raw(raw: i32) -> (Self, i32) {
        let mode = (raw.rem_euclid(8)) as u8;
        let value = raw / 8;
        let kind = match mode {
            0 | 4 | 7 => OperandKind::RegisterOrImmediate,
            1 => OperandKind::ConstantIndex,
            2 => OperandKind::ChildProtoIndex,
            3 => OperandKind::RelativeJumpForward,
            6 => OperandKind::RelativeJumpBack,
            other => OperandKind::Unknown(other),
        };
        (kind, value)
    }
}

/// Upvalue Descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct UpvalueDesc {
    pub in_stack: bool, // Mode 0: local in parent stack; Mode 1: upvalue in parent
    pub index: u16,     // Slot index in parent
}

impl UpvalueDesc {
    pub fn new(in_stack: bool, index: u16) -> Self {
        Self { in_stack, index }
    }

    pub fn from_raw(word: u64) -> Self {
        let mode = (word % 4) as u8;
        let index = (word / 4) as u16;
        Self {
            in_stack: mode == 0,
            index,
        }
    }
}

/// Constant Pool Value.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstantValue {
    Nil,
    Boolean(bool),
    Number(f64),
    Float32(f32),
    Int64(i64),
    String(Arc<str>),
    Binary(Arc<[u8]>),
    Table(Vec<(ConstantValue, ConstantValue)>),
}

impl ConstantValue {
    pub fn is_nil(&self) -> bool {
        matches!(self, ConstantValue::Nil)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            ConstantValue::String(s) => Some(s.as_ref()),
            _ => None,
        }
    }

    pub fn as_number(&self) -> Option<f64> {
        match self {
            ConstantValue::Number(n) => Some(*n),
            ConstantValue::Float32(f) => Some(*f as f64),
            ConstantValue::Int64(i) => Some(*i as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConstantValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Arena-Scoped Prototype Representation for zero-allocation lifting passes.
#[derive(Clone, Debug)]
pub struct Prototype<'arena> {
    pub id: usize,
    pub instructions: &'arena [Instruction],
    pub constants: &'arena [ConstantValue],
    pub child_protos: Vec<usize>,
    pub upvalues: &'arena [UpvalueDesc],
    pub num_params: u8,
    pub is_vararg: bool,
    pub max_stack: u32,
}

/// Heap-Allocated ProtoBytecode Container (Pipeline Transmission Object).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct ProtoBytecode {
    pub id: usize,
    pub instructions: Vec<Instruction>,
    pub constants: Vec<ConstantValue>,
    pub child_protos: Vec<usize>,
    pub upvalues: Vec<UpvalueDesc>,
    pub num_params: u8,
    pub is_vararg: bool,
    pub max_stack: u32,
    pub sub_protos: Vec<ProtoBytecode>,
}

impl ProtoBytecode {
    /// Flattens the recursive prototype hierarchy into a flat list indexed by prototype ID.
    pub fn flatten(&self) -> Vec<ProtoBytecode> {
        let mut list = Vec::new();
        self.collect_flat(&mut list);
        list.sort_by_key(|p| p.id);
        list
    }

    fn collect_flat(&self, out: &mut Vec<ProtoBytecode>) {
        let mut copy = self.clone();
        let sub_ids: Vec<usize> = self.sub_protos.iter().map(|s| s.id).collect();
        copy.child_protos = sub_ids;
        copy.sub_protos.clear();
        out.push(copy);

        for child in &self.sub_protos {
            child.collect_flat(out);
        }
    }
}
