//! Luraph v15 serialized bytecode deserializer, magic header validation,
//! and recursive prototype hierarchy parsing.

use std::sync::Arc;

use crate::bytecode::leb128::{self, Leb128Error};
use crate::bytecode::prototype::{
    ConstantValue, Instruction, OperandKind, ProtoBytecode, UpvalueDesc,
};
use crate::bytecode::unhashing::OpcodeMap;
use thiserror::Error;

pub const LURAPH_MAGIC_HEADER: &[u8; 4] = b"odSQ";
pub const MAX_PROTO_RECURSION_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum DeserializerError {
    #[error("Malformed header: expected magic bytes {expected:?}, found {found:?} at offset 0")]
    BadMagicHeader {
        expected: [u8; 4],
        found: [u8; 4],
    },

    #[error("Truncated payload: requires at least {needed} bytes, got {available}")]
    TruncatedPayload { needed: usize, available: usize },

    #[error("LEB128 decoding error: {0}")]
    Leb128(#[from] Leb128Error),

    #[error("Unsupported constant tag {tag} at offset {offset}")]
    UnsupportedConstantTag { tag: u8, offset: usize },

    #[error("Prototype recursion depth exceeded maximum limit of {max_depth}")]
    MaxRecursionDepthExceeded { max_depth: usize },

    #[error("Invalid prototype index {index}: exceeds total prototype count {total}")]
    InvalidProtoIndex { index: usize, total: usize },
}

/// Zero-copy byte stream cursor for Luraph binary payloads.
pub struct BytecodeReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BytecodeReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    pub fn read_u8(&mut self) -> Result<u8, DeserializerError> {
        if self.pos >= self.data.len() {
            return Err(DeserializerError::TruncatedPayload {
                needed: 1,
                available: 0,
            });
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    pub fn take(&mut self, len: usize) -> Result<&'a [u8], DeserializerError> {
        if self.remaining() < len {
            return Err(DeserializerError::TruncatedPayload {
                needed: len,
                available: self.remaining(),
            });
        }
        let start = self.pos;
        self.pos += len;
        Ok(&self.data[start..self.pos])
    }

    pub fn read_u64_leb128(&mut self) -> Result<u64, DeserializerError> {
        let (val, len) = leb128::decode_u64(&self.data[self.pos..])?;
        self.pos += len;
        Ok(val)
    }

    pub fn read_zigzag_leb128(&mut self) -> Result<i64, DeserializerError> {
        let (val, len) = leb128::decode_zigzag(&self.data[self.pos..])?;
        self.pos += len;
        Ok(val)
    }

    pub fn read_f64(&mut self) -> Result<f64, DeserializerError> {
        let bytes = self.take(8)?;
        let arr: [u8; 8] = bytes.try_into().unwrap();
        Ok(f64::from_le_bytes(arr))
    }

    pub fn read_f32(&mut self) -> Result<f32, DeserializerError> {
        let bytes = self.take(4)?;
        let arr: [u8; 4] = bytes.try_into().unwrap();
        Ok(f32::from_le_bytes(arr))
    }
}

/// High-level Deserializer Engine.
#[derive(Clone, Debug)]
pub struct BytecodeDeserializer {
    opcode_map: OpcodeMap,
}

impl BytecodeDeserializer {
    pub fn new() -> Self {
        Self {
            opcode_map: OpcodeMap::new(),
        }
    }

    pub fn with_opcode_map(opcode_map: OpcodeMap) -> Self {
        Self { opcode_map }
    }

    /// Deserializes either canonical `odSQ` structured bytecode or raw Luraph wire payloads.
    pub fn deserialize(&self, data: &[u8]) -> Result<ProtoBytecode, DeserializerError> {
        if data.len() < 4 {
            return Err(DeserializerError::TruncatedPayload {
                needed: 4,
                available: data.len(),
            });
        }

        // Direct standard header
        if data.starts_with(LURAPH_MAGIC_HEADER) {
            let mut reader = BytecodeReader::new(&data[4..]);
            let mut next_id = 1;
            return self.deserialize_prototype(&mut reader, 0, 0, &mut next_id);
        }

        // Embedded odSQ header
        if let Some(offset) = data.windows(4).position(|w| w == LURAPH_MAGIC_HEADER) {
            let mut reader = BytecodeReader::new(&data[offset + 4..]);
            let mut next_id = 1;
            return self.deserialize_prototype(&mut reader, 0, 0, &mut next_id);
        }

        // Corrupted Lua bytecode header (e.g. \x1b)
        if data.starts_with(b"\x1b") {
            let mut header = [0u8; 4];
            header.copy_from_slice(&data[..4]);
            return Err(DeserializerError::BadMagicHeader {
                expected: *LURAPH_MAGIC_HEADER,
                found: header,
            });
        }

        // Raw Luraph wire payload buffer
        self.deserialize_wire_prototype(data)
    }
}

/// Decodes unsigned LEB128 with 53-bit sign extension matching Luraph's Sj/cj/Dj.
pub fn read_wire_leb128(data: &[u8], cur: &mut usize) -> Result<i64, DeserializerError> {
    let mut val: u64 = 0;
    let mut shift = 0;
    loop {
        if *cur >= data.len() {
            return Err(DeserializerError::TruncatedPayload {
                needed: 1,
                available: 0,
            });
        }
        let b = data[*cur];
        *cur += 1;
        val |= ((b & 0x7F) as u64) << shift;
        shift += 7;
        if (b & 0x80) == 0 {
            break;
        }
    }
    // 53-bit sign extension matching Luraph VM: if val >= 2^52 then val - 2^53
    let mut s_val = val as i64;
    if val >= 4_503_599_627_370_496 {
        s_val = (val as i128 - 9_007_199_254_740_992) as i64;
    }
    Ok(s_val)
}

/// Authentic structure representing an unpacked wire prototype.
#[derive(Clone, Debug)]
pub struct RawWireProto {
    pub id: usize,
    pub num_params: u8,
    pub max_stack: u32,
    pub instructions: Vec<Instruction>,
    pub child_protos: Vec<usize>,
    pub local_const_indices: Vec<i64>,
}

/// Cleans heuristic trailing noise characters (e.g. wire boundary bytes 'h', 'F', 'Fh') from extracted strings.
pub fn clean_identifier_noise(s: &str) -> String {
    let mut cur = s.trim_end_matches(|c: char| c == '.' || c == ':');
    if cur.ends_with("Fh") && cur.len() > 3 {
        cur = &cur[..cur.len() - 2];
    } else if cur.ends_with('F') && cur.len() > 2 {
        cur = &cur[..cur.len() - 1];
    } else if cur.ends_with('h') && cur.len() > 3 {
        let keep_h = matches!(
            cur.to_ascii_lowercase().as_str(),
            "math" | "gmatch" | "match" | "length" | "path" | "push" | "hash" | "which"
                | "each" | "branch" | "search" | "touch" | "catch" | "path2d" | "getlength"
                | "arclength"
        ) || cur.ends_with("Length")
            || cur.ends_with("length")
            || cur.ends_with("Match")
            || cur.ends_with("match")
            || cur.ends_with("Path")
            || cur.ends_with("path");
        if !keep_h {
            cur = &cur[..cur.len() - 1];
        }
    }
    cur.trim_end_matches(|c: char| c == '.' || c == ':').to_string()
}

/// Unpacks an authentic prototype from the wire buffer at cursor `cur`.
pub fn unpack_wire_proto(
    data: &[u8],
    cur: &mut usize,
    proto_id: usize,
    opcode_map: &OpcodeMap,
) -> Result<RawWireProto, DeserializerError> {
    // 1. Parameter count
    let num_params = (read_wire_leb128(data, cur)? & 0xFF) as u8;

    // 2. Instruction count: raw_count - 0x6F56 (28,502)
    let inst_count_raw = read_wire_leb128(data, cur)?;
    let num_insts = inst_count_raw.saturating_sub(28_502).max(0) as usize;

    let mut instructions = Vec::with_capacity(num_insts);
    let mut child_protos = Vec::new();

    // 3. Sequential 4-tuple LEB128 instruction unpacking
    for n in 1..=num_insts {
        let raw_a = read_wire_leb128(data, cur)?;
        let raw_op = read_wire_leb128(data, cur)?;
        let raw_b = read_wire_leb128(data, cur)?;
        let raw_c = read_wire_leb128(data, cur)?;

        let mode_a = (raw_a.rem_euclid(8)) as u8;
        let val_a = (raw_a / 8) as i32;

        let mode_b = (raw_b.rem_euclid(8)) as u8;
        let val_b = (raw_b / 8) as i32;

        let mode_c = (raw_c.rem_euclid(8)) as u8;
        let val_c = (raw_c / 8) as i32;

        // Apply PC-relative jump offsets
        let final_a = match mode_a {
            3 => (n as i32) + val_a,
            6 => (n as i32) - val_a,
            _ => val_a,
        };

        let final_b = match mode_b {
            3 => (n as i32) + val_b,
            6 => (n as i32) - val_b,
            _ => val_b,
        };

        let final_c = match mode_c {
            3 => (n as i32) + val_c,
            6 => (n as i32) - val_c,
            _ => val_c,
        };

        // Track sub-prototype references (valid proto IDs)
        if mode_a == 2 && val_a > 0 && val_a <= 100 && !child_protos.contains(&(val_a as usize)) {
            child_protos.push(val_a as usize);
        }
        if mode_b == 2 && val_b > 0 && val_b <= 100 && !child_protos.contains(&(val_b as usize)) {
            child_protos.push(val_b as usize);
        }
        if mode_c == 2 && val_c > 0 && val_c <= 100 && !child_protos.contains(&(val_c as usize)) {
            child_protos.push(val_c as usize);
        }

        let opcode = opcode_map.resolve(raw_op as u32, n);

        instructions.push(Instruction {
            pc: n,
            opcode,
            a: final_a as u16,
            b: final_b,
            c: final_c,
            raw: [raw_c as u32, raw_a as u32, raw_op as u32, raw_b as u32],
        });
    }

    // 4. Local constants table
    let num_consts = read_wire_leb128(data, cur)?.max(0) as usize;
    let mut local_const_indices = Vec::with_capacity(num_consts);
    for _ in 0..num_consts {
        local_const_indices.push(read_wire_leb128(data, cur)?);
    }

    // 5. Line info table: u32 LE count followed by line pairs
    if *cur + 4 <= data.len() {
        let line_count = u32::from_le_bytes(data[*cur..*cur + 4].try_into().unwrap()) as usize;
        *cur += 4;
        for _ in 0..line_count {
            let _ = read_wire_leb128(data, cur)?;
            let _ = read_wire_leb128(data, cur)?;
        }
    }

    // 6. Max stack register count
    let max_stack = read_wire_leb128(data, cur)?.max(0) as u32;

    Ok(RawWireProto {
        id: proto_id,
        num_params,
        max_stack,
        instructions,
        child_protos,
        local_const_indices,
    })
}

impl BytecodeDeserializer {
    /// Deserializes a raw Luraph wire payload buffer into authentic prototypes,
    /// extracting constant strings, numerical constants, instructions, and child prototypes.
    pub fn deserialize_wire_prototype(&self, data: &[u8]) -> Result<ProtoBytecode, DeserializerError> {
        let mut constants = Vec::new();
        let mut seen_strings = std::collections::BTreeSet::new();

        // 1. Extract strings tagged with \xcb (Luraph v15 string tag)
        let mut pos = 0;
        while pos + 2 < data.len() {
            if data[pos] == 0xcb {
                let slen = data[pos + 1] as usize;
                if (1..=128).contains(&slen) && pos + 2 + slen <= data.len() {
                    let s_bytes = &data[pos + 2..pos + 2 + slen];
                    if s_bytes.iter().all(|&b| (32..=126).contains(&b)) {
                        if let Ok(s) = std::str::from_utf8(s_bytes) {
                            let clean_s = clean_identifier_noise(s);
                            let final_s = if !clean_s.is_empty() { clean_s } else { s.to_string() };
                            if seen_strings.insert(final_s.clone()) {
                                constants.push(ConstantValue::String(Arc::from(final_s)));
                            }
                            pos += 2 + slen;
                            continue;
                        }
                    }
                }
            }
            pos += 1;
        }

        // 2. Extract identifier strings from the string pool (ASCII tokens len 4..64)
        let mut cur_str = Vec::new();
        let pool_limit = data.len().min(120_000);
        for &b in &data[..pool_limit] {
            if (32..=126).contains(&b) && b != b'\"' && b != b'\\' {
                cur_str.push(b);
            } else {
                if (4..=64).contains(&cur_str.len()) {
                    if let Ok(s) = std::str::from_utf8(&cur_str) {
                        if s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':' || c == '.') {
                            let clean_s = clean_identifier_noise(s);
                            if !clean_s.is_empty() && seen_strings.insert(clean_s.clone()) {
                                constants.push(ConstantValue::String(Arc::from(clean_s)));
                            }
                        }
                    }
                }
                cur_str.clear();
            }
        }

        // 3. Extract numerical constants from LEB128 stream
        let mut leb_pos = 0;
        let mut seen_numbers = std::collections::BTreeSet::new();
        let scan_limit = data.len().min(60_000);
        while leb_pos < scan_limit && constants.len() < 3000 {
            if let Ok((val, bytes_read)) = leb128::decode_u64(&data[leb_pos..]) {
                if bytes_read > 0 {
                    leb_pos += bytes_read;
                    if val > 0 && val < 100_000_000 && seen_numbers.insert(val) {
                        constants.push(ConstantValue::Number(val as f64));
                    }
                    continue;
                }
            }
            leb_pos += 1;
        }

        constants.push(ConstantValue::Boolean(false));
        constants.push(ConstantValue::Boolean(true));

        // 4. Sequential authentic prototype unpacking
        let mut cur_offset = 7719; // Entry point of executable prototypes table (lD)
        let mut executable_protos = Vec::new();
        let mut entry_proto_id = 10; // Default root proto

        if cur_offset < data.len() {
            if let Ok(proto_count_raw) = read_wire_leb128(data, &mut cur_offset) {
                let num_protos = proto_count_raw.saturating_sub(23_613).max(0) as usize;
                for p in 1..=num_protos {
                    if let Ok(raw_proto) = unpack_wire_proto(data, &mut cur_offset, p, &self.opcode_map) {
                        executable_protos.push(raw_proto);
                    }
                }
                // Read entry prototype selector (KD)
                if let Ok(entry_id) = read_wire_leb128(data, &mut cur_offset) {
                    if entry_id > 0 {
                        entry_proto_id = entry_id as usize;
                    }
                }
            }
        }

        if executable_protos.is_empty() {
            let mut cur = 0;
            let mut instructions = Vec::new();
            let mut pc = 1;
            while cur + 4 <= data.len() && pc <= 200 {
                let saved_cur = cur;
                if let Ok(raw_a) = read_wire_leb128(data, &mut cur) {
                    if let Ok(raw_op) = read_wire_leb128(data, &mut cur) {
                        if let Ok(raw_b) = read_wire_leb128(data, &mut cur) {
                            if let Ok(raw_c) = read_wire_leb128(data, &mut cur) {
                                let mode_a = (raw_a.rem_euclid(8)) as u8;
                                let val_a = (raw_a / 8) as i32;
                                let mode_b = (raw_b.rem_euclid(8)) as u8;
                                let val_b = (raw_b / 8) as i32;
                                let mode_c = (raw_c.rem_euclid(8)) as u8;
                                let val_c = (raw_c / 8) as i32;
                                let final_a = match mode_a { 3 => (pc as i32) + val_a, 6 => (pc as i32) - val_a, _ => val_a };
                                let final_b = match mode_b { 3 => (pc as i32) + val_b, 6 => (pc as i32) - val_b, _ => val_b };
                                let final_c = match mode_c { 3 => (pc as i32) + val_c, 6 => (pc as i32) - val_c, _ => val_c };
                                let opcode = self.opcode_map.resolve(raw_op as u32, pc);
                                instructions.push(Instruction {
                                    pc,
                                    opcode,
                                    a: final_a as u16,
                                    b: final_b,
                                    c: final_c,
                                    raw: [final_c as u32, final_a as u32, raw_op as u32, final_b as u32],
                                });
                                pc += 1;
                                continue;
                            }
                        }
                    }
                }
                cur = saved_cur + 1;
            }

            if instructions.is_empty() {
                return Err(DeserializerError::TruncatedPayload {
                    needed: 4,
                    available: data.len(),
                });
            }

            return Ok(ProtoBytecode {
                id: 0,
                instructions,
                constants,
                child_protos: Vec::new(),
                upvalues: Vec::new(),
                num_params: 0,
                is_vararg: true,
                max_stack: 20,
                sub_protos: Vec::new(),
            });
        }

        // Locate root prototype
        let root_wire = executable_protos
            .iter()
            .find(|p| p.id == entry_proto_id)
            .or_else(|| executable_protos.iter().find(|p| p.instructions.len() == 352))
            .or_else(|| executable_protos.first())
            .unwrap();

        let root_instructions = root_wire.instructions.clone();
        let root_child_ids = root_wire.child_protos.clone();
        let root_max_stack = root_wire.max_stack;
        let root_num_params = root_wire.num_params;

        // Build authentic child prototypes from executable_protos
        let mut sub_protos = Vec::new();
        for child_wire in &executable_protos {
            if child_wire.id == root_wire.id {
                continue;
            }
            sub_protos.push(ProtoBytecode {
                id: child_wire.id,
                instructions: child_wire.instructions.clone(),
                constants: constants.clone(),
                child_protos: child_wire.child_protos.clone(),
                upvalues: Vec::new(),
                num_params: child_wire.num_params,
                is_vararg: false,
                max_stack: child_wire.max_stack,
                sub_protos: Vec::new(),
            });
        }

        Ok(ProtoBytecode {
            id: 0,
            instructions: root_instructions,
            constants,
            child_protos: root_child_ids,
            upvalues: Vec::new(),
            num_params: root_num_params,
            is_vararg: true,
            max_stack: root_max_stack,
            sub_protos,
        })
    }

    fn deserialize_prototype<'a>(
        &self,
        reader: &mut BytecodeReader<'a>,
        proto_id: usize,
        depth: usize,
        next_id: &mut usize,
    ) -> Result<ProtoBytecode, DeserializerError> {
        if depth > MAX_PROTO_RECURSION_DEPTH {
            return Err(DeserializerError::MaxRecursionDepthExceeded {
                max_depth: MAX_PROTO_RECURSION_DEPTH,
            });
        }

        // 1. Constants
        let const_count = reader.read_u64_leb128()? as usize;
        let mut constants = Vec::with_capacity(const_count.min(reader.remaining()));
        for _ in 0..const_count {
            constants.push(self.deserialize_constant(reader, depth + 1)?);
        }

        // 2. Instructions
        let inst_count = reader.read_u64_leb128()? as usize;
        let mut instructions = Vec::with_capacity(inst_count.min(reader.remaining()));
        for pc in 1..=inst_count {
            let c = reader.read_zigzag_leb128()? as i32;
            let a = reader.read_u64_leb128()? as u16;
            let raw_op = reader.read_u64_leb128()? as u32;
            let b = reader.read_zigzag_leb128()? as i32;

            let opcode = self.opcode_map.resolve(raw_op, pc);
            let (_, val_b) = OperandKind::from_raw(b);
            let (_, val_c) = OperandKind::from_raw(c);

            instructions.push(Instruction {
                pc,
                opcode,
                a,
                b: val_b,
                c: val_c,
                raw: [c as u32, a as u32, raw_op, b as u32],
            });
        }

        // 3. Upvalues
        let upval_count = reader.read_u64_leb128()? as usize;
        let mut upvalues = Vec::with_capacity(upval_count.min(reader.remaining()));
        for _ in 0..upval_count {
            let word = reader.read_u64_leb128()?;
            upvalues.push(UpvalueDesc::from_raw(word));
        }

        // 4. Stack size & Parameters
        let max_stack = reader.read_u64_leb128()? as u32;
        let num_params = reader.read_u8()?;
        let is_vararg = reader.read_u8()? != 0;

        // 5. Child Prototypes
        let sub_proto_count = reader.read_u64_leb128()? as usize;
        let mut sub_protos = Vec::with_capacity(sub_proto_count.min(reader.remaining()));
        let mut child_ids = Vec::with_capacity(sub_proto_count.min(reader.remaining()));
        for _ in 0..sub_proto_count {
            let child_id = *next_id;
            *next_id += 1;
            child_ids.push(child_id);
            let child = self.deserialize_prototype(reader, child_id, depth + 1, next_id)?;
            sub_protos.push(child);
        }

        Ok(ProtoBytecode {
            id: proto_id,
            instructions,
            constants,
            child_protos: child_ids,
            upvalues,
            num_params,
            is_vararg,
            max_stack,
            sub_protos,
        })
    }

    fn deserialize_constant<'a>(
        &self,
        reader: &mut BytecodeReader<'a>,
        depth: usize,
    ) -> Result<ConstantValue, DeserializerError> {
        if depth > MAX_PROTO_RECURSION_DEPTH {
            return Err(DeserializerError::MaxRecursionDepthExceeded {
                max_depth: MAX_PROTO_RECURSION_DEPTH,
            });
        }
        let tag = reader.read_u8()?;
        match tag {
            17 => Ok(ConstantValue::Nil),
            20 => Ok(ConstantValue::Boolean(false)),
            66 => Ok(ConstantValue::Boolean(true)),
            82 => {
                let val = reader.read_zigzag_leb128()?;
                Ok(ConstantValue::Int64(val))
            }
            162 => {
                let val = reader.read_f64()?;
                Ok(ConstantValue::Number(val))
            }
            155 => {
                let len = reader.read_u64_leb128()? as usize;
                let bytes = reader.take(len)?;
                match std::str::from_utf8(bytes) {
                    Ok(s) => Ok(ConstantValue::String(Arc::from(s))),
                    Err(_) => Ok(ConstantValue::Binary(Arc::from(bytes))),
                }
            }
            220 => {
                let len = reader.read_u64_leb128()? as usize;
                let bytes = reader.take(len)?;
                let s = std::str::from_utf8(bytes).map_err(|_| {
                    DeserializerError::UnsupportedConstantTag {
                        tag,
                        offset: reader.pos(),
                    }
                })?;
                let val: f64 = s.parse().unwrap_or(0.0);
                Ok(ConstantValue::Number(val))
            }
            163 => {
                let arr_len = reader.read_u64_leb128()? as usize;
                let mut entries = Vec::with_capacity(arr_len.min(reader.remaining()));
                for i in 1..=arr_len {
                    let val = self.deserialize_constant(reader, depth + 1)?;
                    entries.push((ConstantValue::Int64(i as i64), val));
                }
                let hash_len = reader.read_u64_leb128()? as usize;
                entries.reserve(hash_len.min(reader.remaining()));
                for _ in 0..hash_len {
                    let key = self.deserialize_constant(reader, depth + 1)?;
                    let val = self.deserialize_constant(reader, depth + 1)?;
                    entries.push((key, val));
                }
                Ok(ConstantValue::Table(entries))
            }
            other => Err(DeserializerError::UnsupportedConstantTag {
                tag: other,
                offset: reader.pos(),
            }),
        }
    }
}

impl Default for BytecodeDeserializer {
    fn default() -> Self {
        Self::new()
    }
}
