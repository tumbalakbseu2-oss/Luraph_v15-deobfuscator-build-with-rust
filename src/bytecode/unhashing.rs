//! Polynomial opcode unhashing and dynamic VM dispatch table mapping.

use std::collections::HashMap;

pub const TA: u32 = 1049162819;
pub const POLY_MOD: i64 = 1000003;

/// Computes the intermediate Tz value for a given 1-based Program Counter (PC).
#[inline]
pub fn compute_tz(pc: usize) -> i64 {
    let pc_i = pc as i64;
    ((pc_i * 211854 + 667701) % POLY_MOD).rem_euclid(POLY_MOD)
}

/// Computes the polynomial delta for a given Tz and PC.
#[inline]
pub fn compute_poly_delta(tz: i64, pc: usize) -> i64 {
    let pc_i = pc as i64;
    ((tz * tz + pc_i * 117911 + 74301) % POLY_MOD).rem_euclid(POLY_MOD)
}

/// Unhashes the raw opcode value for a specific PC into the unhashed tj hash.
#[inline]
pub fn unhash_opcode_raw(raw_tj: u32, pc: usize) -> i64 {
    let tz = compute_tz(pc);
    let delta = compute_poly_delta(tz, pc);
    let xor_val = (raw_tj ^ TA) as i64;
    xor_val - delta
}

/// Unhashes the raw opcode with Euclidean positive wrap-around.
/// Matches `T2_B44` boundary requirement.
#[inline]
pub fn unhash_opcode_wrapped(raw_tj: u32, pc: usize) -> u32 {
    let tj = unhash_opcode_raw(raw_tj, pc);
    tj.rem_euclid(POLY_MOD) as u32
}

/// Direct formula evaluation matching `t1_f03_opcode_unhashing_poly.lua`:
/// `bit32.bxor(raw_tj, TA) - ((tz*tz + pc*117911 + 74301) % 1000003)`
pub fn unhash_opcode_with_custom_tz(raw_tj: u32, pc: usize, tz: i64) -> i64 {
    let delta = compute_poly_delta(tz, pc);
    let xor_val = (raw_tj ^ TA) as i64;
    xor_val - delta
}

/// Standard Luraph v15 VM Opcode Mapping Table.
#[derive(Clone, Debug)]
pub struct OpcodeMap {
    lookup: HashMap<i64, u8>,
}

impl OpcodeMap {
    pub fn new() -> Self {
        let mut map = Self {
            lookup: HashMap::with_capacity(128),
        };
        map.populate_default_v15_opcodes();
        map
    }

    pub fn insert(&mut self, hash: i64, vm_op: u8) {
        self.lookup.insert(hash, vm_op);
    }

    /// Resolves raw opcode word at a specific PC into a canonical VM opcode.
    pub fn resolve(&self, raw_tj: u32, pc: usize) -> u8 {
        // 1. Direct unhashed lookup: if raw_tj is a direct opcode index in VM dispatch table
        if let Some(&op) = self.lookup.get(&(raw_tj as i64)) {
            return op;
        }

        // 2. Polynomial unhashing: for odSQ payloads where opcode word is hashed with TA and PC polynomial
        let tj = unhash_opcode_raw(raw_tj, pc);
        if let Some(&op) = self.lookup.get(&tj) {
            return op;
        }

        // 3. Fallback: check wrapped Euclidean hash
        let wrapped = tj.rem_euclid(POLY_MOD);
        if let Some(&op) = self.lookup.get(&wrapped) {
            return op;
        }

        // 4. Fallback: if raw_tj is a small wire opcode byte, preserve it directly
        if raw_tj < 256 {
            return raw_tj as u8;
        }

        // 5. Fallback: return lower 8 bits of hash as synthetic op
        (tj.rem_euclid(256)) as u8
    }

    fn populate_default_v15_opcodes(&mut self) {
        // Known opcode signatures from Luraph v15 VM dispatchers
        let canonical_ops: &[(i64, u8)] = &[
            // Standard Luraph v15 VM Canonical Dispatchers
            (512, 1),  // JMP
            (487, 1),  // JMP alternative
            (50, 2),   // CALL
            (33, 2),   // CALL
            (341, 2),  // CALL
            (113, 3),  // MOVE
            (607, 3),  // MOVE
            (40, 4),   // NEWTABLE
            (280, 5),  // LOADK
            (485, 5),  // LOADK
            (553, 6),  // GETTABLE
            (186, 7),  // SETTABLE
            (199, 8),  // GETGLOBAL
            (504, 9),  // SETGLOBAL
            (55, 10),  // ADD
            (529, 11), // MOD
            (57, 12),  // EQ
            (426, 13), // CONCAT
            (123, 14), // LEN
            (45, 15),  // NOT
            (4, 16),   // CLOSURE
            (231, 17), // RETURN
            (75, 18),  // LOADNIL
            (365, 19), // VARARG
            (361, 20), // GETUPVAL
            (152, 21), // SETUPVAL

            // Extended Wire Opcode Dispatches (from Luraph v15 VM dispatchers)
            (189, 7),  // SETTABLE
            (13, 1),   // JMP / branch
            (24, 1),   // JMP
            (35, 1),   // JMP
            (38, 1),   // JMP
            (42, 1),   // JMP
            (11, 2),   // CALL
            (27, 2),   // CALL
            (142, 2),  // CALL
            (25, 3),   // MOVE
            (31, 3),   // MOVE
            (36, 3),   // MOVE
            (144, 3),  // MOVE
            (148, 3),  // MOVE
            (196, 3),  // MOVE
            (201, 3),  // MOVE
            (1, 4),    // NEWTABLE
            (30, 4),   // NEWTABLE
            (161, 4),  // NEWTABLE
            (180, 4),  // NEWTABLE
            (103, 5),  // LOADK
            (166, 5),  // LOADK
            (209, 5),  // LOADK
            (20, 6),   // GETTABLE
            (39, 6),   // GETTABLE
            (52, 6),   // GETTABLE
            (71, 6),   // GETTABLE
            (110, 6),  // GETTABLE
            (133, 6),  // GETTABLE
            (143, 6),  // GETTABLE
            (146, 6),  // GETTABLE
            (149, 6),  // GETTABLE
            (211, 6),  // GETTABLE
            (21, 7),   // SETTABLE
            (29, 7),   // SETTABLE
            (34, 7),   // SETTABLE
            (41, 7),   // SETTABLE
            (61, 7),   // SETTABLE
            (63, 7),   // SETTABLE
            (122, 7),  // SETTABLE
            (157, 7),  // SETTABLE
            (2, 8),    // GETGLOBAL
            (3, 8),    // GETGLOBAL
            (12, 8),   // GETGLOBAL
            (26, 8),   // GETGLOBAL
            (97, 8),   // GETGLOBAL
            (99, 8),   // GETGLOBAL
            (116, 8),  // GETGLOBAL
            (145, 8),  // GETGLOBAL
            (5, 10),   // ADD
            (6, 10),   // ADD
            (7, 10),   // ADD
            (9, 10),   // ADD
            (10, 10),  // ADD
            (15, 10),  // ADD
            (19, 10),  // ADD
            (22, 10),  // ADD
            (23, 10),  // ADD
            (48, 10),  // ADD
            (58, 10),  // ADD
            (60, 10),  // ADD
            (62, 10),  // ADD
            (64, 10),  // ADD
            (65, 10),  // ADD
            (77, 10),  // ADD
            (79, 10),  // ADD
            (137, 10), // ADD
            (153, 10), // ADD
            (163, 10), // ADD
            (168, 10), // ADD
            (175, 10), // ADD
            (177, 10), // ADD
            (190, 10), // ADD / DIV
            (197, 10), // ADD
            (87, 11),  // MOD
            (170, 11), // MOD
            (8, 12),   // EQ
            (14, 12),  // EQ
            (32, 12),  // EQ
            (73, 12),  // EQ
            (188, 17), // RETURN
            (17, 17),  // RETURN
        ];
        for &(hash, op) in canonical_ops {
            self.lookup.insert(hash, op);
        }
    }
}

impl Default for OpcodeMap {
    fn default() -> Self {
        Self::new()
    }
}
