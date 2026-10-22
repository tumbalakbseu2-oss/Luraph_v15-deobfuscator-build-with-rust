//! Milestone 2 Bytecode Deserializer, LEB128 & Polynomial Unhashing Test Suite.
//!
//! Validates:
//! - T1_F03_01: Biased LEB128 positive integers (bias = 217)
//! - T1_F03_02: Biased LEB128 negative integers (bias = 217)
//! - T1_F03_03: Signed Zig-Zag integer decoding
//! - T1_F03_04: Dynamic opcode polynomial unhashing
//! - T2_B05: Malformed magic header rejection
//! - T2_B06: Truncated LEB128 stream detection
//! - T2_B07: Max 64-bit LEB128 value boundary
//! - T2_B08: Min i32 Zig-Zag (-2^31)
//! - T2_B09: Max i32 Zig-Zag (2^31 - 1)
//! - T2_B44: Opcode unhashing negative intermediate wrap-around

use luraph_deobf_rs::bytecode::deserializer::{
    BytecodeDeserializer, DeserializerError, LURAPH_MAGIC_HEADER,
};
use luraph_deobf_rs::bytecode::leb128::{
    decode_byte_biased_leb128, decode_leb128_biased, decode_u64, zigzag_decode,
    zigzag_encode, Leb128Error, LURAPH_LEB128_BIAS,
};
use luraph_deobf_rs::bytecode::unhashing::{
    compute_poly_delta, compute_tz, unhash_opcode_with_custom_tz, OpcodeMap,
    POLY_MOD, TA,
};

#[test]
fn test_t1_f03_01_leb128_bias217_pos() {
    // Encodes 300: (300 + 217) = 517 -> 0x85 0x04
    let buffer = [0x85, 0x04];
    let (val, bytes_read) = decode_leb128_biased(&buffer, LURAPH_LEB128_BIAS).unwrap();
    assert_eq!(bytes_read, 2);
    assert_eq!(val, 300);
}

#[test]
fn test_t1_f03_02_leb128_bias217_neg() {
    // Encodes -100: (-100 + 217) = 117 -> 0x75
    let buffer = [0x75];
    let (val, bytes_read) = decode_leb128_biased(&buffer, LURAPH_LEB128_BIAS).unwrap();
    assert_eq!(bytes_read, 1);
    assert_eq!(val, -100);
}

#[test]
fn test_byte_biased_leb128_bt_variant() {
    // Tests Luraph VM bT: (byte - 217) % 256
    // If tw = (byte - 217) % 256, for tw = 42: byte = (42 + 217) % 256 = 3
    let buffer = [3u8];
    let (val, bytes_read) = decode_byte_biased_leb128(&buffer, 217).unwrap();
    assert_eq!(bytes_read, 1);
    assert_eq!(val, 42);
}

#[test]
fn test_t1_f03_03_zigzag_decode() {
    // Zig-zag sequence: 0 -> 0, 1 -> -1, 2 -> 1, 3 -> -2
    assert_eq!(zigzag_decode(0), 0);
    assert_eq!(zigzag_decode(1), -1);
    assert_eq!(zigzag_decode(2), 1);
    assert_eq!(zigzag_decode(3), -2);

    // Round-trip verification
    for n in -500..=500 {
        assert_eq!(zigzag_decode(zigzag_encode(n)), n);
    }
}

#[test]
fn test_t2_b08_min_i32_zigzag() {
    // Min i32 = -2147483648
    let min_i32: i64 = -2147483648;
    let enc = zigzag_encode(min_i32);
    assert_eq!(zigzag_decode(enc), min_i32);
}

#[test]
fn test_t2_b09_max_i32_zigzag() {
    // Max i32 = 2147483647
    let max_i32: i64 = 2147483647;
    let enc = zigzag_encode(max_i32);
    assert_eq!(zigzag_decode(enc), max_i32);
}

#[test]
fn test_t2_b07_max_u64_leb128() {
    // [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01] = u64::MAX
    let max_leb = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
    let (val, bytes_read) = decode_u64(&max_leb).unwrap();
    assert_eq!(bytes_read, 10);
    assert_eq!(val, u64::MAX);
}

#[test]
fn test_t2_b06_truncated_leb128() {
    // [0x85, 0x80] has MSB set on both bytes, missing final terminating byte
    let truncated = [0x85, 0x80];
    let err = decode_u64(&truncated).unwrap_err();
    assert_eq!(err, Leb128Error::UnexpectedEof { offset: 2 });
}

#[test]
fn test_leb128_overflow_11_bytes() {
    let overflow = [0x80; 11];
    let err = decode_u64(&overflow).unwrap_err();
    assert!(matches!(err, Leb128Error::Overflow { .. }));
}

#[test]
fn test_t1_f03_04_opcode_unhashing_poly() {
    // From fixture t1_f03_opcode_unhashing_poly.lua:
    // TA = 1049162819, tz = 42, pc = 5, raw_tj = 1049162850
    // poly = (42*42 + 5*117911 + 74301) % 1000003 = 665620
    // xor = 1049162850 ^ 1049162819 = 33
    // opcode = 33 - 665620 = -665587
    let tz: i64 = 42;
    let pc: usize = 5;
    let raw_tj: u32 = 1049162850;

    let opcode = unhash_opcode_with_custom_tz(raw_tj, pc, tz);
    assert_eq!(opcode, -665587);

    // Test compute_tz with standard formula:
    let natural_tz = compute_tz(pc);
    let natural_delta = compute_poly_delta(natural_tz, pc);
    assert!(natural_tz >= 0 && natural_tz < POLY_MOD);
    assert!(natural_delta >= 0 && natural_delta < POLY_MOD);
}

#[test]
fn test_t2_b44_opcode_poly_negative_intermediate() {
    // From fixture t2_b44_opcode_poly_negative_intermediate.lua:
    // raw = 500, poly = 2000000 -> (500 - 2000000) % 1000003 == 506
    let raw: i64 = 500;
    let poly: i64 = 2000000;
    let res = (raw - poly).rem_euclid(POLY_MOD);
    assert_eq!(res, 506);
}

#[test]
fn test_opcode_map_resolution() {
    let mut map = OpcodeMap::new();
    map.insert(512, 1); // JMP
    map.insert(50, 2);  // CALL

    // Natural resolution for JMP
    assert_eq!(map.resolve(512 ^ TA, 1), map.resolve(512 ^ TA, 1));
}

#[test]
fn test_t2_b05_malformed_header() {
    let deserializer = BytecodeDeserializer::new();
    let malformed = b"\x1bLPH_CORRUPTED_MAGIC_HEADER_0xDEADBEEF\x00\xff";
    let err = deserializer.deserialize(malformed).unwrap_err();
    assert!(matches!(err, DeserializerError::BadMagicHeader { .. }));
}

#[test]
fn test_deserializer_truncated_empty() {
    let deserializer = BytecodeDeserializer::new();
    let err = deserializer.deserialize(&[]).unwrap_err();
    assert!(matches!(err, DeserializerError::TruncatedPayload { .. }));
}

#[test]
fn test_deserializer_valid_minimal_payload() {
    let deserializer = BytecodeDeserializer::new();
    let mut payload = Vec::new();
    // 1. Magic Header
    payload.extend_from_slice(LURAPH_MAGIC_HEADER);
    // 2. Constants count = 0
    payload.push(0x00);
    // 3. Instructions count = 0
    payload.push(0x00);
    // 4. Upvalues count = 0
    payload.push(0x00);
    // 5. Max stack = 10
    payload.push(0x0A);
    // 6. Params = 0, Vararg = 0
    payload.push(0x00);
    payload.push(0x00);
    // 7. Sub-protos = 0
    payload.push(0x00);

    let proto = deserializer.deserialize(&payload).unwrap();
    assert_eq!(proto.id, 0);
    assert_eq!(proto.instructions.len(), 0);
    assert_eq!(proto.constants.len(), 0);
    assert_eq!(proto.sub_protos.len(), 0);
    assert_eq!(proto.max_stack, 10);
    assert_eq!(proto.num_params, 0);
    assert!(!proto.is_vararg);
}

#[test]
fn test_deserializer_nested_prototype_hierarchy() {
    let deserializer = BytecodeDeserializer::new();
    let mut payload = Vec::new();

    // Header
    payload.extend_from_slice(LURAPH_MAGIC_HEADER);

    // Root proto (ID 0)
    // 1. Constants count = 1: Tag 20 (Boolean false)
    payload.push(0x01);
    payload.push(20);

    // 2. Instructions count = 1
    payload.push(0x01);
    payload.push(0x00); // c = 0
    payload.push(0x01); // a = 1
    payload.push(0x02); // raw_op = 2
    payload.push(0x00); // b = 0

    // 3. Upvalues count = 1: (mode 0, index 3 -> word = 12)
    payload.push(0x01);
    payload.push(12);

    // 4. Max stack = 5, params = 2, vararg = 1
    payload.push(0x05);
    payload.push(0x02);
    payload.push(0x01);

    // 5. Sub-protos count = 1 (Child proto ID 1)
    payload.push(0x01);

    // Child proto body:
    // Constants count = 0
    payload.push(0x00);
    // Instructions count = 0
    payload.push(0x00);
    // Upvalues count = 0
    payload.push(0x00);
    // Max stack = 2, params = 0, vararg = 0
    payload.push(0x02);
    payload.push(0x00);
    payload.push(0x00);
    // Sub-protos count = 0
    payload.push(0x00);

    let root = deserializer.deserialize(&payload).unwrap();
    assert_eq!(root.id, 0);
    assert_eq!(root.constants.len(), 1);
    assert_eq!(root.instructions.len(), 1);
    assert_eq!(root.upvalues.len(), 1);
    assert!(root.upvalues[0].in_stack);
    assert_eq!(root.upvalues[0].index, 3);
    assert_eq!(root.num_params, 2);
    assert!(root.is_vararg);
    assert_eq!(root.sub_protos.len(), 1);
    assert_eq!(root.sub_protos[0].id, 1);

    // Flattening check
    let flat = root.flatten();
    assert_eq!(flat.len(), 2);
    assert_eq!(flat[0].id, 0);
    assert_eq!(flat[0].child_protos, vec![1]);
    assert_eq!(flat[1].id, 1);
}

#[test]
fn test_deserializer_branching_prototype_hierarchy_unique_ids() {
    let deserializer = BytecodeDeserializer::new();
    let mut payload = Vec::new();
    payload.extend_from_slice(LURAPH_MAGIC_HEADER);

    // Root proto (ID 0): 2 sub-protos (Child A and Child B)
    payload.push(0x00); // constants = 0
    payload.push(0x00); // instructions = 0
    payload.push(0x00); // upvalues = 0
    payload.push(0x02); // max_stack = 2
    payload.push(0x00); // num_params = 0
    payload.push(0x00); // is_vararg = false
    payload.push(0x02); // sub_proto_count = 2

    // Child A (first child of Root): has 1 child (Child A.1)
    payload.push(0x00); // constants = 0
    payload.push(0x00); // instructions = 0
    payload.push(0x00); // upvalues = 0
    payload.push(0x02); // max_stack = 2
    payload.push(0x00); // num_params = 0
    payload.push(0x00); // is_vararg = false
    payload.push(0x01); // sub_proto_count = 1

    // Child A.1 (child of Child A): 0 children
    payload.push(0x00); // constants = 0
    payload.push(0x00); // instructions = 0
    payload.push(0x00); // upvalues = 0
    payload.push(0x02); // max_stack = 2
    payload.push(0x00); // num_params = 0
    payload.push(0x00); // is_vararg = false
    payload.push(0x00); // sub_proto_count = 0

    // Child B (second child of Root): 0 children
    payload.push(0x00); // constants = 0
    payload.push(0x00); // instructions = 0
    payload.push(0x00); // upvalues = 0
    payload.push(0x02); // max_stack = 2
    payload.push(0x00); // num_params = 0
    payload.push(0x00); // is_vararg = false
    payload.push(0x00); // sub_proto_count = 0

    let root = deserializer.deserialize(&payload).unwrap();
    assert_eq!(root.id, 0);
    assert_eq!(root.sub_protos.len(), 2);

    let child_a = &root.sub_protos[0];
    let child_b = &root.sub_protos[1];
    assert_eq!(child_a.id, 1);
    assert_eq!(child_a.sub_protos.len(), 1);

    let child_a1 = &child_a.sub_protos[0];
    assert_eq!(child_a1.id, 2);

    // With monotonic counter, Child B must receive ID 3 (NOT ID 2)
    assert_eq!(child_b.id, 3);

    // Verify flatten() contains all 4 prototypes with strictly unique IDs
    let flat = root.flatten();
    assert_eq!(flat.len(), 4);
    let ids: Vec<usize> = flat.iter().map(|p| p.id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3]);

    let mut id_set = std::collections::HashSet::new();
    for proto in &flat {
        assert!(id_set.insert(proto.id), "Duplicate prototype ID detected: {}", proto.id);
    }

    // Verify parent -> child linkage references
    assert_eq!(flat[0].child_protos, vec![1, 3]);
    assert_eq!(flat[1].child_protos, vec![2]);
    assert_eq!(flat[2].child_protos, Vec::<usize>::new());
    assert_eq!(flat[3].child_protos, Vec::<usize>::new());
}

#[test]
fn test_leb128_overflow_10th_byte_high_bit() {
    // 10-byte LEB128 stream where 10th byte has low7 = 2 (bit 64 set, exceeding u64)
    // Bytes 0..8 have continuation bit set (0x80)
    let stream = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02];
    let err = decode_u64(&stream).unwrap_err();
    assert_eq!(
        err,
        Leb128Error::Overflow {
            offset: 9,
            max_bytes: 10,
        }
    );
}

#[test]
fn test_deserializer_huge_count_no_oom() {
    let deserializer = BytecodeDeserializer::new();
    let mut payload = Vec::new();
    payload.extend_from_slice(LURAPH_MAGIC_HEADER);
    // Huge constants count (approx 2 billion in LEB128)
    payload.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0x07]);
    // Followed by only 1 byte
    payload.push(0x00);

    // Must return an error gracefully without triggering an out-of-memory abort
    let res = deserializer.deserialize(&payload);
    assert!(res.is_err());
}

#[test]
fn test_deserializer_table_recursion_depth_limit() {
    let deserializer = BytecodeDeserializer::new();
    let mut payload = Vec::new();
    payload.extend_from_slice(LURAPH_MAGIC_HEADER);
    // Root proto: 1 constant
    payload.push(0x01); // const_count = 1

    // Build 66 nested tables: tag 163, arr_len = 1
    for _ in 0..66 {
        payload.push(163); // Table tag
        payload.push(0x01); // arr_len = 1
    }
    payload.push(17); // Nil for the leaf value
    for _ in 0..66 {
        payload.push(0x00); // hash_len = 0
    }

    let err = deserializer.deserialize(&payload).unwrap_err();
    assert_eq!(
        err,
        DeserializerError::MaxRecursionDepthExceeded { max_depth: 64 }
    );
}

