//! Milestone 2 Empirical Adversarial Stress-Test Suite.
//!
//! Validates:
//! - Section 1: Empty key prefix string decryptor (`T2_B42`) and adversarial variants.
//! - Section 2: 10th-byte LEB128 overflow boundaries (low7 == 0, 1, 2, 0x80, byte-biased).
//! - Section 3: Complex branching prototype hierarchy monotonicity and zero ID collisions.
//! - Section 4: Cyclic metatables and deeply nested table constants recursion depth bounds.

use luraph_deobf_rs::bytecode::decryptor::*;
use luraph_deobf_rs::bytecode::deserializer::{
    BytecodeDeserializer, DeserializerError, LURAPH_MAGIC_HEADER, MAX_PROTO_RECURSION_DEPTH,
};
use luraph_deobf_rs::bytecode::leb128::{
    decode_byte_biased_leb128, decode_u64, Leb128Error,
};
use std::collections::HashSet;

// ============================================================================
// SECTION 1: EMPTY KEY PREFIX STRING DECRYPTOR (T2_B42 & ADVERSARIAL MUTATIONS)
// ============================================================================

#[test]
fn test_adv_decrypt_prefix_empty_key_exact_spec() {
    // Exact dispatch condition: &[0u8, 65, 66] -> b"AB"
    let payload = [0u8, 65, 66];
    let result = decrypt_prefix_derived_stream(&payload);
    assert_eq!(result, b"AB");
}

#[test]
fn test_adv_decrypt_prefix_boundary_conditions() {
    // 1. Empty buffer
    assert_eq!(decrypt_prefix_derived_stream(&[]), Vec::<u8>::new());

    // 2. Single zero-byte buffer (key_len = 0, no ciphertext)
    assert_eq!(decrypt_prefix_derived_stream(&[0u8]), Vec::<u8>::new());

    // 3. Zero-byte key with multiple null bytes
    let null_payload = [0u8, 0, 0, 0];
    assert_eq!(decrypt_prefix_derived_stream(&null_payload), vec![0, 0, 0]);

    // 4. Truncated key payload: key_len says 1, but buffer len is 1
    // (payload.len() < 1 + key_len -> returns payload[1..] = [])
    assert_eq!(decrypt_prefix_derived_stream(&[1u8]), Vec::<u8>::new());

    // 5. Truncated key payload: key_len says 5, buffer len is 3 -> returns payload[1..]
    let truncated_short = [5u8, 10, 20];
    assert_eq!(decrypt_prefix_derived_stream(&truncated_short), vec![10, 20]);

    // 6. Truncated key payload: key_len says 255, buffer len is 2 -> returns payload[1..]
    let truncated_max = [255u8, 42];
    assert_eq!(decrypt_prefix_derived_stream(&truncated_max), vec![42]);

    // 7. Exactly key_len bytes, zero ciphertext bytes
    let key_only = [2u8, 0xAA, 0xBB];
    assert_eq!(decrypt_prefix_derived_stream(&key_only), Vec::<u8>::new());

    // 8. Normal decryption with 2-byte cyclic key
    // Key: [0x11, 0x22], Ciphertext: [0x41 ^ 0x11, 0x42 ^ 0x22, 0x43 ^ 0x11]
    let payload_norm = [2u8, 0x11, 0x22, 0x41 ^ 0x11, 0x42 ^ 0x22, 0x43 ^ 0x11];
    assert_eq!(decrypt_prefix_derived_stream(&payload_norm), b"ABC");
}

// ============================================================================
// SECTION 2: 10TH-BYTE LEB128 OVERFLOW & BOUNDARY TESTS
// ============================================================================

#[test]
fn test_adv_leb128_10th_byte_boundary_low7_zero_valid() {
    // 9 bytes with MSB set (0xFF) + 10th byte with low7 = 0 (0x00)
    // Value = (1 << 63) - 1 = 0x7FFF_FFFF_FFFF_FFFF
    let mut bytes = vec![0xFF; 9];
    bytes.push(0x00);

    let (val, read) = decode_u64(&bytes).expect("low7 == 0 on 10th byte must be valid");
    assert_eq!(read, 10);
    assert_eq!(val, 0x7FFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_adv_leb128_10th_byte_boundary_low7_one_valid_max_u64() {
    // 9 bytes with MSB set (0xFF) + 10th byte with low7 = 1 (0x01)
    // Value = u64::MAX
    let mut bytes = vec![0xFF; 9];
    bytes.push(0x01);

    let (val, read) = decode_u64(&bytes).expect("low7 == 1 on 10th byte must parse u64::MAX");
    assert_eq!(read, 10);
    assert_eq!(val, u64::MAX);
}

#[test]
fn test_adv_leb128_10th_byte_overflow_low7_two_rejects() {
    // 9 bytes with MSB set (0xFF) + 10th byte with low7 = 2 (0x02)
    // Bit 64 would be set -> MUST return Leb128Error::Overflow
    let mut bytes = vec![0xFF; 9];
    bytes.push(0x02);

    let err = decode_u64(&bytes).expect_err("low7 == 2 on 10th byte must overflow");
    assert_eq!(
        err,
        Leb128Error::Overflow {
            offset: 9,
            max_bytes: 10
        }
    );
}

#[test]
fn test_adv_leb128_10th_byte_overflow_low7_various_invalid() {
    // Test low7 values from 2 to 127 on 10th byte
    for low7 in [2u8, 3, 4, 15, 63, 64, 126, 127] {
        let mut bytes = vec![0x80; 9];
        bytes.push(low7);

        let err = decode_u64(&bytes).expect_err(&format!("low7 == {} on 10th byte must overflow", low7));
        assert_eq!(
            err,
            Leb128Error::Overflow {
                offset: 9,
                max_bytes: 10
            }
        );
    }
}

#[test]
fn test_adv_leb128_10th_byte_with_continuation_bit_exceeded() {
    // 10th byte with low7 > 1 AND continuation bit set (e.g. 0x82)
    let mut bytes = vec![0x80; 9];
    bytes.push(0x82);

    let err = decode_u64(&bytes).expect_err("10th byte with 0x82 must overflow");
    assert_eq!(
        err,
        Leb128Error::Overflow {
            offset: 9,
            max_bytes: 10
        }
    );

    // 11-byte stream with all 0x80 bytes
    let stream_11 = vec![0x80; 11];
    let err_11 = decode_u64(&stream_11).expect_err("11 bytes must overflow");
    assert_eq!(
        err_11,
        Leb128Error::Overflow {
            offset: 10,
            max_bytes: 10
        }
    );

    // 10 bytes with 0x80 + 11th byte with 0x01
    let mut stream_10_cont = vec![0x80; 10];
    stream_10_cont.push(0x01);
    let err_10_cont = decode_u64(&stream_10_cont).expect_err("11 bytes must overflow");
    assert_eq!(
        err_10_cont,
        Leb128Error::Overflow {
            offset: 10,
            max_bytes: 10
        }
    );
}

#[test]
fn test_adv_byte_biased_leb128_10th_byte_overflow() {
    let bias: u8 = 217;

    // Byte 0..8: tw = 0x80 -> byte = (128 + 217) % 256 = 89
    let byte_cont = (128u16 + bias as u16) as u8;
    let stream = vec![byte_cont; 9];

    // 10th byte where tw = 1 -> byte = (1 + 217) % 256 = 218
    let byte_valid_1 = (1u16 + bias as u16) as u8;
    let mut stream_valid = stream.clone();
    stream_valid.push(byte_valid_1);
    let (val, read) = decode_byte_biased_leb128(&stream_valid, bias)
        .expect("Valid 10th byte in byte_biased_leb128");
    assert_eq!(read, 10);
    assert_eq!(val, 1 << 63);

    // 10th byte where tw = 2 -> byte = (2 + 217) % 256 = 219
    let byte_overflow_2 = (2u16 + bias as u16) as u8;
    let mut stream_overflow = stream.clone();
    stream_overflow.push(byte_overflow_2);
    let err = decode_byte_biased_leb128(&stream_overflow, bias)
        .expect_err("tw == 2 on 10th byte in byte_biased_leb128 must overflow");
    assert_eq!(
        err,
        Leb128Error::Overflow {
            offset: 9,
            max_bytes: 10
        }
    );
}

// ============================================================================
// SECTION 3: BRANCHING PROTOTYPE HIERARCHY MONOTONICITY & ZERO ID COLLISIONS
// ============================================================================

#[test]
fn test_adv_branching_prototype_hierarchy_deep_and_wide() {
    // Tree topology:
    // Root (ID 0)
    // ├── Child A (ID 1)
    // │   ├── Child A1 (ID 2)
    // │   │   ├── Child A1_a (ID 3)
    // │   │   └── Child A1_b (ID 4)
    // │   └── Child A2 (ID 5)
    // ├── Child B (ID 6)
    // │   ├── Child B1 (ID 7)
    // │   └── Child B2 (ID 8)
    // │       └── Child B2_a (ID 9)
    // └── Child C (ID 10)
    // Total 11 prototypes. All must have unique monotonic IDs [0..=10].

    let deserializer = BytecodeDeserializer::new();
    let mut payload = Vec::new();
    payload.extend_from_slice(LURAPH_MAGIC_HEADER);

    fn write_proto_header(buf: &mut Vec<u8>, sub_proto_count: u8) {
        buf.push(0x00); // constants = 0
        buf.push(0x00); // instructions = 0
        buf.push(0x00); // upvalues = 0
        buf.push(0x02); // max_stack = 2
        buf.push(0x00); // num_params = 0
        buf.push(0x00); // is_vararg = false
        buf.push(sub_proto_count);
    }

    // Root: 3 children (A, B, C)
    write_proto_header(&mut payload, 3);

    // Child A: 2 children (A1, A2)
    write_proto_header(&mut payload, 2);
    // Child A1: 2 children (A1_a, A1_b)
    write_proto_header(&mut payload, 2);
    // Child A1_a: 0 children
    write_proto_header(&mut payload, 0);
    // Child A1_b: 0 children
    write_proto_header(&mut payload, 0);
    // Child A2: 0 children
    write_proto_header(&mut payload, 0);

    // Child B: 2 children (B1, B2)
    write_proto_header(&mut payload, 2);
    // Child B1: 0 children
    write_proto_header(&mut payload, 0);
    // Child B2: 1 child (B2_a)
    write_proto_header(&mut payload, 1);
    // Child B2_a: 0 children
    write_proto_header(&mut payload, 0);

    // Child C: 0 children
    write_proto_header(&mut payload, 0);

    let root = deserializer.deserialize(&payload).expect("Deserialization of complex hierarchy must succeed");
    assert_eq!(root.id, 0);
    assert_eq!(root.sub_protos.len(), 3);

    let child_a = &root.sub_protos[0];
    assert_eq!(child_a.id, 1);
    assert_eq!(child_a.sub_protos.len(), 2);

    let child_a1 = &child_a.sub_protos[0];
    assert_eq!(child_a1.id, 2);
    assert_eq!(child_a1.sub_protos.len(), 2);

    let child_a1_a = &child_a1.sub_protos[0];
    assert_eq!(child_a1_a.id, 3);
    assert_eq!(child_a1_a.sub_protos.len(), 0);

    let child_a1_b = &child_a1.sub_protos[1];
    assert_eq!(child_a1_b.id, 4);
    assert_eq!(child_a1_b.sub_protos.len(), 0);

    let child_a2 = &child_a.sub_protos[1];
    assert_eq!(child_a2.id, 5);
    assert_eq!(child_a2.sub_protos.len(), 0);

    let child_b = &root.sub_protos[1];
    assert_eq!(child_b.id, 6);
    assert_eq!(child_b.sub_protos.len(), 2);

    let child_b1 = &child_b.sub_protos[0];
    assert_eq!(child_b1.id, 7);
    assert_eq!(child_b1.sub_protos.len(), 0);

    let child_b2 = &child_b.sub_protos[1];
    assert_eq!(child_b2.id, 8);
    assert_eq!(child_b2.sub_protos.len(), 1);

    let child_b2_a = &child_b2.sub_protos[0];
    assert_eq!(child_b2_a.id, 9);
    assert_eq!(child_b2_a.sub_protos.len(), 0);

    let child_c = &root.sub_protos[2];
    assert_eq!(child_c.id, 10);
    assert_eq!(child_c.sub_protos.len(), 0);

    // Verify flatten() contains all 11 prototypes with strictly unique IDs
    let flat = root.flatten();
    assert_eq!(flat.len(), 11);

    let ids: Vec<usize> = flat.iter().map(|p| p.id).collect();
    assert_eq!(ids, (0..11).collect::<Vec<usize>>());

    let mut set = HashSet::new();
    for proto in &flat {
        assert!(set.insert(proto.id), "Duplicate prototype ID detected: {}", proto.id);
    }
    assert_eq!(set.len(), 11);

    // Verify parent -> child ID linkage vectors
    assert_eq!(flat[0].child_protos, vec![1, 6, 10]); // Root -> A, B, C
    assert_eq!(flat[1].child_protos, vec![2, 5]);       // A -> A1, A2
    assert_eq!(flat[2].child_protos, vec![3, 4]);       // A1 -> A1_a, A1_b
    assert_eq!(flat[3].child_protos, Vec::<usize>::new());
    assert_eq!(flat[4].child_protos, Vec::<usize>::new());
    assert_eq!(flat[5].child_protos, Vec::<usize>::new());
    assert_eq!(flat[6].child_protos, vec![7, 8]);       // B -> B1, B2
    assert_eq!(flat[7].child_protos, Vec::<usize>::new());
    assert_eq!(flat[8].child_protos, vec![9]);          // B2 -> B2_a
    assert_eq!(flat[9].child_protos, Vec::<usize>::new());
    assert_eq!(flat[10].child_protos, Vec::<usize>::new());
}

// ============================================================================
// SECTION 4: CYCLIC METATABLES & RECURSION DEPTH BOUNDS
// ============================================================================

#[test]
fn test_adv_metatable_3_node_cycle() {
    // 3-node cycle: t1.__index = t2, t2.__index = t3, t3.__index = t1
    let mut pool = TablePool::new();
    let mut cache = MetamethodCache::new();

    let t1 = pool.allocate();
    let t2 = pool.allocate();
    let t3 = pool.allocate();

    pool.get_mut(t1).unwrap().metatable = Some(t1);
    pool.get_mut(t1).unwrap().raw_set(
        ConstantKey::Str("__index".to_string()),
        ConstantTag::Table(t2),
    );

    pool.get_mut(t2).unwrap().metatable = Some(t2);
    pool.get_mut(t2).unwrap().raw_set(
        ConstantKey::Str("__index".to_string()),
        ConstantTag::Table(t3),
    );

    pool.get_mut(t3).unwrap().metatable = Some(t3);
    pool.get_mut(t3).unwrap().raw_set(
        ConstantKey::Str("__index".to_string()),
        ConstantTag::Table(t1),
    );

    let key = ConstantKey::Str("nonexistent_cycle_key".to_string());
    let res = resolve_table_index(&mut pool, &mut cache, t1, &key, None);

    assert_eq!(res, ConstantTag::Nil, "3-node metatable cycle must terminate with Nil");
}

#[test]
fn test_adv_metatable_linear_deep_chain_depth_32_cutoff() {
    // Linear chain of 40 tables: t0 -> t1 -> t2 -> ... -> t39
    // Must stop cleanly at MAX_METATABLE_DEPTH (32) without stack overflow
    let mut pool = TablePool::new();
    let mut cache = MetamethodCache::new();

    let count = 40;
    let tables: Vec<TableId> = (0..count).map(|_| pool.allocate()).collect();

    for i in 0..count - 1 {
        let curr = tables[i];
        let next = tables[i + 1];
        pool.get_mut(curr).unwrap().metatable = Some(curr);
        pool.get_mut(curr).unwrap().raw_set(
            ConstantKey::Str("__index".to_string()),
            ConstantTag::Table(next),
        );
    }

    // Place target value at table 35 (beyond depth 32)
    let key = ConstantKey::Str("deep_target".to_string());
    pool.get_mut(tables[35]).unwrap().raw_set(key.clone(), ConstantTag::Float64(999.0));

    let res = resolve_table_index(&mut pool, &mut cache, tables[0], &key, None);
    assert_eq!(res, ConstantTag::Nil, "Lookup beyond MAX_METATABLE_DEPTH must terminate with Nil");
}

#[test]
fn test_adv_deserializer_table_constant_max_recursion_depth_boundary() {
    let deserializer = BytecodeDeserializer::new();

    // In deserialize_prototype(root at depth 0):
    // Constant 0 is deserialized at depth 1.
    // 63 nested tables:
    // Table 1 at depth 1, Table 2 at depth 2, ..., Table 63 at depth 63.
    // Inside Table 63, the leaf Nil constant is deserialized at depth 64.
    // Since depth 64 <= MAX_PROTO_RECURSION_DEPTH (64), this MUST SUCCEED.
    let mut payload_depth_64 = Vec::new();
    payload_depth_64.extend_from_slice(LURAPH_MAGIC_HEADER);
    payload_depth_64.push(0x01); // const_count = 1
    for _ in 0..63 {
        payload_depth_64.push(163); // Table tag
        payload_depth_64.push(0x01); // arr_len = 1
    }
    payload_depth_64.push(17); // Nil leaf at depth 64
    for _ in 0..63 {
        payload_depth_64.push(0x00); // hash_len = 0
    }
    // Instructions = 0, upvalues = 0, stack = 2, params = 0, vararg = 0, subprotos = 0
    payload_depth_64.extend_from_slice(&[0, 0, 2, 0, 0, 0]);

    let res_64 = deserializer.deserialize(&payload_depth_64);
    assert!(res_64.is_ok(), "Table constant with leaf at max depth 64 must succeed: {:?}", res_64.err());

    // 64 nested tables:
    // Table 1 at depth 1, ..., Table 64 at depth 64.
    // Inside Table 64, the leaf Nil constant is deserialized at depth 65.
    // Since depth 65 > MAX_PROTO_RECURSION_DEPTH (64), this MUST FAIL with MaxRecursionDepthExceeded.
    let mut payload_depth_65 = Vec::new();
    payload_depth_65.extend_from_slice(LURAPH_MAGIC_HEADER);
    payload_depth_65.push(0x01); // const_count = 1
    for _ in 0..64 {
        payload_depth_65.push(163); // Table tag
        payload_depth_65.push(0x01); // arr_len = 1
    }
    payload_depth_65.push(17); // Nil leaf at depth 65
    for _ in 0..64 {
        payload_depth_65.push(0x00); // hash_len = 0
    }
    payload_depth_65.extend_from_slice(&[0, 0, 2, 0, 0, 0]);

    let err_65 = deserializer.deserialize(&payload_depth_65).expect_err("Depth 65 must exceed max depth");
    assert_eq!(
        err_65,
        DeserializerError::MaxRecursionDepthExceeded {
            max_depth: MAX_PROTO_RECURSION_DEPTH
        }
    );
}

#[test]
fn test_adv_deserializer_proto_hierarchy_max_recursion_depth_boundary() {
    let deserializer = BytecodeDeserializer::new();

    // Nested prototype hierarchy of 65 levels:
    // Root (depth 0) -> child (depth 1) -> ... -> child (depth 65)
    let mut payload = Vec::new();
    payload.extend_from_slice(LURAPH_MAGIC_HEADER);

    for _ in 0..65 {
        payload.push(0x00); // constants = 0
        payload.push(0x00); // instructions = 0
        payload.push(0x00); // upvalues = 0
        payload.push(0x02); // max_stack = 2
        payload.push(0x00); // num_params = 0
        payload.push(0x00); // is_vararg = false
        payload.push(0x01); // sub_proto_count = 1
    }

    let err = deserializer.deserialize(&payload).expect_err("Proto depth 65 must exceed max depth");
    assert_eq!(
        err,
        DeserializerError::MaxRecursionDepthExceeded {
            max_depth: MAX_PROTO_RECURSION_DEPTH
        }
    );
}
