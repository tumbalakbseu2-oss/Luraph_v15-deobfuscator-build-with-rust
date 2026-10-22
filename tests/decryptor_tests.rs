//! Milestone 2 Constant Decryptor, Metamethod Interception & LPH_ENCSTR Decryption Test Suite.
//!
//! Validates:
//! - T1_F02_01: Single-byte XOR stream decryption
//! - T1_F02_02: Multi-byte key unrolling
//! - T1_F02_03: Dynamic affine PRNG keystream decryption
//! - T1_F02_04: Prefix-derived key stream decryption
//! - T1_F02_05: Metamethod same_operand caching across tables
//! - T2_B41: Cyclic metatable fallback chain cycle termination (depth <= 32, returns Nil, zero stack overflow)
//! - T2_B42: Empty cipher key guard (no modulo zero panic)
//! - T2_B43: All-zeros ciphertext and embedded null byte preservation

use luraph_deobf_rs::bytecode::decryptor::*;

#[test]
fn test_single_byte_xor_stream_hello_world() {
    // Matches fixture: t1_f02_key_encstr_xor_stream.lua
    // raw_bytes = {0x14, 0x37, 0x3e, 0x3e, 0x3d, 0x72, 0x25, 0x3d, 0x20, 0x3e, 0x36}
    // key = 0x52
    // Exact plaintext = "Fello world"
    let raw_bytes = [0x14, 0x37, 0x3e, 0x3e, 0x3d, 0x72, 0x25, 0x3d, 0x20, 0x3e, 0x36];
    let key = [0x52];
    let decrypted = decrypt_xor_stream_cyclic(&raw_bytes, &key);
    assert_eq!(decrypted, b"Fello world");
}

#[test]
fn test_multibyte_key_unrolling() {
    // Matches fixture: t1_f02_key_multibyte_unroll.lua
    // key_stream = { 0xAA, 0xBB, 0xCC, 0xDD }
    // encrypted = { 0xEB, 0xDE, 0xA9, 0xB8 }
    // 0xEB ^ 0xAA = 0x41 ('A')
    // 0xDE ^ 0xBB = 0x65 ('e')
    // 0xA9 ^ 0xCC = 0x65 ('e')
    // 0xB8 ^ 0xDD = 0x65 ('e')
    let key_stream = [0xAA, 0xBB, 0xCC, 0xDD];
    let encrypted = [0xEB, 0xDE, 0xA9, 0xB8];
    let decrypted = decrypt_xor_stream_cyclic(&encrypted, &key_stream);
    assert_eq!(decrypted, [0x41, 0x65, 0x65, 0x65]);
}

#[test]
fn test_t2_b42_empty_cipher_key() {
    // T2_B42: Empty key must not divide by zero; must return ciphertext unchanged.
    let ciphertext = b"PlaintextWithoutKey";
    let empty_key: [u8; 0] = [];
    let decrypted = decrypt_xor_stream_cyclic(ciphertext, &empty_key);
    assert_eq!(decrypted, ciphertext);
}

#[test]
fn test_t2_b43_all_zeros_ciphertext() {
    // T2_B43: All-zeros ciphertext with embedded null bytes.
    let zeros = [0u8, 0, 0, 0, 0];
    let key = [0x55, 0xAA];
    let decrypted = decrypt_xor_stream_cyclic(&zeros, &key);
    assert_eq!(decrypted, [0x55, 0xAA, 0x55, 0xAA, 0x55]);

    // Decrypting with 0 key preserves all null bytes
    let zero_key = [0u8];
    let decrypted_zero = decrypt_xor_stream_cyclic(&zeros, &zero_key);
    assert_eq!(decrypted_zero, zeros);
    assert_eq!(decrypted_zero.len(), 5);
}

#[test]
fn test_affine_prng_stream() {
    // Matches fixture: t1_f02_key_affine_prng.lua
    let config = AffinePrngConfig::default();
    let ciphertext = [0x10, 0x20, 0x30, 0x40];
    let decrypted = decrypt_affine_prng_stream(&ciphertext, &config);
    assert_eq!(decrypted.len(), 4);

    // Symmetric decryption: decrypting decrypted with same stream yields ciphertext
    let roundtrip = decrypt_affine_prng_stream(&decrypted, &config);
    assert_eq!(roundtrip, ciphertext);
}

#[test]
fn test_prefix_derived_key_stream() {
    // Header format: [key_len, key..., cipher...]
    // Key: [0x52], Cipher: "Fello world" ^ 0x52
    let mut payload = vec![1, 0x52];
    let raw_bytes = [0x14, 0x37, 0x3e, 0x3e, 0x3d, 0x72, 0x25, 0x3d, 0x20, 0x3e, 0x36];
    payload.extend_from_slice(&raw_bytes);

    let decrypted = decrypt_prefix_derived_stream(&payload);
    assert_eq!(decrypted, b"Fello world");
}

#[test]
fn test_t2_b41_cyclic_metatable_fallback() {
    // T2_B41: t1.__index = t2, t2.__index = t1, setmetatable(t1, t1)
    let mut pool = TablePool::new();
    let mut cache = MetamethodCache::new();

    let t1 = pool.allocate();
    let t2 = pool.allocate();

    // t1 metatable is t1; t1.__index = t2
    pool.get_mut(t1).unwrap().metatable = Some(t1);
    pool.get_mut(t1).unwrap().raw_set(
        ConstantKey::Str("__index".to_string()),
        ConstantTag::Table(t2),
    );

    // t2 metatable is t2; t2.__index = t1
    pool.get_mut(t2).unwrap().metatable = Some(t2);
    pool.get_mut(t2).unwrap().raw_set(
        ConstantKey::Str("__index".to_string()),
        ConstantTag::Table(t1),
    );

    // Query non-existent key on t1
    let key = ConstantKey::Str("nonexistent".to_string());
    let res = resolve_table_index(&mut pool, &mut cache, t1, &key, None);

    // Cycle must be terminated safely without stack overflow, returning Nil
    assert_eq!(res, ConstantTag::Nil);
}

#[test]
fn test_metamethod_same_operand_caching() {
    // Matches fixture: t1_f02_key_metamethod_same_operand.lua
    let mut pool = TablePool::new();
    let mut cache = MetamethodCache::new();

    let decoder_id = MetamethodId(42);

    // Setup raw operand table: slot 1 -> raw 101
    let raw_ops = pool.allocate();
    pool.get_mut(raw_ops).unwrap().raw_set(
        ConstantKey::Int(1),
        ConstantTag::Float64(101.0),
    );

    // Setup table A with metatable pointing to decoder
    let tbl_a = pool.allocate();
    let mt_a = pool.allocate();
    pool.get_mut(mt_a).unwrap().raw_set(
        ConstantKey::Str("__index".to_string()),
        ConstantTag::Metamethod(decoder_id),
    );
    pool.get_mut(tbl_a).unwrap().metatable = Some(mt_a);
    pool.get_mut(tbl_a).unwrap().raw_operands = Some(raw_ops);

    // Decoder XORs raw float with 173
    let eval_count = std::cell::Cell::new(0);
    let eval_fn = |dec_id: MetamethodId, operand: &ConstantTag| -> Option<ConstantTag> {
        if dec_id == decoder_id {
            if let Some(f) = operand.as_float() {
                eval_count.set(eval_count.get() + 1);
                let val = (f as u32) ^ 173;
                return Some(ConstantTag::Float64(val as f64));
            }
        }
        None
    };

    let key = ConstantKey::Int(1);
    let val_a = resolve_table_index(&mut pool, &mut cache, tbl_a, &key, Some(&eval_fn));
    assert_eq!(val_a, ConstantTag::Float64((101 ^ 173) as f64));
    assert_eq!(eval_count.get(), 1);

    // Setup table B with the SAME decoder and raw operands
    let tbl_b = pool.allocate();
    let mt_b = pool.allocate();
    pool.get_mut(mt_b).unwrap().raw_set(
        ConstantKey::Str("__index".to_string()),
        ConstantTag::Metamethod(decoder_id),
    );
    pool.get_mut(tbl_b).unwrap().metatable = Some(mt_b);
    pool.get_mut(tbl_b).unwrap().raw_operands = Some(raw_ops);

    // Lookup on Table B must hit cache without invoking eval_fn again!
    let val_b = resolve_table_index(&mut pool, &mut cache, tbl_b, &key, Some(&eval_fn));
    assert_eq!(val_b, ConstantTag::Float64((101 ^ 173) as f64));
    assert_eq!(eval_count.get(), 1); // Eval count remains 1 due to same_operand cache hit!
}

#[test]
fn test_prefix_derived_key_stream_empty_key() {
    // Verifies payload with key_len = 0 and ciphertext "AB" [65, 66]
    // Does not panic with slice range error (1..=0), returns unencrypted ciphertext.
    let payload = [0u8, 65, 66];
    let decrypted = decrypt_prefix_derived_stream(&payload);
    assert_eq!(decrypted, b"AB");

    // Single zero-byte payload returns empty vector
    let empty_payload = [0u8];
    assert_eq!(decrypt_prefix_derived_stream(&empty_payload), Vec::<u8>::new());
}

