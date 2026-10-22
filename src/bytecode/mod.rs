//! Luraph v15 bytecode deserializer, biased LEB128, opcode unhashing,
//! constant table metamethod interception, and offline string decryption.

pub mod decryptor;
pub mod deserializer;
pub mod leb128;
pub mod prototype;
pub mod unhashing;

pub use decryptor::{
    decrypt_affine_prng_stream, decrypt_prefix_derived_stream, decrypt_rolling_xor_feedback,
    decrypt_xor_stream_cyclic, resolve_table_index, AffinePrngConfig, ConstantKey, ConstantTable,
    ConstantTag, DecryptMethod, LphEncStrDecryptor, MetamethodCache, MetamethodId, TableId,
    TablePool, MAX_METATABLE_DEPTH,
};
pub use deserializer::{
    BytecodeDeserializer, BytecodeReader, DeserializerError, LURAPH_MAGIC_HEADER,
    MAX_PROTO_RECURSION_DEPTH,
};
pub use leb128::{
    decode_byte_biased_leb128, decode_byte_biased_zigzag, decode_leb128_biased, decode_u64,
    decode_zigzag, zigzag_decode, zigzag_encode, Leb128Error, LURAPH_LEB128_BIAS,
    MAX_EXACT_53BIT_INT, MAX_LEB128_BYTES_U64,
};
pub use prototype::{
    ConstantValue, Instruction, OperandKind, ProtoBytecode, Prototype, UpvalueDesc,
};
pub use unhashing::{
    compute_poly_delta, compute_tz, unhash_opcode_raw, unhash_opcode_with_custom_tz,
    unhash_opcode_wrapped, OpcodeMap, POLY_MOD, TA,
};
