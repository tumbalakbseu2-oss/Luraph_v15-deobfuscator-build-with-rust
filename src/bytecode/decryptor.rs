//! Luraph v15 constant table deserialization, metamethod fallback resolution,
//! and offline `LPH_ENCSTR` string decryption engine.

use smallvec::SmallVec;
use std::collections::HashMap;

/// Strongly typed table identifier within a TablePool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct TableId(pub usize);

/// Strongly typed identifier for a metamethod or unflattened decoder function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct MetamethodId(pub usize);

/// All supported constant tag representations in Luraph v15.
#[derive(Clone, Debug, PartialEq)]
pub enum ConstantTag {
    /// Nil value
    Nil,
    /// Boolean value (true / false)
    Boolean(bool),
    /// 64-bit IEEE-754 floating point number
    Float64(f64),
    /// Raw binary string constant
    String(Vec<u8>),
    /// Reference to a structured table constant within the TablePool
    Table(TableId),
    /// Reference to a nested child prototype ID
    Proto(usize),
    /// Metamethod or function closure reference
    Metamethod(MetamethodId),
}

impl ConstantTag {
    pub fn from_str(s: &str) -> Self {
        ConstantTag::String(s.as_bytes().to_vec())
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        ConstantTag::String(bytes.to_vec())
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, ConstantTag::Nil)
    }

    pub fn to_lossy_string(&self) -> Option<String> {
        match self {
            ConstantTag::String(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            ConstantTag::String(bytes) => Some(bytes),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConstantTag::Float64(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ConstantTag::Boolean(b) => Some(*b),
            _ => None,
        }
    }
}

/// Keys supported for indexing constant tables.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConstantKey {
    Int(i64),
    Str(String),
    Bytes(Vec<u8>),
}

impl From<i64> for ConstantKey {
    fn from(v: i64) -> Self {
        ConstantKey::Int(v)
    }
}

impl From<usize> for ConstantKey {
    fn from(v: usize) -> Self {
        ConstantKey::Int(v as i64)
    }
}

impl From<&str> for ConstantKey {
    fn from(s: &str) -> Self {
        ConstantKey::Str(s.to_string())
    }
}

impl From<String> for ConstantKey {
    fn from(s: String) -> Self {
        ConstantKey::Str(s)
    }
}

/// Structured table constant representing Luraph VM table instances.
#[derive(Clone, Debug, Default)]
pub struct ConstantTable {
    pub id: TableId,
    pub array_part: Vec<Option<ConstantTag>>,
    pub map_part: HashMap<ConstantKey, ConstantTag>,
    pub metatable: Option<TableId>,
    pub metamethod_decoder: Option<MetamethodId>,
    pub raw_operands: Option<TableId>,
}

impl ConstantTable {
    pub fn new(id: TableId) -> Self {
        Self {
            id,
            array_part: Vec::new(),
            map_part: HashMap::new(),
            metatable: None,
            metamethod_decoder: None,
            raw_operands: None,
        }
    }

    pub fn raw_get(&self, key: &ConstantKey) -> Option<&ConstantTag> {
        match key {
            ConstantKey::Int(idx) if *idx >= 1 => {
                let uidx = *idx as usize;
                if uidx < self.array_part.len() {
                    self.array_part[uidx].as_ref()
                } else {
                    self.map_part.get(key)
                }
            }
            _ => self.map_part.get(key),
        }
    }

    pub fn raw_set(&mut self, key: ConstantKey, val: ConstantTag) {
        match key {
            ConstantKey::Int(idx) if idx >= 1 && idx <= 100_000 => {
                let uidx = idx as usize;
                if uidx >= self.array_part.len() {
                    self.array_part.resize(uidx + 1, None);
                }
                self.array_part[uidx] = Some(val);
            }
            _ => {
                self.map_part.insert(key, val);
            }
        }
    }
}

/// Global table pool managing allocated tables.
#[derive(Default)]
pub struct TablePool {
    tables: Vec<ConstantTable>,
}

impl TablePool {
    pub fn new() -> Self {
        Self { tables: Vec::new() }
    }

    pub fn allocate(&mut self) -> TableId {
        let id = TableId(self.tables.len());
        self.tables.push(ConstantTable::new(id));
        id
    }

    pub fn get(&self, id: TableId) -> Option<&ConstantTable> {
        self.tables.get(id.0)
    }

    pub fn get_mut(&mut self, id: TableId) -> Option<&mut ConstantTable> {
        self.tables.get_mut(id.0)
    }
}

/// Cache key for cross-table same_operand factoring.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OperandFactKey {
    pub decoder_id: MetamethodId,
    pub operand_repr: String,
}

/// Cache mapping (decoder_id, operand_hash) -> Decoded ConstantTag.
#[derive(Default)]
pub struct MetamethodCache {
    facts: HashMap<OperandFactKey, ConstantTag>,
}

impl MetamethodCache {
    pub fn new() -> Self {
        Self {
            facts: HashMap::new(),
        }
    }

    pub fn get(&self, decoder_id: MetamethodId, operand: &str) -> Option<&ConstantTag> {
        self.facts.get(&OperandFactKey {
            decoder_id,
            operand_repr: operand.to_string(),
        })
    }

    pub fn insert(&mut self, decoder_id: MetamethodId, operand: &str, val: ConstantTag) {
        self.facts.insert(
            OperandFactKey {
                decoder_id,
                operand_repr: operand.to_string(),
            },
            val,
        );
    }
}

pub const MAX_METATABLE_DEPTH: usize = 32;

/// Resolves table index accesses with cycle protection and metamethod caching.
pub fn resolve_table_index(
    pool: &mut TablePool,
    cache: &mut MetamethodCache,
    target_table: TableId,
    key: &ConstantKey,
    decoder_fn: Option<&dyn Fn(MetamethodId, &ConstantTag) -> Option<ConstantTag>>,
) -> ConstantTag {
    let mut visited: SmallVec<[TableId; 8]> = SmallVec::new();
    resolve_table_index_internal(
        pool,
        cache,
        target_table,
        key,
        0,
        &mut visited,
        decoder_fn,
    )
}

fn resolve_table_index_internal(
    pool: &mut TablePool,
    cache: &mut MetamethodCache,
    current_table: TableId,
    key: &ConstantKey,
    depth: usize,
    visited: &mut SmallVec<[TableId; 8]>,
    decoder_fn: Option<&dyn Fn(MetamethodId, &ConstantTag) -> Option<ConstantTag>>,
) -> ConstantTag {
    if depth >= MAX_METATABLE_DEPTH || visited.contains(&current_table) {
        return ConstantTag::Nil;
    }
    visited.push(current_table);

    if let Some(table) = pool.get(current_table) {
        if let Some(val) = table.raw_get(key) {
            if !val.is_nil() {
                return val.clone();
            }
        }
    } else {
        return ConstantTag::Nil;
    }

    let mt_id = match pool.get(current_table).and_then(|t| t.metatable) {
        Some(mt) => mt,
        None => return ConstantTag::Nil,
    };

    let mt_table = match pool.get(mt_id) {
        Some(mt) => mt.clone(),
        None => return ConstantTag::Nil,
    };

    let index_key = ConstantKey::Str("__index".to_string());
    let index_hook = mt_table.raw_get(&index_key).cloned();

    match index_hook {
        Some(ConstantTag::Table(proxy_table_id)) => resolve_table_index_internal(
            pool,
            cache,
            proxy_table_id,
            key,
            depth + 1,
            visited,
            decoder_fn,
        ),
        Some(ConstantTag::Metamethod(decoder_id)) => {
            let raw_operand = {
                let table = pool.get(current_table).unwrap();
                table
                    .raw_operands
                    .and_then(|op_tbl_id| pool.get(op_tbl_id))
                    .and_then(|op_tbl| op_tbl.raw_get(key))
                    .cloned()
                    .unwrap_or(ConstantTag::Nil)
            };

            let operand_repr = format!("{:?}", raw_operand);

            if let Some(cached_val) = cache.get(decoder_id, &operand_repr) {
                let cached_clone = cached_val.clone();
                if let Some(tbl) = pool.get_mut(current_table) {
                    tbl.raw_set(key.clone(), cached_clone.clone());
                }
                return cached_clone;
            }

            if let Some(eval) = decoder_fn {
                if let Some(decoded) = eval(decoder_id, &raw_operand) {
                    cache.insert(decoder_id, &operand_repr, decoded.clone());
                    if let Some(tbl) = pool.get_mut(current_table) {
                        tbl.raw_set(key.clone(), decoded.clone());
                    }
                    return decoded;
                }
            }

            ConstantTag::Nil
        }
        _ => ConstantTag::Nil,
    }
}

/// Decrypts ciphertext using multi-byte cyclic key unrolling.
///
/// Invariants:
/// - T2_B42: Returns ciphertext as-is if key is empty (no division by zero panic).
/// - T2_B43: Preserves all null bytes without string truncation.
pub fn decrypt_xor_stream_cyclic(ciphertext: &[u8], key: &[u8]) -> Vec<u8> {
    if key.is_empty() {
        return ciphertext.to_vec();
    }
    if ciphertext.is_empty() {
        return Vec::new();
    }

    let key_len = key.len();
    let mut plaintext = Vec::with_capacity(ciphertext.len());
    for (i, &c) in ciphertext.iter().enumerate() {
        let k = key[i % key_len];
        plaintext.push(c ^ k);
    }
    plaintext
}

#[derive(Clone, Debug)]
pub struct AffinePrngConfig {
    pub seed: u64,
    pub multiplier: u64,
    pub increment: u64,
    pub modulus: u64,
    pub shift: u32,
}

impl Default for AffinePrngConfig {
    fn default() -> Self {
        Self {
            seed: 1049162819,
            multiplier: 1103515245,
            increment: 12345,
            modulus: 2147483648, // 2^31
            shift: 16,
        }
    }
}

/// Decrypts ciphertext using an offline Affine PRNG keystream.
pub fn decrypt_affine_prng_stream(ciphertext: &[u8], config: &AffinePrngConfig) -> Vec<u8> {
    let mut state = config.seed;
    let mut plaintext = Vec::with_capacity(ciphertext.len());

    for &c in ciphertext {
        state = (state
            .wrapping_mul(config.multiplier)
            .wrapping_add(config.increment))
            % config.modulus;
        let k = ((state >> config.shift) & 0xFF) as u8;
        plaintext.push(c ^ k);
    }

    plaintext
}

/// Decrypts a payload that encodes its key length and key bytes in the initial bytes.
pub fn decrypt_prefix_derived_stream(payload: &[u8]) -> Vec<u8> {
    if payload.is_empty() {
        return Vec::new();
    }
    let key_len = payload[0] as usize;
    if payload.len() < 1 + key_len {
        return payload[1..].to_vec();
    }

    let key = &payload[1..1 + key_len];
    let cipher = &payload[1 + key_len..];
    decrypt_xor_stream_cyclic(cipher, key)
}

/// Decrypts ciphertext using rolling accumulator feedback.
pub fn decrypt_rolling_xor_feedback(ciphertext: &[u8], initial_acc: u8) -> Vec<u8> {
    let mut acc = initial_acc;
    let mut plaintext = Vec::with_capacity(ciphertext.len());

    for &c in ciphertext {
        let p = c ^ acc;
        plaintext.push(p);
        acc = acc.wrapping_add(c);
    }

    plaintext
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecryptMethod {
    CyclicXor,
    AffinePrng,
    PrefixDerived,
    RollingFeedback,
}

pub struct LphEncStrDecryptor;

impl LphEncStrDecryptor {
    pub fn decrypt(
        ciphertext: &[u8],
        key: &[u8],
        method: DecryptMethod,
        prng_seed: Option<u64>,
    ) -> Vec<u8> {
        match method {
            DecryptMethod::CyclicXor => decrypt_xor_stream_cyclic(ciphertext, key),
            DecryptMethod::AffinePrng => {
                let mut config = AffinePrngConfig::default();
                if let Some(s) = prng_seed {
                    config.seed = s;
                }
                decrypt_affine_prng_stream(ciphertext, &config)
            }
            DecryptMethod::PrefixDerived => decrypt_prefix_derived_stream(ciphertext),
            DecryptMethod::RollingFeedback => {
                let init = key.first().copied().unwrap_or(0);
                decrypt_rolling_xor_feedback(ciphertext, init)
            }
        }
    }
}
