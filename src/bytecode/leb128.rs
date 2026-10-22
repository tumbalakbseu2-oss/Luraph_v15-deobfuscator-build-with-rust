//! LEB128 variable-length integer decoding, value-biased and byte-biased modes,
//! and signed Zig-Zag integer conversion for Luraph v15 bytecode streams.

use thiserror::Error;

pub const LURAPH_LEB128_BIAS: i64 = 217;
pub const MAX_LEB128_BYTES_U64: usize = 10;
pub const MAX_EXACT_53BIT_INT: u64 = 1u64 << 53; // 9,007,199,254,740,992

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Leb128Error {
    #[error("Unexpected end of slice while reading LEB128 at offset {offset}")]
    UnexpectedEof { offset: usize },

    #[error("LEB128 integer overflow: exceeded maximum {max_bytes} bytes at offset {offset}")]
    Overflow { offset: usize, max_bytes: usize },

    #[error("LEB128 value {value} exceeds exact 53-bit integer limit at offset {offset}")]
    Exceeds53BitRange { offset: usize, value: u64 },

    #[error("Invalid stream count {count} at offset {offset}: requires at least {required_bytes} bytes, but only {remaining_bytes} remain")]
    ImpossibleCount {
        offset: usize,
        count: usize,
        remaining_bytes: usize,
        required_bytes: usize,
    },
}

/// Decodes an unsigned 64-bit LEB128 integer from a byte slice.
/// Returns `(value, bytes_read)`.
pub fn decode_u64(slice: &[u8]) -> Result<(u64, usize), Leb128Error> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for (i, &byte) in slice.iter().enumerate() {
        if i >= MAX_LEB128_BYTES_U64 {
            return Err(Leb128Error::Overflow {
                offset: i,
                max_bytes: MAX_LEB128_BYTES_U64,
            });
        }

        let low7 = (byte & 0x7F) as u64;

        // Overflow check for shift >= 64 (byte 9 has shift == 63; low7 > 1 overflows u64)
        if shift >= 63 && low7 > 1 {
            return Err(Leb128Error::Overflow {
                offset: i,
                max_bytes: MAX_LEB128_BYTES_U64,
            });
        }

        result |= low7 << shift;

        if (byte & 0x80) == 0 {
            return Ok((result, i + 1));
        }

        shift += 7;
    }

    Err(Leb128Error::UnexpectedEof { offset: slice.len() })
}

/// Decodes a value-biased LEB128 integer (`val - bias`).
/// Matches test fixtures `T1_F03_01` (300) and `T1_F03_02` (-100).
pub fn decode_leb128_biased(slice: &[u8], bias: i64) -> Result<(i64, usize), Leb128Error> {
    let (raw, bytes_read) = decode_u64(slice)?;
    let val = (raw as i64) - bias;
    Ok((val, bytes_read))
}

/// Decodes a byte-biased LEB128 integer where each byte is shifted: `(b - bias) % 256`.
/// Used in Luraph v15 VM bytecode stream reader `bT`.
pub fn decode_byte_biased_leb128(slice: &[u8], bias: u8) -> Result<(u64, usize), Leb128Error> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for (i, &byte) in slice.iter().enumerate() {
        if i >= MAX_LEB128_BYTES_U64 {
            return Err(Leb128Error::Overflow {
                offset: i,
                max_bytes: MAX_LEB128_BYTES_U64,
            });
        }

        let tw = byte.wrapping_sub(bias);
        let low7 = (tw & 0x7F) as u64;

        // Overflow check for shift >= 64 (byte 9 has shift == 63; low7 > 1 overflows u64)
        if shift >= 63 && low7 > 1 {
            return Err(Leb128Error::Overflow {
                offset: i,
                max_bytes: MAX_LEB128_BYTES_U64,
            });
        }

        result |= low7 << shift;

        if tw < 128 {
            return Ok((result, i + 1));
        }

        shift += 7;
    }

    Err(Leb128Error::UnexpectedEof { offset: slice.len() })
}

/// Maps an unsigned 64-bit integer to a signed 64-bit integer using Zig-Zag encoding.
/// Formula: `(n >> 1) ^ (-(n & 1))`
/// Mappings: 0 -> 0, 1 -> -1, 2 -> 1, 3 -> -2, ...
#[inline(always)]
pub fn zigzag_decode(n: u64) -> i64 {
    let half = (n >> 1) as i64;
    let sign_mask = -((n & 1) as i64);
    half ^ sign_mask
}

/// Inverse Zig-Zag encoding (maps signed i64 to unsigned u64).
#[inline(always)]
pub fn zigzag_encode(val: i64) -> u64 {
    ((val << 1) ^ (val >> 63)) as u64
}

/// Decodes a signed Zig-Zag LEB128 integer from slice.
pub fn decode_zigzag(slice: &[u8]) -> Result<(i64, usize), Leb128Error> {
    let (raw, bytes_read) = decode_u64(slice)?;
    Ok((zigzag_decode(raw), bytes_read))
}

/// Decodes a byte-biased Zig-Zag LEB128 integer (`bo` in Luraph v15).
pub fn decode_byte_biased_zigzag(slice: &[u8], bias: u8) -> Result<(i64, usize), Leb128Error> {
    let (raw, bytes_read) = decode_byte_biased_leb128(slice, bias)?;
    Ok((zigzag_decode(raw), bytes_read))
}
