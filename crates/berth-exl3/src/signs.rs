use crate::{Error, Result};

/// fp16 bit pattern of +1.0. A clear sign bit unpacks to this.
pub const FP16_ONE: u16 = 0x3C00;

/// fp16 bit pattern of −1.0. A set sign bit unpacks to this.
pub const FP16_NEG_ONE: u16 = 0xBC00;

/// Unpack packed `su` / `sv` sign words into fp16 ±1 bit patterns.
///
/// `packed_le` is the raw little-endian `I16` bytes of one tensor. Each word
/// holds 16 signs. Bit 0 (the least significant bit) is the first sign, matching
/// `pack_signs_kernel`: the kernel shifts the fp16 sign bit in from the top, so
/// element `i` lands in bit `i`. A set bit is −1 and a clear bit is +1, matching
/// `LinearEXL3.unpack_bf` (`1 - bit * 2`).
///
/// # Examples
///
/// ```
/// use berth_exl3::{unpack_signs, FP16_NEG_ONE, FP16_ONE};
/// let signs = unpack_signs(&1u16.to_le_bytes()).unwrap();
/// assert_eq!(signs[0], FP16_NEG_ONE);
/// assert_eq!(signs[1], FP16_ONE);
/// ```
pub fn unpack_signs(packed_le: &[u8]) -> Result<Vec<u16>> {
    if !packed_le.len().is_multiple_of(2) {
        return Err(Error::bad(
            "su",
            format!(
                "packed sign bytes {} are not a multiple of 2",
                packed_le.len()
            ),
        ));
    }
    let words = packed_le.len() / 2;
    let mut out = Vec::with_capacity(words * 16);
    for chunk in packed_le.chunks_exact(2) {
        let word = u16::from_le_bytes([chunk[0], chunk[1]]);
        for bit in 0..16 {
            let negative = (word >> bit) & 1 == 1;
            out.push(if negative { FP16_NEG_ONE } else { FP16_ONE });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit0_is_negative_one() {
        let signs = unpack_signs(&1u16.to_le_bytes()).unwrap();
        assert_eq!(signs.len(), 16);
        assert_eq!(signs[0], FP16_NEG_ONE);
        assert!(signs[1..].iter().all(|&b| b == FP16_ONE));
    }

    #[test]
    fn bit15_is_the_last_sign() {
        let signs = unpack_signs(&0x8000u16.to_le_bytes()).unwrap();
        assert!(signs[..15].iter().all(|&b| b == FP16_ONE));
        assert_eq!(signs[15], FP16_NEG_ONE);
    }

    #[test]
    fn all_clear_and_all_set() {
        let pos = unpack_signs(&0u16.to_le_bytes()).unwrap();
        assert!(pos.iter().all(|&b| b == FP16_ONE));
        let neg = unpack_signs(&0xffffu16.to_le_bytes()).unwrap();
        assert!(neg.iter().all(|&b| b == FP16_NEG_ONE));
    }

    #[test]
    fn two_words_are_contiguous() {
        let mut bytes = 0x0005u16.to_le_bytes().to_vec();
        bytes.extend_from_slice(&0x0001u16.to_le_bytes());
        let signs = unpack_signs(&bytes).unwrap();
        assert_eq!(signs.len(), 32);
        assert_eq!(signs[0], FP16_NEG_ONE);
        assert_eq!(signs[1], FP16_ONE);
        assert_eq!(signs[2], FP16_NEG_ONE);
        assert_eq!(signs[16], FP16_NEG_ONE);
        assert_eq!(signs[17], FP16_ONE);
    }

    #[test]
    fn odd_length_is_an_error() {
        let err = unpack_signs(&[0x01]).unwrap_err();
        assert!(err.to_string().contains("su"), "{err}");
    }
}
