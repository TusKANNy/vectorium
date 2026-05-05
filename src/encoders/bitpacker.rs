//! Minimal bitpacker for sparse-value quantization codes.
//!
//! Specialised for nbits in `[1, 7]` and codes `< 128` (i.e. `u8`). The implementation
//! drops every safety check the caller can guarantee on its own:
//!
//! - the caller passes `nbits in [1, 7]`,
//! - the caller passes `code < (1 << nbits)`,
//! - the caller knows how many codes are stored and asks for exactly that many on read.
//!
//! Byte-aligned per logical block: callers `flush()` after each vector so vector boundaries
//! always land on a byte boundary. Wastes at most 7 bits per vector. For Splade-cocondenser
//! (~120 nnz/vec, ~60 bytes/vec at 4 bits) that overhead is well under 1 %.
//!
//! Inspired by tantivy's `bitpacker` (https://github.com/quickwit-oss/tantivy/blob/main/bitpacker/src/bitpacker.rs)
//! but trimmed to what we need: byte-flushed (not u64-flushed), no aligned/unaligned read
//! distinction, no SIMD path. Once the experiment numbers are in we can revisit the
//! kernel for speed.

/// Streaming writer that packs codes into a byte buffer.
///
/// All accumulated bits live in `buf`. As soon as we have at least one full byte,
/// it is shifted out into the output `Vec<u8>` and `bits_in_buf` decreases by 8.
/// Call [`flush`](Self::flush) at the end of each vector to commit the residual byte.
#[derive(Debug, Default)]
pub struct BitPacker {
    buf: u64,
    bits_in_buf: u32,
}

impl BitPacker {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `code` (which fits in `nbits` bits) to the bitstream. May emit one or more
    /// bytes into `out`. No bounds check; caller guarantees `code < (1 << nbits)`.
    #[inline]
    pub fn write(&mut self, code: u8, nbits: u8, out: &mut Vec<u8>) {
        debug_assert!(
            (1..=7).contains(&nbits),
            "BitPacker is specialised for nbits in [1, 7], got {nbits}"
        );
        debug_assert!(
            (code as u32) < (1u32 << nbits),
            "code {code} does not fit in {nbits} bits"
        );

        // Shift the new code into the high bits of the buffer that are not yet in use.
        self.buf |= (code as u64) << self.bits_in_buf;
        self.bits_in_buf += nbits as u32;

        // Drain whole bytes. With nbits <= 7 we drain at most one byte per call,
        // but the loop is correct for any value.
        while self.bits_in_buf >= 8 {
            out.push(self.buf as u8);
            self.buf >>= 8;
            self.bits_in_buf -= 8;
        }
    }

    /// Commit the partial byte (zero-padded) and reset state. Idempotent if already aligned.
    #[inline]
    pub fn flush(&mut self, out: &mut Vec<u8>) {
        if self.bits_in_buf > 0 {
            out.push(self.buf as u8);
            self.buf = 0;
            self.bits_in_buf = 0;
        }
    }
}

/// Streaming reader that unpacks codes from a byte buffer produced by [`BitPacker`].
///
/// The caller must know how many codes were written; calling `read` past the end
/// returns whatever residual bits remain (typically zero) — this is intentional
/// to avoid a hot-path bounds check.
#[derive(Debug)]
pub struct BitUnpacker<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    bits_in_buf: u32,
}

impl<'a> BitUnpacker<'a> {
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            buf: 0,
            bits_in_buf: 0,
        }
    }

    /// Read the next code of `nbits` bits. Caller guarantees enough bits remain.
    #[inline]
    pub fn read(&mut self, nbits: u8) -> u8 {
        debug_assert!(
            (1..=7).contains(&nbits),
            "BitUnpacker is specialised for nbits in [1, 7], got {nbits}"
        );

        // Refill the buffer until we have at least `nbits` bits ready to consume.
        // With nbits <= 7 and bits_in_buf in [0, 63] this loop runs at most once.
        while self.bits_in_buf < nbits as u32 {
            // SAFETY: caller guarantees they don't read past the end.
            let byte = unsafe { *self.data.get_unchecked(self.pos) };
            self.buf |= (byte as u64) << self.bits_in_buf;
            self.pos += 1;
            self.bits_in_buf += 8;
        }

        let mask = (1u32 << nbits) - 1;
        let val = (self.buf as u32) & mask;
        self.buf >>= nbits;
        self.bits_in_buf -= nbits as u32;
        val as u8
    }
}

/// Bytes required to hold `n_codes` codes of `nbits` bits each, padded to a byte boundary.
#[inline]
pub const fn packed_byte_len(n_codes: usize, nbits: u8) -> usize {
    let total_bits = n_codes * nbits as usize;
    total_bits.div_ceil(8)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(codes: &[u8], nbits: u8) {
        let mut buf = Vec::new();
        let mut packer = BitPacker::new();
        for &c in codes {
            packer.write(c, nbits, &mut buf);
        }
        packer.flush(&mut buf);

        assert_eq!(
            buf.len(),
            packed_byte_len(codes.len(), nbits),
            "padded byte length mismatch (nbits={nbits}, n={})",
            codes.len()
        );

        let mut unpacker = BitUnpacker::new(&buf);
        let decoded: Vec<u8> = (0..codes.len()).map(|_| unpacker.read(nbits)).collect();
        assert_eq!(decoded, codes, "roundtrip failed at nbits={nbits}");
    }

    #[test]
    fn roundtrip_all_nbits() {
        for nbits in 1..=7u8 {
            let max = (1u32 << nbits) as u8 - 1;
            let codes: Vec<u8> = (0..200).map(|i| ((i * 7) as u8) & max).collect();
            roundtrip(&codes, nbits);
        }
    }

    #[test]
    fn empty_stream_is_empty_buffer() {
        let mut buf = Vec::new();
        let mut packer = BitPacker::new();
        packer.flush(&mut buf);
        assert!(buf.is_empty());
    }

    #[test]
    fn single_code_writes_one_byte() {
        for nbits in 1..=7u8 {
            let mut buf = Vec::new();
            let mut packer = BitPacker::new();
            packer.write(1, nbits, &mut buf);
            packer.flush(&mut buf);
            assert_eq!(buf.len(), 1, "nbits={nbits}: residual byte not flushed");
            assert_eq!(buf[0] & 1, 1);
        }
    }

    #[test]
    fn boundary_crossing() {
        // 8 codes at 3 bits = 24 bits = exactly 3 bytes (no padding).
        let codes = [0u8, 1, 2, 3, 4, 5, 6, 7];
        let mut buf = Vec::new();
        let mut packer = BitPacker::new();
        for &c in &codes {
            packer.write(c, 3, &mut buf);
        }
        packer.flush(&mut buf);
        assert_eq!(buf.len(), 3);

        let mut unpacker = BitUnpacker::new(&buf);
        for &c in &codes {
            assert_eq!(unpacker.read(3), c);
        }
    }
}
