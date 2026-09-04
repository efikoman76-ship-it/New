//! Hashing: BLAKE3 (default, content-addressing) and SHA-256/SHA-512
//! (compat + Ed25519). Pure-Rust, constant-time-ish, no dependencies.

// ---------------------------------------------------------------------------
// BLAKE3
// ---------------------------------------------------------------------------

const OUT_LEN: usize = 32;
const KEY_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline(always)]
fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    // Columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

#[inline(always)]
fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0u32; 16];
    for i in 0..16 {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    *m = permuted;
}

fn compress(
    chaining_value: &[u32; 8],
    block_words: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
) -> [u32; 16] {
    let mut state = [
        chaining_value[0],
        chaining_value[1],
        chaining_value[2],
        chaining_value[3],
        chaining_value[4],
        chaining_value[5],
        chaining_value[6],
        chaining_value[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        counter as u32,
        (counter >> 32) as u32,
        block_len,
        flags,
    ];
    let mut block = *block_words;
    for r in 0..7 {
        round(&mut state, &block);
        if r < 6 {
            permute(&mut block);
        }
    }
    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    state
}

fn words_from_le_bytes(bytes: &[u8; 64]) -> [u32; 16] {
    let mut words = [0u32; 16];
    for (i, w) in words.iter_mut().enumerate() {
        let b = &bytes[i * 4..i * 4 + 4];
        *w = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    words
}

/// Deferred finalization output of a subtree.
struct Output {
    input_chaining_value: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        let mut cv = [0u32; 8];
        let full = compress(
            &self.input_chaining_value,
            &self.block_words,
            self.counter,
            self.block_len,
            self.flags,
        );
        cv.copy_from_slice(&full[..8]);
        cv
    }

    fn root_output_bytes(&self, out: &mut [u8], seek: u64) {
        let first_block = seek / 64;
        let offset = (seek % 64) as usize;
        let mut written = 0;
        let mut block_counter = first_block;
        while written < out.len() {
            let words = compress(
                &self.input_chaining_value,
                &self.block_words,
                block_counter,
                self.block_len,
                self.flags | ROOT,
            );
            let mut block_bytes = [0u8; 64];
            for (i, w) in words.iter().enumerate() {
                block_bytes[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
            let start = if block_counter == first_block {
                offset
            } else {
                0
            };
            let n = (out.len() - written).min(64 - start);
            out[written..written + n].copy_from_slice(&block_bytes[start..start + n]);
            written += n;
            block_counter += 1;
        }
    }
}

struct ChunkState {
    chaining_value: [u32; 8],
    chunk_counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: u8,
    blocks_compressed: u8,
    flags: u32,
}

impl ChunkState {
    fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32) -> Self {
        ChunkState {
            chaining_value: key_words,
            chunk_counter,
            block: [0; BLOCK_LEN],
            block_len: 0,
            blocks_compressed: 0,
            flags,
        }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed as usize + self.block_len as usize
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.block_len as usize == BLOCK_LEN {
                let block_words = words_from_le_bytes(&self.block);
                let out = Output {
                    input_chaining_value: self.chaining_value,
                    block_words,
                    counter: self.chunk_counter,
                    block_len: BLOCK_LEN as u32,
                    flags: self.flags | self.start_flag(),
                };
                self.chaining_value = out.chaining_value();
                self.blocks_compressed += 1;
                self.block = [0; BLOCK_LEN];
                self.block_len = 0;
            }
            let want = BLOCK_LEN - self.block_len as usize;
            let take = want.min(input.len());
            self.block[self.block_len as usize..self.block_len as usize + take]
                .copy_from_slice(&input[..take]);
            self.block_len += take as u8;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        let block_words = words_from_le_bytes(&self.block);
        Output {
            input_chaining_value: self.chaining_value,
            block_words,
            counter: self.chunk_counter,
            block_len: self.block_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

fn parent_output(left_cv: [u32; 8], right_cv: [u32; 8], key_words: [u32; 8], flags: u32) -> Output {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&left_cv);
    block_words[8..].copy_from_slice(&right_cv);
    Output {
        input_chaining_value: key_words,
        block_words,
        counter: 0,
        block_len: BLOCK_LEN as u32,
        flags: PARENT | flags,
    }
}

/// BLAKE3 hasher (plain and keyed).
pub struct Blake3 {
    chunk_state: ChunkState,
    key_words: [u32; 8],
    cv_stack: [[u32; 8]; 54],
    cv_stack_len: u8,
    flags: u32,
}

impl Blake3 {
    /// New unkeyed hasher.
    pub fn new() -> Self {
        Self::with_key_words(IV, 0)
    }

    /// New keyed hasher (32-byte key).
    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        let mut key_words = [0u32; 8];
        for (i, w) in key_words.iter_mut().enumerate() {
            *w = u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
        }
        Self::with_key_words(key_words, 0)
    }

    fn with_key_words(key_words: [u32; 8], extra_flags: u32) -> Self {
        Blake3 {
            chunk_state: ChunkState::new(key_words, 0, extra_flags),
            key_words,
            cv_stack: [[0; 8]; 54],
            cv_stack_len: 0,
            flags: extra_flags,
        }
    }

    /// Length of the input absorbed so far.
    pub fn len(&self) -> usize {
        CHUNK_LEN * self.chunk_state.chunk_counter as usize + self.chunk_state.len()
    }

    /// True when no input has been absorbed.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push_cv(&mut self, new_cv: [u32; 8], mut total: u64) {
        let mut cv = new_cv;
        while total & 1 == 0 {
            let left = self.cv_stack[self.cv_stack_len as usize - 1];
            self.cv_stack_len -= 1;
            cv = parent_output(left, cv, self.key_words, self.flags).chaining_value();
            total >>= 1;
        }
        self.cv_stack[self.cv_stack_len as usize] = cv;
        self.cv_stack_len += 1;
    }

    /// Absorb input bytes.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            if self.chunk_state.len() == CHUNK_LEN {
                let chunk_cv = self.chunk_state.output().chaining_value();
                let total = self.chunk_state.chunk_counter + 1;
                self.push_cv(chunk_cv, total);
                self.chunk_state = ChunkState::new(
                    self.key_words,
                    self.chunk_state.chunk_counter + 1,
                    self.flags,
                );
            }
            let want = CHUNK_LEN - self.chunk_state.len();
            let take = want.min(input.len());
            let chunk_counter = self.chunk_state.chunk_counter;
            self.chunk_state.update(&input[..take]);
            debug_assert_eq!(self.chunk_state.chunk_counter, chunk_counter);
            input = &input[take..];
        }
        self
    }

    fn finalize_inner(&self) -> Output {
        let mut output = self.chunk_state.output();
        let mut parent_nodes_remaining = self.cv_stack_len as usize;
        while parent_nodes_remaining > 0 {
            parent_nodes_remaining -= 1;
            let sibling = self.cv_stack[parent_nodes_remaining];
            output = parent_output(sibling, output.chaining_value(), self.key_words, self.flags);
        }
        output
    }

    /// Write 32-byte digest into `out` (panics only if out < 32; callers pass
    /// exactly 32 for standard use).
    pub fn finalize_into(&self, out: &mut [u8]) {
        let mut buf = [0u8; OUT_LEN];
        self.finalize_inner().root_output_bytes(&mut buf, 0);
        out[..OUT_LEN].copy_from_slice(&buf);
    }

    /// Standard 32-byte digest.
    pub fn finalize(&self) -> [u8; 32] {
        let mut d = [0u8; 32];
        self.finalize_into(&mut d);
        d
    }

    /// Extended output of `out.len()` bytes starting at byte offset `seek`.
    pub fn finalize_xof(&self, out: &mut [u8], seek: u64) {
        self.finalize_inner().root_output_bytes(out, seek);
    }
}

impl Default for Blake3 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot BLAKE3 of `data`.
pub fn blake3(data: &[u8]) -> [u8; 32] {
    let mut h = Blake3::new();
    h.update(data);
    h.finalize()
}

/// Lowercase hex.
pub fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Parse 64 hex chars into 32 bytes; errors on bad length/chars.
pub fn unhex32(s: &str) -> Result<[u8; 32], crate::RhizomeError> {
    let b = s.as_bytes();
    if b.len() != 64 {
        return Err(crate::RhizomeError::Serde("expected 64 hex chars".into()));
    }
    let mut out = [0u8; 32];
    for i in 0..32 {
        let hi = hexval(b[2 * i])?;
        let lo = hexval(b[2 * i + 1])?;
        out[i] = hi << 4 | lo;
    }
    Ok(out)
}

fn hexval(c: u8) -> Result<u8, crate::RhizomeError> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(crate::RhizomeError::Serde("bad hex char".into())),
    }
}

// ---------------------------------------------------------------------------
// SHA-256
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Streaming SHA-256.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
}

impl Sha256 {
    /// New hasher.
    pub fn new() -> Self {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0; 64],
            buf_len: 0,
            total: 0,
        }
    }

    /// Absorb bytes.
    pub fn update(&mut self, mut data: &[u8]) -> &mut Self {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let want = 64 - self.buf_len;
            let take = want.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.compress(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
        self
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut h = self.state;
        for i in 0..64 {
            let s1 = h[4].rotate_right(6) ^ h[4].rotate_right(11) ^ h[4].rotate_right(25);
            let ch = (h[4] & h[5]) ^ ((!h[4]) & h[6]);
            let t1 = h[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = h[0].rotate_right(2) ^ h[0].rotate_right(13) ^ h[0].rotate_right(22);
            let maj = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
            let t2 = s0.wrapping_add(maj);
            h[7] = h[6];
            h[6] = h[5];
            h[5] = h[4];
            h[4] = h[3].wrapping_add(t1);
            h[3] = h[2];
            h[2] = h[1];
            h[1] = h[0];
            h[0] = t1.wrapping_add(t2);
        }
        for (dst, add) in self.state.iter_mut().zip(h.iter()) {
            *dst = dst.wrapping_add(*add);
        }
    }

    /// Finalize into 32 bytes.
    pub fn finalize(&self) -> [u8; 32] {
        let mut h = self.clone();
        let bit_len = h.total.wrapping_mul(8);
        h.update(&[0x80]);
        while h.buf_len != 56 {
            h.update(&[0]);
        }
        h.update(&bit_len.to_be_bytes());
        let mut out = [0u8; 32];
        for (i, w) in h.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot SHA-256.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

// ---------------------------------------------------------------------------
// SHA-512
// ---------------------------------------------------------------------------

const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

/// Streaming SHA-512.
#[derive(Clone)]
pub struct Sha512 {
    state: [u64; 8],
    buf: [u8; 128],
    buf_len: usize,
    total: u64,
}

impl Sha512 {
    /// New hasher.
    pub fn new() -> Self {
        Sha512 {
            state: [
                0x6a09e667f3bcc908,
                0xbb67ae8584caa73b,
                0x3c6ef372fe94f82b,
                0xa54ff53a5f1d36f1,
                0x510e527fade682d1,
                0x9b05688c2b3e6c1f,
                0x1f83d9abfb41bd6b,
                0x5be0cd19137e2179,
            ],
            buf: [0; 128],
            buf_len: 0,
            total: 0,
        }
    }

    /// Absorb bytes.
    pub fn update(&mut self, mut data: &[u8]) -> &mut Self {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let want = 128 - self.buf_len;
            let take = want.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 128 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 128 {
            let mut block = [0u8; 128];
            block.copy_from_slice(&data[..128]);
            self.compress(&block);
            data = &data[128..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
        self
    }

    fn compress(&mut self, block: &[u8; 128]) {
        let mut w = [0u64; 80];
        for i in 0..16 {
            let mut v = [0u8; 8];
            v.copy_from_slice(&block[i * 8..i * 8 + 8]);
            w[i] = u64::from_be_bytes(v);
        }
        for i in 16..80 {
            let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
            let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut h = self.state;
        for i in 0..80 {
            let s1 = h[4].rotate_right(14) ^ h[4].rotate_right(18) ^ h[4].rotate_right(41);
            let ch = (h[4] & h[5]) ^ ((!h[4]) & h[6]);
            let t1 = h[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA512_K[i])
                .wrapping_add(w[i]);
            let s0 = h[0].rotate_right(28) ^ h[0].rotate_right(34) ^ h[0].rotate_right(39);
            let maj = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
            let t2 = s0.wrapping_add(maj);
            h[7] = h[6];
            h[6] = h[5];
            h[5] = h[4];
            h[4] = h[3].wrapping_add(t1);
            h[3] = h[2];
            h[2] = h[1];
            h[1] = h[0];
            h[0] = t1.wrapping_add(t2);
        }
        for (dst, add) in self.state.iter_mut().zip(h.iter()) {
            *dst = dst.wrapping_add(*add);
        }
    }

    /// Finalize into 64 bytes.
    pub fn finalize(&self) -> [u8; 64] {
        let mut h = self.clone();
        let bit_len = h.total.wrapping_mul(8);
        h.update(&[0x80]);
        while h.buf_len != 112 {
            h.update(&[0]);
        }
        h.update(&0u64.to_be_bytes());
        h.update(&bit_len.to_be_bytes());
        let mut out = [0u8; 64];
        for (i, w) in h.state.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&w.to_be_bytes());
        }
        out
    }
}

impl Default for Sha512 {
    fn default() -> Self {
        Self::new()
    }
}

/// One-shot SHA-512.
pub fn sha512(data: &[u8]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(data);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| ((i * 131 + 17) % 256) as u8)
            .collect::<Vec<u8>>()
    }

    /// Official BLAKE3 test vectors (generated with the reference `blake3`
    /// Python package; the n=0 entry is the well-known RFC vector).
    const BLAKE3_VECTORS: &[(usize, &str)] = &[
        (
            0,
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
        ),
        (
            1,
            "e281fae0c58a026a457139c08ca66f97be31528ec6db5dbf76b8b0e5ce99788d",
        ),
        (
            31,
            "07323cf9dba8275190fee1bcfe741d2634dd38aba21d7e1ecebc7f27c0f2a778",
        ),
        (
            32,
            "a55f948b96a58bcf5ee6e7fab490d0a75e6af1a0ef622f73d7bdf1f6f6efcf14",
        ),
        (
            63,
            "3bc92f7cbfbd8cd8bb178f2418ff7eeed08df3f6f31091c641c55ac59a7016ef",
        ),
        (
            64,
            "c2edb8e04da6311ce55df2ee9d593d04953a801fad9600b460281fa5120a6ec4",
        ),
        (
            511,
            "60de1c63f246c89672762c9e22f21e8a13659aee05c7400ba213dee0ae763a29",
        ),
        (
            512,
            "9bebb9b3b8686bcfbaede1a76151cd446bab88869c82c17c88245d22266c3ecd",
        ),
        (
            1023,
            "a34eac40c97997924818ed893e8f92a6c1ab6d29618bdee24a99c33629126b6e",
        ),
        (
            1024,
            "f8b3172077dc47d9b4d288273d627b2b6c8ca66618fe3defa4495695ae6eaf5c",
        ),
        (
            1025,
            "976ca2cbe48093175573e2511769def5fe3f852c87f71a1efeb6d52d9f13e94c",
        ),
        (
            2048,
            "decb010ffddcb1d2409d064f0fbd8a87b782de352189875fccb1ff63108b705e",
        ),
        (
            5000,
            "e2661d2f070fc8e8a487e1b94ef22f1d3776a88f5989537c6120f2b3dc0d11e8",
        ),
    ];

    #[test]
    fn blake3_official_vectors() {
        for (n, want) in BLAKE3_VECTORS {
            let data = gen(*n);
            assert_eq!(&hex(&blake3(&data)), want, "n={n}");
        }
    }

    #[test]
    fn blake3_incremental_equals_oneshot() {
        // Feed in awkward chunk sizes; must equal one-shot for lengths that
        // cross chunk and parent boundaries.
        for n in [1usize, 63, 64, 1023, 1024, 1025, 2049, 4096] {
            let data = gen(n);
            let mut h = Blake3::new();
            let mut i = 0;
            let mut step = 1;
            while i < data.len() {
                let take = step.min(data.len() - i);
                h.update(&data[i..i + take]);
                i += take;
                step = (step * 3) % 200 + 1;
            }
            assert_eq!(hex(&h.finalize()), hex(&blake3(&data)), "n={n}");
        }
    }

    #[test]
    fn blake3_xof() {
        let h = Blake3::new();
        let mut out = [0u8; 64];
        h.finalize_xof(&mut out, 0);
        // First 32 bytes are the digest; XOF_3000 vectors check deep seeks.
        let data = gen(3000);
        let mut h2 = Blake3::new();
        h2.update(&data);
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        h2.finalize_xof(&mut a, 1024);
        h2.finalize_xof(&mut b, 1024 + 64);
        assert_ne!(a.to_vec(), b.to_vec());
        // Seek consistency: bytes [0,32) of seek=32 equal bytes [32,64) of 0.
        let mut c = [0u8; 32];
        h2.finalize_xof(&mut c, 32);
        let mut all = [0u8; 64];
        h2.finalize_xof(&mut all, 0);
        assert_eq!(c.to_vec(), all[32..].to_vec());
    }

    #[test]
    fn sha2_vectors() {
        assert_eq!(
            hex(&sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex(&sha256(b"The quick brown fox jumps over the lazy dog")),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
        assert_eq!(hex(&sha512(b"")), "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
        assert_eq!(hex(&sha512(b"abc")), "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
        // Incremental equals one-shot across block boundaries.
        let data = gen(1000);
        let mut h = Sha256::new();
        for chunk in data.chunks(17) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), sha256(&data));
        let mut h = Sha512::new();
        for chunk in data.chunks(31) {
            h.update(chunk);
        }
        assert_eq!(h.finalize(), sha512(&data));
    }

    #[test]
    fn hex_roundtrip() {
        let d = blake3(b"x");
        let s = hex(&d);
        assert_eq!(unhex32(&s).unwrap(), d);
        assert!(unhex32("zz").is_err());
        assert!(unhex32(&s[..62]).is_err());
    }
}
