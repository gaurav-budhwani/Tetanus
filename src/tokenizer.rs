//! Pure-Rust BPE tokenizer loaded directly from GGUF metadata.
//!
//! Supports both major open-weights tokenization schemes found in GGUF files:
//!   - `"gpt2"` byte-level BPE (Qwen 2 / 2.5 / 3, Llama 3): uses `tokenizer.ggml.merges`
//!     and the GPT-2 byte-to-unicode bijection (`Ġ` -> `' '`, `Ċ` -> `'\n'`).
//!   - `"llama"` SentencePiece BPE (Llama 2, Mistral): uses `tokenizer.ggml.scores`,
//!     `▁` (U+2581) word boundaries, and `<0xXX>` byte-fallback tokens.

use std::collections::HashMap;

use crate::gguf::{GgufError, GgufFile, GgufValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    SentencePiece,
    Gpt2ByteBpe,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreTokenizerStyle {
    Llama3,
    Qwen2,
}

pub struct Tokenizer {
    kind: TokenizerKind,
    pre_style: PreTokenizerStyle,
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    scores: Vec<f32>,
    /// `(left_id, right_id) -> (merge_rank, merged_id)` for O(1) integer-ID merge lookup.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Control / special tokens (`token_type == 3`) matched greedily before BPE.
    special_tokens: Vec<(String, u32)>,
    byte_tokens: [Option<u32>; 256],
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub bos_id: Option<u32>,
    pub eos_id: u32,
    pub add_bos: bool,
    add_space_prefix: bool,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, GgufError> {
        let model_kind = gguf.get_str("tokenizer.ggml.model").unwrap_or("llama");
        let kind = match model_kind {
            "gpt2" => TokenizerKind::Gpt2ByteBpe,
            _ => TokenizerKind::SentencePiece,
        };

        let pre_name = gguf.get_str("tokenizer.ggml.pre").unwrap_or("");
        let pre_style = if pre_name.contains("qwen") || gguf.arch() == Some("qwen2") || gguf.arch() == Some("qwen3") {
            PreTokenizerStyle::Qwen2
        } else {
            PreTokenizerStyle::Llama3
        };

        let tokens_arr = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .ok_or_else(|| GgufError::Format("missing tokenizer.ggml.tokens".into()))?;

        let mut vocab = Vec::with_capacity(tokens_arr.len());
        let mut token_to_id = HashMap::with_capacity(tokens_arr.len());
        for (i, val) in tokens_arr.iter().enumerate() {
            let s = val
                .as_str()
                .ok_or_else(|| GgufError::Format("non-string entry in tokenizer.ggml.tokens".into()))?;
            vocab.push(s.to_owned());
            token_to_id.insert(s.to_owned(), i as u32);
        }

        let scores: Vec<f32> = gguf
            .metadata
            .get("tokenizer.ggml.scores")
            .and_then(GgufValue::as_array)
            .map(|arr| arr.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect())
            .unwrap_or_else(|| vec![0.0; vocab.len()]);

        // Identify special/control tokens (`token_type == 3`) for greedy pre-BPE matching.
        let mut special_tokens = Vec::new();
        if let Some(types_arr) = gguf
            .metadata
            .get("tokenizer.ggml.token_type")
            .and_then(GgufValue::as_array)
        {
            for (i, v) in types_arr.iter().enumerate() {
                if v.as_u32() == Some(3) {
                    if let Some(tok) = vocab.get(i) {
                        if !tok.is_empty() {
                            special_tokens.push((tok.clone(), i as u32));
                        }
                    }
                }
            }
        }
        // Longest match first avoids prefix collisions (e.g. `<|im_start|>` vs `<|im_`).
        special_tokens.sort_by_key(|entry| std::cmp::Reverse(entry.0.len()));

        // Map `<0x00>..=<0xFF>` byte-fallback tokens for SentencePiece models.
        let mut byte_tokens = [None; 256];
        for b in 0u8..=255 {
            let name = format!("<0x{b:02X}>");
            if let Some(&id) = token_to_id.get(&name) {
                byte_tokens[b as usize] = Some(id);
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_tables();

        // Pre-resolve GGUF merges `"left right"` into integer `(left_id, right_id) -> (rank, merged_id)`
        // so the hot BPE loop works purely on `u32` IDs with zero string allocations.
        let mut merges = HashMap::new();
        if let Some(merges_arr) = gguf
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(GgufValue::as_array)
        {
            merges.reserve(merges_arr.len());
            for (rank, val) in merges_arr.iter().enumerate() {
                let Some(line) = val.as_str() else { continue };
                let Some((left, right)) = line.split_once(' ') else { continue };
                let (Some(&lid), Some(&rid)) = (token_to_id.get(left), token_to_id.get(right)) else {
                    continue;
                };
                let merged = format!("{left}{right}");
                if let Some(&mid) = token_to_id.get(&merged) {
                    merges.insert((lid, rid), (rank as u32, mid));
                }
            }
        }

        let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id");
        let eos_id = gguf
            .get_u32("tokenizer.ggml.eos_token_id")
            .or_else(|| token_to_id.get("<|im_end|>").copied())
            .or_else(|| token_to_id.get("<|eot_id|>").copied())
            .unwrap_or(2);
        let add_bos = gguf
            .get_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(kind == TokenizerKind::SentencePiece);
        let add_space_prefix = gguf
            .get_bool("tokenizer.ggml.add_space_prefix")
            .unwrap_or(kind == TokenizerKind::SentencePiece);

        Ok(Self {
            kind,
            pre_style,
            vocab,
            token_to_id,
            scores,
            merges,
            special_tokens,
            byte_tokens,
            byte_to_char,
            char_to_byte,
            bos_id,
            eos_id,
            add_bos,
            add_space_prefix,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn kind(&self) -> TokenizerKind {
        self.kind
    }

    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str)
    }

    pub fn is_eos(&self, id: u32) -> bool {
        if id == self.eos_id {
            return true;
        }
        matches!(
            self.token_str(id),
            Some("<|im_end|>" | "<|eot_id|>" | "<|endoftext|>" | "</s>")
        )
    }

    /// Encodes `text` into token IDs, optionally prepending BOS and honoring special tokens.
    pub fn encode(&self, text: &str, prepend_bos: bool) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 4);
        if prepend_bos {
            if let Some(bos) = self.bos_id {
                ids.push(bos);
            }
        }
        if text.is_empty() {
            return ids;
        }

        // Split around special/control tokens first so tags like `<|im_start|>` stay atomic.
        let mut pos = 0;
        while pos < text.len() {
            if let Some((tok_len, tok_id)) = self.match_special(&text[pos..]) {
                ids.push(tok_id);
                pos += tok_len;
                continue;
            }
            let next_special = self.find_next_special(&text[pos..]).unwrap_or(text.len() - pos);
            let segment = &text[pos..pos + next_special];
            match self.kind {
                TokenizerKind::Gpt2ByteBpe => self.encode_gpt2_segment(segment, &mut ids),
                TokenizerKind::SentencePiece => self.encode_sp_segment(segment, &mut ids),
            }
            pos += next_special;
        }
        ids
    }

    /// Appends raw decoded bytes of `token_id` to `out`.
    pub fn decode_token_bytes(&self, token_id: u32, out: &mut Vec<u8>) {
        let Some(tok) = self.vocab.get(token_id as usize) else {
            return;
        };
        if self.special_tokens.iter().any(|(_, id)| *id == token_id) {
            out.extend_from_slice(tok.as_bytes());
            return;
        }
        match self.kind {
            TokenizerKind::Gpt2ByteBpe => {
                for ch in tok.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        out.push(b);
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            TokenizerKind::SentencePiece => {
                if tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                    if let Ok(byte) = u8::from_str_radix(&tok[3..5], 16) {
                        out.push(byte);
                        return;
                    }
                }
                for ch in tok.chars() {
                    if ch == '\u{2581}' {
                        out.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.decode_token_bytes(id, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn match_special(&self, s: &str) -> Option<(usize, u32)> {
        for (tok, id) in &self.special_tokens {
            if s.starts_with(tok.as_str()) {
                return Some((tok.len(), *id));
            }
        }
        None
    }

    fn find_next_special(&self, s: &str) -> Option<usize> {
        let mut earliest: Option<usize> = None;
        for (tok, _) in &self.special_tokens {
            if let Some(idx) = s.find(tok.as_str()) {
                earliest = Some(match earliest {
                    Some(cur) => cur.min(idx),
                    None => idx,
                });
            }
        }
        earliest
    }

    fn encode_gpt2_segment(&self, text: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = Vec::with_capacity(64);
        let mut single_char_buf = [0u8; 4];

        for chunk in pretokenize_gpt2(text, self.pre_style) {
            symbols.clear();
            for &b in chunk.as_bytes() {
                let ch = self.byte_to_char[b as usize];
                let s = ch.encode_utf8(&mut single_char_buf);
                if let Some(&id) = self.token_to_id.get(s) {
                    symbols.push(id);
                }
            }
            if symbols.len() <= 1 {
                out.extend_from_slice(&symbols);
                continue;
            }

            // Standard greedy rank-ordered BPE merge over integer token IDs.
            loop {
                let mut best_rank = u32::MAX;
                let mut best_idx = usize::MAX;
                let mut best_merged = 0u32;

                for i in 0..symbols.len() - 1 {
                    if let Some(&(rank, merged)) = self.merges.get(&(symbols[i], symbols[i + 1])) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                            best_merged = merged;
                        }
                    }
                }
                if best_idx == usize::MAX {
                    break;
                }
                symbols[best_idx] = best_merged;
                symbols.remove(best_idx + 1);
                if symbols.len() <= 1 {
                    break;
                }
            }
            out.extend_from_slice(&symbols);
        }
    }

    fn encode_sp_segment(&self, text: &str, out: &mut Vec<u32>) {
        let mut normalized = String::with_capacity(text.len() + 4);
        if self.add_space_prefix && !text.starts_with(' ') {
            normalized.push('\u{2581}');
        }
        for ch in text.chars() {
            if ch == ' ' {
                normalized.push('\u{2581}');
            } else {
                normalized.push(ch);
            }
        }

        // Seed initial symbol IDs (character -> vocab ID, or `<0xXX>` byte fallback).
        let mut symbols: Vec<u32> = Vec::with_capacity(normalized.len());
        let mut buf = [0u8; 4];
        for ch in normalized.chars() {
            let s = ch.encode_utf8(&mut buf);
            if let Some(&id) = self.token_to_id.get(s) {
                symbols.push(id);
            } else {
                for &b in s.as_bytes() {
                    if let Some(bid) = self.byte_tokens[b as usize] {
                        symbols.push(bid);
                    }
                }
            }
        }
        if symbols.len() <= 1 {
            out.extend_from_slice(&symbols);
            return;
        }

        let mut merged_buf = String::with_capacity(32);
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..symbols.len() - 1 {
                merged_buf.clear();
                merged_buf.push_str(&self.vocab[symbols[i] as usize]);
                merged_buf.push_str(&self.vocab[symbols[i + 1] as usize]);
                if let Some(&mid) = self.token_to_id.get(&merged_buf) {
                    let score = self.scores.get(mid as usize).copied().unwrap_or(0.0);
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = mid;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }
            symbols[best_idx] = best_id;
            symbols.remove(best_idx + 1);
            if symbols.len() <= 1 {
                break;
            }
        }
        out.extend_from_slice(&symbols);
    }
}

/// Streaming UTF-8 buffer so partial multi-byte characters across BPE tokens
/// wait for their remaining continuation bytes before printing.
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Default for Utf8Stream {
    fn default() -> Self {
        Self::new()
    }
}

impl Utf8Stream {
    pub fn new() -> Self {
        Self {
            pending: Vec::with_capacity(8),
        }
    }

    pub fn push_token(&mut self, tok: &Tokenizer, id: u32) -> String {
        tok.decode_token_bytes(id, &mut self.pending);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(valid) => {
                    out.push_str(valid);
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid_up_to = err.valid_up_to();
                    if valid_up_to > 0 {
                        let valid = unsafe { std::str::from_utf8_unchecked(&self.pending[..valid_up_to]) };
                        out.push_str(valid);
                        self.pending.drain(..valid_up_to);
                    }
                    if let Some(err_len) = err.error_len() {
                        out.push('\u{FFFD}');
                        self.pending.drain(..err_len);
                    } else {
                        // Incomplete UTF-8 sequence at tail; wait for next token.
                        break;
                    }
                }
            }
        }
        out
    }
}

/// Deterministic pre-tokenizer covering GPT-2 / Llama-3 / Qwen2 regex rules without a regex crate.
fn pretokenize_gpt2(text: &str, style: PreTokenizerStyle) -> Vec<&str> {
    let mut chunks = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        // 1. English contractions: `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d`
        if bytes[i] == b'\'' {
            if let Some(len) = match_contraction(&text[i..]) {
                chunks.push(&text[i..i + len]);
                i += len;
                continue;
            }
        }

        let rest = &text[i..];
        let mut chars = rest.char_indices();
        let (_, first) = chars.next().unwrap();

        // 2. Newlines (`\r\n` or `\n` runs).
        if first == '\r' || first == '\n' {
            let mut end = first.len_utf8();
            while i + end < text.len() {
                let next = text[i + end..].chars().next().unwrap();
                if next == '\r' || next == '\n' {
                    end += next.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 3. Optional single space followed by a letter run (` ?\p{L}+`).
        if first == ' ' {
            if let Some((second_off, second)) = chars.next() {
                if second.is_alphabetic() {
                    let mut end = second_off + second.len_utf8();
                    while i + end < text.len() {
                        let c = text[i + end..].chars().next().unwrap();
                        if c.is_alphabetic() {
                            end += c.len_utf8();
                        } else {
                            break;
                        }
                    }
                    chunks.push(&text[i..i + end]);
                    i += end;
                    continue;
                }
                if !second.is_whitespace() && !second.is_numeric() {
                    let mut end = second_off + second.len_utf8();
                    while i + end < text.len() {
                        let c = text[i + end..].chars().next().unwrap();
                        if !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() {
                            end += c.len_utf8();
                        } else {
                            break;
                        }
                    }
                    chunks.push(&text[i..i + end]);
                    i += end;
                    continue;
                }
            }
        }

        // 4. Letter run (`\p{L}+`).
        if first.is_alphabetic() {
            let mut end = first.len_utf8();
            while i + end < text.len() {
                let c = text[i + end..].chars().next().unwrap();
                if c.is_alphabetic() {
                    end += c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 5. Numeric digits: Qwen2 splits every single digit (`\p{N}`); Llama-3 groups 1..=3.
        if first.is_numeric() {
            let max_digits = match style {
                PreTokenizerStyle::Qwen2 => 1,
                PreTokenizerStyle::Llama3 => 3,
            };
            let mut end = first.len_utf8();
            let mut count = 1;
            while count < max_digits && i + end < text.len() {
                let c = text[i + end..].chars().next().unwrap();
                if c.is_numeric() {
                    end += c.len_utf8();
                    count += 1;
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 6. Whitespace run (excluding newlines).
        if first.is_whitespace() {
            let mut end = first.len_utf8();
            while i + end < text.len() {
                let c = text[i + end..].chars().next().unwrap();
                if c.is_whitespace() && c != '\r' && c != '\n' {
                    end += c.len_utf8();
                } else {
                    break;
                }
            }
            chunks.push(&text[i..i + end]);
            i += end;
            continue;
        }

        // 7. Punctuation / symbols run.
        let mut end = first.len_utf8();
        while i + end < text.len() {
            let c = text[i + end..].chars().next().unwrap();
            if !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() {
                end += c.len_utf8();
            } else {
                break;
            }
        }
        chunks.push(&text[i..i + end]);
        i += end;
    }

    chunks
}

fn match_contraction(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    if b.len() >= 3 {
        let two = [b[1].to_ascii_lowercase(), b[2].to_ascii_lowercase()];
        if matches!(&two, b"re" | b"ve" | b"ll") {
            return Some(3);
        }
    }
    if b.len() >= 2 {
        let one = b[1].to_ascii_lowercase();
        if matches!(one, b's' | b't' | b'm' | b'd') {
            return Some(2);
        }
    }
    None
}

/// Builds OpenAI GPT-2's reversible 256-entry byte-to-Unicode-char mapping.
fn build_gpt2_byte_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut next_extra: u32 = 256;

    for b in 0u16..=255 {
        let is_direct = matches!(b, 33..=126 | 161..=172 | 174..=255);
        let code = if is_direct {
            b as u32
        } else {
            let c = next_extra;
            next_extra += 1;
            c
        };
        let ch = char::from_u32(code).unwrap();
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b as u8);
    }

    (byte_to_char, char_to_byte)
}
