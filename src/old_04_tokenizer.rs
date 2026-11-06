//! Tokenizers: the tokenizer embedded in the GGUF file itself (like llama.cpp).

use anyhow::{anyhow, bail};
use candle_core::quantized::gguf_file;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use tokenizers::Tokenizer;

// ---- Tokenizer ------------------------------------------------------------

/// Either a Hugging Face tokenizer (built from GGUF metadata) or the
/// built-in SentencePiece parser.
pub enum Tok {
    Hf(Tokenizer),
    Spm(SpmTokenizer),
}

impl Tok {
    /// Text -> token ids (adds the BOS token when the model expects it).
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        match self {
            Tok::Hf(t) => {
                let enc = t
                    .encode(text, true)
                    .map_err(|e| anyhow!("tokenization failed: {e}"))?;
                Ok(enc.get_ids().to_vec())
            }
            Tok::Spm(t) => Ok(t.encode(text)),
        }
    }

    /// Token ids -> text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            Tok::Hf(t) => t.decode(ids, true).unwrap_or_default(),
            Tok::Spm(t) => t.decode(ids),
        }
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        match self {
            Tok::Hf(t) => t.token_to_id(token),
            Tok::Spm(t) => t.token_id(token),
        }
    }

    /// Build a tokenizer directly from the GGUF metadata (llama.cpp behaviour).
    /// Handles both SentencePiece (`llama`) and byte-level BPE (`gpt2`, Qwen)
    /// families. If candle's builder fails, falls back to the built-in
    /// SentencePiece parser for old `llama`-type GGUFs.
    pub fn load(
        _models_dir: &str,
        _model_name: &str,
        content: &gguf_file::Content,
    ) -> anyhow::Result<Tok> {
        use candle_core::quantized::tokenizer::TokenizerFromGguf;

        // The primary path: build the tokenizer from GGUF metadata.
        // PR #3245 added this for Qwen2 / Llama3 BPE / GPT-2 style models.
        match Tokenizer::from_gguf(content) {
            Ok(hf) => return Ok(Tok::Hf(hf)),
            Err(e) => {
                tracing::debug!("candle GGUF tokenizer builder failed: {e}");
            }
        }

        // Fallback: our own SentencePiece parser, for old GGUFs whose
        // metadata the candle builder does not understand yet.
        match SpmTokenizer::from_gguf(&content.metadata) {
            Ok(spm) => Ok(Tok::Spm(spm)),
            Err(embedded_err) => Err(embedded_err.context(
                "no usable tokenizer found in the GGUF metadata",
            )),
        }
    }
}

// ---- SentencePiece (built-in fallback) ------------------------------------

/// SentencePiece-style tokenizer (Llama, TinyLlama, Mistral, ...) built from the
/// `tokenizer.ggml.*` metadata of a GGUF file. It follows llama.cpp's algorithm:
/// split into characters, repeatedly merge the adjacent pair whose merged piece
/// has the highest score, and fall back to `<0xNN>` byte tokens.
pub struct SpmTokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    mergeable: HashMap<String, u32>,
    all: HashMap<String, u32>,
    byte_ids: Vec<Option<u32>>,
    byte_value: Vec<Option<u8>>,
    skip_in_decode: Vec<bool>,
    specials: Vec<(String, u32)>,
    bos: Option<u32>,
    unk: Option<u32>,
    add_bos: bool,
    add_space_prefix: bool,
}

fn value_i64(v: &gguf_file::Value) -> Option<i64> {
    use gguf_file::Value::*;
    match v {
        U8(x) => Some(*x as i64),
        I8(x) => Some(*x as i64),
        U16(x) => Some(*x as i64),
        I16(x) => Some(*x as i64),
        U32(x) => Some(*x as i64),
        I32(x) => Some(*x as i64),
        U64(x) => Some(*x as i64),
        I64(x) => Some(*x),
        _ => None,
    }
}

fn value_f32(v: &gguf_file::Value) -> Option<f32> {
    match v {
        gguf_file::Value::F32(x) => Some(*x),
        gguf_file::Value::F64(x) => Some(*x as f32),
        _ => None,
    }
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

struct Symbol {
    start: usize,
    len: usize,
    prev: isize,
    next: isize,
}

struct Bigram {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Bigram {}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl SpmTokenizer {
    pub fn from_gguf(md: &HashMap<String, gguf_file::Value>) -> anyhow::Result<Self> {
        let model = md
            .get("tokenizer.ggml.model")
            .and_then(|v| v.to_string().ok())
            .map(String::as_str)
            .unwrap_or("llama");
        if model != "llama" {
            bail!(
                "this GGUF uses tokenizer type '{model}', which is not supported by \
                 the built-in SentencePiece parser (only 'llama' is)"
            );
        }

        let tokens = md
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("GGUF has no tokenizer.ggml.tokens metadata"))?
            .to_vec()?;
        let pieces: Vec<String> = tokens
            .iter()
            .map(|v| v.to_string().map(|s| s.clone()))
            .collect::<Result<_, _>>()?;

        let scores: Vec<f32> = match md.get("tokenizer.ggml.scores").map(|v| v.to_vec()) {
            Some(Ok(arr)) => arr.iter().map(|v| value_f32(v).unwrap_or(0.0)).collect(),
            _ => vec![0.0; pieces.len()],
        };
        let kinds: Vec<i64> = match md.get("tokenizer.ggml.token_type").map(|v| v.to_vec()) {
            Some(Ok(arr)) => arr.iter().map(|v| value_i64(v).unwrap_or(1)).collect(),
            _ => vec![1; pieces.len()],
        };

        let id_of = |key: &str, default: Option<u32>| -> Option<u32> {
            match md.get(key).and_then(value_i64) {
                Some(x) if x >= 0 && (x as usize) < pieces.len() => Some(x as u32),
                Some(_) => None,
                None => default,
            }
        };
        let bos = id_of("tokenizer.ggml.bos_token_id", Some(1));
        let eos = id_of("tokenizer.ggml.eos_token_id", Some(2));
        let unk = id_of("tokenizer.ggml.unknown_token_id", Some(0));
        let add_bos = md
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);
        let add_space_prefix = md
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);

        let mut mergeable = HashMap::new();
        let mut all = HashMap::new();
        let mut byte_ids = vec![None; 256];
        let mut byte_value = vec![None; pieces.len()];
        let mut skip_in_decode = vec![false; pieces.len()];
        let mut specials = Vec::new();

        for (i, piece) in pieces.iter().enumerate() {
            let id = i as u32;
            let kind = kinds.get(i).copied().unwrap_or(1);
            all.entry(piece.clone()).or_insert(id);

            if let Some(b) = parse_byte_piece(piece) {
                byte_ids[b as usize] = Some(id);
                byte_value[i] = Some(b);
                continue;
            }
            let is_special = matches!(kind, 2 | 3 | 4)
                || Some(id) == bos
                || Some(id) == eos
                || Some(id) == unk;
            if is_special {
                skip_in_decode[i] = true;
                if !piece.is_empty() {
                    specials.push((piece.clone(), id));
                }
            } else if kind == 5 {
                skip_in_decode[i] = true;
            } else if !piece.is_empty() {
                mergeable.entry(piece.clone()).or_insert(id);
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        Ok(SpmTokenizer {
            pieces,
            scores,
            mergeable,
            all,
            byte_ids,
            byte_value,
            skip_in_decode,
            specials,
            bos,
            unk,
            add_bos,
            add_space_prefix,
        })
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.all.get(token).copied()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos {
            if let Some(bos) = self.bos {
                out.push(bos);
            }
        }

        let mut raw = String::new();
        let mut i = 0;
        while i < text.len() {
            let rest = &text[i..];
            if let Some((piece, id)) =
                self.specials.iter().find(|(p, _)| rest.starts_with(p.as_str()))
            {
                if !raw.is_empty() {
                    self.encode_raw(&raw, &mut out);
                    raw.clear();
                }
                out.push(*id);
                i += piece.len();
            } else if let Some(ch) = rest.chars().next() {
                raw.push(ch);
                i += ch.len_utf8();
            } else {
                break;
            }
        }
        if !raw.is_empty() {
            self.encode_raw(&raw, &mut out);
        }
        out
    }

    pub fn encode_raw(&self, text: &str, out: &mut Vec<u32>) {
        let mut s = String::with_capacity(text.len() + 4);
        if self.add_space_prefix {
            s.push('▁');
        }
        for ch in text.chars() {
            s.push(if ch == ' ' { '▁' } else { ch });
        }
        self.merge_pieces(&s, out);
    }

    pub fn merge_pieces(&self, s: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut offset = 0;
        for ch in s.chars() {
            let len = ch.len_utf8();
            let idx = symbols.len() as isize;
            symbols.push(Symbol {
                start: offset,
                len,
                prev: idx - 1,
                next: idx + 1,
            });
            offset += len;
        }
        if let Some(last) = symbols.last_mut() {
            last.next = -1;
        }

        let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();

        let try_add =
            |symbols: &Vec<Symbol>, heap: &mut BinaryHeap<Bigram>, left: isize, right: isize| {
                if left < 0 || right < 0 {
                    return;
                }
                let (l, r) = (left as usize, right as usize);
                let text = &s[symbols[l].start..symbols[r].start + symbols[r].len];
                if let Some(&id) = self.mergeable.get(text) {
                    heap.push(Bigram {
                        score: self.scores.get(id as usize).copied().unwrap_or(0.0),
                        left: l,
                        right: r,
                        size: text.len(),
                    });
                }
            };

        for i in 1..symbols.len() as isize {
            try_add(&symbols, &mut heap, i - 1, i);
        }

        while let Some(bg) = heap.pop() {
            let (l, r) = (bg.left, bg.right);
            if symbols[l].len == 0
                || symbols[r].len == 0
                || symbols[l].len + symbols[r].len != bg.size
            {
                continue;
            }
            symbols[l].len += symbols[r].len;
            symbols[r].len = 0;
            symbols[l].next = symbols[r].next;
            if symbols[r].next >= 0 {
                let n = symbols[r].next as usize;
                symbols[n].prev = l as isize;
            }
            let (prev, next) = (symbols[l].prev, symbols[l].next);
            try_add(&symbols, &mut heap, prev, l as isize);
            try_add(&symbols, &mut heap, l as isize, next);
        }

        let mut i: isize = if symbols.is_empty() { -1 } else { 0 };
        while i >= 0 {
            let sym = &symbols[i as usize];
            let text = &s[sym.start..sym.start + sym.len];
            if let Some(&id) = self.mergeable.get(text) {
                out.push(id);
            } else {
                for b in text.bytes() {
                    match self.byte_ids[b as usize].or(self.unk) {
                        Some(id) => out.push(id),
                        None => {}
                    }
                }
            }
            i = sym.next;
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let i = id as usize;
            if i >= self.pieces.len() || self.skip_in_decode[i] {
                continue;
            }
            if let Some(b) = self.byte_value[i] {
                bytes.push(b);
            } else {
                bytes.extend(self.pieces[i].replace('▁', " ").as_bytes());
            }
        }
        let mut text = String::from_utf8_lossy(&bytes).into_owned();
        if self.add_space_prefix && text.starts_with(' ') {
            text.remove(0);
        }
        text
    }
}