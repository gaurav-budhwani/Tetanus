//! Zero-copy GGUF v2/v3 file parser.
//!
//! Reads metadata key-value pairs and tensor headers upfront, validates bounds and alignment,
//! and returns tensor byte slices directly out of the underlying `MappedFile` without copying
//! weights to the heap.
//!
//! Important GGUF layout note:
//! GGUF stores tensor dimensions in innermost-first (column-major) order: `dims[0]` is the
//! contiguous row length (`in_features`), and `dims[1]` is `out_features`. This is the
//! transpose of PyTorch/Safetensors `[out_features, in_features]` shape conventions.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::mmap::MappedFile;

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32;

/// GGML tensor storage formats supported by Tetanus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q8_0,
    Q4K,
    Q6K,
    Unsupported(u32),
}

impl GgmlType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            8 => Self::Q8_0,
            12 => Self::Q4K,
            14 => Self::Q6K,
            30 => Self::BF16,
            other => Self::Unsupported(other),
        }
    }

    /// Number of scalar weights encoded per quantized block (1 for unquantized formats).
    #[inline]
    pub const fn block_size(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::Q4_0 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q6K => 256,
            Self::Unsupported(_) => 0,
        }
    }

    /// Byte size of one encoded block.
    #[inline]
    pub const fn type_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => 18,  // 2B f16 scale + 16B packed nibbles (32 weights)
            Self::Q8_0 => 34,  // 2B f16 scale + 32B i8 quants (32 weights)
            Self::Q4K => 144,  // 2B d + 2B dmin + 12B scales + 128B qs (256 weights)
            Self::Q6K => 210,  // 128B ql + 64B qh + 16B scales + 2B d (256 weights)
            Self::Unsupported(_) => 0,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Q4_0 => "Q4_0",
            Self::Q8_0 => "Q8_0",
            Self::Q4K => "Q4_K",
            Self::Q6K => "Q6_K",
            Self::Unsupported(_) => "UNSUPPORTED",
        }
    }
}

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> {
        match *self {
            Self::U32(v) => Some(v),
            Self::I32(v) => u32::try_from(v).ok(),
            Self::U64(v) => u32::try_from(v).ok(),
            Self::I64(v) => u32::try_from(v).ok(),
            Self::U16(v) => Some(u32::from(v)),
            Self::U8(v) => Some(u32::from(v)),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Self::F32(v) => Some(v),
            Self::F64(v) => Some(v as f32),
            Self::U32(v) => Some(v as f32),
            Self::I32(v) => Some(v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            Self::Array(arr) => Some(arr.as_slice()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<usize>,
    pub dtype: GgmlType,
    /// Absolute byte offset from the start of the mapped GGUF file.
    pub file_offset: usize,
    pub byte_size: usize,
    pub n_elements: usize,
}

#[derive(Debug)]
pub enum GgufError {
    Io(std::io::Error),
    Format(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Format(msg) => write!(f, "GGUF format error: {msg}"),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

pub struct GgufFile {
    file: MappedFile,
    pub version: u32,
    pub alignment: usize,
    pub tensor_data_offset: usize,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    /// Preserves file order for `inspect` output.
    pub tensor_order: Vec<String>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = MappedFile::open(path)?;
        Self::from_mapped(file)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, GgufError> {
        Self::from_mapped(MappedFile::from_bytes(bytes))
    }

    pub fn from_mapped(file: MappedFile) -> Result<Self, GgufError> {
        let data = file.as_slice();
        let mut cursor = Cursor::new(data);

        let magic = cursor.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::Format(format!(
                "invalid magic 0x{magic:08x} (expected GGUF)"
            )));
        }

        let version = cursor.read_u32()?;
        if version != 2 && version != 3 {
            return Err(GgufError::Format(format!(
                "unsupported GGUF version {version} (expected 2 or 3)"
            )));
        }

        let tensor_count = cursor.read_usize_u64()?;
        let kv_count = cursor.read_usize_u64()?;

        let mut metadata = HashMap::with_capacity(kv_count.min(4096));
        for _ in 0..kv_count {
            let key = cursor.read_string()?;
            let val_type = cursor.read_u32()?;
            let value = read_value(&mut cursor, val_type, 0)?;
            metadata.insert(key, value);
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .map(|a| a as usize)
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::Format(format!(
                "invalid general.alignment {alignment}"
            )));
        }

        // Parse tensor headers first; data offsets are relative to `tensor_data_offset`
        // which begins immediately after the header padded up to `alignment`.
        struct RawTensor {
            name: String,
            dims: Vec<usize>,
            dtype: GgmlType,
            rel_offset: usize,
        }

        let mut raw_tensors = Vec::with_capacity(tensor_count.min(4096));
        for _ in 0..tensor_count {
            let name = cursor.read_string()?;
            let n_dims = cursor.read_u32()? as usize;
            if n_dims > 8 {
                return Err(GgufError::Format(format!(
                    "tensor '{name}' has implausible rank {n_dims}"
                )));
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(cursor.read_usize_u64()?);
            }
            let dtype = GgmlType::from_u32(cursor.read_u32()?);
            let rel_offset = cursor.read_usize_u64()?;
            raw_tensors.push(RawTensor {
                name,
                dims,
                dtype,
                rel_offset,
            });
        }

        let tensor_data_offset = align_up(cursor.pos(), alignment);
        if tensor_data_offset > data.len() {
            return Err(GgufError::Format(
                "GGUF tensor data offset exceeds file size".into(),
            ));
        }

        let mut tensors = HashMap::with_capacity(raw_tensors.len());
        let mut tensor_order = Vec::with_capacity(raw_tensors.len());
        for rt in raw_tensors {
            let n_elements = rt
                .dims
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or_else(|| GgufError::Format(format!("element count overflow for {}", rt.name)))?;

            let byte_size = if let GgmlType::Unsupported(_) = rt.dtype {
                0
            } else {
                let bs = rt.dtype.block_size();
                if n_elements % bs != 0 {
                    return Err(GgufError::Format(format!(
                        "tensor '{}' elements ({n_elements}) not divisible by block size {bs}",
                        rt.name
                    )));
                }
                (n_elements / bs)
                    .checked_mul(rt.dtype.type_size())
                    .ok_or_else(|| GgufError::Format(format!("byte size overflow for {}", rt.name)))?
            };

            let file_offset = tensor_data_offset
                .checked_add(rt.rel_offset)
                .ok_or_else(|| GgufError::Format(format!("offset overflow for {}", rt.name)))?;
            let end = file_offset
                .checked_add(byte_size)
                .ok_or_else(|| GgufError::Format(format!("end offset overflow for {}", rt.name)))?;
            if end > data.len() {
                return Err(GgufError::Format(format!(
                    "tensor '{}' [{file_offset}..{end}) exceeds file size {}",
                    rt.name,
                    data.len()
                )));
            }

            tensor_order.push(rt.name.clone());
            tensors.insert(
                rt.name.clone(),
                TensorInfo {
                    name: rt.name,
                    dims: rt.dims,
                    dtype: rt.dtype,
                    file_offset,
                    byte_size,
                    n_elements,
                },
            );
        }

        Ok(Self {
            file,
            version,
            alignment,
            tensor_data_offset,
            metadata,
            tensors,
            tensor_order,
        })
    }

    pub fn advise_willneed(&self) {
        self.file.advise_willneed();
    }

    pub fn file_size(&self) -> usize {
        self.file.len()
    }

    #[inline]
    pub fn raw_bytes(&self) -> &[u8] {
        self.file.as_slice()
    }

    pub fn arch(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key)?.as_u32()
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key)?.as_f32()
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key)?.as_str()
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.metadata.get(key)?.as_bool()
    }

    pub fn arch_u32(&self, suffix: &str) -> Option<u32> {
        let arch = self.arch()?;
        self.get_u32(&format!("{arch}.{suffix}"))
    }

    pub fn arch_f32(&self, suffix: &str) -> Option<f32> {
        let arch = self.arch()?;
        self.get_f32(&format!("{arch}.{suffix}"))
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, GgufError> {
        self.tensors
            .get(name)
            .ok_or_else(|| GgufError::Format(format!("missing required tensor '{name}'")))
    }

    /// Returns a zero-copy byte slice into the mapped file for `info`.
    #[inline]
    pub fn tensor_bytes(&self, info: &TensorInfo) -> Result<&[u8], GgufError> {
        if let GgmlType::Unsupported(code) = info.dtype {
            return Err(GgufError::Format(format!(
                "tensor '{}' uses unsupported GGML type code {code}",
                info.name
            )));
        }
        Ok(&self.file.as_slice()[info.file_offset..info.file_offset + info.byte_size])
    }
}

#[inline]
const fn align_up(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    #[inline]
    fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], GgufError> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| GgufError::Format("cursor overflow".into()))?;
        if end > self.data.len() {
            return Err(GgufError::Format("unexpected EOF in GGUF stream".into()));
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        let b = self.read_bytes(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(i64::from_le_bytes(b.try_into().unwrap()))
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        let b = self.read_bytes(8)?;
        Ok(f64::from_bits(u64::from_le_bytes(b.try_into().unwrap())))
    }

    fn read_usize_u64(&mut self) -> Result<usize, GgufError> {
        let v = self.read_u64()?;
        usize::try_from(v).map_err(|_| GgufError::Format("64-bit length exceeds usize".into()))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_usize_u64()?;
        let bytes = self.read_bytes(len)?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

fn read_value(cursor: &mut Cursor<'_>, val_type: u32, depth: usize) -> Result<GgufValue, GgufError> {
    if depth > 8 {
        return Err(GgufError::Format("nested metadata array too deep".into()));
    }
    match val_type {
        0 => Ok(GgufValue::U8(cursor.read_u8()?)),
        1 => Ok(GgufValue::I8(cursor.read_i8()?)),
        2 => Ok(GgufValue::U16(cursor.read_u16()?)),
        3 => Ok(GgufValue::I16(cursor.read_i16()?)),
        4 => Ok(GgufValue::U32(cursor.read_u32()?)),
        5 => Ok(GgufValue::I32(cursor.read_i32()?)),
        6 => Ok(GgufValue::F32(cursor.read_f32()?)),
        7 => Ok(GgufValue::Bool(cursor.read_u8()? != 0)),
        8 => Ok(GgufValue::String(cursor.read_string()?)),
        9 => {
            let elem_type = cursor.read_u32()?;
            let len = cursor.read_usize_u64()?;
            // Guard against bogus header counts before pre-allocating.
            if len > cursor.remaining() {
                return Err(GgufError::Format(format!(
                    "metadata array length {len} exceeds remaining bytes"
                )));
            }
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_value(cursor, elem_type, depth + 1)?);
            }
            Ok(GgufValue::Array(items))
        }
        10 => Ok(GgufValue::U64(cursor.read_u64()?)),
        11 => Ok(GgufValue::I64(cursor.read_i64()?)),
        12 => Ok(GgufValue::F64(cursor.read_f64()?)),
        other => Err(GgufError::Format(format!(
            "unknown metadata value type {other}"
        ))),
    }
}
