//! Tetanus: zero-dependency local Transformer inference engine in Rust.

pub mod arena;
pub mod f16;
pub mod gguf;
pub mod kernels;
pub mod kv_cache;
pub mod mmap;
pub mod model;
pub mod pool;
pub mod quant;
pub mod sampler;
pub mod tokenizer;
