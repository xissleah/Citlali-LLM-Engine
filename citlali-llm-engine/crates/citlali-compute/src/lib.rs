// crates/citlali-compute/src/lib.rs
//! 第3层：计算层
//! 基础数学算子的实现：matmul, rmsnorm, rope, softmax, silu, gelu, swiglu, geglu
//! 内存管理：线性分配器 (Linear Arena)

pub mod arena;
pub mod kv_cache;
pub mod ops;
