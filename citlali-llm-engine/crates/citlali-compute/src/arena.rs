// crates/citlali-compute/src/arena.rs
//! 线性分配器 (Linear Arena)
//! 算子暂存区：每层 Transformer 计算完立刻 reset 复用
//!
//! 设计原则：
//! - 一次性预分配固定大小的连续内存
//! - 分配只需移动偏移指针，O(1)
//! - 不支持单独释放，只能整体 reset
//! - 适用于生命周期一致的临时缓冲区
//!
//! alloc_slice 接受 &self（通过 Cell 实现内部可变性），
//! 使得多个分配的切片可以同时存活，不触发借用冲突。

use std::alloc::{self, Layout};
use std::cell::Cell;
use std::ptr::NonNull;

/// 线性分配器
/// 内部维护一块连续的 u8 缓冲区和一个偏移指针
pub struct Arena {
    /// 底层内存块起始地址
    ptr: NonNull<u8>,
    /// 总容量（字节）
    capacity: usize,
    /// 当前已分配偏移（字节）— 使用 Cell 实现内部可变性
    offset: Cell<usize>,
    /// 本轮分配的峰值（用于调优）
    peak: Cell<usize>,
}

// Safety: Arena 内部内存由自身独占管理，不跨线程共享
unsafe impl Send for Arena {}

impl Arena {
    /// 创建一个指定容量（字节）的 Arena
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Arena capacity must be > 0");
        let layout = Layout::from_size_align(capacity, 64).expect("invalid layout");
        let ptr = unsafe { alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).expect("Arena allocation failed (OOM)");
        Self {
            ptr,
            capacity,
            offset: Cell::new(0),
            peak: Cell::new(0),
        }
    }

    /// 从 Arena 中分配 count 个 T 元素的可变切片
    /// 接受 &self，允许多个返回切片同时存活
    ///
    /// # Safety 论证
    /// 每次分配返回不重叠的内存区域，因此多个 &mut [T] 不会别名
    #[inline]
    pub fn alloc_slice<T>(&self, count: usize) -> &mut [T] {
        let size = std::mem::size_of::<T>() * count;
        let align = std::mem::align_of::<T>();

        let current = self.offset.get();
        let aligned_offset = (current + align - 1) & !(align - 1);
        let new_offset = aligned_offset + size;

        assert!(
            new_offset <= self.capacity,
            "Arena OOM: need {} bytes at offset {}, capacity {}",
            size,
            aligned_offset,
            self.capacity
        );

        self.offset.set(new_offset);
        if new_offset > self.peak.get() {
            self.peak.set(new_offset);
        }

        let slice_ptr = unsafe { self.ptr.as_ptr().add(aligned_offset) as *mut T };
        unsafe { std::slice::from_raw_parts_mut(slice_ptr, count) }
    }

    /// 分配 count 个 T 元素并清零
    #[inline]
    pub fn alloc_zeroed_slice<T>(&self, count: usize) -> &mut [T] {
        let slice = self.alloc_slice::<T>(count);
        let byte_len = std::mem::size_of::<T>() * count;
        unsafe {
            std::ptr::write_bytes(slice.as_mut_ptr() as *mut u8, 0, byte_len);
        }
        slice
    }

    /// 重置偏移指针，释放本轮所有分配
    /// 注意：调用者必须确保之前分配的切片不再被使用
    #[inline]
    pub fn reset(&mut self) {
        self.offset.set(0);
    }

    /// 返回历史峰值使用量（字节）
    #[inline]
    pub fn peak_usage(&self) -> usize {
        self.peak.get()
    }

    /// 返回当前已使用量（字节）
    #[inline]
    pub fn used(&self) -> usize {
        self.offset.get()
    }

    /// 返回总容量（字节）
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 64).expect("invalid layout in drop");
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}
