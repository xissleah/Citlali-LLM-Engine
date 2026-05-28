// crates/citlali-compute/src/kv_cache.rs
//! 分页式 KV Cache
//! 固定页(16KB) + LRU 淘汰 + Swap 磁盘溢出
//!
//! 设计原则：
//! - 所有页从 PagePool 预分配，运行时零 malloc
//! - 页表(PageTable)管理逻辑块到物理页的映射
//! - LRU 跟踪页访问顺序，内存不足时淘汰冷页到 swap
//! - 对外暴露简洁的 push/get 接口，forward.rs 无感切换

use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::ptr::NonNull;

/// 默认页大小：16KB
pub const PAGE_SIZE: usize = 16 * 1024;

/// KV 类型标记
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvType {
    Key,
    Value,
}

/// 页池配置
#[derive(Debug, Clone)]
pub struct KvCacheConfig {
    /// 每个 position 的 KV 向量维度 (num_kv_heads * head_dim)
    pub kv_dim: usize,
    /// 模型层数
    pub num_layers: usize,
    /// 内存中最大页数（超出则 LRU 淘汰到 swap）
    pub max_pages_in_memory: usize,
    /// Swap 文件路径（None 则禁用 swap，OOM 时 panic）
    pub swap_path: Option<String>,
}

/// 每页能容纳的 position 数
/// positions_per_page = PAGE_SIZE / (kv_dim * sizeof(f32))
#[inline]
pub fn positions_per_page(kv_dim: usize) -> usize {
    PAGE_SIZE / (kv_dim * std::mem::size_of::<f32>())
}

// ─── 页标识 ───────────────────────────────────────────

/// 逻辑页键：唯一标识一个逻辑块
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LogicalPageKey {
    pub layer: usize,
    pub kv_type: KvType,
    /// 块索引：position / positions_per_page
    pub block_idx: usize,
}

/// 物理页 ID（在 PagePool 中的索引）
type PageId = usize;

/// Swap 中的偏移（字节）
type SwapOffset = u64;

// ─── 物理页池 ─────────────────────────────────────────

/// 物理页池：预分配一块连续内存，切分为固定大小的页
struct PagePool {
    /// 底层内存块
    ptr: NonNull<u8>,
    /// 总页数
    num_pages: usize,
    /// 空闲页栈（LIFO，快速分配）
    free_list: Vec<PageId>,
}

impl PagePool {
    fn new(num_pages: usize) -> Self {
        let total_bytes = num_pages * PAGE_SIZE;
        let layout = Layout::from_size_align(total_bytes, 64).expect("PagePool layout invalid");
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("PagePool allocation failed (OOM)");
        // 初始化空闲列表：所有页都可用
        let free_list: Vec<PageId> = (0..num_pages).rev().collect();
        Self {
            ptr,
            num_pages,
            free_list,
        }
    }

    /// 分配一页，返回页 ID
    fn alloc_page(&mut self) -> Option<PageId> {
        self.free_list.pop()
    }

    /// 回收一页
    fn free_page(&mut self, page_id: PageId) {
        // 清零页内容（可选，防止数据残留）
        let offset = page_id * PAGE_SIZE;
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr().add(offset), 0, PAGE_SIZE);
        }
        self.free_list.push(page_id);
    }

    /// 获取页的可变字节切片
    fn page_bytes_mut(&mut self, page_id: PageId) -> &mut [u8] {
        let offset = page_id * PAGE_SIZE;
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(offset), PAGE_SIZE) }
    }

    /// 获取页的只读字节切片
    fn page_bytes(&self, page_id: PageId) -> &[u8] {
        let offset = page_id * PAGE_SIZE;
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(offset), PAGE_SIZE) }
    }

    /// 将页解释为 f32 切片
    fn page_as_f32(&self, page_id: PageId) -> &[f32] {
        let bytes = self.page_bytes(page_id);
        let count = PAGE_SIZE / std::mem::size_of::<f32>();
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, count) }
    }

    fn page_as_f32_mut(&mut self, page_id: PageId) -> &mut [f32] {
        let bytes = self.page_bytes_mut(page_id);
        let count = PAGE_SIZE / std::mem::size_of::<f32>();
        unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, count) }
    }

    fn free_count(&self) -> usize {
        self.free_list.len()
    }

    /// 清零指定页的内容
    fn clear_page(&mut self, page_id: PageId) {
        let offset = page_id * PAGE_SIZE;
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr().add(offset), 0, PAGE_SIZE);
        }
    }
}

impl Drop for PagePool {
    fn drop(&mut self) {
        let total_bytes = self.num_pages * PAGE_SIZE;
        let layout =
            Layout::from_size_align(total_bytes, 64).expect("PagePool layout invalid in drop");
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), layout);
        }
    }
}

// Safety: PagePool 独占管理其内存，不跨线程共享
unsafe impl Send for PagePool {}

// ─── LRU 跟踪器 ──────────────────────────────────────

/// LRU 节点（侵入式双向链表）
struct LruNode {
    key: LogicalPageKey,
    page_id: PageId,
    prev: Option<usize>, // 在 nodes Vec 中的索引
    next: Option<usize>,
}

/// LRU 跟踪器：维护页的访问顺序
/// head = 最近访问，tail = 最久未访问（淘汰候选）
struct LruTracker {
    nodes: Vec<LruNode>,
    /// key → nodes 索引
    index: HashMap<LogicalPageKey, usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

impl LruTracker {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            index: HashMap::new(),
            head: None,
            tail: None,
        }
    }

    /// 插入新页到 LRU 头部
    fn insert(&mut self, key: LogicalPageKey, page_id: PageId) {
        let idx = self.nodes.len();
        self.nodes.push(LruNode {
            key,
            page_id,
            prev: None,
            next: self.head,
        });
        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
        self.index.insert(key, idx);
    }

    /// 将已有页提升到头部（标记为最近访问）
    fn touch(&mut self, key: &LogicalPageKey) {
        let Some(&idx) = self.index.get(key) else {
            return;
        };
        if self.head == Some(idx) {
            return;
        } // 已在头部
        self.detach(idx);
        // 插入头部
        self.nodes[idx].prev = None;
        self.nodes[idx].next = self.head;
        if let Some(old_head) = self.head {
            self.nodes[old_head].prev = Some(idx);
        }
        self.head = Some(idx);
    }

    /// 淘汰尾部页（最久未访问），返回 (key, page_id)
    fn evict_tail(&mut self) -> Option<(LogicalPageKey, PageId)> {
        let tail_idx = self.tail?;
        let node = &self.nodes[tail_idx];
        let key = node.key;
        let page_id = node.page_id;
        self.detach(tail_idx);
        self.index.remove(&key);
        Some((key, page_id))
    }

    /// 从链表中摘除节点（不从 index 中删除）
    fn detach(&mut self, idx: usize) {
        let prev = self.nodes[idx].prev;
        let next = self.nodes[idx].next;
        if let Some(p) = prev {
            self.nodes[p].next = next;
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            self.nodes[n].prev = prev;
        } else {
            self.tail = prev;
        }
        self.nodes[idx].prev = None;
        self.nodes[idx].next = None;
    }

    /// 查询页是否在内存中
    fn contains(&self, key: &LogicalPageKey) -> bool {
        self.index.contains_key(key)
    }

    /// 获取页的物理 ID
    fn get_page_id(&self, key: &LogicalPageKey) -> Option<PageId> {
        self.index.get(key).map(|&idx| self.nodes[idx].page_id)
    }

    /// 移除指定 key，返回其 page_id（用于 truncate）
    fn remove_key(&mut self, key: &LogicalPageKey) -> Option<PageId> {
        let idx = *self.index.get(key)?;
        let page_id = self.nodes[idx].page_id;
        self.detach(idx);
        self.index.remove(key);
        Some(page_id)
    }

    /// 当前在内存中的页数
    fn len(&self) -> usize {
        self.index.len()
    }
}

// ─── Swap 文件 ───────────────────────────────────────

/// 磁盘溢出存储：被淘汰的页写入文件，需要时再读回
struct SwapFile {
    file: File,
    /// key → swap 文件中的偏移
    directory: HashMap<LogicalPageKey, SwapOffset>,
    /// 下一个可写偏移
    next_offset: u64,
}

impl SwapFile {
    fn open(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            file,
            directory: HashMap::new(),
            next_offset: 0,
        })
    }

    /// 将一页写入 swap
    fn write_page(&mut self, key: LogicalPageKey, data: &[u8]) -> std::io::Result<()> {
        assert_eq!(data.len(), PAGE_SIZE);
        self.file.seek(SeekFrom::Start(self.next_offset))?;
        self.file.write_all(data)?;
        self.directory.insert(key, self.next_offset);
        self.next_offset += PAGE_SIZE as u64;
        Ok(())
    }

    /// 从 swap 读回一页
    fn read_page(&mut self, key: &LogicalPageKey, buf: &mut [u8]) -> std::io::Result<()> {
        let offset = self
            .directory
            .get(key)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "page not in swap"))?;
        self.file.seek(SeekFrom::Start(*offset))?;
        self.file.read_exact(buf)?;
        Ok(())
    }

    /// 检查页是否在 swap 中
    fn contains(&self, key: &LogicalPageKey) -> bool {
        self.directory.contains_key(key)
    }

    /// 从 swap 目录中移除（页已被读回内存）
    fn remove(&mut self, key: &LogicalPageKey) {
        self.directory.remove(key);
    }
}

// ─── 分页式 KV Cache（对外接口）───────────────────

/// 分页式 KV Cache
/// 对外提供简洁的 push/get 接口，内部管理页分配、LRU 淘汰、Swap 溢出
pub struct PagedKvCache {
    config: KvCacheConfig,
    pool: PagePool,
    lru: LruTracker,
    swap: Option<SwapFile>,
    /// 每页容纳的 position 数
    positions_per_page: usize,
    /// 当前序列长度（已 push 的 position 数）
    seq_len: usize,
}

impl PagedKvCache {
    /// 创建分页式 KV Cache
    pub fn new(config: KvCacheConfig) -> Self {
        let ppp = positions_per_page(config.kv_dim);
        assert!(ppp > 0, "kv_dim too large for page size");

        let swap = config
            .swap_path
            .as_ref()
            .map(|path| SwapFile::open(path).expect("failed to open swap file"));

        Self {
            pool: PagePool::new(config.max_pages_in_memory),
            lru: LruTracker::new(),
            swap,
            positions_per_page: ppp,
            seq_len: 0,
            config,
        }
    }

    /// 当前序列长度
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// 推入一个 position 的 K/V 向量到所有层
    /// k_vectors[layer] = &[f32; kv_dim], v_vectors[layer] = &[f32; kv_dim]
    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.config.kv_dim);
        debug_assert_eq!(v.len(), self.config.kv_dim);

        let pos = self.seq_len; // 当前写入的 position
        let block_idx = pos / self.positions_per_page;
        let slot_in_page = pos % self.positions_per_page;

        // 写 K
        self.write_slot(layer, KvType::Key, block_idx, slot_in_page, k);
        // 写 V
        self.write_slot(layer, KvType::Value, block_idx, slot_in_page, v);
    }

    /// 增加序列长度（在所有层的 push 完成后调用一次）
    pub fn advance_seq(&mut self) {
        self.seq_len += 1;
    }

    /// 推入一个 position 的 K/V 向量到指定层的指定位置
    /// 用于 batch prefill：多个 token 在同一层内按各自位置写入
    pub fn push_at(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.config.kv_dim);
        debug_assert_eq!(v.len(), self.config.kv_dim);

        let block_idx = pos / self.positions_per_page;
        let slot_in_page = pos % self.positions_per_page;

        self.write_slot(layer, KvType::Key, block_idx, slot_in_page, k);
        self.write_slot(layer, KvType::Value, block_idx, slot_in_page, v);
    }

    /// 读取指定层、指定 position 的 K 向量（堆分配，已弃用）
    #[deprecated(note = "use copy_k_into instead")]
    pub fn get_k(&mut self, layer: usize, pos: usize) -> Vec<f32> {
        self.read_slot(layer, KvType::Key, pos)
    }

    /// 读取指定层、指定 position 的 V 向量（堆分配，已弃用）
    #[deprecated(note = "use copy_v_into instead")]
    pub fn get_v(&mut self, layer: usize, pos: usize) -> Vec<f32> {
        self.read_slot(layer, KvType::Value, pos)
    }

    /// 将指定 position 的 K 向量复制到 dest（零堆分配）
    pub fn copy_k_into(&mut self, layer: usize, pos: usize, dest: &mut [f32]) {
        self.copy_slot_into(layer, KvType::Key, pos, dest);
    }

    /// 将指定 position 的 V 向量复制到 dest（零堆分配）
    pub fn copy_v_into(&mut self, layer: usize, pos: usize, dest: &mut [f32]) {
        self.copy_slot_into(layer, KvType::Value, pos, dest);
    }

    // ─── 内部方法 ───────────────────────────────────

    /// 写入一个 slot
    fn write_slot(
        &mut self,
        layer: usize,
        kv_type: KvType,
        block_idx: usize,
        slot_in_page: usize,
        data: &[f32],
    ) {
        let key = LogicalPageKey {
            layer,
            kv_type,
            block_idx,
        };
        let page_id = self.ensure_page(key);

        // 写入数据到页内对应 slot
        let page_f32 = self.pool.page_as_f32_mut(page_id);
        let offset = slot_in_page * self.config.kv_dim;
        page_f32[offset..offset + self.config.kv_dim].copy_from_slice(data);

        self.lru.touch(&key);
    }

    /// 读取一个 slot（堆分配版本，仅 get_k/get_v 使用）
    fn read_slot(&mut self, layer: usize, kv_type: KvType, pos: usize) -> Vec<f32> {
        let block_idx = pos / self.positions_per_page;
        let slot_in_page = pos % self.positions_per_page;
        let key = LogicalPageKey {
            layer,
            kv_type,
            block_idx,
        };

        let page_id = self.ensure_page(key);
        self.lru.touch(&key);

        let page_f32 = self.pool.page_as_f32(page_id);
        let offset = slot_in_page * self.config.kv_dim;
        page_f32[offset..offset + self.config.kv_dim].to_vec()
    }

    /// 将 slot 数据复制到 dest（零堆分配版本）
    fn copy_slot_into(&mut self, layer: usize, kv_type: KvType, pos: usize, dest: &mut [f32]) {
        let block_idx = pos / self.positions_per_page;
        let slot_in_page = pos % self.positions_per_page;
        let key = LogicalPageKey {
            layer,
            kv_type,
            block_idx,
        };

        let page_id = self.ensure_page(key);
        self.lru.touch(&key);

        let page_f32 = self.pool.page_as_f32(page_id);
        let offset = slot_in_page * self.config.kv_dim;
        dest.copy_from_slice(&page_f32[offset..offset + self.config.kv_dim]);
    }

    /// 确保逻辑页在内存中，返回物理页 ID
    /// 如果页已在内存 → 直接返回
    /// 如果页在 swap → 读回内存（可能触发淘汰）
    /// 如果页不存在 → 分配新页（可能触发淘汰）
    /// Drop 策略下，已淘汰的页会返回零页（数据丢失）
    fn ensure_page(&mut self, key: LogicalPageKey) -> PageId {
        // 已在内存中
        if let Some(page_id) = self.lru.get_page_id(&key) {
            return page_id;
        }

        // 需要一个空闲页
        let page_id = self.acquire_free_page();

        // 检查是否在 swap 中（仅 Swap 策略有效）
        if let Some(ref mut swap) = self.swap {
            if swap.contains(&key) {
                let page_bytes = self.pool.page_bytes_mut(page_id);
                swap.read_page(&key, page_bytes)
                    .expect("failed to read from swap");
                swap.remove(&key);
            }
        }
        // Drop 策略：无 swap，页已被清零，返回的是全零页
        // 调用方会读到全零向量（等价于“这个 position 不存在”）

        self.lru.insert(key, page_id);
        page_id
    }

    /// 获取一个空闲页，必要时淘汰
    fn acquire_free_page(&mut self) -> PageId {
        // 尝试从空闲列表分配
        if let Some(page_id) = self.pool.alloc_page() {
            return page_id;
        }
        // 空闲列表耗尽，淘汰一页
        self.evict_one()
    }

    /// 淘汰 LRU 尾部的一页
    /// 如果配置了 swap → 写入磁盘（可恢复）
    /// 如果未配置 swap → 直接丢弃（Drop 策略，不可恢复）
    fn evict_one(&mut self) -> PageId {
        let (evicted_key, page_id) = self
            .lru
            .evict_tail()
            .expect("no pages to evict (pool exhausted)");

        // 写入 swap（如果启用）
        if let Some(ref mut swap) = self.swap {
            let page_bytes = self.pool.page_bytes(page_id);
            swap.write_page(evicted_key, page_bytes)
                .expect("failed to write to swap");
        }
        // 未启用 swap 时，数据直接丢弃（Drop 策略）
        // 清零页内容并返回
        self.pool.clear_page(page_id);
        page_id
    }

    /// 截断 KV Cache 到指定序列长度
    /// 释放超出 new_seq_len 的所有页，保留前缀部分的 KV 状态
    pub fn truncate_to(&mut self, new_seq_len: usize) {
        if new_seq_len >= self.seq_len {
            return;
        }
        let pages_needed = if new_seq_len == 0 {
            0
        } else {
            (new_seq_len - 1) / self.positions_per_page + 1
        };
        // 收集需要释放的页 key
        let keys_to_remove: Vec<LogicalPageKey> = self
            .lru
            .index
            .keys()
            .filter(|k| k.block_idx >= pages_needed)
            .copied()
            .collect();
        for key in keys_to_remove {
            if let Some(page_id) = self.lru.remove_key(&key) {
                self.pool.free_page(page_id);
            }
        }
        // 清理 swap 中超出范围的页
        if let Some(ref mut swap) = self.swap {
            let swap_keys: Vec<LogicalPageKey> = swap
                .directory
                .keys()
                .filter(|k| k.block_idx >= pages_needed)
                .copied()
                .collect();
            for key in swap_keys {
                swap.remove(&key);
            }
        }
        self.seq_len = new_seq_len;
    }

    /// 统计信息
    pub fn stats(&self) -> KvCacheStats {
        KvCacheStats {
            seq_len: self.seq_len,
            pages_in_memory: self.lru.len(),
            pages_in_swap: self.swap.as_ref().map(|s| s.directory.len()).unwrap_or(0),
            free_pages: self.pool.free_count(),
            positions_per_page: self.positions_per_page,
        }
    }

    /// 批量读取指定层的 K 向量 [0..seq_len) 到连续缓冲区
    /// dest: [seq_len * kv_dim]，按 position 连续存储
    /// 相比逐个 copy_k_into，减少重复的 page lookup 和 LRU touch
    pub fn batch_copy_k_into(&mut self, layer: usize, seq_len: usize, dest: &mut [f32]) {
        debug_assert_eq!(dest.len(), seq_len * self.config.kv_dim);
        self.batch_copy_slot_into(layer, KvType::Key, seq_len, dest);
    }

    /// 批量读取指定层的 V 向量 [0..seq_len) 到连续缓冲区
    /// dest: [seq_len * kv_dim]，按 position 连续存储
    pub fn batch_copy_v_into(&mut self, layer: usize, seq_len: usize, dest: &mut [f32]) {
        debug_assert_eq!(dest.len(), seq_len * self.config.kv_dim);
        self.batch_copy_slot_into(layer, KvType::Value, seq_len, dest);
    }

    /// 批量复制内部实现：按页遍历，每页只做一次 ensure_page + touch
    fn batch_copy_slot_into(
        &mut self,
        layer: usize,
        kv_type: KvType,
        seq_len: usize,
        dest: &mut [f32],
    ) {
        let kv_dim = self.config.kv_dim;
        let ppp = self.positions_per_page;
        let num_blocks = if seq_len == 0 { 0 } else { (seq_len - 1) / ppp + 1 };

        for block_idx in 0..num_blocks {
            let key = LogicalPageKey { layer, kv_type, block_idx };
            let page_id = self.ensure_page(key);
            self.lru.touch(&key);

            let page_f32 = self.pool.page_as_f32(page_id);
            let pos_start = block_idx * ppp;
            let pos_end = ((block_idx + 1) * ppp).min(seq_len);

            for pos in pos_start..pos_end {
                let slot = pos % ppp;
                let src_offset = slot * kv_dim;
                let dst_offset = pos * kv_dim;
                dest[dst_offset..dst_offset + kv_dim]
                    .copy_from_slice(&page_f32[src_offset..src_offset + kv_dim]);
            }
        }
    }
}

/// KV Cache 统计信息
#[derive(Debug)]
pub struct KvCacheStats {
    pub seq_len: usize,
    pub pages_in_memory: usize,
    pub pages_in_swap: usize,
    pub free_pages: usize,
    pub positions_per_page: usize,
}
