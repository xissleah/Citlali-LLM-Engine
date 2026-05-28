// crates/citlali-gguf/src/lib.rs
pub mod dequant;
pub mod header;
pub mod metadata;
pub mod tensor;

use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

use header::GgufHeader;
use metadata::{MetadataKv, MetadataValue};
use tensor::TensorInfo;

/// 解析后的 GGUF 文件（不含权重数据）
#[derive(Debug)]
pub struct GgufFile {
    pub header: GgufHeader,
    pub metadata: Vec<MetadataKv>,
    pub tensors: Vec<TensorInfo>,
    /// 权重数据段在文件中的起始偏移（字节）
    pub data_offset: u64,
}

/// 默认对齐值
const DEFAULT_ALIGNMENT: u64 = 32;

impl GgufFile {
    /// 从文件路径加载并解析 GGUF
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }

    /// 从任意 Read+Seek 源解析
    pub fn read_from<R: Read + Seek>(reader: &mut R) -> io::Result<Self> {
        // 1. 解析文件头
        let header = GgufHeader::read_from(reader)?;

        // 2. 解析 metadata
        let mut metadata = Vec::with_capacity(header.metadata_kv_count as usize);
        for _ in 0..header.metadata_kv_count {
            metadata.push(MetadataKv::read_from(reader)?);
        }

        // 3. 解析 tensor info
        let mut tensors = Vec::with_capacity(header.tensor_count as usize);
        for _ in 0..header.tensor_count {
            tensors.push(TensorInfo::read_from(reader)?);
        }

        // 4. 计算数据段起始偏移（对齐到 alignment）
        let alignment = Self::find_alignment(&metadata);
        let current_pos = reader.stream_position()?;
        let data_offset = align_to(current_pos, alignment);

        Ok(Self {
            header,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// 从 metadata 中查找对齐值
    fn find_alignment(metadata: &[MetadataKv]) -> u64 {
        for kv in metadata {
            if kv.key == "general.alignment" {
                return match &kv.value {
                    MetadataValue::Uint32(v) => *v as u64,
                    MetadataValue::Uint64(v) => *v,
                    _ => DEFAULT_ALIGNMENT,
                };
            }
        }
        DEFAULT_ALIGNMENT
    }

    /// 按 key 查找 metadata 值
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| &kv.value)
    }
}

/// 向上对齐到 alignment 的倍数
fn align_to(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}
