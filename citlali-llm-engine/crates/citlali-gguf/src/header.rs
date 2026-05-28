// crates/citlali-gguf/src/header.rs

use std::io::{self, Read};

/// GGUF 文件头 magic 常量
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" 小端: G(47) G(47) U(55) F(46)

/// GGUF 文件头
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// 从 reader 中读取并验证文件头
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let magic = read_u32(reader)?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "无效的 GGUF magic: 0x{:08X}, 期望 0x{:08X}",
                    magic, GGUF_MAGIC
                ),
            ));
        }

        let version = read_u32(reader)?;
        if version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("不支持的 GGUF 版本: {}, 仅支持 v3", version),
            ));
        }

        let tensor_count = read_u64(reader)?;
        let metadata_kv_count = read_u64(reader)?;

        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

/// 读取小端 u32
pub fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

/// 读取小端 u64
pub fn read_u64<R: Read>(reader: &mut R) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}
