// crates/citlali-gguf/src/metadata.rs

use crate::header::{read_u32, read_u64};
use std::io::{self, Read};

/// GGUF metadata 值类型标识
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufMetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

/// 元数据值
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// 一个 metadata 键值对
#[derive(Debug, Clone)]
pub struct MetadataKv {
    pub key: String,
    pub value: MetadataValue,
}

impl MetadataKv {
    /// 从 reader 读取一个完整的 KV 对
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let key = read_string(reader)?;
        let value = read_value(reader)?;
        Ok(Self { key, value })
    }
}

/// 读取 GGUF 格式的字符串（u64 长度前缀 + utf8 数据）
pub fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
    let len = read_u64(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("无效 UTF-8: {}", e)))
}

/// 读取一个 metadata value
pub fn read_value<R: Read>(reader: &mut R) -> io::Result<MetadataValue> {
    let type_id = read_u32(reader)?;
    read_value_of_type(reader, type_id)
}

/// 根据类型 ID 读取对应的值
fn read_value_of_type<R: Read>(reader: &mut R, type_id: u32) -> io::Result<MetadataValue> {
    match type_id {
        0 => {
            // Uint8
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Uint8(buf[0]))
        }
        1 => {
            // Int8
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Int8(buf[0] as i8))
        }
        2 => {
            // Uint16
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Uint16(u16::from_le_bytes(buf)))
        }
        3 => {
            // Int16
            let mut buf = [0u8; 2];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Int16(i16::from_le_bytes(buf)))
        }
        4 => {
            // Uint32
            Ok(MetadataValue::Uint32(read_u32(reader)?))
        }
        5 => {
            // Int32
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Int32(i32::from_le_bytes(buf)))
        }
        6 => {
            // Float32
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Float32(f32::from_le_bytes(buf)))
        }
        7 => {
            // Bool
            let mut buf = [0u8; 1];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Bool(buf[0] != 0))
        }
        8 => {
            // String
            let s = read_string(reader)?;
            Ok(MetadataValue::String(s))
        }
        9 => {
            // Array
            let elem_type = read_u32(reader)?;
            let count = read_u64(reader)? as usize;
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                items.push(read_value_of_type(reader, elem_type)?);
            }
            Ok(MetadataValue::Array(items))
        }
        10 => {
            // Uint64
            Ok(MetadataValue::Uint64(read_u64(reader)?))
        }
        11 => {
            // Int64
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Int64(i64::from_le_bytes(buf)))
        }
        12 => {
            // Float64
            let mut buf = [0u8; 8];
            reader.read_exact(&mut buf)?;
            Ok(MetadataValue::Float64(f64::from_le_bytes(buf)))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("未知的 metadata 类型 ID: {}", type_id),
        )),
    }
}
