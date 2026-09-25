#!/usr/bin/env python3
"""Read-only GGUF tensor inspector.

Usage:
  python gguf_inspect.py MODEL.gguf
  python gguf_inspect.py MODEL.gguf --type Q6_K
  python gguf_inspect.py MODEL.gguf --summary
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
from pathlib import Path


TENSOR_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}

# GGUF metadata value types.
META_FIXED = {
    0: ("UINT8", "B"),
    1: ("INT8", "b"),
    2: ("UINT16", "H"),
    3: ("INT16", "h"),
    4: ("UINT32", "I"),
    5: ("INT32", "i"),
    6: ("FLOAT32", "f"),
    7: ("BOOL", "?"),
    10: ("UINT64", "Q"),
    11: ("INT64", "q"),
    12: ("FLOAT64", "d"),
}


class Reader:
    def __init__(self, fp):
        self.fp = fp

    def read(self, n: int) -> bytes:
        data = self.fp.read(n)
        if len(data) != n:
            raise ValueError("unexpected end of file")
        return data

    def u8(self) -> int:
        return self.read(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        size = self.u64()
        raw = self.read(size)
        return raw.decode("utf-8", errors="replace")

    def skip_value(self, value_type: int) -> None:
        if value_type in META_FIXED:
            self.read(struct.calcsize("<" + META_FIXED[value_type][1]))
        elif value_type == 8:  # STRING
            self.string()
        elif value_type == 9:  # ARRAY
            element_type = self.u32()
            count = self.u64()
            for _ in range(count):
                self.skip_value(element_type)
        else:
            raise ValueError(f"unsupported GGUF metadata type {value_type}")


def tensor_bytes(element_count: int, type_id: int) -> int | None:
    info = TENSOR_TYPES.get(type_id)
    if info is None:
        return None
    block, block_bytes = info[1], info[2]
    if element_count % block:
        return None
    return element_count // block * block_bytes


def read_tensors(path: Path):
    with path.open("rb") as fp:
        r = Reader(fp)
        if r.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = r.u32()
        tensor_count = r.u64()
        metadata_count = r.u64()
        for _ in range(metadata_count):
            r.string()
            r.skip_value(r.u32())

        tensors = []
        for _ in range(tensor_count):
            name = r.string()
            ndim = r.u32()
            dims = [r.u64() for _ in range(ndim)]
            type_id = r.u32()
            offset = r.u64()
            count = 1
            for dim in dims:
                count *= dim
            info = TENSOR_TYPES.get(type_id)
            tensors.append({
                "name": name,
                "dims": dims,
                "type_id": type_id,
                "type": info[0] if info else f"UNKNOWN({type_id})",
                "offset": offset,
                "elements": count,
                "bytes": tensor_bytes(count, type_id),
            })
        return version, tensors


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GGUF tensor names and types")
    parser.add_argument("model", type=Path)
    parser.add_argument("--type", dest="type_name", help="show only this type, e.g. Q6_K")
    parser.add_argument("--match", help="show tensor names containing this text")
    parser.add_argument("--summary", action="store_true", help="show counts only")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    try:
        version, tensors = read_tensors(args.model)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.type_name:
        tensors = [t for t in tensors if t["type"] == args.type_name]
    if args.match:
        needle = args.match.lower()
        tensors = [t for t in tensors if needle in t["name"].lower()]

    if args.json:
        print(json.dumps({"version": version, "tensors": tensors}, ensure_ascii=False, indent=2))
        return 0

    print(f"GGUF version: {version}")
    print(f"tensor count: {len(tensors)}" + (" (filtered)" if args.type_name or args.match else ""))
    if args.summary:
        counts = collections.Counter(t["type"] for t in tensors)
        for type_name, count in sorted(counts.items()):
            print(f"  {type_name}: {count}")
        return 0

    print("type      elements       bytes  dims                         name")
    print("--------  -------------  ------  ---------------------------  ----")
    for t in tensors:
        byte_text = str(t["bytes"]) if t["bytes"] is not None else "?"
        dims_text = "x".join(str(x) for x in t["dims"])
        print(f"{t['type']:<8}  {t['elements']:>13}  {byte_text:>6}  {dims_text:<27}  {t['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
