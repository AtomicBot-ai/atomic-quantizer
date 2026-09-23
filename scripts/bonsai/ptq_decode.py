#!/usr/bin/env python3
"""
Decode PrismML PTQ1_0 / PQ2_0 tensors from a Bonsai GGUF.

The PrismML GGUF uses private ggml type ids:
  PQ2_0 = 142, PTQ1_0 = 143
so stock gguf-py/GGUFReader rejects the file before exposing the tensors.

This script keeps the small custom GGUF header reader from the reconstruction
and implements the same PTQ byte->ternary decoding used by PrismML's public
runtime transcoder.  Matrices are returned in [out, in] convention.

Examples:
  python ptq_decode.py --gguf PTQ1_0.gguf --list
  python ptq_decode.py --gguf PTQ1_0.gguf --tensor blk.0.ffn_down.weight --out w.npy
  python ptq_decode.py --gguf PTQ1_0.gguf --tensor blk.0.ffn_down.weight --stats

References:
  PrismML Bonsai runtime/codec.py
  PrismML llama.cpp gguf-py constants.py
  OrcaRouter Bonsai exporter / gguf_min.py
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

BLOCK_BYTES = {
    142: 34,  # PQ2_0: fp16 scale + 128 x 2-bit codes
    143: 28,  # PTQ1_0: 24 packed bytes + 2 packed trits + fp16 scale
}
TYPE_NAME = {142: "PQ2_0", 143: "PTQ1_0"}
ITEMSIZE = {
    0: 4,   # F32
    1: 2,   # F16
    30: 2,  # BF16
}
VALUE_FORMAT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
ARRAY_TYPE = 9
STRING_TYPE = 8


def _unpack(f, fmt: str):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise EOFError("unexpected end of GGUF while reading header")
    return struct.unpack(fmt, b)


def _read_string(f) -> str:
    (n,) = _unpack(f, "<Q")
    b = f.read(n)
    if len(b) != n:
        raise EOFError("unexpected end of GGUF while reading string")
    return b.decode("utf-8")


def _read_value(f, t: int):
    if t == STRING_TYPE:
        return _read_string(f)
    if t == ARRAY_TYPE:
        (et,) = _unpack(f, "<I")
        (n,) = _unpack(f, "<Q")
        return [_read_value(f, et) for _ in range(n)]
    fmt = VALUE_FORMAT.get(t)
    if fmt is None:
        raise NotImplementedError(f"unsupported GGUF KV type {t}")
    return _unpack(f, fmt)[0]


class Gguf:
    """Minimal GGUF reader that accepts PrismML's private tensor types."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{self.path} is not a GGUF file")
            (self.version,) = _unpack(f, "<I")
            (n_tensors,) = _unpack(f, "<Q")
            (n_kv,) = _unpack(f, "<Q")

            self.kv = {}
            for _ in range(n_kv):
                key = _read_string(f)
                (typ,) = _unpack(f, "<I")
                self.kv[key] = _read_value(f, typ)

            self.tensors = {}
            for _ in range(n_tensors):
                name = _read_string(f)
                (ndim,) = _unpack(f, "<I")
                dims = [_unpack(f, "<Q")[0] for _ in range(ndim)]
                (typ,) = _unpack(f, "<I")
                (offset,) = _unpack(f, "<Q")
                self.tensors[name] = {
                    "ne": dims,
                    "type": typ,
                    "offset": offset,
                    "type_name": TYPE_NAME.get(typ, str(typ)),
                }

            alignment = int(self.kv.get("general.alignment", 32))
            pos = f.tell()
            self.data_start = pos + (-pos) % alignment

    def shape_out_in(self, name: str) -> tuple[int, int]:
        dims = self.tensors[name]["ne"]
        if len(dims) != 2:
            raise ValueError(f"{name}: expected a matrix, GGUF dims={dims}")
        return tuple(int(x) for x in dims[::-1])

    def raw(self, name: str) -> bytes:
        t = self.tensors[name]
        numel = 1
        for d in t["ne"]:
            numel *= int(d)
        typ = int(t["type"])
        if typ in BLOCK_BYTES:
            if numel % 128:
                raise ValueError(f"{name}: {numel} elements is not divisible by 128")
            nbytes = numel // 128 * BLOCK_BYTES[typ]
        elif typ in ITEMSIZE:
            nbytes = numel * ITEMSIZE[typ]
        else:
            raise NotImplementedError(f"{name}: unsupported GGML type {typ}")
        with self.path.open("rb") as f:
            f.seek(self.data_start + int(t["offset"]))
            data = f.read(nbytes)
        if len(data) != nbytes:
            raise EOFError(f"{name}: short tensor read ({len(data)} != {nbytes})")
        return data


def _decode_ptq1_blocks(data: np.ndarray) -> np.ndarray:
    """
    Prism PTQ1_0:
      16 bytes x 5 trits
       8 bytes x 5 trits
       2 bytes x 4 trits
       2 bytes fp16 scale

    The extraction below intentionally matches PrismML's public transcoder:
      remainder = (packed * 3**trit) & 255
      code      = (remainder * 3) >> 8
    """
    if data.shape[1] != 28:
        raise ValueError("PTQ1_0 block must be exactly 28 bytes")

    pieces = []
    for lo, hi, count in ((0, 16, 5), (16, 24, 5), (24, 26, 4)):
        packed = data[:, lo:hi].astype(np.uint16)
        for trit in range(count):
            remainder = (packed * (3 ** trit)) & 0xFF
            code = ((remainder * 3) >> 8).astype(np.uint8)
            pieces.append(code)

    codes = np.concatenate(pieces, axis=1)  # [blocks, 128], values 0/1/2
    q = codes.astype(np.float32) - 1.0

    scale_bytes = np.ascontiguousarray(data[:, 26:28])
    scales = scale_bytes.view("<f2").reshape(-1).astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError("PTQ1_0 contains non-finite FP16 scales")

    return q * scales[:, None]


def _decode_pq2_blocks(data: np.ndarray) -> np.ndarray:
    """Prism PQ2_0: fp16 scale followed by 128 packed 2-bit codes."""
    if data.shape[1] != 34:
        raise ValueError("PQ2_0 block must be exactly 34 bytes")

    scale_bytes = np.ascontiguousarray(data[:, :2])
    scales = scale_bytes.view("<f2").reshape(-1).astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError("PQ2_0 contains non-finite FP16 scales")

    qs = data[:, 2:].reshape(data.shape[0], 32, 1).astype(np.uint8)
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4)
    codes = ((qs >> shifts) & 0x03).reshape(data.shape[0], 128)
    q = codes.astype(np.float32) - 1.0
    return q * scales[:, None]


def decode_bytes(raw: bytes, shape_out_in: tuple[int, int], type_id: int) -> np.ndarray:
    rows, width = shape_out_in
    if width % 128:
        raise ValueError(f"input dimension {width} is not divisible by 128")

    blocks = rows * width // 128
    block_bytes = BLOCK_BYTES[type_id]
    if len(raw) != blocks * block_bytes:
        raise ValueError(
            f"raw size {len(raw)} != {blocks} blocks * {block_bytes} bytes"
        )

    data = np.frombuffer(raw, dtype=np.uint8).reshape(blocks, block_bytes)
    if type_id == 143:
        values = _decode_ptq1_blocks(data)
    elif type_id == 142:
        values = _decode_pq2_blocks(data)
    else:
        raise NotImplementedError(type_id)

    return values.reshape(rows, width).astype(np.float32, copy=False)


def decode_tensor(gguf: Gguf, name: str) -> np.ndarray:
    if name not in gguf.tensors:
        raise KeyError(f"tensor not found: {name}")

    t = gguf.tensors[name]
    typ = int(t["type"])
    shape = gguf.shape_out_in(name)

    if typ in BLOCK_BYTES:
        return decode_bytes(gguf.raw(name), shape, typ)

    raw = gguf.raw(name)
    if typ == 0:
        return np.frombuffer(raw, dtype="<f4").reshape(shape).astype(np.float32, copy=False)
    if typ == 1:
        return np.frombuffer(raw, dtype="<f2").reshape(shape).astype(np.float32)
    if typ == 30:
        # numpy has no native BF16 dtype; convert uint16 bits to float32.
        u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (u << 16).view("<f4").reshape(shape)
    raise NotImplementedError(f"{name}: type {typ}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", required=True, help="PrismML GGUF file")
    ap.add_argument("--tensor", help="tensor name to decode")
    ap.add_argument("--out", help="write decoded tensor as .npy")
    ap.add_argument("--list", action="store_true", help="list tensor names/shapes/types")
    ap.add_argument("--stats", action="store_true", help="print min/max/norm for decoded tensor")
    args = ap.parse_args()

    g = Gguf(args.gguf)

    print(f"[gguf] {args.gguf}")
    print(f"[gguf] version={g.version} tensors={len(g.tensors)}")
    print(f"[gguf] architecture={g.kv.get('general.architecture')!r}")

    if args.list:
        for name in sorted(g.tensors):
            t = g.tensors[name]
            shape = tuple(reversed(t["ne"]))
            print(f"{name}\tshape={shape}\ttype={t['type_name']}({t['type']})")

    if not args.tensor:
        if not args.list:
            ap.error("--tensor is required unless --list is used")
        return

    arr = decode_tensor(g, args.tensor)
    print(f"[tensor] {args.tensor} shape={arr.shape} dtype={arr.dtype}")

    if args.stats:
        print(
            "[stats] min={:.7g} max={:.7g} mean={:.7g} norm={:.7g}".format(
                float(arr.min()), float(arr.max()), float(arr.mean()), float(np.linalg.norm(arr))
            )
        )

    if args.out:
        np.save(args.out, arr)
        print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
