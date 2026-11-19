# SNGPKG layout after the 6-byte magic + uint32 version:
#   [16 bytes]  XOR mask key (applied to file data only, not headers/metadata)
#   [uint64]    byte offset to the file data section (from end of file index)
#   [uint64]    metadata pair count
#   [N * (uint32 key_len, key_bytes, uint32 val_len, val_bytes)]  metadata pairs
#   [uint64]    file count
#   [N * (uint32 name_len, name_bytes, uint64 data_len, uint64 data_offset)]  file index
#   [raw bytes] file data, each byte XORed with xor_key[i % 16]

import struct
from pathlib import Path

SNG_MAGIC = b'SNGPKG'


def _read_uint32(f):
    return struct.unpack('<I', f.read(4))[0]


def _read_uint64(f):
    return struct.unpack('<Q', f.read(8))[0]


def _read_string(f):
    length = _read_uint32(f)
    return f.read(length).decode('utf-8', errors='replace')


def extract_sng(sng_path, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with Path(sng_path).open('rb') as f:
        # --- Header ---
        magic = f.read(6)
        if magic != SNG_MAGIC:
            raise ValueError(f'Not a valid .sng file (bad magic): {magic!r}')
        _version = _read_uint32(f)  # uint32 version (currently 1)
        xor_key = f.read(16)  # 16-byte rolling XOR mask

        _file_data_offset = _read_uint64(f)  # offset to file data from end of index
        meta_count = _read_uint64(f)

        # --- Metadata ---
        metadata = {}
        for _ in range(meta_count):
            key = _read_string(f)
            val = _read_string(f)
            metadata[key] = val

        # --- File index ---
        file_count = _read_uint64(f)
        file_index = []
        for _ in range(file_count):
            name = _read_string(f)
            data_len = _read_uint64(f)
            data_offset = _read_uint64(f)
            file_index.append((name, data_len, data_offset))

        # File data starts right after the index
        data_section_start = f.tell()

        # --- Extract each file ---
        for name, data_len, data_offset in file_index:
            f.seek(data_section_start + data_offset)
            masked = f.read(data_len)
            # Unmask: rolling XOR with the 16-byte key
            data = bytes(masked[i] ^ xor_key[i % 16] for i in range(len(masked)))

            out_path = dest_dir / Path(name).name
            with out_path.open('wb') as out:
                out.write(data)

    return metadata
