"""
SNG (SNGPKG) format parser for Python.

Binary layout after SNGPKG magic (6 bytes) + uint32 version:
  [16 bytes]  XOR mask key (applied to file data only, NOT to headers/metadata)
  [uint64]    Byte offset to the file data section (from end of file index)
  [uint64]    Metadata pair count
  [N * (uint32 key_len, key_bytes, uint32 val_len, val_bytes)]  Metadata pairs
  [uint64]    File count
  [N * (uint32 name_len, name_bytes, uint64 data_len, uint64 data_offset)]  File index
  [raw bytes] File data — each byte XORed with xor_key[i % 16]
"""
import os
import struct

SNG_MAGIC = b'SNGPKG'


def _read_uint32(f):
    return struct.unpack('<I', f.read(4))[0]


def _read_uint64(f):
    return struct.unpack('<Q', f.read(8))[0]


def _read_string(f):
    length = _read_uint32(f)
    return f.read(length).decode('utf-8', errors='replace')


def extract_sng(sng_path, dest_dir):
    """
    Extracts all files from a .sng archive into dest_dir.
    Returns the metadata dict {key: value}.
    Raises ValueError on bad magic.
    """
    os.makedirs(dest_dir, exist_ok=True)

    with open(sng_path, 'rb') as f:
        # --- Header ---
        magic = f.read(6)
        if magic != SNG_MAGIC:
            raise ValueError(f'Not a valid .sng file (bad magic): {magic!r}')
        _version = _read_uint32(f)          # uint32 version (currently 1)
        xor_key  = f.read(16)              # 16-byte rolling XOR mask

        _file_data_offset = _read_uint64(f)  # offset to file data from end of index
        meta_count        = _read_uint64(f)

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
            name       = _read_string(f)
            data_len   = _read_uint64(f)
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

            out_path = os.path.join(dest_dir, os.path.basename(name))
            with open(out_path, 'wb') as out:
                out.write(data)

    return metadata


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print('Usage: sng_parser.py <file.sng> <dest_dir>')
        sys.exit(1)
    meta = extract_sng(sys.argv[1], sys.argv[2])
    print('Metadata:')
    for k, v in meta.items():
        print(f'  {k} = {v}')
    print(f'Files extracted to: {sys.argv[2]}')
