#!/usr/bin/env python3
"""
Build main_ram.init from firmware binary + golden data blobs.

Loads firmware.bin and test data binary blobs, combines them with
proper padding so blobs land at fixed addresses:
  - 0x40002000: INT8 blob
  - 0x40006000: INT16 blob

Usage: python3 build_main_ram.py firmware.bin test_data.bin output.init
"""

import struct
import sys

TARGET_ADDR_INT8 = 0x40002000
MAIN_RAM_BASE = 0x40000000


def bin_to_words(data):
    """Convert binary bytes to list of uint32 words (little-endian)."""
    words = []
    for i in range(0, len(data), 4):
        chunk = data[i:i+4]
        if len(chunk) < 4:
            chunk = chunk + b'\x00' * (4 - len(chunk))
        words.append(struct.unpack('<I', chunk)[0])
    return words


def words_to_hex(words, out_path):
    """Write uint32 words as Verilog hex init file (one word per line)."""
    with open(out_path, 'w') as f:
        for w in words:
            f.write(f'{w:08X}\n')


def main():
    if len(sys.argv) != 4:
        print(f'Usage: {sys.argv[0]} firmware.bin test_data.bin output.init')
        sys.exit(1)

    firmware_bin = sys.argv[1]
    test_data_bin = sys.argv[2]
    output_init = sys.argv[3]

    with open(firmware_bin, 'rb') as f:
        fw_data = f.read()
    with open(test_data_bin, 'rb') as f:
        td_data = f.read()

    fw_words = bin_to_words(fw_data)
    td_words = bin_to_words(td_data)

    # Calculate padding: blob target is 0x40001000
    # Absolute word offset of blob: (0x40001000 - 0x40000000) / 4 = 1024
    blob_word_offset = (TARGET_ADDR_INT8 - MAIN_RAM_BASE) // 4
    pad_words = blob_word_offset - len(fw_words)

    if pad_words < 0:
        print(f'ERROR: Firmware too large ({len(fw_words)} words > {blob_word_offset})')
        sys.exit(1)

    print(f'Firmware: {len(fw_words)} words ({len(fw_data)} bytes)')
    print(f'Test data: {len(td_words)} words ({len(td_data)} bytes)')
    print(f'Blob offset: {blob_word_offset} words (addr 0x{TARGET_ADDR_INT8:08X})')
    print(f'Padding: {pad_words} words')

    # Build combined init
    combined = list(fw_words)
    combined.extend([0] * pad_words)
    combined.extend(td_words)

    words_to_hex(combined, output_init)
    print(f'Output: {output_init} ({len(combined)} words, '
          f'{len(combined)*4} bytes in main RAM)')


if __name__ == '__main__':
    main()
