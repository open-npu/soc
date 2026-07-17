#!/usr/bin/env python3
"""
Build main_ram.init from firmware binary + chained model blob.

DDR-aware scatter: places each layer's wgt/param/input/output at their
remapped DDR addresses (0x40XXXXXX) in main RAM. Uses @address hex format
for $readmemh to keep file small (skip zero regions).

Usage: python3 build_main_ram.py firmware.bin test_data.bin output.init
"""

import struct
import sys
import os

MAIN_RAM_BASE = 0x40000000
BLOB_BASE = 0x40010000      # blob header + layer entries at 64KB offset

MAGIC = 0x4E505532
LAYER_ENTRY_HDR_WORDS = 36


def bin_to_words(data):
    """Convert binary bytes to list of uint32 words (little-endian)."""
    words = []
    for i in range(0, len(data), 4):
        chunk = data[i:i+4]
        if len(chunk) < 4:
            chunk = chunk + b'\x00' * (4 - len(chunk))
        words.append(struct.unpack('<I', chunk)[0])
    return words


def parse_blob(blob_words):
    """Parse chained blob into layer entries with DDR addresses.

    Returns list of dicts: [{header: [36 words], wgt: [], param: [], input: [], output: []}]
    """
    magic = blob_words[0]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X}")

    num_layers = blob_words[1]
    layers = []

    offset = 3  # skip 3-word blob header
    for l in range(num_layers):
        hdr = blob_words[offset:offset + LAYER_ENTRY_HDR_WORDS]
        offset += LAYER_ENTRY_HDR_WORDS

        n_wgt = hdr[0]
        n_param = hdr[1]
        n_input = hdr[2]
        n_output = hdr[3]

        wgt = blob_words[offset:offset + n_wgt]
        offset += n_wgt

        param = blob_words[offset:offset + n_param]
        offset += n_param

        inp = blob_words[offset:offset + n_input]
        offset += n_input

        out = blob_words[offset:offset + n_output]
        offset += n_output

        layers.append({
            'header': hdr,
            'wgt': wgt,
            'param': param,
            'input': inp,
            'output': out,
            'ddr_wgt': hdr[29],
            'ddr_param': hdr[30],
            'ddr_out': hdr[31],
            'ddr_in': hdr[32],
        })

    return layers


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

    print(f"Firmware: {len(fw_words)} words ({len(fw_data)} bytes)")
    print(f"Test data: {len(td_words)} words ({len(td_data)} bytes)")

    # Parse blob to get DDR addresses
    layers = parse_blob(td_words)
    print(f"Blob: {len(layers)} layers")

    # Build sparse memory: word_offset → value
    ram = {}

    # 1. Firmware at offset 0
    #    Firmware is in ROM (0x00000000), not main_ram. But build_main_ram
    #    also writes it to main_ram offset 0 for CPU copy.
    #    Actually firmware is in ROM, not main_ram. Skip.
    fw_wb_offset = 0  # Firmware not in main_ram; ROM is separate
    # (main_ram is initialized via $readmemh, ROM via separate sim_rom.init)

    # 2. Blob header at BLOB_BASE
    blob_offset = ((BLOB_BASE - MAIN_RAM_BASE) // 4) & 0xFFFFFF  # array index for main_ram
    for i, w in enumerate(td_words[:3]):  # 3-word blob header
        ram[blob_offset + i] = w

    # 3. Scatter layer payloads to DDR addresses
    #    main_ram array index = (ddr_byte_addr - MAIN_RAM_BASE) // 4
    for l, layer in enumerate(layers):
        hdr = layer['header']
        n_wgt = hdr[0]
        n_param = hdr[1]
        n_input = hdr[2]
        n_output = hdr[3]

        # Weights at ddr_wgt_addr
        if layer['ddr_wgt'] and n_wgt > 0:
            base = ((layer['ddr_wgt'] - MAIN_RAM_BASE) // 4) & 0xFFFFFF
            for i, w in enumerate(layer['wgt']):
                ram[base + i] = w

        # Params at ddr_param_addr
        if layer['ddr_param'] and n_param > 0:
            base = ((layer['ddr_param'] - MAIN_RAM_BASE) // 4) & 0xFFFFFF
            for i, w in enumerate(layer['param']):
                ram[base + i] = w

        # Input at ddr_in_addr (layer 0 only)
        if layer['ddr_in'] and n_input > 0:
            base = ((layer['ddr_in'] - MAIN_RAM_BASE) // 4) & 0xFFFFFF
            for i, w in enumerate(layer['input']):
                ram[base + i] = w

        # Golden output at ddr_out_addr
        if layer['ddr_out'] and n_output > 0:
            base = ((layer['ddr_out'] - MAIN_RAM_BASE) // 4) & 0xFFFFFF
            for i, w in enumerate(layer['output']):
                ram[base + i] = w

    # 4. Place the entire blob (headers + inline payloads) at BLOB_BASE
    #    so firmware can parse layer_entry_t headers and read golden outputs.
    blob_offset_full = (BLOB_BASE // 4) & 0xFFFFFF
    for i, w in enumerate(td_words):
        ram[blob_offset_full + i] = w

    # 5. Write @address hex format for $readmemh
    with open(output_init, 'w') as f:
        prev_addr = -1
        for addr in sorted(ram.keys()):
            if addr != prev_addr + 1:
                f.write(f'@{addr:08X}\n')
            f.write(f'{ram[addr]:08X}\n')
            prev_addr = addr

    max_addr = max(ram.keys())
    print(f"Output: {output_init} ({len(ram)} non-zero words, "
          f"max addr 0x{MAIN_RAM_BASE + max_addr*4:08X})")


if __name__ == '__main__':
    main()
