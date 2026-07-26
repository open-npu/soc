#!/usr/bin/env python3
"""
Generate SoC firmware test data for chained model inference.

Reads golden .npy files and metadata.json for model_a/b/c/d INT16,
produces:
  - test_data.bin     — binary blob with 36-word layer_entry_t headers
  - soc_test_data.h   — C header with blob format and model base address

Chained inference: Layer N output in DDR becomes Layer N+1 input.
Only layer 0 has inline input data; all layers have golden output for verification.

Usage:
  python3 gen_soc_test.py --model model_b_int16
  python3 gen_soc_test.py --model model_d_int16

SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import struct
import sys
import argparse
import numpy as np

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SOC_DIR)
GOLDEN_DIR = os.path.join(PROJECT_ROOT, 'rtl', 'tb', 'golden', 'golden_dma_e2e')

OUTPUT_DIR = SCRIPT_DIR
BLOB_FILE = os.path.join(OUTPUT_DIR, 'test_data.bin')
HEADER_FILE = os.path.join(OUTPUT_DIR, 'soc_test_data.h')

# ── Memory map ──
MAIN_RAM_BASE = 0x40000000
BLOB_BASE = 0x40010000      # blob header + layer entries + inline data at 64KB offset
MODEL_DDR_BASE = 0x40200000  # model DDR data (wgt/param/in/out) at 2MB offset
DDR_REMAP_OFFSET = MODEL_DDR_BASE - 0x30000000  # metadata 0x30000000 → 0x40200000

# ── Binary blob format ──
MAGIC = 0x4E505532  # "NPU2"
LAYER_ENTRY_HDR_WORDS = 36
LAYER_ENTRY_HDR_SIZE = LAYER_ENTRY_HDR_WORDS * 4  # 144 bytes


def pack_u32(v):
    """Pack uint32 to little-endian bytes."""
    return struct.pack('<I', v & 0xFFFFFFFF)


def load_model_golden(model_name):
    """Load model INT16 golden data from .npy files."""
    d = os.path.join(GOLDEN_DIR, model_name)
    with open(os.path.join(d, 'metadata.json')) as f:
        meta = json.load(f)
    data = []
    for i in range(len(meta)):
        prefix = f'layer_{i:02d}'
        entry = {
            'wgt': np.load(os.path.join(d, f'{prefix}_wgt.npy')),
            'param': np.load(os.path.join(d, f'{prefix}_param.npy')),
            'input': np.load(os.path.join(d, f'{prefix}_input.npy')),
            'output': np.load(os.path.join(d, f'{prefix}_output.npy')),
        }
        # Check for input_b (Add layers)
        input_b_path = os.path.join(d, f'{prefix}_input_b.npy')
        if os.path.exists(input_b_path):
            entry['input_b'] = np.load(input_b_path)
        data.append(entry)
    return meta, data


def remap_ddr(addr):
    """Remap metadata DDR address (0x30XXXXXX) to SoC main RAM (0x40100000+)."""
    if addr == 0:
        return 0
    return (addr & 0x0FFFFFFF) + MODEL_DDR_BASE


def build_chained_blob(meta, data, standalone_layer=-1):
    """Build binary blob bytes for chained model inference.

    Each layer has 36-word header + wgt + param + input(L0 only) + golden output.
    DDR addresses are remapped from 0x30XXXXXX to 0x40XXXXXX.
    """
    buf = bytearray()
    buf += pack_u32(MAGIC)
    buf += pack_u32(len(meta))
    buf += pack_u32(1)  # version

    for i, (m, d) in enumerate(zip(meta, data)):
        # Build 36-word header
        wgt = d['wgt']
        param = d['param']
        inp = d['input']
        out = d['output']

        n_wgt = len(wgt)
        n_param = len(param)
        n_input = len(inp) if (i == 0 or i == standalone_layer) else 0
        n_output = len(out)

        # Remap DDR addresses
        ddr_wgt = remap_ddr(m.get('ddr_wgt_addr', 0))
        ddr_param = remap_ddr(m.get('ddr_param_addr', 0))
        ddr_out = remap_ddr(m.get('ddr_out_addr', 0))
        ddr_in = remap_ddr(m.get('ddr_in_addr', 0)) if (i == 0 or i == standalone_layer) else 0
        ddr_add_b = remap_ddr(m.get('ddr_add_b_addr', 0))

        # Pack fields
        in_hw = m['in_h'] | (m['in_w'] << 16)
        out_hw = m['out_h'] | (m['out_w'] << 16)
        kernel_dil = m['kernel_h'] | (m['kernel_w'] << 8)
        stride = m['stride_h'] | (m['stride_w'] << 8)
        padding = m.get('pad_top', 0) | (m.get('pad_left', 0) << 8)

        tile_cfg = m.get('tile_h', 0) | (m.get('tile_w', 0) << 16)
        tile_count = m.get('tile_num_h', 1) | (m.get('tile_num_w', 1) << 16)
        sched_ctrl = m.get('sched_ctrl', 0)
        store_mode = m.get('store_mode', 0)
        tile_in_size = m.get('tile_in_size', 0)
        tile_out_size = m.get('tile_out_size', 0)
        row_cfg = m.get('row_cfg', 0)
        wgt_per_oc = m.get('wgt_per_oc_words', 0)
        clamp_max = m.get('clamp_max', 32767)
        in_zp = m.get('in_zp', 0)
        input_src = m.get('input_src', -1)
        residual_src = m.get('residual_src', -1)

        # cfg_aux: operator-specific config
        cfg_aux = 0
        if m['op_type'] == 3:  # Pool
            cfg_aux = m.get('pool_cfg', 0)
        elif m['op_type'] == 5:  # Resize
            cfg_aux = m.get('resize_cfg', 0)
        elif m['op_type'] == 6:  # Deconv
            cfg_aux = m.get('deconv_cfg', 0)
        elif m['op_type'] == 7:  # Concat
            cfg_aux = m.get('concat_cfg', 0)

        # Write 36-word header
        hdr = [
            n_wgt,              # [0]
            n_param,            # [1]
            n_input,            # [2]
            n_output,           # [3]
            m['op_type'],       # [4]
            m.get('data_type', 1),  # [5]
            in_hw,              # [6]
            m['in_c'],          # [7]
            out_hw,             # [8]
            m['out_c'],         # [9]
            kernel_dil,         # [10]
            stride,             # [11]
            padding,            # [12]
            m.get('post_ctrl', 0),  # [13]
            m.get('dma_param_count', 0),  # [14]
            m.get('dma_in_size', 0),      # [15]
            m.get('dma_wgt_size', 0),     # [16]
            m.get('dma_out_size', 0),     # [17]
            cfg_aux,            # [18]
            tile_cfg,           # [19]
            tile_count,         # [20]
            sched_ctrl,         # [21]
            store_mode,         # [22]
            tile_in_size,       # [23]
            tile_out_size,      # [24]
            row_cfg,            # [25]
            wgt_per_oc,         # [26]
            clamp_max,          # [27]
            in_zp,              # [28]
            ddr_wgt,            # [29]
            ddr_param,          # [30]
            ddr_out,            # [31]
            ddr_in,             # [32]
            ddr_add_b,          # [33]
            input_src & 0xFFFFFFFF,   # [34] -1 → 0xFFFFFFFF
            residual_src & 0xFFFFFFFF,  # [35] -1 → 0xFFFFFFFF
        ]
        for w in hdr:
            buf += pack_u32(w)

        # Payload: wgt + param + input(L0 only) + golden output
        for w in wgt:
            buf += pack_u32(int(w))
        for w in param:
            buf += pack_u32(int(w))
        if i == 0 or i == standalone_layer:
            for w in inp:
                buf += pack_u32(int(w))
        for w in out:
            buf += pack_u32(int(w))

    return bytes(buf)


def generate_header(model_name, blob_size):
    """Generate soc_test_data.h C header."""
    blob_base = BLOB_BASE

    with open(HEADER_FILE, 'w') as f:
        f.write("/* Auto-generated by gen_soc_test.py — do not edit */\n")
        f.write("#ifndef SOC_TEST_DATA_H\n")
        f.write("#define SOC_TEST_DATA_H\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"/* Model: {model_name} */\n")
        f.write(f"#define BLOB_MODEL_BASE  0x{blob_base:08X}\n")
        f.write(f"#define BLOB_MODEL_SIZE  {blob_size}\n\n")

        f.write("/* Per-layer blob entry header (36 uint32 words = 144 bytes) */\n")
        f.write("typedef struct {\n")
        f.write("    uint32_t n_wgt;            /* [0]  */\n")
        f.write("    uint32_t n_param;          /* [1]  */\n")
        f.write("    uint32_t n_input;          /* [2]  */\n")
        f.write("    uint32_t n_output;         /* [3]  */\n")
        f.write("    uint32_t op_type;          /* [4]  */\n")
        f.write("    uint32_t data_type;        /* [5]  */\n")
        f.write("    uint32_t in_hw;            /* [6]  */\n")
        f.write("    uint32_t in_c;             /* [7]  */\n")
        f.write("    uint32_t out_hw;           /* [8]  */\n")
        f.write("    uint32_t out_c;            /* [9]  */\n")
        f.write("    uint32_t kernel_dil;       /* [10] */\n")
        f.write("    uint32_t stride;           /* [11] */\n")
        f.write("    uint32_t padding;          /* [12] */\n")
        f.write("    uint32_t post_ctrl;        /* [13] */\n")
        f.write("    uint32_t param_count;      /* [14] */\n")
        f.write("    uint32_t dma_in_size;      /* [15] */\n")
        f.write("    uint32_t dma_wgt_size;     /* [16] */\n")
        f.write("    uint32_t dma_out_size;     /* [17] */\n")
        f.write("    uint32_t cfg_aux;          /* [18] */\n")
        f.write("    uint32_t tile_cfg;         /* [19] */\n")
        f.write("    uint32_t tile_count;       /* [20] */\n")
        f.write("    uint32_t sched_ctrl;       /* [21] */\n")
        f.write("    uint32_t store_mode;       /* [22] */\n")
        f.write("    uint32_t tile_in_size;     /* [23] */\n")
        f.write("    uint32_t tile_out_size;    /* [24] */\n")
        f.write("    uint32_t row_cfg;          /* [25] */\n")
        f.write("    uint32_t wgt_per_oc_words; /* [26] */\n")
        f.write("    uint32_t clamp_max;        /* [27] */\n")
        f.write("    uint32_t in_zp;            /* [28] */\n")
        f.write("    uint32_t ddr_wgt_addr;     /* [29] */\n")
        f.write("    uint32_t ddr_param_addr;   /* [30] */\n")
        f.write("    uint32_t ddr_out_addr;     /* [31] */\n")
        f.write("    uint32_t ddr_in_addr;      /* [32] */\n")
        f.write("    uint32_t ddr_add_b_addr;   /* [33] */\n")
        f.write("    int32_t  input_src;        /* [34] -1=chain, N=skip */\n")
        f.write("    int32_t  residual_src;     /* [35] -1=none, N=branch B */\n")
        f.write("} __attribute__((packed)) layer_entry_t;\n\n")

        f.write(f"#define LAYER_ENTRY_HDR_SIZE  {LAYER_ENTRY_HDR_SIZE}\n")
        f.write(f"#define LAYER_ENTRY_HDR_WORDS {LAYER_ENTRY_HDR_WORDS}\n\n")

        f.write("/* Blob header */\n")
        f.write("#define BLOB_MAGIC 0x4E505532\n")
        f.write("#define BLOB_OFF_MAGIC   0\n")
        f.write("#define BLOB_OFF_LAYERS  4\n")
        f.write("#define BLOB_OFF_VERSION 8\n")
        f.write("#define BLOB_OFF_DATA    12\n\n")

        f.write("#endif /* SOC_TEST_DATA_H */\n")

    print(f"Generated {HEADER_FILE}")


def main():
    parser = argparse.ArgumentParser(description='Generate SoC test data for chained model inference')
    parser.add_argument('--model', required=True,
                        choices=['model_a_int16', 'model_a_int8',
                                 'model_b_int16', 'model_b_int8',
                                 'model_c_int16', 'model_c_int8',
                                 'model_d_int16', 'model_d_int8',
                                 'model_e_int16', 'model_e_int8'],
                        help='Model to generate')
    parser.add_argument('--standalone-layer', type=int, default=-1,
                        help='Also pack input data for this layer (for standalone testing)')
    args = parser.parse_args()

    print(f"Loading golden data for {args.model}...")
    meta, data = load_model_golden(args.model)
    print(f"  {len(meta)} layers loaded")

    # Check which layers have valid golden data
    valid = sum(1 for d in data if len(d['output']) > 0)
    print(f"  {valid}/{len(meta)} layers have golden output")

    # Filter to only layers with golden data
    # (fused block intermediates may not have golden output)
    # For chained inference, we need ALL layers — skip those without output
    # Actually, we need all layers for chaining, but can only verify those with output
    # For now, include all layers; firmware will skip verification for empty output

    print(f"Building chained blob...")
    blob = build_chained_blob(meta, data, args.standalone_layer)
    print(f"  Blob size: {len(blob)} bytes ({len(blob)/1024/1024:.1f} MB)")

    with open(BLOB_FILE, 'wb') as f:
        f.write(blob)
    print(f"Written {BLOB_FILE}")

    generate_header(args.model, len(blob))

    # Print DDR usage summary
    ddr_addrs = []
    for m in meta:
        for k in ['ddr_in_addr', 'ddr_out_addr', 'ddr_wgt_addr', 'ddr_param_addr', 'ddr_add_b_addr']:
            v = m.get(k, 0)
            if v > 0:
                ddr_addrs.append(remap_ddr(v))
    if ddr_addrs:
        print(f"  DDR range: 0x{min(ddr_addrs):08X} - 0x{max(ddr_addrs):08X} "
              f"({(max(ddr_addrs)-min(ddr_addrs))/1024/1024:.1f} MB)")


if __name__ == '__main__':
    main()
