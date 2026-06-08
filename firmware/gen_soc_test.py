#!/usr/bin/env python3
"""
Generate SoC firmware test data from DMA E2E golden data.

Reads golden .npy files and metadata.json for both INT8 and INT16
MobileNetV2-Tiny models, produces:
  - test_data.hex     — Verilog hex init for main RAM ($readmemh)
  - soc_test_data.h   — C header with blob format and layer offsets
  - soc_test_main.c   — Generated firmware main() replacement (optional)

Memory layout in main RAM (0x40000000):
  0x40000000 — blob header + configs
  0x40000100 — INT8 test data
  0x40010000 — INT16 test data
  0x40020000 — NPU WGT workspace (reused per layer)
  0x40024000 — NPU PARAM workspace
  0x40028000 — NPU INPUT workspace
  0x4002C000 — NPU OUTPUT workspace

SPDX-License-Identifier: Apache-2.0
"""

import json
import os
import struct
import sys
import numpy as np

# Add rtl/tb to path for golden generator imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SOC_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'rtl', 'tb'))
from gen_dma_e2e_golden import gen_pooling_test, gen_resize_test, gen_deconv_test, gen_concat_test, gen_add_test

rtl_tb_dir = os.path.join(PROJECT_ROOT, 'rtl', 'tb')
sys.path.insert(0, rtl_tb_dir)
from gen_full_model_golden import build_allops_mini

# ── Paths ──
GOLDEN_DIR = os.path.join(PROJECT_ROOT, 'rtl', 'tb', 'golden_dma_e2e')

OUTPUT_DIR = SCRIPT_DIR
BLOB_FILE = os.path.join(OUTPUT_DIR, 'test_data.bin')
HEADER_FILE = os.path.join(OUTPUT_DIR, 'soc_test_data.h')

# ── Memory map ──
MAIN_RAM_BASE = 0x40000000
BLOB_INT8_BASE = 0x40002000   # 8KB into main RAM (past firmware)
# INT16_BASE will be computed based on actual INT8 blob size
BLOB_INT16_BASE = None  # computed below
NPU_WGT_BASE = 0x40028000
NPU_PARAM_BASE = 0x4002C000
NPU_INPUT_BASE = 0x40030000
NPU_OUTPUT_BASE = 0x40034000

# ── Binary blob format ──
MAGIC = 0x4E505532  # "NPU2"

# Per-layer blob entry size (19 uint32_t header words per layer)
LAYER_ENTRY_HDR_WORDS = 19


def pack_u32(v):
    """Pack uint32 to little-endian bytes."""
    return struct.pack('<I', v & 0xFFFFFFFF)


def load_golden(mode):
    """Load golden data from .npy files."""
    d = os.path.join(GOLDEN_DIR, mode)
    meta = json.load(open(os.path.join(d, 'metadata.json')))
    data = []
    for i in range(len(meta)):
        prefix = f'layer_{i:02d}'
        data.append({
            'wgt': np.load(os.path.join(d, f'{prefix}_wgt.npy')),
            'param': np.load(os.path.join(d, f'{prefix}_param.npy')),
            'input': np.load(os.path.join(d, f'{prefix}_input.npy')),
            'output': np.load(os.path.join(d, f'{prefix}_output.npy')),
        })
    return meta, data


def build_blob(meta, layer_data):
    """Build binary blob bytes for one test case."""
    buf = bytearray()
    buf += pack_u32(MAGIC)
    buf += pack_u32(len(meta))
    buf += pack_u32(1)  # version

    for i, m in enumerate(meta):
        d = layer_data[i]
        wgt_words = np.asarray(d['wgt'], dtype=np.uint32)
        param_words = np.asarray(d['param'], dtype=np.uint32)
        input_words = np.asarray(d['input'], dtype=np.uint32)
        output_words = np.asarray(d['output'], dtype=np.uint32)

        # Entry header (16 words)
        buf += pack_u32(len(wgt_words))
        buf += pack_u32(len(param_words))
        buf += pack_u32(len(input_words))
        buf += pack_u32(len(output_words))
        buf += pack_u32(m['op_type'])
        buf += pack_u32(m['data_type'])
        buf += pack_u32(m['in_h'] | (m['in_w'] << 16))
        buf += pack_u32(m['in_c'])
        buf += pack_u32(m['out_h'] | (m['out_w'] << 16))
        buf += pack_u32(m['out_c'])
        buf += pack_u32(m['kernel_h'] | (m['kernel_w'] << 8)
                        | (m.get('dilation_h', 1) << 16) | (m.get('dilation_w', 1) << 24))
        buf += pack_u32(m['stride_h'] | (m['stride_w'] << 8))
        buf += pack_u32(m['pad_top'] | (m.get('pad_bottom', m['pad_top']) << 8)
                        | (m['pad_left'] << 16) | (m.get('pad_right', m['pad_left']) << 24))
        buf += pack_u32(m['post_ctrl'])
        buf += pack_u32(m['dma_param_count'])
        buf += pack_u32(m['dma_in_size'])
        buf += pack_u32(m['dma_wgt_size'])
        buf += pack_u32(m['dma_out_size'])
        buf += pack_u32(m.get('pool_cfg', 0)      # word [18]: operator-specific
                        | m.get('resize_cfg', 0)
                        | m.get('deconv_cfg', 0)
                        | m.get('concat_cfg', 0))

        # Data payload
        buf += wgt_words.tobytes()
        buf += param_words.tobytes()
        buf += input_words.tobytes()
        buf += output_words.tobytes()
        if 'input_b' in d:
            buf += np.asarray(d['input_b'], dtype=np.uint32).tobytes()

    # Pad to 4-byte boundary
    while len(buf) % 4 != 0:
        buf.append(0)

    return bytes(buf)


def bytes_to_hex_lines(data, words_per_line=1):
    """Convert bytes to Verilog hex init format (one uint32 per line)."""
    lines = []
    for i in range(0, len(data), 4):
        word = struct.unpack('<I', data[i:i+4])[0]
        lines.append(f'{word:08X}')
    return '\n'.join(lines)


def generate_pooling_test_data():
    """Generate 4 Pooling test cases: Max INT8, Avg INT8, Global Avg INT8, Max INT16.

    Returns list of (name, [meta], [data]) tuples, each a single-layer test.
    """
    tests = []

    def _add(name, meta, raw_data):
        data = {
            'wgt': np.array([], dtype=np.uint32),
            'param': raw_data['param_words'],
            'input': raw_data['input_words'],
            'output': raw_data['output_words'],
        }
        tests.append((name, [meta], [data]))

    # Test 0: MaxPool 2x2 stride 2, INT8 (4x4x8 -> 2x2x8)
    meta, raw = gen_pooling_test(mode='max', pool_h=2, pool_w=2,
                                 pool_sh=2, pool_sw=2,
                                 in_h=4, in_w=4, in_c=8,
                                 global_pool=False, int16_mode=False, seed=42)
    _add('pool_max_int8', meta, raw)

    # Test 1: AvgPool 2x2 stride 2, INT8 (4x4x8 -> 2x2x8)
    meta, raw = gen_pooling_test(mode='avg', pool_h=2, pool_w=2,
                                 pool_sh=2, pool_sw=2,
                                 in_h=4, in_w=4, in_c=8,
                                 global_pool=False, int16_mode=False, seed=43)
    _add('pool_avg_int8', meta, raw)

    # Test 2: Global AvgPool, INT8 (4x4x4 -> 1x1x4)
    meta, raw = gen_pooling_test(mode='avg', pool_h=4, pool_w=4,
                                 pool_sh=4, pool_sw=4,
                                 in_h=4, in_w=4, in_c=4,
                                 global_pool=True, int16_mode=False, seed=44)
    _add('pool_global_int8', meta, raw)

    # Test 3: MaxPool 2x2 stride 2, INT16 (4x4x8 -> 2x2x8)
    meta, raw = gen_pooling_test(mode='max', pool_h=2, pool_w=2,
                                 pool_sh=2, pool_sw=2,
                                 in_h=4, in_w=4, in_c=8,
                                 global_pool=False, int16_mode=True, seed=45)
    _add('pool_max_int16', meta, raw)

    return tests


def generate_resize_test_data():
    """Generate 2 Resize test cases: nearest INT8, bilinear INT16.

    Returns list of (name, [meta], [data]) tuples, each a single-layer test.
    """
    tests = []

    def _add(name, meta, raw_data):
        data = {
            'wgt': np.array([], dtype=np.uint32),
            'param': raw_data['param_words'],
            'input': raw_data['input_words'],
            'output': raw_data['output_words'],
        }
        tests.append((name, [meta], [data]))

    # Test 0: Nearest resize INT8 (4x4x4 -> 8x8x4)
    meta, raw = gen_resize_test(in_h=4, in_w=4, in_c=4,
                                out_h=8, out_w=8,
                                resize_mode=0, int16_mode=False, seed=120)
    _add('resize_nearest_int8', meta, raw)

    # Test 1: Bilinear resize INT16 (4x4x4 -> 8x8x4)
    meta, raw = gen_resize_test(in_h=4, in_w=4, in_c=4,
                                out_h=8, out_w=8,
                                resize_mode=1, int16_mode=True, seed=121)
    _add('resize_bilinear_int16', meta, raw)

    return tests


def generate_deconv_test_data():
    """Generate 2 Deconv test cases: INT8, INT16.

    Returns list of (name, [meta], [data]) tuples, each a single-layer test.
    """
    tests = []

    def _add(name, meta, raw_data):
        data = {
            'wgt': raw_data['wgt_words'],
            'param': raw_data['param_words'],
            'input': raw_data['input_words'],
            'output': raw_data['output_words'],
        }
        tests.append((name, [meta], [data]))

    # Test 0: Deconv 2x2, insert_h=1, INT8 (4x4x4 -> 7x7x4)
    meta, raw = gen_deconv_test(in_h=4, in_w=4, in_c=4, out_c=4,
                                kernel_h=2, kernel_w=2, insert_h=1, insert_w=1,
                                pad_top=0, pad_left=0,
                                int16_mode=False, seed=200)
    _add('deconv_int8', meta, raw)

    # Test 1: Deconv 3x3, insert_h=1, INT16 (4x4x4 -> 9x9x4)
    meta, raw = gen_deconv_test(in_h=4, in_w=4, in_c=4, out_c=4,
                                kernel_h=3, kernel_w=3, insert_h=1, insert_w=1,
                                pad_top=1, pad_left=1,
                                int16_mode=True, seed=201)
    _add('deconv_int16', meta, raw)

    return tests


def generate_concat_test_data():
    """Generate 2 Concat test cases: INT8, INT16 (single branch = rescale only).

    Returns list of (name, [meta], [data]) tuples.
    """
    tests = []

    def _add(name, branch_results, output_words, total_c, dtype):
        meta, raw_data = branch_results[0]
        meta['dma_param_count'] = 1
        meta['n_output_words'] = len(output_words)
        # Recompute dma_out_size for packed output
        meta['dma_out_size'] = len(output_words) * 4
        data = {
            'wgt': np.array([], dtype=np.uint32),
            'param': np.array(raw_data['add_param_words'], dtype=np.uint32),
            'input': raw_data['input_words'],
            'output': output_words,
        }
        tests.append((name, [meta], [data]))

    # Test 0: Concat INT8 (single branch, 4x4x4 → 4x4x4, rescale + relu)
    branch_results, output_words, total_c = gen_concat_test(
        h=4, w=4, branches=[{'in_c': 4}], relu=True,
        int16_mode=False, seed=200)
    _add('concat_int8', branch_results, output_words, total_c, 'int8')

    # Test 1: Concat INT16 (single branch, 4x4x4 → 4x4x4, rescale + relu)
    branch_results, output_words, total_c = gen_concat_test(
        h=4, w=4, branches=[{'in_c': 4}], relu=True,
        int16_mode=True, seed=201)
    _add('concat_int16', branch_results, output_words, total_c, 'int16')

    return tests


def generate_add_test_data():
    """Generate 2 Eltwise Add test cases: INT8, INT16.

    Returns list of (name, [meta], [data]) tuples.
    """
    tests = []

    def _add(name, meta, raw_data):
        meta['dma_param_count'] = 1
        data = {
            'wgt': np.array([], dtype=np.uint32),
            'param': np.array(raw_data['add_param_words'], dtype=np.uint32),
            'input': raw_data['input_a_words'],
            'input_b': raw_data['input_b_words'],
            'output': raw_data['output_words'],
        }
        tests.append((name, [meta], [data]))

    # Test 0: Eltwise Add INT8 (4x4x8, relu)
    meta, raw = gen_add_test(h=4, w=4, c=8, relu=True, int16_mode=False, seed=100)
    _add('add_int8', meta, raw)

    # Test 1: Eltwise Add INT16 (4x4x8, relu)
    meta, raw = gen_add_test(h=4, w=4, c=8, relu=True, int16_mode=True, seed=101)
    _add('add_int16', meta, raw)

    return tests


def convert_allops_invocations(invocations):
    """Convert AllOps-Mini invocations to (meta_list, data_list) for build_blob()."""
    metas, datas = [], []
    for inv in invocations:
        meta = dict(inv['meta'])
        data = inv['data']
        op = meta['op_type']

        converted = {
            'wgt': np.array(data.get('wgt_words', []), dtype=np.uint32),
            'param': np.array([], dtype=np.uint32),
            'input': np.array(data['input_words'], dtype=np.uint32),
            'output': np.array(data['output_words'], dtype=np.uint32),
        }

        if op in (4, 7):  # Add, Concat: use add_param_words
            converted['param'] = np.array(data.get('add_param_words', []), dtype=np.uint32)
            meta['dma_param_count'] = 1
        else:
            converted['param'] = np.array(data.get('param_words', []), dtype=np.uint32)

        if op == 4 and 'input_b_words' in data:
            converted['input_b'] = np.array(data['input_b_words'], dtype=np.uint32)

        # Compute DMA sizes
        meta['dma_in_size'] = len(converted['input']) * 4
        meta['dma_wgt_size'] = len(converted['wgt']) * 4

        # Concat: use dma_out_size_override (both branches DMA the full combined output)
        if op == 7 and 'dma_out_size_override' in inv['meta']:
            meta['dma_out_size'] = inv['meta']['dma_out_size_override']
        else:
            meta['dma_out_size'] = len(converted['output']) * 4

        metas.append(meta)
        datas.append(converted)
    return metas, datas


def generate():
    print('=== SoC Test Data Generator ===')
    print(f'Golden source: {GOLDEN_DIR}')

    # Load MobileNet models
    meta_int8, data_int8 = load_golden('int8')
    meta_int16, data_int16 = load_golden('int16')

    print(f'INT8:  {len(meta_int8)} layers')
    print(f'INT16: {len(meta_int16)} layers')

    # Build MobileNet blobs
    blob_int8 = build_blob(meta_int8, data_int8)
    blob_int16 = build_blob(meta_int16, data_int16)

    print(f'INT8 blob:  {len(blob_int8)} bytes')
    print(f'INT16 blob: {len(blob_int16)} bytes')

    # Generate pool test data
    pool_tests = generate_pooling_test_data()
    pool_blobs = {}
    for name, meta, data in pool_tests:
        pool_blobs[name] = build_blob(meta, data)
        print(f'{name} blob:  {len(pool_blobs[name])} bytes')

    # Generate resize test data
    resize_tests = generate_resize_test_data()
    resize_blobs = {}
    for name, meta, data in resize_tests:
        resize_blobs[name] = build_blob(meta, data)
        print(f'{name} blob:  {len(resize_blobs[name])} bytes')

    # Generate deconv test data
    deconv_tests = generate_deconv_test_data()
    deconv_blobs = {}
    for name, meta, data in deconv_tests:
        deconv_blobs[name] = build_blob(meta, data)
        print(f'{name} blob:  {len(deconv_blobs[name])} bytes')

    # Generate concat test data
    concat_tests = generate_concat_test_data()
    concat_blobs = {}
    for name, meta, data in concat_tests:
        concat_blobs[name] = build_blob(meta, data)
        print(f'{name} blob:  {len(concat_blobs[name])} bytes')

    # Generate add test data
    add_tests = generate_add_test_data()
    add_blobs = {}
    for name, meta, data in add_tests:
        add_blobs[name] = build_blob(meta, data)
        print(f'{name} blob:  {len(add_blobs[name])} bytes')

    # Generate AllOps-Mini full model (18 layers, all 7 ops) INT8 + INT16
    print('Building AllOps-Mini INT8...')
    invocations_int8 = build_allops_mini(int16_mode=False)
    meta_allops_int8, data_allops_int8 = convert_allops_invocations(invocations_int8)
    blob_allops_int8 = build_blob(meta_allops_int8, data_allops_int8)
    print(f'AllOps-Mini INT8 blob: {len(blob_allops_int8)} bytes ({len(meta_allops_int8)} layers)')

    print('Building AllOps-Mini INT16...')
    invocations_int16 = build_allops_mini(int16_mode=True)
    meta_allops_int16, data_allops_int16 = convert_allops_invocations(invocations_int16)
    blob_allops_int16 = build_blob(meta_allops_int16, data_allops_int16)
    print(f'AllOps-Mini INT16 blob: {len(blob_allops_int16)} bytes ({len(meta_allops_int16)} layers)')

    # Compute addresses
    blob_int8_bytes = len(blob_int8)
    blob_int8_aligned = (blob_int8_bytes + 3) & ~3
    global BLOB_INT16_BASE
    BLOB_INT16_BASE = BLOB_INT8_BASE + blob_int8_aligned
    print(f'INT16 blob @ 0x{BLOB_INT16_BASE:08X}')

    blob_int16_aligned = (len(blob_int16) + 3) & ~3
    pool_base = BLOB_INT16_BASE + blob_int16_aligned
    pool_addrs = {}
    for name in ['pool_max_int8', 'pool_avg_int8', 'pool_global_int8', 'pool_max_int16']:
        pool_addrs[name] = pool_base
        print(f'{name} blob @ 0x{pool_base:08X}')
        pool_base += (len(pool_blobs[name]) + 3) & ~3

    resize_addrs = {}
    for name in ['resize_nearest_int8', 'resize_bilinear_int16']:
        resize_addrs[name] = pool_base
        print(f'{name} blob @ 0x{pool_base:08X}')
        pool_base += (len(resize_blobs[name]) + 3) & ~3

    deconv_addrs = {}
    for name in ['deconv_int8', 'deconv_int16']:
        deconv_addrs[name] = pool_base
        print(f'{name} blob @ 0x{pool_base:08X}')
        pool_base += (len(deconv_blobs[name]) + 3) & ~3

    concat_addrs = {}
    for name in ['concat_int8', 'concat_int16']:
        concat_addrs[name] = pool_base
        print(f'{name} blob @ 0x{pool_base:08X}')
        pool_base += (len(concat_blobs[name]) + 3) & ~3

    add_addrs = {}
    for name in ['add_int8', 'add_int16']:
        add_addrs[name] = pool_base
        print(f'{name} blob @ 0x{pool_base:08X}')
        pool_base += (len(add_blobs[name]) + 3) & ~3

    # AllOps-Mini blobs
    allops_int8_base = pool_base
    print(f'AllOps-Mini INT8 blob @ 0x{allops_int8_base:08X}')
    pool_base += (len(blob_allops_int8) + 3) & ~3

    allops_int16_base = pool_base
    print(f'AllOps-Mini INT16 blob @ 0x{allops_int16_base:08X}')
    pool_base += (len(blob_allops_int16) + 3) & ~3

    # Write concatenated test_data.bin
    with open(BLOB_FILE, 'wb') as f:
        f.write(blob_int8)
        f.write(blob_int16)
        for name in ['pool_max_int8', 'pool_avg_int8', 'pool_global_int8', 'pool_max_int16']:
            f.write(pool_blobs[name])
        for name in ['resize_nearest_int8', 'resize_bilinear_int16']:
            f.write(resize_blobs[name])
        for name in ['deconv_int8', 'deconv_int16']:
            f.write(deconv_blobs[name])
        for name in ['concat_int8', 'concat_int16']:
            f.write(concat_blobs[name])
        for name in ['add_int8', 'add_int16']:
            f.write(add_blobs[name])
        f.write(blob_allops_int8)
        f.write(blob_allops_int16)
    print(f'Generated {BLOB_FILE} ({os.path.getsize(BLOB_FILE)} bytes)')

    # Generate C header
    generate_header(meta_int8, meta_int16, blob_int8_bytes, pool_addrs, resize_addrs,
                    deconv_addrs, concat_addrs, add_addrs, allops_int8_base, allops_int16_base)

    print(f'\nDone. Next: make && cd ../sim && make run')


def generate_header(meta_int8, meta_int16, blob_int8_size, pool_addrs=None, resize_addrs=None, deconv_addrs=None, concat_addrs=None, add_addrs=None, allops_int8_base=None, allops_int16_base=None):
    """Generate C header with blob addresses and sizes."""
    h = []
    h.append('/* Auto-generated by gen_soc_test.py — do not edit */')
    h.append('#ifndef SOC_TEST_DATA_H')
    h.append('#define SOC_TEST_DATA_H')
    h.append('#include <stdint.h>')
    h.append('')

    # Blob address macros
    h.append('/* Blob base addresses in main RAM */')
    h.append(f'#define BLOB_INT8_BASE  0x{BLOB_INT8_BASE:08X}')
    h.append(f'#define BLOB_INT16_BASE 0x{BLOB_INT16_BASE:08X}')
    if pool_addrs:
        for name, addr in pool_addrs.items():
            define_name = 'BLOB_' + name.upper() + '_BASE'
            h.append(f'#define {define_name}  0x{addr:08X}')
    if resize_addrs:
        for name, addr in resize_addrs.items():
            define_name = 'BLOB_' + name.upper() + '_BASE'
            h.append(f'#define {define_name}  0x{addr:08X}')
    if deconv_addrs:
        for name, addr in deconv_addrs.items():
            define_name = 'BLOB_' + name.upper() + '_BASE'
            h.append(f'#define {define_name}  0x{addr:08X}')
    if concat_addrs:
        for name, addr in concat_addrs.items():
            define_name = 'BLOB_' + name.upper() + '_BASE'
            h.append(f'#define {define_name}  0x{addr:08X}')
    if add_addrs:
        for name, addr in add_addrs.items():
            define_name = 'BLOB_' + name.upper() + '_BASE'
            h.append(f'#define {define_name}  0x{addr:08X}')
    if allops_int8_base:
        h.append(f'#define BLOB_ALLOPS_INT8_BASE  0x{allops_int8_base:08X}')
    if allops_int16_base:
        h.append(f'#define BLOB_ALLOPS_INT16_BASE  0x{allops_int16_base:08X}')
    h.append('')

    # NPU DMA workspace addresses
    h.append('/* NPU DMA workspace addresses in main RAM */')
    h.append(f'#define NPU_WGT_BASE     0x{NPU_WGT_BASE:08X}')
    h.append(f'#define NPU_PARAM_BASE   0x{NPU_PARAM_BASE:08X}')
    h.append(f'#define NPU_INPUT_BASE   0x{NPU_INPUT_BASE:08X}')
    h.append(f'#define NPU_OUTPUT_BASE  0x{NPU_OUTPUT_BASE:08X}')
    h.append('')

    # Per-entry header layout (16 uint32 words)
    h.append('/* Per-layer blob entry header layout (16 uint32 words per layer) */')
    h.append('typedef struct {')
    h.append('    uint32_t n_wgt;          /* [0]  weight word count */')
    h.append('    uint32_t n_param;        /* [1]  param word count */')
    h.append('    uint32_t n_input;        /* [2]  input word count */')
    h.append('    uint32_t n_output;       /* [3]  output word count */')
    h.append('    uint32_t op_type;        /* [4]  operator type */')
    h.append('    uint32_t data_type;      /* [5]  INT8=0, INT16=1 */')
    h.append('    uint32_t in_hw;          /* [6]  in_h | (in_w << 16) */')
    h.append('    uint32_t in_c;           /* [7]  input channels */')
    h.append('    uint32_t out_hw;         /* [8]  out_h | (out_w << 16) */')
    h.append('    uint32_t out_c;          /* [9]  output channels */')
    h.append('    uint32_t kernel_dil;     /* [10] kh | (kw<<8) | (dh<<16) | (dw<<24) */')
    h.append('    uint32_t stride;         /* [11] sh | (sw << 8) */')
    h.append('    uint32_t padding;        /* [12] top | (bot<<8) | (left<<16) | (right<<24) */')
    h.append('    uint32_t post_ctrl;      /* [13] PPU post-processing control */')
    h.append('    uint32_t param_count;    /* [14] per-channel param count */')
    h.append('    uint32_t dma_in_size;    /* [15] DMA input transfer size (bytes) */')
    h.append('    uint32_t dma_wgt_size;   /* [16] DMA weight transfer size (bytes) */')
    h.append('    uint32_t dma_out_size;   /* [17] DMA output transfer size (bytes) */')
    h.append('    uint32_t cfg_aux;        /* [18] operator-specific config (pool_cfg etc) */')
    h.append('    /* Data follows: wgt[n_wgt*4], param[n_param*4], input[n_input*4], output[n_output*4] */')
    h.append('} __attribute__((packed)) layer_entry_t;')
    h.append('')

    # Header sizes
    h.append(f'#define LAYER_ENTRY_HDR_SIZE  {LAYER_ENTRY_HDR_WORDS * 4}')
    h.append('')

    # Blob header macros
    h.append('/* Blob header offsets */')
    h.append('#define BLOB_OFF_MAGIC      0')
    h.append('#define BLOB_OFF_LAYERS    4')
    h.append('#define BLOB_OFF_VERSION   8')
    h.append('#define BLOB_OFF_DATA      12  /* first layer entry starts here */')
    h.append('')

    # Blob magic
    h.append(f'#define BLOB_MAGIC 0x{MAGIC:08X}')
    h.append('')

    h.append('#endif /* SOC_TEST_DATA_H */')
    h.append('')

    with open(HEADER_FILE, 'w') as f:
        f.write('\n'.join(h))
    print(f'Generated {HEADER_FILE}')


if __name__ == '__main__':
    generate()
