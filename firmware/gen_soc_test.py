#!/usr/bin/env python3
"""
Generate SoC firmware test data for chained model inference.

Reads golden .npy files and metadata.json for model_a/b/c/d INT16,
produces:
  - test_data.bin     — binary blob with 37-word layer_entry_t headers
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
LAYER_ENTRY_HDR_WORDS = 37
LAYER_ENTRY_HDR_SIZE = LAYER_ENTRY_HDR_WORDS * 4  # 148 bytes

# The row keeps one 64-channel group in the 80KB weight SRAM.
WGT_SRAM_BYTES = 80 * 1024
ROW_LANES = 64
IFM_DEPTH_WORDS = 24 * 256


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
        fuse_input_path = os.path.join(d, f'{prefix}_fuse_input.npy')
        if os.path.exists(fuse_input_path):
            entry['fuse_input'] = np.load(fuse_input_path)
        data.append(entry)
    return meta, data


def enable_conv_pool_strip_fusion(meta, data):
    """Mark L0→L1 for firmware strip-major SRAM fusion."""
    if len(meta) < 2 or 'fuse_input' not in data[0]:
        return False
    conv, pool = meta[0], meta[1]
    if (conv.get('op_type') != 0 or pool.get('op_type') != 3
            or conv.get('out_w') != pool.get('in_w')
            or conv.get('out_c') != pool.get('in_c')):
        raise RuntimeError("layer_00_fuse_input exists but L0→L1 is not Conv→Pool")

    pool_cfg = pool.get('pool_cfg', 0)
    pool_h = (pool_cfg >> 4) & 0xF
    pool_sh = (pool_cfg >> 12) & 0xF
    elem_bytes = 2 if conv.get('data_type', 0) else 1
    pool_strip_h = 0
    for candidate in range(1, pool['out_h'] + 1):
        candidate_producer_h = (candidate - 1) * pool_sh + pool_h
        producer_bytes = (candidate_producer_h * conv['out_w']
                          * conv['out_c'] * elem_bytes)
        pool_out_bytes = (candidate * pool['out_w']
                          * pool['out_c'] * elem_bytes)
        conv_in_h = ((candidate_producer_h - 1) * conv['stride_h']
                     + conv['kernel_h'])
        conv_in_w = ((conv['out_w'] - 1) * conv['stride_w']
                     + conv['kernel_w'])
        conv_in_bytes = conv_in_h * conv_in_w * conv['in_c'] * elem_bytes
        if (producer_bytes > 48 * 1024 or pool_out_bytes > 24 * 1024
                or conv_in_bytes > 24 * 1024):
            break
        pool_strip_h = candidate
    if pool_strip_h == 0:
        raise RuntimeError("L0→L1 has no legal Conv→Pool fusion strip")

    producer_h = (pool_strip_h - 1) * pool_sh + pool_h
    producer_w = conv['out_w']
    strip_count = (pool['out_h'] + pool_strip_h - 1) // pool_strip_h
    slot_bytes = len(data[0]['fuse_input']) * 4 // strip_count
    local_in_h = ((producer_h - 1) * conv['stride_h'] + conv['kernel_h'])
    local_in_w = ((producer_w - 1) * conv['stride_w'] + conv['kernel_w'])
    expected_slot = ((local_in_h * local_in_w * conv['in_c']
                      * elem_bytes + 3) // 4) * 4
    if slot_bytes != expected_slot:
        raise RuntimeError(f"unexpected L0 fuse slot size {slot_bytes}, "
                           f"expected {expected_slot}")

    data[0]['input'] = data[0]['fuse_input']
    conv['sched_ctrl'] = conv.get('sched_ctrl', 0) | (1 << 5)
    # In strip mode these descriptor fields are consumed by firmware, not RTL.
    conv['tile_h'] = producer_h
    conv['tile_w'] = producer_w
    conv['tile_num_h'] = strip_count
    conv['tile_num_w'] = pool_strip_h
    conv['tile_in_size'] = slot_bytes
    print(f"  L0→L1 strip fusion: {strip_count} strips, "
          f"producer={producer_h}x{producer_w}, pool_h={pool_strip_h}, "
          f"input_slot={slot_bytes}B")
    return True


def retile_oversize_untiled_adds(meta):
    """Split untiled Add tensors whose two inputs cannot coexist in IFM."""
    changed = []
    half_words = IFM_DEPTH_WORDS // 2
    for i, m in enumerate(meta):
        if m.get('op_type') != 4 or m.get('tile_h', 0) != 0:
            continue
        eb = 2 if m.get('data_type', 0) else 1
        full_words = (m['in_h'] * m['in_w'] * m['in_c'] * eb + 3) // 4
        if 2 * full_words <= IFM_DEPTH_WORDS:
            continue
        row_words = (m['in_w'] * m['in_c'] * eb + 3) // 4
        tile_h = min(m['out_h'], half_words // (2 * row_words))
        if tile_h == 0:
            raise RuntimeError(f"L{i} Add one row cannot fit DB IFM half")
        m['tile_h'] = tile_h
        m['tile_w'] = m['out_w']
        m['tile_num_h'] = (m['out_h'] + tile_h - 1) // tile_h
        m['tile_num_w'] = 1
        m['tile_in_size'] = tile_h * row_words * 4
        m['tile_out_size'] = tile_h * row_words * 4
        m['row_cfg'] = (tile_h << 16) | row_words
        m['store_mode'] = 1
        m['sched_ctrl'] = (m.get('sched_ctrl', 0) | (1 << 4)) & ~(1 << 5)
        changed.append((i, tile_h, m['out_w']))
    if changed:
        print(f"  Re-tiled oversized Add layers: {changed}")
    return changed


def retile_for_double_buffer(meta, model_name):
    """Shrink oversized tiles so both IFM ping/pong halves are usable."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tools'))
    from tiling import compute_tiling
    from hw_config import DEFAULT_HW

    op_names = {0: 'conv', 1: 'dw', 2: 'fc', 3: 'pool',
                4: 'add', 5: 'resize', 6: 'deconv', 7: 'concat'}
    half_bytes = (IFM_DEPTH_WORDS // 2) * 4
    changed = []
    for i, m in enumerate(meta):
        if i == 0 or m.get('tile_h', 0) == 0:
            continue
        if m.get('op_type') not in (0, 1):
            continue
        if m.get('tile_in_size', 0) <= half_bytes:
            continue
        eb = 2 if m.get('data_type', 0) else 1
        tile = compute_tiling(
            op_type=op_names[m['op_type']],
            in_h=m['in_h'], in_w=m['in_w'], in_c=m['in_c'],
            out_h=m['out_h'], out_w=m['out_w'], out_c=m['out_c'],
            kernel_h=m['kernel_h'], kernel_w=m['kernel_w'],
            stride_h=m['stride_h'], stride_w=m['stride_w'],
            elem_size=eb, double_buffer=True, profile=DEFAULT_HW)
        th, tw = tile['tile_h'], tile['tile_w']
        in_th = (th - 1) * m['stride_h'] + m['kernel_h']
        in_tw = (tw - 1) * m['stride_w'] + m['kernel_w']
        in_words = (in_th * in_tw * m['in_c'] * eb + 3) // 4
        out_words = (th * tw * m['out_c'] * eb + 3) // 4
        if in_words * 4 > half_bytes:
            raise RuntimeError(f"L{i} DB tile input still exceeds IFM half")
        old = (m['tile_h'], m['tile_w'])
        m['tile_h'], m['tile_w'] = th, tw
        m['tile_num_h'] = tile['tile_num_h']
        m['tile_num_w'] = tile['tile_num_w']
        m['tile_in_size'] = in_words * 4
        m['tile_out_size'] = out_words * 4
        m['row_cfg'] = (th << 16) | ((tw * m['out_c'] * eb + 3) // 4)
        m['store_mode'] = 1
        m['sched_ctrl'] = (m.get('sched_ctrl', 0) | (1 << 4)) & ~(1 << 6)
        changed.append((i, old, (th, tw)))
    if changed:
        print(f"  {model_name} DB re-tiles: {changed}")
    return changed


def enable_untiled_sram_fusion(meta, model_name):
    """Fuse safe untiled producer/consumer pairs through activation SRAM."""
    referenced = set()
    for m in meta:
        for key in ('input_src', 'residual_src'):
            if m.get(key, -1) >= 0:
                referenced.add(m[key])

    changed = []
    i = 0
    while i + 1 < len(meta):
        prod, cons = meta[i], meta[i + 1]
        if i + 2 < len(meta):
            end = meta[i + 2]
            end_words = ((end['out_h'] * end['out_w'] * end['out_c']
                          * (2 if end.get('data_type', 0) else 1)) + 3) // 4
            triple = (
                prod.get('op_type') == 0
                and cons.get('op_type') == 1
                and end.get('op_type') == 0
                and prod.get('kernel_h') == 1 and prod.get('kernel_w') == 1
                and end.get('kernel_h') == 1 and end.get('kernel_w') == 1
                and prod.get('tile_h', 0) == 0
                and cons.get('tile_h', 0) == 0
                and end.get('tile_h', 0) == 0
                and i not in referenced and (i + 1) not in referenced
                and prod['out_h'] == cons['in_h']
                and prod['out_w'] == cons['in_w']
                and prod['out_c'] == cons['in_c']
                and cons['out_h'] == end['in_h']
                and cons['out_w'] == end['in_w']
                and cons['out_c'] == end['in_c']
                and end_words <= IFM_DEPTH_WORDS
            )
            if triple:
                prod['sched_ctrl'] = (prod.get('sched_ctrl', 0) & ~0x0E) | (1 << 1)
                cons['sched_ctrl'] = (cons.get('sched_ctrl', 0) & ~0x0E) | (1 << 2)
                end['sched_ctrl'] = (end.get('sched_ctrl', 0) & ~0x0E) | (1 << 3)
                changed.append((i, i + 1, i + 2))
                i += 3
                continue
        eb = 2 if prod.get('data_type', 0) else 1
        prod_words = (prod['out_h'] * prod['out_w'] * prod['out_c'] * eb + 3) // 4
        cons_words = (cons['out_h'] * cons['out_w'] * cons['out_c'] * eb + 3) // 4
        safe = (
            prod.get('op_type') in (0, 1, 3)
            and cons.get('op_type') in (0, 1, 2, 3)
            and prod.get('tile_h', 0) == 0
            and cons.get('tile_h', 0) == 0
            and i not in referenced
            and prod['out_h'] == cons['in_h']
            and prod['out_w'] == cons['in_w']
            and prod['out_c'] == cons['in_c']
            and prod_words <= IFM_DEPTH_WORDS
            and cons_words <= IFM_DEPTH_WORDS
        )
        if safe:
            prod['sched_ctrl'] = (prod.get('sched_ctrl', 0) & ~0x0E) | (1 << 1)
            cons['sched_ctrl'] = (cons.get('sched_ctrl', 0) & ~0x0E) | (1 << 3)
            changed.append((i, i + 1))
            i += 2
        else:
            i += 1
    if changed:
        print(f"  {model_name} SRAM-fused pairs: {changed}")
    return changed


def remap_ddr(addr):
    """Remap metadata DDR address (0x30XXXXXX) to SoC main RAM.

    MODEL_DDR_BASE is set dynamically in main() so the DDR scatter region
    starts AFTER the end of the blob. Otherwise the blob's inline payload
    (which includes the golden outputs firmware compares against) and the
    DDR regions overlap: build_main_ram.py writes both into the same
    main_ram array, and at runtime the NPU's output DMA overwrites the
    golden data the firmware needs.
    """
    if addr == 0:
        return 0
    return (addr & 0x0FFFFFFF) + MODEL_DDR_BASE


def _dw_stream(m):
    """Match npu_ctrl dw_stream: global depthwise whose activation overflows IFM."""
    return (m['op_type'] == 1 and m.get('tile_h', 0) == 0
            and m['kernel_h'] == m['in_h'] and m['kernel_w'] == m['in_w']
            and m['out_h'] == 1 and m['out_w'] == 1
            and (m.get('dma_in_size', 0) >> 2) > IFM_DEPTH_WORDS)


def _words_to_bytes(wgt):
    return np.ascontiguousarray(wgt, dtype=np.uint32).tobytes()


def _bytes_to_words(blob):
    if len(blob) % 4:
        blob = blob + bytes(4 - (len(blob) % 4))
    return np.frombuffer(blob, dtype=np.uint32).copy()


def _row_group_channels(k_depth, elem_bytes):
    """Widest power-of-two OC group whose full K vector fits WGT SRAM."""
    for channels in (64, 32, 16, 8, 4, 2, 1):
        if channels * k_depth * elem_bytes <= WGT_SRAM_BYTES:
            return channels
    return 0


def prepare_row_weights(meta, data):
    """Rewrite OC-major conv, FC, and depthwise blobs into the 64-lane layout.

    Checked-in goldens omit wgt_layout. The row reads K-major, one 64-channel
    group at a time when the whole layer does not fit the weight SRAM.
    FC is a 1x1 conv on that row. Global-pool depthwise stays on the
    streaming engine and keeps its bytes.
    """
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tools'))
    from model_packer import (
        conv_weights_from_blob, conv_weights_to_blob,
        dw_weights_from_blob, dw_weights_to_blob,
        WGT_LAYOUT_OC_MAJOR, WGT_LAYOUT_K_MAJOR,
    )
    for i, (m, d) in enumerate(zip(meta, data)):
        op = m['op_type']
        if op not in (0, 1, 2) or len(d['wgt']) == 0:
            continue
        if op == 1 and _dw_stream(m):
            # The streaming engine reloads contiguous 16-channel blocks where
            # every channel owns KH*KW adjacent elements ([C][KH][KW]).  Fresh
            # converter output can already be K-major ([KH][KW][C]); changing
            # only the descriptor bit would then make every reload interpret
            # the right bytes in the wrong order.
            if m.get('wgt_layout', 0) == WGT_LAYOUT_K_MAJOR:
                es = 2 if m.get('data_type', 0) == 1 else 1
                dtype = np.int16 if es == 2 else np.int8
                kh, kw, oc = m['kernel_h'], m['kernel_w'], m['out_c']
                raw = _words_to_bytes(d['wgt'])
                expect = oc * kh * kw * es
                if len(raw) != expect:
                    raise RuntimeError(
                        f"L{i} DW-stream weight bytes {len(raw)} != {expect}")
                weights = dw_weights_from_blob(
                    raw, WGT_LAYOUT_K_MAJOR, oc, kh, kw, dtype)
                d['wgt'] = _bytes_to_words(
                    dw_weights_to_blob(weights, WGT_LAYOUT_OC_MAJOR))
                print(f"  L{i} depthwise stream K-major -> OC-major")
            else:
                print(f"  L{i} depthwise stream stays OC-major")
            m['wgt_layout'] = WGT_LAYOUT_OC_MAJOR
            m['wgt_per_oc_words'] = 0
            continue
        if m.get('wgt_layout', 0) == WGT_LAYOUT_K_MAJOR:
            # Current converters already emit grouped K-major weights and
            # bin2golden supplies the matching reload stride. Repacking them
            # as if they were legacy OC-major silently corrupts the blob.
            continue
        es = 2 if m.get('data_type', 0) == 1 else 1
        dtype = np.int16 if es == 2 else np.int8
        kh, kw = m['kernel_h'], m['kernel_w']
        ic, oc = m['in_c'], m['out_c']
        raw = _words_to_bytes(d['wgt'])
        if op != 1:
            expect = oc * kh * kw * ic * es
            if len(raw) != expect:
                raise RuntimeError(f"L{i} weight bytes {len(raw)} != {expect}")
            k_depth = kh * kw * ic
            auto_group_ch = _row_group_channels(k_depth, es)
            if auto_group_ch == 0:
                raise RuntimeError(
                    f"L{i} one output channel ({k_depth * es}B) does not fit "
                    f"the {WGT_SRAM_BYTES}B weight SRAM")
            # A one-pixel tile cannot fill spare row lanes with more pixels.
            # Keep 64 output channels and stream its K-major weights in SRAM-
            # sized K slices instead. Bit 31 marks this interpretation of wpo.
            kslice = (len(raw) > WGT_SRAM_BYTES
                      and m.get('tile_h', 0) * m.get('tile_w', 0) <= 1
                      and auto_group_ch < ROW_LANES)
            group_ch = ROW_LANES if kslice else auto_group_ch
            weights = conv_weights_from_blob(
                raw, WGT_LAYOUT_OC_MAJOR, oc, kh, kw, ic, dtype)
            blob = conv_weights_to_blob(
                weights, WGT_LAYOUT_K_MAJOR, lanes=group_ch)
            group_bytes = group_ch * k_depth * es
        else:
            expect = oc * kh * kw * es
            if len(raw) != expect:
                raise RuntimeError(f"L{i} dw weight bytes {len(raw)} != {expect}")
            weights = dw_weights_from_blob(
                raw, WGT_LAYOUT_OC_MAJOR, oc, kh, kw, dtype)
            blob = dw_weights_to_blob(weights, WGT_LAYOUT_K_MAJOR)
            group_ch = ROW_LANES
            group_bytes = ROW_LANES * kh * kw * es
            if len(blob) > WGT_SRAM_BYTES:
                raise RuntimeError(
                    f"L{i} depthwise blob {len(blob)}B does not fit the weight SRAM")
        if len(blob) != expect:
            raise RuntimeError(f"L{i} transposed to {len(blob)} != {expect}")
        if len(blob) <= WGT_SRAM_BYTES:
            packed = blob
            wpo = 0
        else:
            groups = (oc + group_ch - 1) // group_ch
            padded = groups * group_bytes
            packed = blob + bytes(padded - len(blob))
            wpo = group_bytes // 4
            if op != 1 and kslice:
                wpo |= 0x80000000
        d['wgt'] = _bytes_to_words(packed)
        m['wgt_layout'] = 1
        m['wgt_per_oc_words'] = wpo
        if wpo or group_bytes > WGT_SRAM_BYTES:
            print(f"  L{i} op={op} K-major wpo={wpo} words={len(d['wgt'])} "
                  f"group_ch={group_ch} group={group_bytes}B"
                  f"{' K-sliced' if op != 1 and kslice else ''}")


def build_chained_blob(meta, data, standalone_layer=-1):
    """Build binary blob bytes for chained model inference.

    Each layer has 37-word header + wgt + param + input(L0 only) + golden output.
    DDR addresses are remapped from 0x30XXXXXX to 0x40XXXXXX.
    """
    # ─── Re-allocate DDR regions non-overlapping ───
    # The packer (bin2golden.generate_golden + generator Add patching) can leave
    # a layer's OUT region overlapping the NEXT layer's WGT region (e.g. model_d
    # L12 Add out vs L13 Conv wgt, 128-word overlap → L13 loads corrupted weights).
    # The metadata's relative spacing is only safe for standalone (single-layer)
    # use; chained inference needs each layer's regions to be disjoint because a
    # layer's runtime output DMA must not clobber another layer's weights/params.
    # Recompute each region sequentially from MODEL_DDR_BASE with 4KB alignment.
    # ddr_add_b is NOT allocated: the firmware always resolves it at runtime to a
    # producing layer's OUT region (layer_out_addr[residual_src] / concat l-1).
    ALIGN = 4096
    def _align(v):
        return (v + ALIGN - 1) & ~(ALIGN - 1)
    alloc = []  # per-layer (ddr_in, ddr_wgt, ddr_param, ddr_out)
    cur = MODEL_DDR_BASE
    for i, (m, d) in enumerate(zip(meta, data)):
        n_wgt = len(d['wgt']); n_param = len(d['param'])
        n_in  = len(d['input']) if (i == 0 or i == standalone_layer) else 0
        n_out = len(d['output'])
        dma_out = m.get('dma_out_size', 0)
        # Fused intermediates may have empty golden npy but still need a
        # runtime OUT buffer so the next tiled layer can 2D-load them.
        out_bytes = max(n_out * 4, dma_out)
        ddr_in = ddr_wgt = ddr_param = ddr_out = 0
        if n_in > 0:
            ddr_in = cur; cur += _align(n_in * 4)
        if n_wgt > 0:
            ddr_wgt = cur; cur += _align(n_wgt * 4)
        if n_param > 0:
            ddr_param = cur; cur += _align(n_param * 4)
        if out_bytes > 0:
            ddr_out = cur; cur += _align(out_bytes)
        alloc.append((ddr_in, ddr_wgt, ddr_param, ddr_out))

    buf = bytearray()
    buf += pack_u32(MAGIC)
    buf += pack_u32(len(meta))
    buf += pack_u32(1)  # version

    for i, (m, d) in enumerate(zip(meta, data)):
        # Build 37-word header
        wgt = d['wgt']
        param = d['param']
        inp = d['input']
        out = d['output']

        n_wgt = len(wgt)
        n_param = len(param)
        n_input = len(inp) if (i == 0 or i == standalone_layer) else 0
        n_output = len(out)

        # Use the re-allocated non-overlapping DDR addresses
        ddr_in, ddr_wgt, ddr_param, ddr_out = alloc[i]
        # ddr_add_b: firmware overrides with the producing layer's OUT region.
        # Keep the metadata remap only as a harmless fallback (never used for
        # Add/Concat where residual_src/l-1 is always valid).
        ddr_add_b = remap_ddr(m.get('ddr_add_b_addr', 0))

        # Pack fields
        # RTL decodes W from [15:0] and H from [31:16].
        in_hw = m['in_w'] | (m['in_h'] << 16)
        out_hw = m['out_w'] | (m['out_h'] << 16)
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
        # prepare_row_weights sets this for conv and depthwise. A missing
        # key stays 0 (pool, add, concat, and the streaming depthwise engine).
        wgt_layout = m.get('wgt_layout', 0)

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

        # Write 37-word header
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
            wgt_layout,         # [36] 0=OC-major, 1=K-major (64-lane row)
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

        f.write("/* Per-layer blob entry header (37 uint32 words = 148 bytes) */\n")
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
        f.write("    uint32_t wgt_layout;       /* [36] 0=OC-major, 1=K-major */\n")
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


def self_test():
    """Check the binary ABI without requiring model golden files."""
    meta = [{
        'op_type': 0, 'data_type': 1,
        'in_h': 0x1234, 'in_w': 0x5678, 'in_c': 3,
        'out_h': 0x2345, 'out_w': 0x6789, 'out_c': 5,
        'kernel_h': 3, 'kernel_w': 5,
        'stride_h': 1, 'stride_w': 2,
        'pad_top': 3, 'pad_left': 4,
        'post_ctrl': 0x11223344, 'sched_ctrl': 0x55667788,
        'wgt_layout': 1,
    }]
    data = [{
        'wgt': [0xA1], 'param': [0xB2],
        'input': [0xC3], 'output': [0xD4],
    }]
    blob = build_chained_blob(meta, data)
    assert len(blob) == 12 + 37 * 4 + 4 * 4
    magic, layers, version = struct.unpack_from('<III', blob, 0)
    assert (magic, layers, version) == (MAGIC, 1, 1)
    hdr = struct.unpack_from('<37I', blob, 12)
    assert hdr[6] == 0x12345678       # H in high 16, W in low 16
    assert hdr[8] == 0x23456789
    assert hdr[10] == 0x00000503
    assert hdr[11] == 0x00000201
    assert hdr[12] == 0x00000403
    assert hdr[13] == 0x11223344
    assert hdr[21] == 0x55667788
    assert hdr[36] == 1
    assert struct.unpack_from('<4I', blob, 12 + 37 * 4) == (
        0xA1, 0xB2, 0xC3, 0xD4)
    meta[0].pop('wgt_layout')
    blob = build_chained_blob(meta, data)
    hdr = struct.unpack_from('<37I', blob, 12)
    assert hdr[36] == 0
    print('gen_soc_test self-test: PASS')


def main():
    parser = argparse.ArgumentParser(description='Generate SoC test data for chained model inference')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--model',
                      choices=['model_a_int16', 'model_a_int8',
                               'model_b_int16', 'model_b_int8',
                               'model_c_int16', 'model_c_int8',
                               'model_d_int16', 'model_d_int8',
                               'model_e_int16', 'model_e_int8'],
                      help='Model to generate')
    mode.add_argument('--self-test', action='store_true',
                      help='Validate the 37-word binary ABI and exit')
    parser.add_argument('--standalone-layer', type=int, default=-1,
                        help='Also pack input data for this layer (for standalone testing)')
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    print(f"Loading golden data for {args.model}...")
    meta, data = load_model_golden(args.model)
    print(f"  {len(meta)} layers loaded")
    prepare_row_weights(meta, data)
    retile_oversize_untiled_adds(meta)
    if args.model.startswith(('model_a_', 'model_d_')):
        retile_for_double_buffer(meta, args.model)
    if args.model.startswith('model_d_'):
        enable_untiled_sram_fusion(meta, args.model)
    if args.model in ('model_b_int8', 'model_c_int8') and args.standalone_layer < 0:
        enable_conv_pool_strip_fusion(meta, data)

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

    # ─── Place DDR scatter region AFTER the blob ───
    # The blob (headers + inline wgt/param/input/golden) lives at BLOB_BASE.
    # The DDR regions must not overlap it, or build_main_ram.py's scatter and
    # the NPU's runtime output DMA will clobber the golden data.
    # Blob size does not depend on DDR addresses, so we can size it first,
    # then rebuild the blob with the corrected MODEL_DDR_BASE.
    global MODEL_DDR_BASE
    blob_end = BLOB_BASE + len(blob)
    ddr_base_needed = (blob_end + 0xFFFFF) & ~0xFFFFF  # round up to 1MB
    if ddr_base_needed > MODEL_DDR_BASE:
        print(f"  Blob ends at 0x{blob_end:08X}, moving DDR base "
              f"0x{MODEL_DDR_BASE:08X} → 0x{ddr_base_needed:08X}")
        MODEL_DDR_BASE = ddr_base_needed
        blob = build_chained_blob(meta, data, args.standalone_layer)

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
