/*
 * Open-NPU SoC Firmware — Chained Model Inference Test Runner
 *
 * Runs INT16 model (a/b/c/d) with chained inference:
 * Layer N output → Layer N+1 input (DDR). Supports tiling,
 * per-OC weight reload, fused blocks, skip-connections, Add/Concat.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include "soc_test_data.h"

/* ═══════════════════════════════════════════════════════════════════
 *  Memory map
 *  ═══════════════════════════════════════════════════════════════════ */
#define MAIN_RAM_BASE   0x40000000UL
#define NPU_BASE_ADDR   0x80000000UL

/* UART (LiteX CSR UART) */
#define UART_RXTX       (*(volatile uint32_t *)0xF0001800UL)
#define UART_TXFULL     (*(volatile uint32_t *)0xF0001804UL)

/* ═══════════════════════════════════════════════════════════════════
 *  NPU register offsets
 *  ═══════════════════════════════════════════════════════════════════ */
#define NPU_REG(off) (*(volatile uint32_t *)(NPU_BASE_ADDR + (off)))

#define REG_CTRL            0x000
#define REG_STATUS          0x004
#define REG_IRQ_EN          0x008
#define REG_IRQ_STATUS      0x00C
#define REG_VERSION         0x014
#define REG_HW_CONFIG       0x018
#define REG_LAYER_MODE      0x040
#define REG_IN_DIM_HW       0x044
#define REG_IN_DIM_C        0x048
#define REG_OUT_DIM_HW      0x04C
#define REG_OUT_DIM_C       0x050
#define REG_KERNEL_SIZE     0x054
#define REG_STRIDE          0x058
#define REG_PADDING         0x05C
#define REG_POOL_CFG        0x060
#define REG_RESIZE_CFG      0x064
#define REG_DECONV_CFG      0x068
#define REG_CONCAT_CFG      0x06C
#define REG_TILE_CFG        0x070
#define REG_TILE_COUNT      0x074
#define REG_SRAM_BASE       0x078
#define REG_DMA_IN_ADDR     0x100
#define REG_DMA_OUT_ADDR    0x104
#define REG_DMA_WGT_ADDR    0x108
#define REG_DMA_PARAM_ADDR  0x10C
#define REG_DMA_IN_STRIDE   0x110
#define REG_DMA_OUT_STRIDE  0x114
#define REG_DMA_CTRL        0x118
#define REG_DMA_ADD_B_ADDR  0x120
#define REG_DMA_IN_SIZE     0x128
#define REG_DMA_WGT_SIZE    0x12C
#define REG_DMA_OUT_SIZE    0x130
#define REG_DMA_TILE_IN_SIZE 0x134
#define REG_DMA_TILE_OUT_SIZE 0x138
#define REG_DMA_STORE_MODE  0x140
#define REG_DMA_ROW_CFG     0x144
#define REG_DMA_WGT_PER_OC  0x148
#define REG_POST_CTRL       0x180
#define REG_POST_PARAM_CNT  0x188
#define REG_POST_CLAMP      0x18C

/* Control bits */
#define CTRL_START      (1U << 0)
#define CTRL_SOFT_RST   (1U << 2)

/* Status bits */
#define STATUS_BUSY     (1U << 0)
#define STATUS_DONE     (1U << 3)
#define STATUS_ERROR    (1U << 2)

#define MAX_LAYERS 80

/* ═══════════════════════════════════════════════════════════════════
 *  UART helpers
 *  ═══════════════════════════════════════════════════════════════════ */
static void uart_putc(char c) {
    while (UART_TXFULL) {}
    UART_RXTX = (uint32_t)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

static void uart_put_hex32(uint32_t v) {
    static const char hex[] = "0123456789ABCDEF";
    for (int i = 28; i >= 0; i -= 4)
        uart_putc(hex[(v >> i) & 0xF]);
}

static void uart_put_dec(int32_t v) {
    if (v < 0) { uart_putc('-'); v = -v; }
    if (v == 0) { uart_putc('0'); return; }
    char buf[12];
    int i = 0;
    while (v > 0) { buf[i++] = '0' + (v % 10); v /= 10; }
    while (--i >= 0) uart_putc(buf[i]);
}

/* ═══════════════════════════════════════════════════════════════════
 *  DCache flush/invalidate (VexRiscv custom instructions)
 *  ═══════════════════════════════════════════════════════════════════ */
static inline void dcache_flush(void) {
    /* Clean D-cache: write-back dirty lines to DDR */
    asm volatile(".word 0x500F" : : : "memory");
}
/* Invalidate cache line containing addr (VexRiscv custom instruction).
 * Use after DMA writes to DDR, before CPU reads. */
static inline void dcache_inval_addr(void *addr) {
    asm volatile(".word 0x003b" : : "r"(addr) : "memory");
}
/* Invalidate a range of DDR addresses (force CPU to re-read from DDR). */
static void dcache_inval_range(volatile void *ptr, uint32_t size) {
    /* VexRiscv: use flush (0x500F) which also invalidates clean lines.
     * For NPU output, lines are clean (CPU didn't write), so flush = invalidate. */
    dcache_flush();
}

#define NPU_WGT_BASE     0x4002C000  /* workspace for wgt (16KB) */
#define NPU_PARAM_BASE   0x40030000  /* workspace for param (16KB) */
#define NPU_INPUT_BASE   0x40034000  /* workspace for input (16KB) */
#define NPU_OUTPUT_BASE  0x40038000  /* workspace for output (16KB) */
#define NPU_ADD_B_BASE   0x4003C000  /* workspace for add_b (16KB) */

/* Provide memcpy to satisfy compiler-generated calls */
__attribute__((noinline))
void *memcpy(void *dst, const void *src, uint32_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    for (uint32_t i = 0; i < n; i++)
        d[i] = s[i];
    return dst;
}

__attribute__((noinline))
static void memcpy_32(uint32_t dst, const uint32_t *src, uint32_t n_words) {
    volatile uint32_t *d = (volatile uint32_t *)dst;
    for (uint32_t i = 0; i < n_words; i++)
        d[i] = src[i];
}

/* ═══════════════════════════════════════════════════════════════════
 *  NPU helpers
 *  ═══════════════════════════════════════════════════════════════════ */
static void npu_reset(void) {
    NPU_REG(REG_CTRL) = CTRL_SOFT_RST;
    for (volatile int i = 0; i < 100; i++) {}
    NPU_REG(REG_IRQ_STATUS) = 0x7;
    NPU_REG(REG_IRQ_EN) = 0;
}

static int npu_wait_done(void) {
    uint32_t timeout = 50000000;  /* 50M cycles — enough for large layers */
    while (timeout--) {
        uint32_t irq_st = NPU_REG(REG_IRQ_STATUS);
        uint32_t status = NPU_REG(REG_STATUS);
        if (irq_st & 0x1) return 0;   /* IRQ_DONE latched */
        if (status & STATUS_ERROR) return -1;
        if (!(status & STATUS_BUSY)) return 0;
    }
    return -2; /* timeout */
}

/*
 * npu_program_layer — program ALL CSRs from 36-word layer_entry_t.
 * Mirrors test_npu_dma_e2e.py:program_layer() CSR sequence.
 *
 * Parameters:
 *   e: pointer to 36-word layer entry
 *   runtime_in_addr: resolved input DDR address (layer 0: from entry[32],
 *                    N>0: producing layer's ddr_out_addr)
 *   runtime_add_b_addr: resolved Add/Concat branch-B DDR address (0 if none)
 *   act_base: SRAM act base (0 for normal; 0 for fused — RTL handles reuse)
 */
static void npu_program_layer(const uint32_t *e,
                              uint32_t runtime_in_addr,
                              uint32_t runtime_add_b_addr,
                              uint32_t act_base) {
    uint32_t op_type   = e[4];
    uint32_t data_type = e[5];
    uint32_t in_zp     = e[28];
    uint32_t sched     = e[21];

    /* Layer mode: op_type | data_type | in_zp */
    NPU_REG(REG_LAYER_MODE) = (op_type & 0xF) | ((data_type & 1) << 4)
                              | ((in_zp & 0xFFFF) << 8);

    /* Dimensions */
    NPU_REG(REG_IN_DIM_HW)  = e[6];
    NPU_REG(REG_IN_DIM_C)   = e[7];
    NPU_REG(REG_OUT_DIM_HW) = e[8];
    NPU_REG(REG_OUT_DIM_C)  = e[9];

    /* Kernel, stride, padding */
    NPU_REG(REG_KERNEL_SIZE) = e[10] & 0xFFFF;
    NPU_REG(REG_STRIDE)      = e[11];
    NPU_REG(REG_PADDING)     = e[12];

    /* Operator-specific config */
    uint32_t cfg_aux = e[18];
    if (op_type == 3)      NPU_REG(REG_POOL_CFG)    = cfg_aux;
    else if (op_type == 5) NPU_REG(REG_RESIZE_CFG)  = cfg_aux;
    else if (op_type == 6) NPU_REG(REG_DECONV_CFG)  = cfg_aux;
    else if (op_type == 7) NPU_REG(REG_CONCAT_CFG)  = cfg_aux;

    /* Tiling */
    NPU_REG(REG_TILE_CFG)   = e[19];  /* tile_h | (tile_w<<16) */
    NPU_REG(REG_TILE_COUNT) = e[20];  /* tile_num_h | (tile_num_w<<16) */

    /* SRAM base: out_base = tile_in_size/4 if tiled, else dma_in_size/4 */
    uint32_t out_base = (e[23] > 0) ? (e[23] / 4) : (e[15] / 4);
    NPU_REG(REG_SRAM_BASE) = (out_base << 16) | act_base;

    /* DMA addresses — runtime-resolved input, metadata for wgt/param/out */
    NPU_REG(REG_DMA_IN_ADDR)    = runtime_in_addr;
    NPU_REG(REG_DMA_OUT_ADDR)   = e[31];
    NPU_REG(REG_DMA_WGT_ADDR)   = e[29];
    NPU_REG(REG_DMA_PARAM_ADDR) = e[30];

    /* Add/Concat branch B — always write to clear stale value from previous layer */
    NPU_REG(REG_DMA_ADD_B_ADDR) = (op_type == 4 || op_type == 7) ? runtime_add_b_addr : 0;

    /* DMA sizes */
    NPU_REG(REG_DMA_IN_SIZE)  = e[15];
    NPU_REG(REG_DMA_WGT_SIZE) = e[16];
    NPU_REG(REG_DMA_OUT_SIZE) = e[17];

    /* Per-OC weight reload */
    NPU_REG(REG_DMA_WGT_PER_OC) = e[26];

    /* Tiled DB_EN prefetch + PTS 2D DMA */
    if (e[23] > 0) NPU_REG(REG_DMA_TILE_IN_SIZE) = e[23];
    if (e[22] > 0) {
        NPU_REG(REG_DMA_STORE_MODE)    = e[22];
        NPU_REG(REG_DMA_TILE_OUT_SIZE) = e[24];
        NPU_REG(REG_DMA_ROW_CFG)       = e[25];
    }

    /* Strides: in_stride = in_w * in_c * elem_bytes for 2D tiled load (chain mode) */
    {
        uint32_t in_w = e[6] >> 16;
        uint32_t in_c = e[7];
        uint32_t eb = (e[5] & 1) ? 2 : 1;
        /* Set in_stride for tiled layers when input is NHWC (chain mode).
         * Layer 0 input is pre-packed contiguous → in_stride=0 (1D mode).
         * Chain input from previous layer → in_stride=NHWC row width. */
        int is_nhwc_input = (runtime_in_addr != e[32]);  /* not layer-0 packed data */
        uint32_t in_stride_val = (e[19] != 0 && is_nhwc_input) ? (in_w * in_c * eb) : 0;
        NPU_REG(REG_DMA_IN_STRIDE) = in_stride_val;
    }
    NPU_REG(REG_DMA_OUT_STRIDE) = 0;

    /* DMA control: sched_ctrl (DB_EN/FUSE/PTS bits from metadata) */
    NPU_REG(REG_DMA_CTRL) = sched;

    /* Post-processing */
    NPU_REG(REG_POST_CTRL)      = e[13];
    NPU_REG(REG_POST_PARAM_CNT) = e[14];
    NPU_REG(REG_POST_CLAMP)     = e[27];  /* clamp_max in [15:0] */
}

/* ═══════════════════════════════════════════════════════════════════
 *  Chained model inference
 *  ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    uint32_t blob_base;
    const char *name;
} test_case_t;

static int run_chained_model(test_case_t *tc) {
    uart_puts("\n  ----------------------------------------\n");
    uart_puts("  Model: ");
    uart_puts(tc->name);
    uart_puts("\n  ----------------------------------------\n");

    volatile const uint32_t *blob = (volatile const uint32_t *)(uintptr_t)tc->blob_base;
    uint32_t magic = blob[0];
    uint32_t num_layers = blob[1];

    if (magic != BLOB_MAGIC) {
        uart_puts("  BLOB MAGIC MISMATCH: 0x");
        uart_put_hex32(magic);
        uart_puts("\n  SKIP\n");
        return -1;
    }

    uart_puts("  Layers: ");
    uart_put_dec(num_layers);
    uart_putc('\n');

    /* First pass: build layer_out_addr[] for skip/residual lookup */
    uint32_t layer_out_addr[MAX_LAYERS];
    const uint32_t *cursor = blob + 3;  /* skip 3-word blob header */

    for (uint32_t l = 0; l < num_layers && l < MAX_LAYERS; l++) {
        const uint32_t *e = cursor;
        layer_out_addr[l] = e[31];  /* ddr_out_addr */

        /* Advance cursor past header + wgt + param + input + output */
        cursor += LAYER_ENTRY_HDR_WORDS;
        cursor += e[0];  /* n_wgt */
        cursor += e[1];  /* n_param */
        cursor += e[2];  /* n_input */
        cursor += e[3];  /* n_output */
    }

    npu_reset();
    int total_err = 0;
    cursor = blob + 3;

    for (uint32_t l = 0; l < num_layers; l++) {
        const uint32_t *e = cursor;

        uint32_t op_type = e[4];
        int32_t input_src = (int32_t)e[34];
        int32_t residual_src = (int32_t)e[35];
        uint32_t sched = e[21];

        /* Resolve runtime input address */
        uint32_t runtime_in_addr;
        if (l == 0) {
            runtime_in_addr = e[32];  /* layer 0: use blob's ddr_in_addr */
        } else if (op_type == 7 && residual_src >= 0 && residual_src < (int32_t)num_layers) {
            /* Concat phase 2: true input is residual_src's output (e.g. L8),
             * NOT the previous layer's output (that's the other concat
             * phase's partial buffer). */
            runtime_in_addr = layer_out_addr[residual_src];
        } else if (input_src >= 0 && input_src < (int32_t)num_layers) {
            runtime_in_addr = layer_out_addr[input_src];  /* skip connection */
        } else {
            runtime_in_addr = layer_out_addr[l - 1];  /* chain from previous */
        }

        /* Resolve Add/Concat branch-B address */
        uint32_t runtime_add_b_addr = 0;
        if (op_type == 7) {
            /* Concat: ADD_B_ADDR carries the OUT-region preload source — the
             * previous concat phase's DDR output (layer l-1). Only phases
             * with concat offset != 0 need the carryover preload; the first
             * phase (offset 0) owns ch[0:in_c) and needs no preload. RTL keys
             * the in-place store on op_type==4, so a nonzero add_b here no
             * longer misroutes the store region. */
            uint32_t concat_off = e[18] & 0xFFFF;
            if (concat_off != 0 && l > 0)
                runtime_add_b_addr = layer_out_addr[l - 1];
        } else if (residual_src >= 0 && residual_src < (int32_t)num_layers) {
            runtime_add_b_addr = layer_out_addr[residual_src];
        } else if (e[33] != 0) {
            runtime_add_b_addr = e[33];  /* fallback: metadata's ddr_add_b_addr */
        }

        /* Soft-reset between non-fused layers */
        if (l > 0 && !(sched & 0x06)) {  /* not FUSE_MID or FUSE_END */
            NPU_REG(REG_CTRL) = CTRL_SOFT_RST;
            for (volatile int d = 0; d < 100; d++) {}
            NPU_REG(REG_IRQ_STATUS) = 0x7;
        }

        /* Use original sched_ctrl from metadata (DB_EN + PTS enabled) */
        npu_program_layer(e, runtime_in_addr, runtime_add_b_addr, 0);
        dcache_flush();
        if (l < 3) {
            uart_puts("  DBG L"); uart_put_dec(l);
            uart_puts(": in_addr=0x"); uart_put_hex32(runtime_in_addr);
            uart_puts(" e32=0x"); uart_put_hex32(e[32]);
            uart_puts(" in_stride=0x"); uart_put_hex32(NPU_REG(0x110));
            uart_puts("\n");
        }

        /* Start NPU */
        NPU_REG(REG_CTRL) = CTRL_START;
        dcache_flush();  /* Ensure CTRL_START reaches NPU */

        /* Wait for completion */
        int ret = npu_wait_done();
        if (ret != 0) {
            uart_puts("  L");
            uart_put_dec(l);
            if (ret == -1) {
                uart_puts(": ERROR (status=0x");
                uart_put_hex32(NPU_REG(REG_STATUS));
                uart_puts(")\n");
            } else {
                uart_puts(": TIMEOUT\n");
            }
            return -1;
        }

        /* Flush DCache so CPU reads DMA-written DDR output (not stale cache) */
        dcache_flush();

        /* Verify output: compare DDR at ddr_out_addr vs golden.
         * In chain mode (RUN_MODEL_B), intermediate layer golden is based on
         * pre-packed input, not chain input. Only verify L0 and last layer. */
        uint32_t n_output = e[3];
#ifdef RUN_MODEL_B
        int skip_verify = 0;  /* Debug: verify all layers to find bugs */
#else
        int skip_verify = 0;
#endif
        if (n_output > 0 && !skip_verify) {
            const uint32_t *golden = cursor + LAYER_ENTRY_HDR_WORDS + e[0] + e[1] + e[2];
            volatile const uint32_t *out_ptr = (volatile const uint32_t *)(uintptr_t)e[31];

            /* Debug: for layer 0-2, print first 5 output words vs golden */
            if (l < 3) {
                uart_puts("  DBG L"); uart_put_dec(l);
                uart_puts(": out_addr=0x");
                uart_put_hex32(e[31]);
                uart_puts("\n");
                for (int dbg = 0; dbg < 5 && dbg < (int)n_output; dbg++) {
                    uart_puts("  w[");
                    uart_put_dec(dbg);
                    uart_puts("] exp=0x");
                    uart_put_hex32(golden[dbg]);
                    uart_puts(" got=0x");
                    uart_put_hex32(out_ptr[dbg]);
                    uart_puts("\n");
                }
            }
            int layer_err = 0;
            if (op_type == 7) {
                /* Concat: verify only the channels this phase owns.
                 * concat_cfg = (total_c << 16) | offset; owned ch range is
                 * [offset, offset + in_c). The rest of the buffer is the
                 * other phase's data (or undefined for the offset-0 phase,
                 * whose golden has zeros that dirty SRAM cannot provide). */
                uint32_t c_off  = e[18] & 0xFFFF;
                uint32_t in_c   = e[7];
                uint32_t out_c  = e[9];
                uint32_t eb     = (e[5] & 1) ? 2 : 1;
                uint32_t emask  = (eb == 2) ? 0xFFFF : 0xFF;
                uint32_t total_elems = n_output * 4 / eb;
                uint32_t ch = 0;
                for (uint32_t idx = 0; idx < total_elems; idx++) {
                    if (ch >= c_off && ch < c_off + in_c) {
                        uint32_t w  = (idx * eb) >> 2;
                        uint32_t sh = (idx * eb & 3) << 3;
                        uint32_t ge = (golden[w] >> sh) & emask;
                        uint32_t oe = (out_ptr[w] >> sh) & emask;
                        if (ge != oe) {
                            if (layer_err < 10) {
                                uart_puts("  L");
                                uart_put_dec(l);
                                uart_puts(" elem[");
                                uart_put_dec(idx);
                                uart_puts("]: exp=0x");
                                uart_put_hex32(ge);
                                uart_puts(" got=0x");
                                uart_put_hex32(oe);
                                uart_putc('\n');
                            }
                            layer_err++;
                        }
                    }
                    if (++ch == out_c) ch = 0;
                }
            } else {
            for (uint32_t i = 0; i < n_output; i++) {
                if (out_ptr[i] != golden[i]) {
            if (layer_err < 10) {
                    uart_puts("  L");
                    uart_put_dec(l);
                    uart_puts(" w[");
                    uart_put_dec(i);
                    uart_puts("]: exp=0x");
                    uart_put_hex32(golden[i]);
                    uart_puts(" got=0x");
                    uart_put_hex32(out_ptr[i]);
                    uart_putc('\n');
                }
                    layer_err++;
                }
            }
            }

            if (layer_err == 0) {
                uart_puts("  L");
                uart_put_dec(l);
                uart_puts(": PASS (");
                uart_put_dec(n_output);
                uart_puts(" words)\n");
            } else {
                uart_puts("  L");
                uart_put_dec(l);
                uart_puts(": FAIL — ");
                uart_put_dec(layer_err);
                uart_puts("/");
                uart_put_dec(n_output);
                uart_puts(" mismatches\n");
                total_err += layer_err;
            }
        } else if (skip_verify) {
            uart_puts("  L");
            uart_put_dec(l);
            uart_puts(": SKIP (chain mode, golden mismatch expected)\n");
        } else {
            uart_puts("  L");
            uart_put_dec(l);
            uart_puts(": (no golden output, skipped verification)\n");
        }

        /* Advance cursor */
        cursor += LAYER_ENTRY_HDR_WORDS + e[0] + e[1] + e[2] + e[3];
    }

    return total_err;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Main entry point
 *  ═══════════════════════════════════════════════════════════════════ */
void main(void) {
    uart_puts("\n\n");
    uart_puts("========================================\n");
    uart_puts("  Open-NPU SoC — Chained Model Inference\n");
    uart_puts("========================================\n\n");

    /* Read NPU version */
    uart_puts("[INIT] NPU Version: ");
    uint32_t ver = NPU_REG(REG_VERSION);
    uart_put_dec((ver >> 16) & 0xFF); uart_putc('.');
    uart_put_dec((ver >> 8) & 0xFF);  uart_putc('.');
    uart_put_dec(ver & 0xFF);
    uart_putc('\n');

    uint32_t hwcfg = NPU_REG(REG_HW_CONFIG);
    uart_puts("      Array size: ");
    uart_put_dec(hwcfg & 0xFF);
    uart_puts(", HW config: 0x");
    uart_put_hex32(hwcfg);
    uart_putc('\n');

    int global_err = 0;

#ifdef RUN_FC_TEST
#ifndef STANDALONE_LAYER_VAL
#define STANDALONE_LAYER_VAL 24
#endif
    /* Standalone FC test: run L24 only, using workspace mode (no tiling, no DDR chaining) */
    uart_puts("\n[FC TEST] Running L24 standalone (workspace mode)\n");
    {
        /* Access blob at BLOB_MODEL_BASE */
        volatile const uint32_t *blob_fc = (volatile const uint32_t *)(uintptr_t)BLOB_MODEL_BASE;
        /* Find L24 in blob */
        const uint32_t *cursor_fc = (const uint32_t *)(blob_fc + 3);
        int skip_layers = STANDALONE_LAYER_VAL;
        for (uint32_t l = 0; l < (uint32_t)skip_layers; l++) {
            cursor_fc += LAYER_ENTRY_HDR_WORDS + cursor_fc[0] + cursor_fc[1] + cursor_fc[2] + cursor_fc[3];
        }
        const uint32_t *e24 = cursor_fc;

        /* Debug: print L24 header fields */
        uart_puts("  L24 hdr: op=");
        uart_put_dec(e24[4]);
        uart_puts(" n_wgt=");
        uart_put_dec(e24[0]);
        uart_puts(" n_param=");
        uart_put_dec(e24[1]);
        uart_puts(" n_in=");
        uart_put_dec(e24[2]);
        uart_puts(" n_out=");
        uart_put_dec(e24[3]);
        uart_puts(" post_ctrl=0x");
        uart_put_hex32(e24[13]);
        uart_puts(" param_cnt=");
        uart_put_dec(e24[14]);
        uart_puts(" clamp=0x");
        uart_put_hex32(e24[27]);
        uart_puts("\n");

        /* Always use DDR for wgt/param/output. Copy only input to workspace
         * if it fits, else use DDR. */
        const uint32_t *payload = cursor_fc + LAYER_ENTRY_HDR_WORDS;
        uint32_t in_addr;
        if (e24[2] * 4 <= 16384) {
            memcpy_32(NPU_INPUT_BASE, payload + e24[0] + e24[1], e24[2]);
            in_addr = NPU_INPUT_BASE;
        } else {
            in_addr = e24[32];  /* ddr_in_addr */
        }
        uint32_t wgt_addr = e24[29];    /* ddr_wgt_addr */
        uint32_t param_addr = e24[30];  /* ddr_param_addr */
        uint32_t out_addr = e24[31];    /* ddr_out_addr */
        const uint32_t *golden24 = payload + e24[0] + e24[1] + e24[2];
        dcache_flush();

        /* Debug */
        uart_puts("  DBG: wgt_addr=0x");
        uart_put_hex32(wgt_addr);
        uart_puts(" in[0]=0x");
        uart_put_hex32(*(volatile uint32_t*)in_addr);
        uart_puts(" golden[0]=0x");
        uart_put_hex32(golden24[0]);
        uart_puts("\n");

        /* Program NPU — use full npu_program_layer for tiling support */
        npu_reset();
        npu_program_layer(e24, in_addr, 0, 0);
        /* Override output address */
        NPU_REG(REG_DMA_OUT_ADDR) = out_addr;
        dcache_flush();

        NPU_REG(REG_CTRL) = CTRL_START;
        dcache_flush();

        int ret = npu_wait_done();
        if (ret != 0) {
            uart_puts("  FC: ERROR/TIMEOUT\n");
            global_err++;
        } else {
            dcache_flush();
            volatile uint32_t *out = (volatile uint32_t *)out_addr;
            int err = 0;
            for (uint32_t i = 0; i < e24[3]; i++) {
                if (out[i] != golden24[i]) {
                    if (err < 5) {
                        uart_puts("  w[");
                        uart_put_dec(i);
                        uart_puts("] exp=0x");
                        uart_put_hex32(golden24[i]);
                        uart_puts(" got=0x");
                        uart_put_hex32(out[i]);
                        uart_puts("\n");
                    }
                    err++;
                }
            }
            if (err == 0) {
                uart_puts("  FC: PASS (");
                uart_put_dec(e24[3]);
                uart_puts(" words)\n");
            } else {
                uart_puts("  FC: FAIL — ");
                uart_put_dec(err);
                uart_puts(" mismatches\n");
                global_err++;
            }
        }
    }
#else
    /* Run selected model (chained inference) */
#if defined(RUN_MODEL_A)
    test_case_t tc = { BLOB_MODEL_BASE, "model_a (MobileNetV2 INT16, 63 layers)" };
#elif defined(RUN_MODEL_B)
    test_case_t tc = { BLOB_MODEL_BASE, "model_b (ResNet-18 CIFAR INT16, 25 layers)" };
#elif defined(RUN_MODEL_C)
    test_case_t tc = { BLOB_MODEL_BASE, "model_c (YOLO-Tiny INT16, 17 layers)" };
#elif defined(RUN_MODEL_D)
    test_case_t tc = { BLOB_MODEL_BASE, "model_d (Palm Vein INT16, 24 layers)" };
#else
    uart_puts("ERROR: No model selected. Compile with -DRUN_MODEL_X\n");
    while (1) {}
#endif

    int ret = run_chained_model(&tc);
    if (ret < 0) {
        uart_puts("\n  MODEL ABORTED\n");
        global_err++;
    } else if (ret > 0) {
        uart_puts("\n  FAIL — ");
        uart_put_dec(ret);
        uart_puts(" total mismatches\n");
        global_err++;
    } else {
        uart_puts("\n  ALL LAYERS PASSED\n");
    }
#endif /* !RUN_FC_TEST */

    /* Final result */
    uart_puts("\n========================================\n");
    if (global_err == 0) {
        uart_puts("  RESULT: PASS\n");
    } else {
        uart_puts("  RESULT: FAIL\n");
    }
    uart_puts("========================================\n");

    while (1) {}
}
