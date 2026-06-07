/*
 * Open-NPU SoC Firmware — Multi-Layer E2E Test Runner
 *
 * Runs MobileNetV2-Tiny (10 layers) in both INT8 and INT16 modes,
 * using pre-loaded golden test data in main RAM.
 *
 * Test data is loaded via $readmemh into main RAM at SoC startup.
 * Blob format: header + per-layer entries with CSR configs + data.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include "soc_test_data.h"

/* ═══════════════════════════════════════════════════════════════════
 *  Memory map
 * ═══════════════════════════════════════════════════════════════════ */
#define MAIN_RAM_BASE   0x40000000UL
#define NPU_BASE_ADDR   0x80000000UL
#define NPU_ADD_B_BASE  0x40030000UL  /* Add Branch B input workspace */

/* UART (LiteX CSR UART) */
#define UART_RXTX       (*(volatile uint32_t *)0xF0001800UL)
#define UART_TXFULL     (*(volatile uint32_t *)0xF0001804UL)

/* ═══════════════════════════════════════════════════════════════════
 *  NPU register offsets (from npu_hal.h)
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_REG(off) (*(volatile uint32_t *)(NPU_BASE_ADDR + (off)))

/* VexRiscv custom instruction: invalidate entire data cache */
static inline void dcache_flush(void) {
    asm volatile(".word 0x500F" : : : "memory");
}

#define REG_CTRL            0x000
#define REG_STATUS          0x004
#define REG_IRQ_EN          0x008
#define REG_IRQ_STATUS      0x00C
#define REG_VERSION         0x014
#define REG_HW_CONFIG       0x018
#define REG_LAYER_COUNT     0x030
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

/* ═══════════════════════════════════════════════════════════════════
 *  UART helpers
 * ═══════════════════════════════════════════════════════════════════ */
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
 *  Data copy helpers (inline-friendly to avoid libc memcpy)
 * ═══════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void memcpy_32(uint32_t dst, const uint32_t *src, uint32_t n_words) {
    volatile uint32_t *d = (volatile uint32_t *)dst;
    for (uint32_t i = 0; i < n_words; i++)
        d[i] = src[i];
}

/* Provide memcpy to satisfy compiler-generated calls */
__attribute__((noinline))
void *memcpy(void *dst, const void *src, uint32_t n) {
    char *d = (char *)dst;
    const char *s = (const char *)src;
    for (uint32_t i = 0; i < n; i++)
        d[i] = s[i];
    return dst;
}

/* ═══════════════════════════════════════════════════════════════════
 *  NPU helpers
 * ═══════════════════════════════════════════════════════════════════ */
static void npu_reset(void) {
    NPU_REG(REG_CTRL) = CTRL_SOFT_RST;
    for (volatile int i = 0; i < 100; i++) {}
    NPU_REG(REG_IRQ_STATUS) = 0x7;
    NPU_REG(REG_IRQ_EN) = 0;
}

static int npu_wait_done(void) {
    uint32_t timeout = 500000;
    while (timeout--) {
        uint32_t irq_st = NPU_REG(REG_IRQ_STATUS);
        uint32_t status = NPU_REG(REG_STATUS);
        if (irq_st & 0x1) return 0;   /* IRQ_DONE latched */
        if (status & STATUS_ERROR) return -1;
        if (!(status & STATUS_BUSY)) return 0;
    }
    return -2; /* timeout */
}

static void npu_program_layer(const uint32_t *entry_hdr, uint32_t n_input_words) {
    /*
     * entry_hdr layout (see soc_test_data.h layer_entry_t):
     *   [0]  n_wgt
     *   [1]  n_param
     *   [2]  n_input
     *   [3]  n_output
     *   [4]  op_type
     *   [5]  data_type
     *   [6]  in_hw
     *   [7]  in_c
     *   [8]  out_hw
     *   [9]  out_c
     *   [10] kernel_dil
     *   [11] stride
     *   [12] padding
     *   [13] post_ctrl
     *   [14] param_count
     *   [15] dma_in_size
     *   [16] dma_wgt_size
     *   [17] dma_out_size
     */
    uint32_t op_type   = entry_hdr[4];
    uint32_t data_type = entry_hdr[5];
    uint32_t in_hw     = entry_hdr[6];
    uint32_t in_c      = entry_hdr[7];
    uint32_t out_hw    = entry_hdr[8];
    uint32_t out_c     = entry_hdr[9];
    uint32_t kernel_dil= entry_hdr[10];
    uint32_t stride    = entry_hdr[11];
    uint32_t padding   = entry_hdr[12];
    uint32_t post_ctrl = entry_hdr[13];
    uint32_t param_cnt = entry_hdr[14];
    uint32_t dma_in_sz = entry_hdr[15];
    uint32_t dma_wgt_sz= entry_hdr[16];
        uint32_t dma_out_sz= entry_hdr[17];
        uint32_t cfg_aux     = entry_hdr[18];

        /* Layer mode: bits[3:0]=op_type, bit[4]=data_type */
    NPU_REG(REG_LAYER_MODE) = (op_type & 0xF) | ((data_type & 1) << 4);

    /* Dimensions */
    NPU_REG(REG_IN_DIM_HW)  = in_hw;
    NPU_REG(REG_IN_DIM_C)   = in_c;
    NPU_REG(REG_OUT_DIM_HW) = out_hw;
    NPU_REG(REG_OUT_DIM_C)  = out_c;

        /* Kernel (only bits 0-15 used; dilation bits ignored by hardware) */
        NPU_REG(REG_KERNEL_SIZE) = kernel_dil & 0xFFFF;
        NPU_REG(REG_STRIDE)      = stride;
        NPU_REG(REG_PADDING)     = padding;

        /* Operator-specific config */
        if (op_type == 3) {
            NPU_REG(REG_POOL_CFG) = cfg_aux;
        } else if (op_type == 5) {
            NPU_REG(REG_RESIZE_CFG) = cfg_aux;
        } else if (op_type == 6) {
            NPU_REG(REG_DECONV_CFG) = cfg_aux;
        } else if (op_type == 7) {
            NPU_REG(REG_CONCAT_CFG) = cfg_aux;
        } else if (op_type == 4) {
            NPU_REG(REG_DMA_ADD_B_ADDR) = NPU_ADD_B_BASE;
            param_cnt = 1;  /* Add uses 1 rescale param set, not per-channel */
        }

    /* No tiling */
    NPU_REG(REG_TILE_CFG)   = 0;
    NPU_REG(REG_TILE_COUNT) = 1 | (1 << 16);

    /* SRAM base: act_base=0, out_base = n_input_words (output after input) */
    NPU_REG(REG_SRAM_BASE) = n_input_words << 16;

    /* DMA addresses */
    NPU_REG(REG_DMA_WGT_ADDR)   = NPU_WGT_BASE;
    NPU_REG(REG_DMA_PARAM_ADDR) = NPU_PARAM_BASE;
    NPU_REG(REG_DMA_IN_ADDR)    = NPU_INPUT_BASE;
    NPU_REG(REG_DMA_OUT_ADDR)   = NPU_OUTPUT_BASE;

    /* DMA sizes (byte-aligned, already aligned to 4) */
    NPU_REG(REG_DMA_IN_SIZE)  = dma_in_sz;
    NPU_REG(REG_DMA_WGT_SIZE) = dma_wgt_sz;
    NPU_REG(REG_DMA_OUT_SIZE) = dma_out_sz;

    /* Stride (not used, set to 0) */
    NPU_REG(REG_DMA_IN_STRIDE)  = 0;
    NPU_REG(REG_DMA_OUT_STRIDE) = 0;
    NPU_REG(REG_DMA_CTRL)       = 0;

    /* Post-processing */
    NPU_REG(REG_POST_CTRL)      = post_ctrl;
    NPU_REG(REG_POST_PARAM_CNT) = param_cnt;
    NPU_REG(REG_POST_CLAMP)     = (uint32_t)(-128 & 0xFFFF) | ((127 & 0xFFFF) << 16);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Test runner
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    uint32_t blob_base;      /* base address of blob in main RAM */
    const char *name;        /* test name for UART output */
    int num_layers;           /* filled by run_test_case */
} test_case_t;

static int run_test_case(test_case_t *tc) {
    uart_puts("\n  ----------------------------------------\n");
    uart_puts("  Test: ");
    uart_puts(tc->name);
    uart_puts("\n  ----------------------------------------\n");

    /* Parse blob header */
    volatile const uint32_t *blob = (volatile const uint32_t *)(uintptr_t)tc->blob_base;
    uint32_t magic = blob[0];
    uint32_t num_layers = blob[1];
    tc->num_layers = (int)num_layers;

    if (magic != BLOB_MAGIC) {
        uart_puts("  BLOB MAGIC MISMATCH: 0x");
        uart_put_hex32(magic);
        uart_puts(" != 0x");
        uart_put_hex32(BLOB_MAGIC);
        uart_puts("\n  SKIP\n");
        return -1;
    }

    uart_puts("  Layers: ");
    uart_put_dec(num_layers);
    uart_putc('\n');

    /* Reset NPU */
    npu_reset();

    /* Current data pointer (skip 3-word blob header) */
    const uint32_t *data_ptr = (const uint32_t *)(blob + 3);
    int total_err = 0;

    for (uint32_t l = 0; l < num_layers; l++) {
        /* Read entry header */
        const uint32_t *hdr = data_ptr;
        uint32_t n_wgt   = hdr[0];
        uint32_t n_param = hdr[1];
        uint32_t n_input = hdr[2];
        uint32_t n_output= hdr[3];

        /* Advance data pointer past header */
        const uint32_t *payload = data_ptr + (LAYER_ENTRY_HDR_SIZE / 4);

        /* Copy weights to NPU WGT workspace */
        memcpy_32(NPU_WGT_BASE, payload, n_wgt);
        payload += n_wgt;

        /* Copy params to NPU PARAM workspace */
        memcpy_32(NPU_PARAM_BASE, payload, n_param);
        payload += n_param;

        /* Copy input to NPU INPUT workspace (first layer only) */
        if (l == 0) {
            memcpy_32(NPU_INPUT_BASE, payload, n_input);
        }
        payload += n_input;

        /* Golden output reference */
        const uint32_t *golden = payload;
        payload += n_output;

        /* For Add: copy Branch B input to separate workspace */
        uint32_t op_type = hdr[4];
        if (op_type == 4) {
            memcpy_32(NPU_ADD_B_BASE, payload, n_input);
            payload += n_input;
        }

        /* Flush DCache so DMA sees CPU writes to workspace buffers */
        dcache_flush();

        /* Program CSRs */
        npu_program_layer(hdr, n_input);

        /* Soft-reset NPU between layers for clean start */
        if (l > 0) {
            NPU_REG(REG_CTRL) = CTRL_SOFT_RST;
            for (volatile int d = 0; d < 100; d++) {}
            NPU_REG(REG_IRQ_STATUS) = 0x7;
        }
        /* Start NPU */
        NPU_REG(REG_CTRL) = CTRL_START;

        /* Wait for completion */
        int ret = npu_wait_done();
        if (l == 0) {
            uint32_t st = NPU_REG(REG_STATUS);
            uart_puts("  DBG: status=0x");
            uart_put_hex32(st);
            uart_puts(" irq=0x");
            uart_put_hex32(NPU_REG(REG_IRQ_STATUS));
            uart_putc('\n');
        }
        if (ret != 0) {
            uart_puts("  Layer ");
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

        /* Invalidate DCache so CPU reads DMA-written output from memory */
        dcache_flush();

        /* Verify output */
        volatile uint32_t *out_ptr = (volatile uint32_t *)NPU_OUTPUT_BASE;
        int layer_err = 0;
        for (uint32_t i = 0; i < n_output; i++) {
            if (out_ptr[i] != golden[i]) {
                if (layer_err < 3) {
                    uart_puts("  L");
                    uart_put_dec(l);
                    uart_puts(" word[");
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

        if (layer_err == 0) {
            uart_puts("  Layer ");
            uart_put_dec(l);
            uart_puts(": PASS (");
            uart_put_dec(n_output);
            uart_puts(" words)\n");
        } else {
            uart_puts("  Layer ");
            uart_put_dec(l);
            uart_puts(": FAIL — ");
            uart_put_dec(layer_err);
            uart_puts(" mismatches\n");
            total_err += layer_err;
        }

        /* Prepare for next layer: copy output to input */
        if (l + 1 < num_layers) {
            uint32_t hdr_words = LAYER_ENTRY_HDR_SIZE / 4;
            uint32_t extra = (op_type == 4) ? n_input : 0;  /* Add has input_b */
            uint32_t next_hdr_off = hdr_words + n_wgt + n_param + n_input + n_output + extra;
            const uint32_t *next_hdr = data_ptr + next_hdr_off;
            uint32_t next_n_input = next_hdr[2];
            /* Copy output as input for next layer */
            for (uint32_t i = 0; i < next_n_input; i++) {
                ((volatile uint32_t *)NPU_INPUT_BASE)[i] = out_ptr[i];
            }
            /* Flush so next layer's DMA sees the copied input */
            dcache_flush();
        }

        /* Move data pointer to next entry */
        uint32_t hdr_words = LAYER_ENTRY_HDR_SIZE / 4;
        uint32_t extra = (op_type == 4) ? n_input : 0;  /* Add has input_b */
        data_ptr = data_ptr + hdr_words + n_wgt + n_param + n_input + n_output + extra;
    }

    return total_err;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Main entry point
 * ═══════════════════════════════════════════════════════════════════ */
void main(void) {
    uart_puts("\n\n");
    uart_puts("========================================\n");
    uart_puts("  Open-NPU SoC — Multi-Layer E2E Test\n");
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

    /* ── Test 1: INT8 MobileNetV2-Tiny ── */
    uart_puts("\n[TEST 1/16] INT8 MobileNetV2-Tiny (10 layers)\n");
    test_case_t tc_int8 = { BLOB_INT8_BASE, "INT8", 0 };
    int ret_int8 = run_test_case(&tc_int8);
    if (ret_int8 < 0) {
        uart_puts("  TEST ABORTED\n");
        global_err++;
    } else if (ret_int8 > 0) {
        uart_puts("  INT8: ");
        uart_put_dec(ret_int8);
        uart_puts(" mismatches — FAIL\n");
        global_err++;
    } else {
        uart_puts("  INT8: ALL ");
        uart_put_dec(tc_int8.num_layers);
        uart_puts(" LAYERS PASSED ✓\n");
    }

    /* ── Test 2: INT16 MobileNetV2-Tiny ── */
    uart_puts("\n[TEST 2/16] INT16 MobileNetV2-Tiny (10 layers)\n");
    test_case_t tc_int16 = { BLOB_INT16_BASE, "INT16", 0 };
    int ret_int16 = run_test_case(&tc_int16);
    if (ret_int16 < 0) {
        uart_puts("  TEST ABORTED\n");
        global_err++;
    } else if (ret_int16 > 0) {
        uart_puts("  INT16: ");
        uart_put_dec(ret_int16);
        uart_puts(" mismatches — FAIL\n");
        global_err++;
    } else {
        uart_puts("  INT16: ALL ");
        uart_put_dec(tc_int16.num_layers);
        uart_puts(" LAYERS PASSED ✓\n");
    }

    /* ── Pooling operator tests ── */
    {
        uart_puts("\n[TEST 3/16] Pool Max INT8\n");
        test_case_t tc = { BLOB_POOL_MAX_INT8_BASE, "Pool Max INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    {
        uart_puts("\n[TEST 4/16] Pool Avg INT8\n");
        test_case_t tc = { BLOB_POOL_AVG_INT8_BASE, "Pool Avg INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    {
        uart_puts("\n[TEST 5/16] Pool Global INT8\n");
        test_case_t tc = { BLOB_POOL_GLOBAL_INT8_BASE, "Pool Global INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    {
        uart_puts("\n[TEST 6/16] Pool Max INT16\n");
        test_case_t tc = { BLOB_POOL_MAX_INT16_BASE, "Pool Max INT16", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    /* ── Resize operator tests ── */
    {
        uart_puts("\n[TEST 7/16] Resize Nearest INT8\n");
        test_case_t tc = { BLOB_RESIZE_NEAREST_INT8_BASE, "Resize Nearest INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    {
        uart_puts("\n[TEST 8/16] Resize Bilinear INT16\n");
        test_case_t tc = { BLOB_RESIZE_BILINEAR_INT16_BASE, "Resize Bilinear INT16", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    /* ── Deconv operator tests ── */
    {
        uart_puts("\n[TEST 9/16] Deconv INT8\n");
        test_case_t tc = { BLOB_DECONV_INT8_BASE, "Deconv INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    {
        uart_puts("\n[TEST 10/16] Deconv INT16\n");
        test_case_t tc = { BLOB_DECONV_INT16_BASE, "Deconv INT16", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    /* ── Concat operator tests ── */
    {
        uart_puts("\n[TEST 11/16] Concat INT8\n");
        test_case_t tc = { BLOB_CONCAT_INT8_BASE, "Concat INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    {
        uart_puts("\n[TEST 12/16] Concat INT16\n");
        test_case_t tc = { BLOB_CONCAT_INT16_BASE, "Concat INT16", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    /* ── Add operator tests ── */
    {
        uart_puts("\n[TEST 13/16] Eltwise Add INT8\n");
        test_case_t tc = { BLOB_ADD_INT8_BASE, "Add INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    {
        uart_puts("\n[TEST 14/16] Eltwise Add INT16\n");
        test_case_t tc = { BLOB_ADD_INT16_BASE, "Add INT16", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) { uart_puts("  FAIL\n"); global_err++; }
        else { uart_puts("  PASS ✓\n"); }
    }

    /* ── AllOps-Mini full model (18 layers, all 7 operators) ── */
    {
        uart_puts("\n[TEST 15/16] AllOps-Mini INT8 (18 layers)\n");
        test_case_t tc = { BLOB_ALLOPS_INT8_BASE, "AllOps-Mini INT8", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) {
            uart_puts("  FAIL — ");
            uart_put_dec(ret);
            uart_puts(" mismatches\n");
            global_err++;
        }
        else {
            uart_puts("  AllOps-Mini INT8: ALL ");
            uart_put_dec(tc.num_layers);
            uart_puts(" LAYERS PASSED ✓\n");
        }
    }

    {
        uart_puts("\n[TEST 16/16] AllOps-Mini INT16 (18 layers)\n");
        test_case_t tc = { BLOB_ALLOPS_INT16_BASE, "AllOps-Mini INT16", 0 };
        int ret = run_test_case(&tc);
        if (ret < 0) { uart_puts("  TEST ABORTED\n"); global_err++; }
        else if (ret > 0) {
            uart_puts("  FAIL — ");
            uart_put_dec(ret);
            uart_puts(" mismatches\n");
            global_err++;
        }
        else {
            uart_puts("  AllOps-Mini INT16: ALL ");
            uart_put_dec(tc.num_layers);
            uart_puts(" LAYERS PASSED ✓\n");
        }
    }

    /* ── Final result ── */
    uart_puts("\n========================================\n");
    if (global_err == 0) {
        uart_puts("  RESULT: ALL TESTS PASSED ✓\n");
    } else {
        uart_puts("  RESULT: ");
        uart_put_dec(global_err);
        uart_puts(" TEST(S) FAILED ✗\n");
    }
    uart_puts("========================================\n");

    while (1) {}
}
