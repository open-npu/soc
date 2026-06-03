/*
 * Open-NPU SoC Firmware — 1-Layer Conv2D Test
 *
 * Minimal bare-metal firmware that runs on the VexRiscv CPU in the
 * LiteX SoC, programs the NPU for a simple Conv2D operation, and
 * verifies the output matches expected golden data.
 *
 * Test parameters:
 *   - Input:  4x4, 1 channel, INT8
 *   - Kernel: 3x3, 1 input channel, 2 output channels
 *   - Stride: 1x1, Padding: 0
 *   - Output: 2x2, 2 channels
 *   - Post-processing: requantize with M/S/zp, ReLU, clamp to [0,127]
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>

/* ═══════════════════════════════════════════════════════════════════
 *  Memory map (from LiteX csr.csv)
 * ═══════════════════════════════════════════════════════════════════ */
#define MAIN_RAM_BASE   0x40000000UL
#define NPU_BASE_ADDR   0x80000000UL

/* UART for printf (LiteX CSR UART) */
#define UART_RXTX       (*(volatile uint32_t *)0xF0001800UL)
#define UART_TXFULL     (*(volatile uint32_t *)0xF0001804UL)

/* ═══════════════════════════════════════════════════════════════════
 *  NPU register offsets (from npu_hal.h)
 * ═══════════════════════════════════════════════════════════════════ */
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

/* Post-processing control */
#define POST_RELU_EN    (1U << 2)
#define POST_ZP_EN      (1U << 5)
#define POST_BIAS_EN    (1U << 6)

/* ═══════════════════════════════════════════════════════════════════
 *  Simple UART print
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

static void uart_put_hex8(uint8_t v) {
    const char hex[] = "0123456789ABCDEF";
    uart_putc(hex[(v >> 4) & 0xF]);
    uart_putc(hex[v & 0xF]);
}

static void uart_put_hex32(uint32_t v) {
    for (int i = 28; i >= 0; i -= 4)
        uart_putc("0123456789ABCDEF"[(v >> i) & 0xF]);
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
 *  Test data layout in main RAM
 *
 *  0x40020000: input  (4*4*1 = 16 bytes)
 *  0x40020100: weights (3*3*1*2 = 18 bytes)
 *  0x40020200: per-channel params (2 channels * 14 bytes = 28 bytes)
 *  0x40020400: output  (2*2*2 = 4 bytes, INT8)
 *  0x40020500: expected output (for comparison)
 * ═══════════════════════════════════════════════════════════════════ */
#define DATA_BASE       (MAIN_RAM_BASE + 0x20000)
#define INPUT_ADDR      (DATA_BASE + 0x000)
#define WEIGHT_ADDR     (DATA_BASE + 0x100)
#define PARAM_ADDR      (DATA_BASE + 0x200)
#define OUTPUT_ADDR     (DATA_BASE + 0x400)
#define EXPECTED_ADDR   (DATA_BASE + 0x500)

/* ═══════════════════════════════════════════════════════════════════
 *  Test vectors — Conv2D 1×1, 2×2×4 → 2×2×2
 *
 *  Input (2x2x4, NHWC layout):
 *    pixel(0,0): [1, 2, 3, 4]
 *    pixel(0,1): [5, 6, 7, 8]
 *    pixel(1,0): [9,10,11,12]
 *    pixel(1,1): [13,14,15,16]
 *
 *  Weights (1x1, IC=4, OC=2): [OC][IC] = [OC][k_depth]
 *    OC0: [1, 1, 1, 1]  → sum of input channels
 *    OC1: [2, 0, 0, 0]  → 2x first channel
 *
 *  Raw conv output (no bias/requant):
 *    pixel(0,0): OC0=1+2+3+4=10, OC1=1*2=2
 *    pixel(0,1): OC0=5+6+7+8=26, OC1=5*2=10
 *    pixel(1,0): OC0=9+10+11+12=42, OC1=9*2=18
 *    pixel(1,1): OC0=13+14+15+16=58, OC1=13*2=26
 *
 *  PPU: M=16384, S=14 → identity requant (acc * 1.0)
 *  ReLU: all positive → no change
 *  Clamp [0,127]: all < 128 → no clamp
 *
 *  Final output (NHWC):
 *    [10, 2, 26, 10, 42, 18, 58, 26]
 * ═══════════════════════════════════════════════════════════════════ */

/* Input: 2×2×4 = 16 bytes, NHWC layout */
static const int8_t test_input[16] = {
     1,  2,  3,  4,   /* pixel (0,0) */
     5,  6,  7,  8,   /* pixel (0,1) */
     9, 10, 11, 12,   /* pixel (1,0) */
    13, 14, 15, 16    /* pixel (1,1) */
};

/* Weights: 1x1 x IC=4 x OC=4, layout: [OC][k_depth=IC*KH*KW=4] */
static const int8_t test_weights[16] = {
    /* OC0: [1, 1, 1, 1] → sum of channels */
     1,  1,  1,  1,
    /* OC1: [2, 0, 0, 0] → 2x ch0 */
     2,  0,  0,  0,
    /* OC2: [0, 1, 0, 0] → ch1 */
     0,  1,  0,  0,
    /* OC3: [1, 0, 1, 0] → ch0 + ch2 */
     1,  0,  1,  0
};

/*
 * Per-channel parameters: 14 bytes each
 *   M(2) + S(1) + pad(1) + zp(2) + bias(8) = 14 bytes
 *   M=16384 (identity requant at S=14), S=14, zp=0, bias=0
 */
/* Per-channel params: 4 words (16 bytes) per channel
 * Hardware layout (from npu_compute.v param_buf[] unpacking):
 *   Word 0 [31:0]: M[14:0] in bits[14:0], S[5:0] in bits[21:16]
 *   Word 1 [31:0]: zp[15:0] in bits[15:0], bias[15:0] in bits[31:16]
 *   Word 2 [31:0]: bias[47:16]
 *   Word 3 [31:0]: bias[63:48] in bits[15:0]
 *
 * M=16384(0x4000), S=14(0x0E), zp=0, bias=0
 * Word 0 = 0x000E4000  (S<<16 | M)
 * Word 1 = 0x00000000  (bias_lo<<16 | zp)
 * Word 2 = 0x00000000  (bias_mid)
 * Word 3 = 0x00000000  (bias_hi)
 */
static const uint32_t test_params[16] = {
    /* OC0: M=16384, S=14, identity requantization */
    0x000E4000, 0x00000000, 0x00000000, 0x00000000,
    /* OC1 */
    0x000E4000, 0x00000000, 0x00000000, 0x00000000,
    /* OC2 */
    0x000E4000, 0x00000000, 0x00000000, 0x00000000,
    /* OC3 */
    0x000E4000, 0x00000000, 0x00000000, 0x00000000
};

/* Expected output: 2×2×4 INT8 layout [H][W][C] (NHWC)
 * pixel(0,0): in=[1,2,3,4] → OC0=1+2+3+4=10, OC1=1*2=2, OC2=2, OC3=1+3=4
 * pixel(0,1): in=[5,6,7,8] → OC0=5+6+7+8=26, OC1=5*2=10, OC2=6, OC3=5+7=12
 * pixel(1,0): in=[9,10,11,12] → OC0=9+10+11+12=42, OC1=9*2=18, OC2=10, OC3=9+11=20
 * pixel(1,1): in=[13,14,15,16] → OC0=13+14+15+16=58, OC1=13*2=26, OC2=14, OC3=13+15=28
 */
static const int8_t expected_output[16] = {
    10,  2,  2,  4,    /* pixel (0,0) */
    26, 10,  6, 12,    /* pixel (0,1) */
    42, 18, 10, 20,    /* pixel (1,0) */
    58, 26, 14, 28     /* pixel (1,1) */
};

/* ═══════════════════════════════════════════════════════════════════
 *  Helper: copy data to RAM address
 * ═══════════════════════════════════════════════════════════════════ */
static void memcpy_to_hw(uint32_t dst_addr, const void *src, uint32_t len) {
    volatile uint8_t *dst = (volatile uint8_t *)dst_addr;
    const uint8_t *s = (const uint8_t *)src;
    for (uint32_t i = 0; i < len; i++)
        dst[i] = s[i];
}

/* ═══════════════════════════════════════════════════════════════════
 *  Main entry point
 * ═══════════════════════════════════════════════════════════════════ */
void main(void) {
    uart_puts("\n\n");
    uart_puts("========================================\n");
    uart_puts("  Open-NPU SoC — Conv2D Test Firmware\n");
    uart_puts("========================================\n\n");

    /* ── Step 1: Read NPU version and HW config ── */
    uart_puts("[1] Reading NPU registers...\n");

    uint32_t ver = NPU_REG(REG_VERSION);
    uart_puts("    Version: ");
    uart_put_dec((ver >> 16) & 0xFF); uart_putc('.');
    uart_put_dec((ver >> 8) & 0xFF);  uart_putc('.');
    uart_put_dec(ver & 0xFF);
    uart_putc('\n');

    uint32_t hwcfg = NPU_REG(REG_HW_CONFIG);
    uart_puts("    Array size: "); uart_put_dec(hwcfg & 0xFF); uart_putc('\n');
    uart_puts("    HW config:  0x"); uart_put_hex32(hwcfg); uart_putc('\n');

    /* ── Step 2: Reset NPU ── */
    uart_puts("[2] Resetting NPU...\n");
    NPU_REG(REG_CTRL) = CTRL_SOFT_RST;

    /* Brief spin to let reset propagate */
    for (volatile int i = 0; i < 100; i++) {}

    /* Clear IRQs */
    NPU_REG(REG_IRQ_STATUS) = 0x7;
    NPU_REG(REG_IRQ_EN) = 0;

    uart_puts("    Status: 0x"); uart_put_hex32(NPU_REG(REG_STATUS)); uart_putc('\n');

    /* ── Step 3: Load test data to main RAM ── */
    uart_puts("[3] Loading test data to RAM...\n");

    memcpy_to_hw(INPUT_ADDR,  test_input,   sizeof(test_input));
    memcpy_to_hw(WEIGHT_ADDR, test_weights, sizeof(test_weights));
    memcpy_to_hw(PARAM_ADDR,  test_params,  sizeof(test_params));

    /* Fill output area with distinctive pattern (0xAA = -86) to detect if NPU writes */
    for (uint32_t i = 0; i < 32; i++)
        *(volatile uint8_t *)(OUTPUT_ADDR + i) = 0xAA;

    uart_puts("    Input @0x");  uart_put_hex32(INPUT_ADDR);  uart_putc('\n');
    uart_puts("    Weight @0x"); uart_put_hex32(WEIGHT_ADDR); uart_putc('\n');
    uart_puts("    Param @0x");  uart_put_hex32(PARAM_ADDR);  uart_putc('\n');
    uart_puts("    Output @0x"); uart_put_hex32(OUTPUT_ADDR); uart_putc('\n');

    /* ── Step 4: Program NPU CSRs ── */
    uart_puts("[4] Programming NPU for Conv2D 1x1 2x2x4 -> 2x2x4...\n");

    /* Layer mode: CONV2D (0), INT8 (0) */
    NPU_REG(REG_LAYER_MODE) = 0x00000000;

    /* Input dimensions: H=4, W=4 */
    NPU_REG(REG_IN_DIM_HW) = (2 << 16) | 2;  /* H=2 in [31:16], W=2 in [15:0] */
    NPU_REG(REG_IN_DIM_C)  = 4;

    /* Output dimensions: H=2, W=2 */
    NPU_REG(REG_OUT_DIM_HW) = (2 << 16) | 2;
    NPU_REG(REG_OUT_DIM_C)  = 4;

    /* Kernel 1×1, dilation 1×1 */
    NPU_REG(REG_KERNEL_SIZE) = (1) | (1 << 8) | (1 << 16) | (1 << 24);

    /* Stride 1×1 */
    NPU_REG(REG_STRIDE) = (1) | (1 << 8);

    /* No padding */
    NPU_REG(REG_PADDING) = 0;

    /* No tiling: tile_h=0 means "use full output dims" */
    NPU_REG(REG_TILE_CFG)   = 0;
    NPU_REG(REG_TILE_COUNT) = (1) | (1 << 16);

    /* SRAM base: act_base=0, out_base=128 words (well past input area) */
    NPU_REG(REG_SRAM_BASE) = (0) | ((128) << 16);

    /* DMA addresses */
    NPU_REG(REG_DMA_IN_ADDR)    = INPUT_ADDR;
    NPU_REG(REG_DMA_OUT_ADDR)   = OUTPUT_ADDR;
    NPU_REG(REG_DMA_WGT_ADDR)   = WEIGHT_ADDR;
    NPU_REG(REG_DMA_PARAM_ADDR) = PARAM_ADDR;

    /* DMA sizes (must be 4-byte aligned for word transfer) */
    NPU_REG(REG_DMA_IN_SIZE)  = 16;  /* 2×2×4 = 16 bytes (4 words) */
    NPU_REG(REG_DMA_WGT_SIZE) = 16;  /* 1×1×4×4 = 16 bytes (4 words) */
    NPU_REG(REG_DMA_OUT_SIZE) = 16;  /* 2×2×4 = 16 bytes (4 words) */

    /* DMA control: no transpose, default burst */
    NPU_REG(REG_DMA_CTRL) = 0;

    /* Post-processing: CONV_REQ mode (0), ReLU enabled */
    NPU_REG(REG_POST_CTRL)      = POST_RELU_EN;  /* Just ReLU, mode=CONV_REQ(0) */
    NPU_REG(REG_POST_PARAM_CNT) = 4;  /* 4 output channels */
    NPU_REG(REG_POST_CLAMP)     = (0 & 0xFFFF) | ((127 & 0xFFFF) << 16);

    uart_puts("    CSR programming done.\n");

    /* ── Step 5: Start NPU ── */
    uart_puts("[5] Starting NPU...\n");
    NPU_REG(REG_CTRL) = CTRL_START;

    /* ── Step 6: Poll for completion ──
     * NOTE: hw_done is a 1-cycle pulse in npu_ctrl. The STATUS register
     * only reflects it instantaneously. Use IRQ_STATUS register which
     * latches the done event, or poll for !BUSY as completion indicator.
     */
    uint32_t timeout = 500000;
    uint32_t status = 0;
    uint32_t irq_st = 0;
    while (timeout--) {
        status = NPU_REG(REG_STATUS);
        irq_st = NPU_REG(REG_IRQ_STATUS);
        /* Done: IRQ status bit 0 latched, or not-busy after start */
        if (irq_st & 0x1)  break;  /* IRQ_DONE latched */
        if (status & STATUS_ERROR) break;
        /* Also check: was busy, now idle = done */
        if (!(status & STATUS_BUSY) && (status & 0xFF00)) break;
    }

    if (status & STATUS_ERROR) {
        uart_puts("    ERROR! Status: 0x");
        uart_put_hex32(status);
        uart_puts("\n    FAIL\n");
        while (1) {}
    }

    if (!(irq_st & 0x1) && (status & STATUS_BUSY)) {
        uart_puts("    TIMEOUT! Status: 0x");
        uart_put_hex32(status);
        uart_puts(" IRQ: 0x");
        uart_put_hex32(irq_st);
        uart_puts("\n    FAIL\n");
        while (1) {}
    }

    uart_puts("    NPU done. Status: 0x");
    uart_put_hex32(status);
    uart_putc('\n');

    /* ── Step 7: Verify output ── */
    uart_puts("[6] Verifying output...\n");

    int errors = 0;
    volatile uint8_t *out_ptr = (volatile uint8_t *)OUTPUT_ADDR;

    uart_puts("    Output[0..15]: ");
    for (int i = 0; i < 16; i++) {
        int8_t got = (int8_t)out_ptr[i];
        uart_put_dec(got);
        uart_putc(' ');
    }
    uart_putc('\n');

    /* Count mismatches for all 16 output bytes */
    for (int i = 0; i < 16; i++) {
        int8_t got = (int8_t)out_ptr[i];
        int8_t exp = expected_output[i];
        if (got != exp) errors++;
    }

    uart_puts("    Expect: ");
    for (int i = 0; i < 16; i++) {
        uart_put_dec(expected_output[i]);
        uart_putc(' ');
    }
    uart_putc('\n');

    /* ── Result ── */
    uart_puts("\n========================================\n");
    if (errors == 0) {
        uart_puts("  RESULT: PASS — Conv2D output matches!\n");
    } else {
        uart_puts("  RESULT: FAIL — ");
        uart_put_dec(errors);
        uart_puts(" mismatches!\n");
    }
    uart_puts("========================================\n");

    /* Halt */
    while (1) {}
}
