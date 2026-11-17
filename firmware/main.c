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
#define REG_ERROR_CODE      0x010
#define REG_VERSION         0x014
#define REG_HW_CONFIG       0x018
#define REG_PERF_CYCLES     0x01C
#define REG_PERF_MACS       0x020
#define REG_PERF_DMA        0x034
#define REG_PERF_COMPUTE    0x038
#define REG_PERF_OVERLAP    0x03C
#define REG_ST_MAC          0x1A0
#define REG_ST_WAIT         0x1A4
#define REG_ST_WB           0x1A8
#define REG_ST_POOL         0x1AC
#define REG_ST_ADD          0x1B0
#define REG_ST_RSZ          0x1B4
#define REG_ST_CFG          0x1B8
#define REG_ST_OTH          0x1BC
#define REG_WT_ACT          0x1C0
#define REG_WT_WGT          0x1C4
#define REG_WT_DB           0x1C8
#define REG_MAC_IDLE        0x1CC
#define REG_WT_STORE        0x1D0
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
#define REG_WGT_LAYOUT      0x14C
#define REG_NEXT_PARAM_ADDR 0x150
#define REG_NEXT_PARAM_CFG  0x154
#define REG_PARAM_BANK_CTRL 0x158
#define REG_NEXT_PARAM_STATUS 0x15C
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
#define IRQ_DONE        (1U << 0)
#define IRQ_ERROR       (1U << 1)

#define MAX_LAYERS 80

/* Must match soc/firmware/build_main_ram.py OUT_POISON.
 * Workspace DDR outputs are initialized to this, not golden. */
#define OUT_POISON      0xDEADBEEFu

/* sched_ctrl bits (model descriptor, programmed into DMA_CTRL). */
#define SCHED_DB_EN         (1U << 0)
#define SCHED_FUSE_START    (1U << 1)
#define SCHED_FUSE_MID      (1U << 2)
#define SCHED_FUSE_END      (1U << 3)
#define SCHED_FUSE_MASK     (SCHED_FUSE_START | SCHED_FUSE_MID | SCHED_FUSE_END)
#define SCHED_FUSE_STRIP    (1U << 5) /* Firmware-only Conv→Pool strip schedule. */
#define SCHED_NO_AUTO_DB    (1U << 6) /* Firmware-only: keep single-buffer PTS. */

/* HW_CONFIG (0x018) [23:16] is the total on-chip SRAM in KB
 * (IFM 24 + OFM 48 + weight 80 + param 12 = 164). Fusion swaps the
 * two activation banks, so the live cap is the smaller one: IFM,
 * 24KB = 6144 words. DB_EN halves that in sram_fuse_reuse.
 * Compile with -DNPU_IFM_KB_OVERRIDE=N to pin the IFM size. */
#ifndef NPU_SPAD_KB_OVERRIDE
#define NPU_SPAD_KB_OVERRIDE 0
#endif
#ifndef NPU_IFM_KB_OVERRIDE
#define NPU_IFM_KB_OVERRIDE 0
#endif
#define HW_CFG_ARRAY_SIZE(v) ((v) & 0xFFu)
#define HW_CFG_SPAD_KB(v)    (((v) >> 16) & 0xFFu)
#define NPU_SPAD_KB_DEFAULT  164u
#define NPU_IFM_KB           24u

static uint32_t g_spad_kb;
static uint32_t g_act_words;

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
static __attribute__((unused)) void
dcache_inval_range(volatile void *ptr, uint32_t size) {
    /* VexRiscv: use flush (0x500F) which also invalidates clean lines.
     * For NPU output, lines are clean (CPU didn't write), so flush = invalidate. */
    (void)ptr;
    (void)size;
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
static __attribute__((unused)) void
memcpy_32(uint32_t dst, const uint32_t *src, uint32_t n_words) {
    volatile uint32_t *d = (volatile uint32_t *)dst;
    for (uint32_t i = 0; i < n_words; i++)
        d[i] = src[i];
}

/* ═══════════════════════════════════════════════════════════════════
 *  NPU SRAM geometry (from HW_CONFIG, not a baked-in 192KB)
 *  ═══════════════════════════════════════════════════════════════════ */
static uint32_t npu_act_words_from_spad_kb(uint32_t spad_kb) {
    /* Fusion cap is IFM, not total/4. spad_kb is only the CSR total. */
    (void)spad_kb;
    return (NPU_IFM_KB_OVERRIDE ? (uint32_t)NPU_IFM_KB_OVERRIDE
                                : (uint32_t)NPU_IFM_KB) * 256u;
}

static void npu_hw_probe(void) {
    uint32_t hwcfg = NPU_REG(REG_HW_CONFIG);
    uint32_t spad = NPU_SPAD_KB_OVERRIDE ? (uint32_t)NPU_SPAD_KB_OVERRIDE
                                         : HW_CFG_SPAD_KB(hwcfg);
    if (spad == 0)
        spad = NPU_SPAD_KB_DEFAULT;
    g_spad_kb = spad;
    g_act_words = npu_act_words_from_spad_kb(spad);
}

static uint16_t layer_tile_h(const uint32_t *e) {
    return (uint16_t)(e[19] & 0xFFFFu);
}

static uint32_t layer_out_base(const uint32_t *e) {
    /* IFM and OFM are separate physical banks. Conv/DW/Pool/Resize/FC/Add
     * write logical OFM at word 0; after role swap the next layer therefore
     * reads from word 0. Concat alone scatters after its resident input. */
    return ((e[4] & 0xFu) == 7u)
        ? ((e[23] > 0) ? (e[23] / 4u) : (e[15] / 4u))
        : 0u;
}

static uint32_t layer_in_words(const uint32_t *e) {
    return e[15] / 4u;
}

static uint32_t layer_out_words(const uint32_t *e) {
    return e[17] / 4u;
}

/* Is it safe to double-buffer this layer's tiles?
 *
 * DB_EN splits Act SRAM in half: the tile being computed plus everything it
 * reads must fit one half, because the other half holds the next tile's
 * prefetch. Add and Concat keep branch B in the same physical bank as branch
 * A, so their footprint is A + B, not just A. Checking only A let a
 * 19x20x32 Add tile (3040 + 3040 words) pass a 3072-word half and stomp the
 * prefetched half, which is what made L5/L8 fail.
 *
 * Returns the A+B words needed in the IFM half, or 0 if the layer cannot
 * use DB_EN at all. */
static int layer_db_ok(const uint32_t *e) {
    uint32_t op = e[4] & 0xFu;
    uint32_t tile_in, need, eb, tile_out, concat_b;

    if (layer_tile_h(e) == 0)          /* untiled: nothing to overlap */
        return 0;
    if (e[19] == 0)                    /* no tile_w either */
        return 0;
    if (g_act_words == 0)
        return 0;
    if (e[21] & SCHED_NO_AUTO_DB)
        return 0;
    /* Group-major traversal and DB_EN are mutually exclusive in the RTL. */
    if ((e[22] & 0x3u) == 0x3u)
        return 0;
    /* Concat scatters into an offset output region and later phases preload
     * that region. Its bank lifetime is not compatible with the current
     * background DB scheduler; keep the proven single-buffer path. */
    if (op == 7u)
        return 0;

    tile_in = e[23] / 4u;              /* tile_in_size is bytes */
    if (tile_in == 0)
        return 0;

    /* Add always has a same-size branch B. Concat phase 0 owns the initial
     * channels and has no carry-over preload; later phases preload a tile
     * with the full output-channel pitch. */
    eb = (e[5] & 1u) ? 2u : 1u;
    if (op == 4u) {
        need = tile_in * 2u;
    } else if (op == 7u && (e[18] & 0xFFFFu) != 0u) {
        concat_b = (uint32_t)(e[19] & 0xFFFFu)
                 * (uint32_t)(e[19] >> 16) * e[9] * eb / 4u;
        need = tile_in + concat_b;
    } else {
        need = tile_in;
    }

    if (need > g_act_words / 2u)
        return 0;

    /* The output tile must fit the OFM half (OFM bank is 2x IFM bank). */
    tile_out = (uint32_t)(e[19] & 0xFFFFu) * (uint32_t)(e[19] >> 16)
             * e[9] * eb / 4u;
    if (tile_out == 0)
        tile_out = e[24] / 4u;
    if (tile_out > g_act_words)
        return 0;

    return 1;
}

/* SRAM reuse is only valid when both layers are untiled (RTL skip_act_load
 * requires tile_h==0) AND the tensors fit in the live Act SRAM.
 * Converter now stamps one large tile grid on each IR (1×1→DW→1×1 [+Add]).
 * Tiled same-grid still cannot skip DMA: compute finishes all tiles of
 * START before MID starts, so only the last tile would remain in SRAM.
 * Mixed tiled START + untiled MID (M_X L16→L17) falls back to DDR. A later
 * untiled FUSE_END can still reuse SRAM after that DDR-loaded MID. */
static int sram_fuse_reuse(const uint32_t *prev, const uint32_t *cur) {
    uint32_t ps, cs, act_base, need_in, need_out, cap;
    if (!prev || !cur || g_act_words == 0)
        return 0;
    if (layer_tile_h(prev) != 0 || layer_tile_h(cur) != 0)
        return 0;
    ps = prev[21];
    cs = cur[21];
    if ((ps & (SCHED_FUSE_START | SCHED_FUSE_MID)) == 0)
        return 0;
    if ((cs & (SCHED_FUSE_MID | SCHED_FUSE_END)) == 0)
        return 0;
    act_base = layer_out_base(prev);
    need_in  = act_base + layer_in_words(cur);
    need_out = layer_out_base(cur) + layer_out_words(cur);
    cap = g_act_words;
    /* DB_EN confines each bank to half of Act SRAM. */
    if ((cs & SCHED_DB_EN) || (ps & SCHED_DB_EN))
        cap = g_act_words / 2u;
    if (need_in > cap || need_out > cap)
        return 0;
    return 1;
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
        /* Capacity guards pulse hw_error for 1 cycle then go idle. The
         * latched IRQ_ERROR bit is the durable signal; STATUS_ERROR is not.
         * Idle (!BUSY) is the reset/error state, not completion — treating
         * it as done made poison cells report npu=0 instead of ERROR. */
        if (irq_st & IRQ_ERROR) return -1;
        if (status & STATUS_ERROR) return -1;
        if (irq_st & IRQ_DONE) return 0;
    }
    return -2; /* timeout */
}

/*
 * npu_program_layer — program ALL CSRs from 37-word layer_entry_t.
 * Mirrors test_npu_dma_e2e.py:program_layer() CSR sequence.
 *
 * Parameters:
 *   e: pointer to 37-word layer entry
 *   runtime_in_addr: resolved input DDR address (layer 0: from entry[32],
 *                    N>0: producing layer's ddr_out_addr)
 *   runtime_add_b_addr: resolved Add/Concat branch-B DDR address (0 if none)
 *   act_base: SRAM word address to read activations from.
 *             0 for a normal DDR load; previous layer's out_base for
 *             non-tiled FUSE_MID/END (skip_act_load).
 *   dma_ctrl: sched_ctrl with FUSE bits kept only when SRAM reuse is safe.
 */
static void npu_program_layer(const uint32_t *e,
                              uint32_t runtime_in_addr,
                              uint32_t runtime_add_b_addr,
                              uint32_t act_base,
                              uint32_t dma_ctrl,
                              int param_preloaded,
                              uint32_t param_bank,
                              uint32_t next_param_addr,
                              uint32_t next_param_words,
                              uint32_t next_param_bank) {
    uint32_t op_type   = e[4];
    uint32_t data_type = e[5];
    uint32_t in_zp     = e[28];

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

    /* OFM is a separate bank, so conv/depthwise/pool write at word 0.
     * The old offset (input tile size) placed the output after the
     * input in a shared SRAM. On a wide layer that offset plus the
     * output tile no longer fits in the OFM half (model_x L60).
     * Add keeps the offset: branch B is loaded after branch A in the IFM. */
    uint32_t out_base = 0;
    if (op_type == 4 || op_type == 7)
        out_base = (e[23] > 0) ? (e[23] / 4) : (e[15] / 4);
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

    /* Weight blob layout: the 64-lane row uses K-major dense weights.
     * Must be written every layer so a stale compatibility value cannot
     * select an unsupported legacy layout. */
    NPU_REG(REG_WGT_LAYOUT) = e[36];
    NPU_REG(REG_NEXT_PARAM_ADDR) = next_param_addr;
    NPU_REG(REG_NEXT_PARAM_CFG) = next_param_words
        ? (0x80000000u | next_param_words) : 0;
    NPU_REG(REG_PARAM_BANK_CTRL) = (param_bank & 1u)
        | (param_preloaded ? 2u : 0u) | ((next_param_bank & 1u) << 2);

    /* Tiled DB_EN prefetch + PTS 2D DMA */
    NPU_REG(REG_DMA_TILE_IN_SIZE) = e[23];
    /* STORE_MODE must be written unconditionally — in a chained run a stale
     * PTS_EN=1 from a previous tiled layer would otherwise divert a non-PTS
     * layer's final store to the S_TILE_STORE path with the wrong source
     * address (hit by model_x L61 after L60: output read back as zeros). */
    NPU_REG(REG_DMA_STORE_MODE)    = e[22];
    NPU_REG(REG_DMA_TILE_OUT_SIZE) = e[24];
    NPU_REG(REG_DMA_ROW_CFG)       = e[25];

    /* Strides: in_stride = in_w * in_c * elem_bytes for 2D tiled load (chain mode) */
    {
        uint32_t in_w = e[6] & 0xFFFF;  /* IN_DIM_HW[15:0] = W */
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

    /* DMA control: keep DB_EN/PTS. Keep FUSE_* only when the caller decided
     * SRAM reuse is safe (dma_ctrl already stripped otherwise). */
    NPU_REG(REG_DMA_CTRL) = dma_ctrl;

    /* Post-processing */
    NPU_REG(REG_POST_CTRL)      = e[13];
    NPU_REG(REG_POST_PARAM_CNT) = e[14];
    NPU_REG(REG_POST_CLAMP)     = e[27];  /* clamp_max in [15:0] */
}

static uint32_t layer_param_words(const uint32_t *e) {
    uint32_t op = e[4] & 0xF;
    return (op == 4 || op == 7) ? e[14] : e[14] * 4;
}

static void copy_layer_entry(uint32_t *dst, const uint32_t *src) {
    for (int i = 0; i < LAYER_ENTRY_HDR_WORDS; i++)
        dst[i] = src[i];
}

/* Execute a large Conv→Pool pair as SRAM-resident full-width strips.
 *
 * L0 recomputes the one-row overlap required by 3x3/s2 Pool. FUSE_START
 * leaves each producer strip in physical OFM; FUSE_END swaps the activation
 * roles and consumes it without a DDR round trip. Only Pool output reaches
 * DDR, at its final global row offset.
 */
static int run_conv_pool_strips(const uint32_t *conv_e,
                                const uint32_t *pool_e) {
    uint32_t conv[LAYER_ENTRY_HDR_WORDS];
    uint32_t pool[LAYER_ENTRY_HDR_WORDS];
    uint32_t producer_h = conv_e[19] & 0xFFFFu;
    uint32_t producer_w = conv_e[19] >> 16;
    uint32_t strip_count = conv_e[20] & 0xFFFFu;
    uint32_t pool_strip_h = conv_e[20] >> 16;
    uint32_t input_slot = conv_e[23];
    uint32_t conv_out_h = conv_e[8] >> 16;
    uint32_t pool_out_h = pool_e[8] >> 16;
    uint32_t pool_out_w = pool_e[8] & 0xFFFFu;
    uint32_t pool_out_c = pool_e[9];
    uint32_t elem_bytes = (pool_e[5] & 1u) ? 2u : 1u;
    uint32_t conv_kh = conv_e[10] & 0xFFu;
    uint32_t conv_kw = (conv_e[10] >> 8) & 0xFFu;
    uint32_t conv_sh = conv_e[11] & 0xFFu;
    uint32_t conv_sw = (conv_e[11] >> 8) & 0xFFu;
    uint32_t local_in_h = (producer_h - 1u) * conv_sh + conv_kh;
    uint32_t local_in_w = (producer_w - 1u) * conv_sw + conv_kw;

    if (!producer_h || !producer_w || !strip_count || !pool_strip_h
        || !input_slot || elem_bytes != 1u)
        return -1;

    for (uint32_t s = 0; s < strip_count; s++) {
        uint32_t pool_y = s * pool_strip_h;
        uint32_t valid_pool_h = pool_out_h - pool_y;
        uint32_t producer_y = pool_y * 2u;
        uint32_t valid_pool_in_h = conv_out_h - producer_y;
        if (valid_pool_h > pool_strip_h)
            valid_pool_h = pool_strip_h;
        if (valid_pool_in_h > producer_h)
            valid_pool_in_h = producer_h;

        copy_layer_entry(conv, conv_e);
        conv[6] = local_in_w | (local_in_h << 16);
        conv[8] = producer_w | (producer_h << 16);
        conv[12] = 0; /* Strip input already contains global zero-point padding. */
        conv[15] = input_slot;
        conv[17] = producer_h * producer_w * conv_e[9] * elem_bytes;
        conv[19] = 0;
        conv[20] = 0x00010001u;
        conv[21] = SCHED_FUSE_START;
        conv[22] = 0;
        conv[23] = 0;
        conv[24] = 0;
        conv[25] = 0;

        NPU_REG(REG_IRQ_STATUS) = 0x7;
        npu_program_layer(conv, conv_e[32] + s * input_slot, 0, 0,
                          SCHED_FUSE_START, 0, 0, 0, 0, 0);
        dcache_flush();
        NPU_REG(REG_CTRL) = CTRL_START;
        dcache_flush();
        if (npu_wait_done() != 0)
            return -2;

        copy_layer_entry(pool, pool_e);
        pool[6] = producer_w | (valid_pool_in_h << 16);
        pool[8] = pool_out_w | (valid_pool_h << 16);
        pool[15] = 0;
        pool[17] = valid_pool_h * pool_out_w * pool_out_c * elem_bytes;
        pool[19] = 0;
        pool[20] = 0x00010001u;
        pool[21] = SCHED_FUSE_END;
        pool[22] = 0;
        pool[23] = 0;
        pool[24] = 0;
        pool[25] = 0;
        pool[31] = pool_e[31]
                 + pool_y * pool_out_w * pool_out_c * elem_bytes;

        NPU_REG(REG_IRQ_STATUS) = 0x7;
        npu_program_layer(pool, 0, 0, 0, SCHED_FUSE_END,
                          0, 0, 0, 0, 0);
        dcache_flush();
        NPU_REG(REG_CTRL) = CTRL_START;
        dcache_flush();
        if (npu_wait_done() != 0)
            return -3;
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Chained model inference
 *  ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    uint32_t blob_base;
    const char *name;
} test_case_t;

#ifdef PERF_ONLY
/* Per-layer histogram snapshot, printed after PERF_END so the UART traffic
 * does not extend the measured interval. 13 words per layer.
 *
 * Reading the counters still costs a few thousand cycles per model, so the
 * snapshot is behind PERF_HIST: a plain PERF_ONLY build measures the
 * production interval untouched, and HIST=1 builds add the breakdown. */
#define PERF_LAYER_MAX 80
static uint32_t perf_layer_log[PERF_LAYER_MAX * 14];
static int      perf_layer_idx;
#endif

static int run_chained_model(test_case_t *tc) {
    uart_puts("\n  ----------------------------------------\n");
    uart_puts("  Model: ");
    uart_puts(tc->name);
    uart_puts("\n  ----------------------------------------\n");

    if (g_act_words == 0)
        npu_hw_probe();

    const uint32_t *blob = (const uint32_t *)(uintptr_t)tc->blob_base;
    uint32_t magic = blob[0];
    uint32_t num_layers = blob[1];

    if (magic != BLOB_MAGIC) {
        uart_puts("  BLOB MAGIC MISMATCH: 0x");
        uart_put_hex32(magic);
        uart_puts("\n  SKIP\n");
        return -1;
    }
    if (num_layers > MAX_LAYERS) {
        uart_puts("  TOO MANY LAYERS: ");
        uart_put_dec(num_layers);
        uart_putc('\n');
        return -1;
    }

    uart_puts("  Layers: ");
    uart_put_dec(num_layers);
    uart_putc('\n');

    /* First pass: layer pointers + skip/residual lookup */
    uint32_t layer_out_addr[MAX_LAYERS];
    const uint32_t *layer_e[MAX_LAYERS];
    const uint32_t *cursor = blob + 3;  /* skip 3-word blob header */

    for (uint32_t l = 0; l < num_layers; l++) {
        const uint32_t *e = cursor;
        layer_e[l] = e;
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
    int param_preloaded = 0;
#ifdef PERF_ONLY
    perf_layer_idx = 0;
#endif
    uint32_t param_bank = 0;
    uint32_t param_half_words = NPU_REG(REG_NEXT_PARAM_STATUS) >> 16;
    cursor = blob + 3;

#ifdef PERF_ONLY
    /* Timestamped by sim_main.cpp.  Everything between these markers is
     * production-style chained inference: layer programming, DMA, compute,
     * and inter-layer control, but no CPU golden comparison or debug UART. */
    uart_puts("PERF_BEGIN\n");
#endif

    for (uint32_t l = 0; l < num_layers; l++) {
        const uint32_t *e = cursor;
        const uint32_t *prev = (l > 0) ? layer_e[l - 1] : 0;
        const uint32_t *next = (l + 1 < num_layers) ? layer_e[l + 1] : 0;

        if ((e[21] & SCHED_FUSE_STRIP) && next) {
            const uint32_t *pool_cursor =
                cursor + LAYER_ENTRY_HDR_WORDS + e[0] + e[1] + e[2] + e[3];
            int ret = run_conv_pool_strips(e, next);
            if (ret != 0) {
                uart_puts("  L0->L1 STRIP FUSION ERROR ");
                uart_put_dec((uint32_t)(-ret));
                uart_putc('\n');
                return -1;
            }
            param_preloaded = 0;
            param_bank = 0;
#ifdef PERF_ONLY
#ifdef PERF_HIST
            if (perf_layer_idx < PERF_LAYER_MAX) {
                uint32_t *p = &perf_layer_log[perf_layer_idx * 14];
                p[0]  = NPU_REG(REG_ST_MAC);
                p[1]  = NPU_REG(REG_ST_WAIT);
                p[2]  = NPU_REG(REG_ST_WB);
                p[3]  = NPU_REG(REG_ST_POOL);
                p[4]  = NPU_REG(REG_ST_ADD);
                p[5]  = NPU_REG(REG_ST_RSZ);
                p[6]  = NPU_REG(REG_ST_CFG);
                p[7]  = NPU_REG(REG_ST_OTH);
                p[8]  = NPU_REG(REG_WT_ACT);
                p[9]  = NPU_REG(REG_WT_WGT);
                p[10] = NPU_REG(REG_WT_DB);
                p[11] = NPU_REG(REG_MAC_IDLE);
                p[12] = NPU_REG(REG_PERF_DMA);
                p[13] = NPU_REG(REG_WT_STORE);
                perf_layer_idx++;
            }
#endif
#else
            dcache_flush();
            {
                uint32_t n_output = next[3];
                const uint32_t *golden =
                    pool_cursor + LAYER_ENTRY_HDR_WORDS
                    + next[0] + next[1] + next[2];
                volatile const uint32_t *out_ptr =
                    (volatile const uint32_t *)(uintptr_t)next[31];
                uint32_t mismatches = 0;
                for (uint32_t i = 0; i < n_output; i++) {
                    if (out_ptr[i] != golden[i]) {
                        if (mismatches < 10) {
                            uart_puts("  L1 fused w[");
                            uart_put_dec(i);
                            uart_puts("]: exp=0x");
                            uart_put_hex32(golden[i]);
                            uart_puts(" got=0x");
                            uart_put_hex32(out_ptr[i]);
                            uart_putc('\n');
                        }
                        mismatches++;
                    }
                }
                if (mismatches) {
                    uart_puts("  L0->L1 STRIP FUSION FAIL: ");
                    uart_put_dec(mismatches);
                    uart_puts(" mismatches\n");
                    total_err += mismatches;
                } else {
                    uart_puts("  L0: PASS (Conv output kept in SRAM strips)\n");
                    uart_puts("  L1: PASS (Conv->Pool strip fusion, ");
                    uart_put_dec(n_output);
                    uart_puts(" words)\n");
                }
            }
#endif
            cursor = pool_cursor + LAYER_ENTRY_HDR_WORDS
                   + next[0] + next[1] + next[2] + next[3];
            l++; /* L1 was consumed by the strip scheduler. */
            continue;
        }

        int reuse = sram_fuse_reuse(prev, e);
        /* Keep FUSE_START on an untiled producer only if the next layer
         * will actually skip_act_load. Otherwise START must store to DDR. */
        int keep_fuse = reuse;
        if (!keep_fuse && next && (e[21] & SCHED_FUSE_START) && layer_tile_h(e) == 0
            && sram_fuse_reuse(e, next))
            keep_fuse = 1;
        uint32_t dma_ctrl = keep_fuse ? e[21] : (e[21] & ~SCHED_FUSE_MASK);
        /* Enable double buffering on any tiled layer whose tile plus its
         * branch-B tensor fits one bank half. Without this every tile
         * boundary stalls compute for the store plus the next prefetch.
         * PTS must already be on: DB_EN halves the bank, so stores have to
         * go out per tile rather than as one final dump. */
        if ((dma_ctrl & SCHED_DB_EN) == 0 && (e[22] & 1u) && layer_db_ok(e))
            dma_ctrl |= SCHED_DB_EN;
#ifndef PERF_ONLY
        int skip_store = keep_fuse && layer_tile_h(e) == 0
                         && (e[21] & (SCHED_FUSE_START | SCHED_FUSE_MID));
#endif

        uint32_t op_type = e[4];
        int32_t input_src = (int32_t)e[34];
        int32_t residual_src = (int32_t)e[35];

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

        /* Preserve Act SRAM across FUSE_MID/END. Soft-reset otherwise so a
         * mixed tiled→untiled block cannot read a stale tiled layout. */
        if (l > 0 && !reuse) {
            NPU_REG(REG_CTRL) = CTRL_SOFT_RST;
            for (volatile int d = 0; d < 100; d++) {}
        }
        NPU_REG(REG_IRQ_STATUS) = 0x7;  /* always W1C before START */

        uint32_t act_base = reuse ? layer_out_base(prev) : 0;

        uint32_t next_param_words = 0;
        uint32_t next_param_addr = 0;
        uint32_t next_param_bank = param_bank ^ 1u;
        if (next && e[19] == 0
            && layer_param_words(e) <= param_half_words
            && layer_param_words(next) <= param_half_words) {
            next_param_words = layer_param_words(next);
            next_param_addr = next[30];
        }

        npu_program_layer(e, runtime_in_addr, runtime_add_b_addr, act_base, dma_ctrl,
                          param_preloaded, param_bank,
                          next_param_addr, next_param_words, next_param_bank);
        dcache_flush();
#ifndef PERF_ONLY
        if (l < 3 || (e[21] & SCHED_FUSE_MASK)) {
            uart_puts("  DBG L"); uart_put_dec(l);
            uart_puts(": in_addr=0x"); uart_put_hex32(runtime_in_addr);
            uart_puts(" e32=0x"); uart_put_hex32(e[32]);
            uart_puts(" in_stride=0x"); uart_put_hex32(NPU_REG(0x110));
            uart_puts(" act_base="); uart_put_dec(act_base);
            uart_puts(" dma_ctrl=0x"); uart_put_hex32(NPU_REG(REG_DMA_CTRL));
            uart_puts(reuse ? " SRAM_FUSE\n" : "\n");
        }
#endif

        /* Start NPU */
        NPU_REG(REG_CTRL) = CTRL_START;
        dcache_flush();  /* Ensure CTRL_START reaches NPU */

        /* Wait for completion */
        int ret = npu_wait_done();
        param_preloaded = next_param_words
            && (NPU_REG(REG_NEXT_PARAM_STATUS) & 1u);
        param_bank = param_preloaded ? next_param_bank : 0;
        /* Perf counters: busy cycles + MAC issues since last soft-rst */
#ifndef PERF_ONLY
        uint32_t perf_npu     = NPU_REG(REG_PERF_CYCLES);
        uint32_t perf_macs    = NPU_REG(REG_PERF_MACS);
        uint32_t perf_dma     = NPU_REG(REG_PERF_DMA);
        uint32_t perf_compute = NPU_REG(REG_PERF_COMPUTE);
        uint32_t perf_overlap = NPU_REG(REG_PERF_OVERLAP);
        uint32_t st_mac       = NPU_REG(REG_ST_MAC);
        uint32_t st_wait      = NPU_REG(REG_ST_WAIT);
        uint32_t st_wb        = NPU_REG(REG_ST_WB);
        uint32_t st_pool      = NPU_REG(REG_ST_POOL);
        uint32_t st_add       = NPU_REG(REG_ST_ADD);
        uint32_t st_rsz       = NPU_REG(REG_ST_RSZ);
        uint32_t st_cfg       = NPU_REG(REG_ST_CFG);
        uint32_t st_oth       = NPU_REG(REG_ST_OTH);
#endif
        if (ret != 0) {
            uart_puts("  L");
            uart_put_dec(l);
            if (ret == -1) {
                uart_puts(": ERROR (status=0x");
                uart_put_hex32(NPU_REG(REG_STATUS));
                uart_puts(" irq=0x");
                uart_put_hex32(NPU_REG(REG_IRQ_STATUS));
                uart_puts(" code=0x");
                uart_put_hex32(NPU_REG(REG_ERROR_CODE));
                uart_puts(")\n");
            } else {
                uart_puts(": TIMEOUT (status=0x");
                uart_put_hex32(NPU_REG(REG_STATUS));
                uart_puts(" irq=0x");
                uart_put_hex32(NPU_REG(REG_IRQ_STATUS));
                uart_puts(" code=0x");
                uart_put_hex32(NPU_REG(REG_ERROR_CODE));
                uart_puts(")\n");
            }
            return -1;
        }

#ifdef PERF_ONLY
        /* No CPU touches the DMA output in performance mode, so no cache
         * flush or golden walk is needed.  The following layer consumes the
         * output directly from DDR through the NPU DMA.
         *
         * The counters are buffered and printed after PERF_END: a UART line
         * per layer costs thousands of cycles and would otherwise inflate
         * the very interval it is supposed to measure. */
#ifdef PERF_HIST
        if (perf_layer_idx < PERF_LAYER_MAX) {
            uint32_t *p = &perf_layer_log[perf_layer_idx * 14];
            p[0]  = NPU_REG(REG_ST_MAC);
            p[1]  = NPU_REG(REG_ST_WAIT);
            p[2]  = NPU_REG(REG_ST_WB);
            p[3]  = NPU_REG(REG_ST_POOL);
            p[4]  = NPU_REG(REG_ST_ADD);
            p[5]  = NPU_REG(REG_ST_RSZ);
            p[6]  = NPU_REG(REG_ST_CFG);
            p[7]  = NPU_REG(REG_ST_OTH);
            p[8]  = NPU_REG(REG_WT_ACT);
            p[9]  = NPU_REG(REG_WT_WGT);
            p[10] = NPU_REG(REG_WT_DB);
            p[11] = NPU_REG(REG_MAC_IDLE);
            p[12] = NPU_REG(REG_PERF_DMA);
            p[13] = NPU_REG(REG_WT_STORE);
            perf_layer_idx++;
        }
#else
        (void)0;
#endif
        cursor += LAYER_ENTRY_HDR_WORDS + e[0] + e[1] + e[2] + e[3];
        continue;
#else
        /* Flush DCache so CPU reads DMA-written DDR output (not stale cache) */
        dcache_flush();

        /* Verify: workspace was poison, not golden. A layer that did not
         * run used to PASS because ddr_out already held the expected words. */
        uint32_t n_output = e[3];
        const uint32_t *golden = cursor + LAYER_ENTRY_HDR_WORDS + e[0] + e[1] + e[2];
        volatile const uint32_t *out_ptr = (volatile const uint32_t *)(uintptr_t)e[31];
        int layer_err = 0;
        int poison_left = 0;

        uart_puts("  PERF_L");
        uart_put_dec(l);
        uart_puts(" npu=");
        uart_put_dec(perf_npu);
        uart_puts(" mac=");
        uart_put_dec(perf_macs);
        uart_puts(" dma=");
        uart_put_dec(perf_dma);
        uart_puts(" comp=");
        uart_put_dec(perf_compute);
        uart_puts(" ovlp=");
        uart_put_dec(perf_overlap);
        uart_puts("\n");
        uart_puts("  STL");
        uart_put_dec(l);
        uart_puts(" mac=");
        uart_put_dec(st_mac);
        uart_puts(" wait=");
        uart_put_dec(st_wait);
        uart_puts(" wb=");
        uart_put_dec(st_wb);
        uart_puts(" pool=");
        uart_put_dec(st_pool);
        uart_puts(" add=");
        uart_put_dec(st_add);
        uart_puts(" rsz=");
        uart_put_dec(st_rsz);
        uart_puts(" cfg=");
        uart_put_dec(st_cfg);
        uart_puts(" oth=");
        uart_put_dec(st_oth);
        uart_puts(" wact=");
        uart_put_dec(NPU_REG(REG_WT_ACT));
        uart_puts(" wwgt=");
        uart_put_dec(NPU_REG(REG_WT_WGT));
        uart_puts(" wdb=");
        uart_put_dec(NPU_REG(REG_WT_DB));
        uart_puts(" midle=");
        uart_put_dec(NPU_REG(REG_MAC_IDLE));
        uart_puts(" wst=");
        uart_put_dec(NPU_REG(REG_WT_STORE));
        uart_puts("\n");
        if (perf_npu == 0) {
            uart_puts("  L");
            uart_put_dec(l);
            uart_puts(": FAIL — npu busy cycles=0 (layer did not run)\n");
            layer_err++;
        }

        if (skip_store) {
            if (layer_err == 0) {
                uart_puts("  L");
                uart_put_dec(l);
                uart_puts(": PASS (SRAM fuse, DDR store skipped)\n");
            } else {
                total_err += layer_err;
            }
        } else if (n_output == 0) {
            if (layer_err == 0) {
                uart_puts("  L");
                uart_put_dec(l);
                uart_puts(": PASS (empty output buffer, DDR compare omitted)\n");
            }
        } else {
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
                    if (out_ptr[i] == OUT_POISON && golden[i] != OUT_POISON)
                        poison_left++;
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

            if (poison_left) {
                uart_puts("  L");
                uart_put_dec(l);
                uart_puts(": FAIL — ");
                uart_put_dec(poison_left);
                uart_puts(" poison words not overwritten\n");
            }
            if (layer_err == 0 && poison_left == 0) {
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
                total_err += layer_err + poison_left;
            }
        }
        if (!skip_store && n_output == 0 && layer_err)
            total_err += layer_err;

        /* Advance cursor */
        cursor += LAYER_ENTRY_HDR_WORDS + e[0] + e[1] + e[2] + e[3];
#endif
    }

#ifdef PERF_ONLY
    uart_puts("PERF_END\n");
#ifdef PERF_HIST
    /* Now that the measured interval is closed, emit the per-layer
     * histogram. Same fields as the checked build's STL lines. */
    for (int s = 0; s < perf_layer_idx; s++) {
        const uint32_t *p = &perf_layer_log[s * 14];
        uart_puts("STL");
        uart_put_dec((uint32_t)s);
        uart_puts(" mac=");   uart_put_dec(p[0]);
        uart_puts(" wait=");  uart_put_dec(p[1]);
        uart_puts(" wb=");    uart_put_dec(p[2]);
        uart_puts(" pool=");  uart_put_dec(p[3]);
        uart_puts(" add=");   uart_put_dec(p[4]);
        uart_puts(" rsz=");   uart_put_dec(p[5]);
        uart_puts(" cfg=");   uart_put_dec(p[6]);
        uart_puts(" oth=");   uart_put_dec(p[7]);
        uart_puts(" wact=");  uart_put_dec(p[8]);
        uart_puts(" wwgt=");  uart_put_dec(p[9]);
        uart_puts(" wdb=");   uart_put_dec(p[10]);
        uart_puts(" midle="); uart_put_dec(p[11]);
        uart_puts(" dma=");   uart_put_dec(p[12]);
        uart_puts(" wst=");   uart_put_dec(p[13]);
        uart_putc('\n');
    }
#endif
#endif
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

    npu_hw_probe();
    uint32_t hwcfg = NPU_REG(REG_HW_CONFIG);
    uart_puts("      Dense MAC lanes: ");
    uart_put_dec(HW_CFG_ARRAY_SIZE(hwcfg));
    uart_puts(", SPAD ");
    uart_put_dec(g_spad_kb);
    uart_puts("KB, Act SRAM ");
    uart_put_dec(g_act_words);
    uart_puts(" words, HW config: 0x");
    uart_put_hex32(hwcfg);
    uart_putc('\n');

    int global_err = 0;

#ifdef RUN_FC_TEST
#ifndef STANDALONE_LAYER_VAL
#define STANDALONE_LAYER_VAL 24
#endif
    /* Standalone FC test: run L24 only, using workspace mode (no tiling, no DDR chaining) */
    uart_puts("\n[FC TEST] Running standalone layer ");
    uart_put_dec(STANDALONE_LAYER_VAL);
    uart_puts("\n");
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
        uart_puts("  hdr: op=");
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
        npu_program_layer(e24, in_addr, 0, 0, e24[21] & ~SCHED_FUSE_MASK,
                          0, 0, 0, 0, 1);
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
#if defined(RUN_model_x)
    test_case_t tc = { BLOB_MODEL_BASE, "model_x (synth_dw INT16, 63 layers)" };
#elif defined(RUN_model_x)
    test_case_t tc = { BLOB_MODEL_BASE, "model_x (synth_res synth_data INT8, 25 layers)" };
#elif defined(RUN_MODEL_C)
    test_case_t tc = { BLOB_MODEL_BASE, "model_c (synth_ms INT16, 17 layers)" };
#elif defined(RUN_model_x)
    test_case_t tc = { BLOB_MODEL_BASE, "model_x (benchmark INT16, 24 layers)" };
#elif defined(RUN_MODEL_E)
    test_case_t tc = { BLOB_MODEL_BASE, "model_e (synth_res rgb_data INT16, 31 layers)" };
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
#ifdef PERF_ONLY
        uart_puts("\n  INFERENCE COMPLETE (outputs unchecked)\n");
#else
        uart_puts("\n  ALL LAYERS PASSED\n");
#endif
    }
#endif /* !RUN_FC_TEST */

    /* Final result */
    uart_puts("\n========================================\n");
    if (global_err == 0) {
#ifdef PERF_ONLY
        uart_puts("  RESULT: PASS PERF_ONLY_UNCHECKED\n");
#else
        uart_puts("  RESULT: PASS\n");
#endif
    } else {
        uart_puts("  RESULT: FAIL\n");
    }
    uart_puts("========================================\n");

    while (1) {}
}
