/*
 * Open-NPU SoC Firmware — Startup code
 * Minimal crt0: set stack pointer, copy .data to SRAM, zero BSS, jump to main.
 */

.section .text.start
.global _start
.global _sidata
.global _sdata
.global _edata

_start:
    /* Copy .data section from ROM to SRAM (LMA → VMA) */
    la a0, _sidata   /* source: ROM (LMA) */
    la a1, _sdata    /* dest: SRAM (VMA) */
    la a2, _edata    /* end */
    beq a1, a2, 2f
1:  lw t0, 0(a0)
    sw t0, 0(a1)
    addi a0, a0, 4
    addi a1, a1, 4
    blt a1, a2, 1b

    /* Zero BSS */
2:  la t0, _fbss
    la t1, _ebss
3:  bge t0, t1, 4f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 3b

    /* Set stack pointer */
4:  la sp, _stack_top

    /* Jump to main */
    jal ra, main

    /* If main returns, loop forever */
5:  j 5b
