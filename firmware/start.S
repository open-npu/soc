/*
 * Open-NPU SoC Firmware — Startup code
 * Minimal crt0: set stack pointer, zero BSS, jump to main.
 */

.section .text.start
.global _start

_start:
    /* Set stack pointer */
    la sp, _stack_top

    /* Zero BSS */
    la t0, _fbss
    la t1, _ebss
1:  bge t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b
2:

    /* Jump to main */
    jal ra, main

    /* If main returns, loop forever */
3:  j 3b
