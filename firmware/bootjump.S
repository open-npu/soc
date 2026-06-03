/*
 * Minimal ROM bootloader: immediately jump to main_ram (0x40000000)
 * Used to bypass LiteX BIOS in simulation.
 */
.section .text
.global _start
_start:
    li t0, 0x40000000
    jr t0
