/*
 * Open-NPU SoC Firmware Linker Script
 * Target: LiteX SoC ROM at 0x0 (replaces BIOS)
 *
 * Memory Map:
 *   ROM:      0x0000_0000 (64KB) — code + rodata
 *   SRAM:     0x1000_0000 (8KB)  — stack + bss + data
 *   MAIN_RAM: 0x4000_0000 (256KB) — NPU test buffers (accessed at runtime)
 */

OUTPUT_ARCH(riscv)
ENTRY(_start)

MEMORY
{
    rom  (rx)  : ORIGIN = 0x00000000, LENGTH = 64K
    sram (rwx) : ORIGIN = 0x10000000, LENGTH = 8K
}

SECTIONS
{
    .text : {
        . = ALIGN(4);
        *(.text.start)
        *(.text .text.*)
    } > rom

    .rodata : {
        . = ALIGN(4);
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    } > rom

    .data : {
        . = ALIGN(4);
        _sdata = .;
        *(.data .data.*)
        *(.sdata .sdata.*)
        _edata = .;
    } > sram AT > rom

    .bss : {
        . = ALIGN(4);
        _fbss = .;
        _sbss = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        _ebss = .;
    } > sram

    /* Stack at top of SRAM */
    _stack_top = ORIGIN(sram) + LENGTH(sram);
}
