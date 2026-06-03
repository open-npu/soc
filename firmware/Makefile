# Open-NPU SoC Firmware Makefile
# Cross-compiles for RISC-V VexRiscv (rv32im)

CROSS   ?= riscv64-unknown-elf-
CC      = $(CROSS)gcc
AS      = $(CROSS)gcc
LD      = $(CROSS)gcc
OBJCOPY = $(CROSS)objcopy
OBJDUMP = $(CROSS)objdump

# VexRiscv standard variant: rv32im (no compressed, no atomic)
ARCH    = -march=rv32im -mabi=ilp32

CFLAGS  = $(ARCH) -Os -Wall -Wextra -ffreestanding -nostdlib -nostartfiles
ASFLAGS = $(ARCH) -ffreestanding -nostdlib
LDFLAGS = $(ARCH) -nostdlib -nostartfiles -Wl,-T,linker.ld -Wl,--gc-sections

SRCS_C = main.c
SRCS_S = start.S
OBJS   = start.o main.o

TARGET  = firmware

all: $(TARGET).bin $(TARGET).hex

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S
	$(AS) $(ASFLAGS) -c $< -o $@

$(TARGET).elf: $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $@ -lgcc
	$(OBJDUMP) -d $@ > $(TARGET).dis

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@
	@echo "Firmware size: $$(wc -c < $@) bytes"

# Generate hex for $readmemh loading into sim
$(TARGET).hex: $(TARGET).bin
	od -An -tx4 -v $< | sed 's/^ *//' | tr ' ' '\n' | grep -v '^$$' > $@
	@echo "Generated $@ ($$(wc -l < $@) words)"

clean:
	rm -f *.o *.elf *.bin *.hex *.dis

.PHONY: all clean
