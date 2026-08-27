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

CFLAGS  = $(ARCH) -Os -Wall -Wextra -ffreestanding -nostdlib -nostartfiles $(DEFINES)
ASFLAGS = $(ARCH) -ffreestanding -nostdlib
LDFLAGS = $(ARCH) -nostdlib -nostartfiles -Wl,-T,linker.ld -Wl,--gc-sections

SRCS_C = main.c
SRCS_S = start.S
OBJS   = start.o main.o

TARGET  = firmware

# Test data (generated from golden)
TEST_DATA_BIN = test_data.bin
TEST_DATA_H  = soc_test_data.h
MAIN_RAM_INIT = main_ram.init

# Sim directory for output files
SIM_DIR = ../sim

all: $(TARGET).bin $(TARGET).hex $(MAIN_RAM_INIT)

# Model selection. MODEL=model_b_int16 also injects -DRUN_MODEL_B for main.c.
MODEL ?= model_b_int16
STANDALONE_LAYER ?= -1
ifeq ($(findstring model_a,$(MODEL)),model_a)
DEFINES += -DRUN_MODEL_A
endif
ifeq ($(findstring model_b,$(MODEL)),model_b)
DEFINES += -DRUN_MODEL_B
endif
ifeq ($(findstring model_c,$(MODEL)),model_c)
DEFINES += -DRUN_MODEL_C
endif
ifeq ($(findstring model_d,$(MODEL)),model_d)
DEFINES += -DRUN_MODEL_D
endif
ifeq ($(findstring model_e,$(MODEL)),model_e)
DEFINES += -DRUN_MODEL_E
endif

# Generate test data header and binary blob together
$(TEST_DATA_H) $(TEST_DATA_BIN):
	python3 gen_soc_test.py --model $(MODEL) --standalone-layer $(STANDALONE_LAYER)

%.o: %.c $(TEST_DATA_H)
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.S
	$(AS) $(ASFLAGS) -c $< -o $@

$(TARGET).elf: $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $@ -lgcc
	$(OBJDUMP) -d $@ > $(TARGET).dis

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@
	@echo "Firmware size: $$(wc -c < $@) bytes"

# Generate hex for $readmemh loading into sim (ROM)
$(TARGET).hex: $(TARGET).bin
	od -An -tx4 -v $< | sed 's/^ *//' | tr ' ' '\n' | grep -v '^$$' > $@
	@echo "Generated $@ ($$(wc -l < $@) words)"

# Build main RAM init: firmware.bin + padding + test_data.bin
$(MAIN_RAM_INIT): $(TARGET).bin $(TEST_DATA_BIN) build_main_ram.py
	python3 build_main_ram.py $(TARGET).bin $(TEST_DATA_BIN) $@
	@# Copy to sim directory
	@cp $(TARGET).hex $(SIM_DIR)/sim_rom.init
	@cp $@ $(SIM_DIR)/sim_main_ram.init
	@echo "Copied sim_rom.init and sim_main_ram.init to $(SIM_DIR)/"

clean:
	rm -f *.o *.elf *.bin *.hex *.dis $(TEST_DATA_BIN) $(MAIN_RAM_INIT) $(TEST_DATA_H)

.PHONY: all clean
