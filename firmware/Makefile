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

# Model selection. MODEL=model_x also injects -DRUN_model_x for main.c.
MODEL ?= model_x
STANDALONE_LAYER ?= -1
SUPER_BLOCK ?= 1
PERF_ONLY ?= 0
PERF_HIST ?= 0
IFM_KB ?= 32
OFM_KB ?= 48
WGT_KB ?= 80
PARAM_KB ?= 12
export NPU_IFM_KB := $(IFM_KB)
export NPU_OFM_KB := $(OFM_KB)
export NPU_WGT_KB := $(WGT_KB)
export NPU_PARAM_KB := $(PARAM_KB)
DEFINES += -DNPU_IFM_KB_OVERRIDE=$(IFM_KB)
DEFINES += -DNPU_OFM_KB_OVERRIDE=$(OFM_KB)
DEFINES += -DNPU_SPAD_KB_OVERRIDE=$(shell expr $(IFM_KB) + $(OFM_KB) + $(WGT_KB) + $(PARAM_KB))
ifeq ($(SUPER_BLOCK),1)
GEN_SUPER_BLOCK = --enable-super-block
endif
ifeq ($(NO_FUSE),1)
GEN_SUPER_BLOCK += --no-fuse
endif
ifeq ($(PERF_ONLY),1)
DEFINES += -DPERF_ONLY
endif
ifeq ($(PERF_HIST),1)
DEFINES += -DPERF_HIST
endif
ifeq ($(findstring model_x,$(MODEL)),model_x)
DEFINES += -DRUN_model_x
endif
ifeq ($(findstring model_x,$(MODEL)),model_x)
DEFINES += -DRUN_model_x
endif
ifeq ($(findstring model_c,$(MODEL)),model_c)
DEFINES += -DRUN_MODEL_C
endif
ifeq ($(findstring model_x,$(MODEL)),model_x)
DEFINES += -DRUN_model_x
endif
ifeq ($(findstring model_e,$(MODEL)),model_e)
DEFINES += -DRUN_MODEL_E
endif
MODEL_CODE := $(word 2,$(subst _, ,$(MODEL)))
ifeq ($(filter a b c d e,$(MODEL_CODE)),)
DEFINES += -DRUN_MODEL_GENERIC -DMODEL_LABEL=\"$(MODEL)\"
endif

# Generate both artifacts on every build. MODEL and STANDALONE_LAYER are make
# variables rather than files, so timestamp-only dependencies can otherwise
# silently reuse a blob generated for a different model.
$(TEST_DATA_BIN): FORCE gen_soc_test.py
	python3 gen_soc_test.py --model $(MODEL) --standalone-layer $(STANDALONE_LAYER) $(GEN_SUPER_BLOCK)

$(TEST_DATA_H): $(TEST_DATA_BIN)
	@test -f $@

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

.PHONY: all clean FORCE
FORCE:
