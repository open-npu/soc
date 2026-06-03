# Verilator simulation Makefile for Open-NPU SoC

VERILOG_TOP = sim
BUILD_DIR   = ../litex/build/npu_soc
GATEWARE    = $(BUILD_DIR)/gateware/$(VERILOG_TOP).v
NPU_RTL_DIR = ../../rtl/src
NPU_INC_DIR = ../../rtl/include
VEXRISCV    = $(shell python3 -c "import pythondata_cpu_vexriscv; import os; print(os.path.join(os.path.dirname(pythondata_cpu_vexriscv.__file__), 'verilog', 'VexRiscv.v'))")

OBJ_DIR     = obj_dir

# NPU RTL sources
NPU_SRCS = $(NPU_RTL_DIR)/npu_pe.v \
           $(NPU_RTL_DIR)/npu_systolic.v \
           $(NPU_RTL_DIR)/npu_ppu.v \
           $(NPU_RTL_DIR)/npu_csr.v \
           $(NPU_RTL_DIR)/npu_sram.v \
           $(NPU_RTL_DIR)/npu_dma.v \
           $(NPU_RTL_DIR)/npu_dw_conv.v \
           $(NPU_RTL_DIR)/npu_ctrl.v \
           $(NPU_RTL_DIR)/npu_compute.v \
           $(NPU_RTL_DIR)/npu_top.v

# Verilator flags
VERILATOR_FLAGS = --cc --exe -O3 \
    -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
    -Wno-UNOPTFLAT -Wno-COMBDLY -Wno-LATCH -Wno-CASEINCOMPLETE \
    --top-module $(VERILOG_TOP) \
    -I$(NPU_INC_DIR)

.PHONY: all build run clean generate

all: run

# Step 1: Generate SoC Verilog (if not already done)
generate:
	@if [ ! -f $(GATEWARE) ]; then \
		echo "Generating SoC Verilog..."; \
		cd ../litex && python3 npu_soc.py --build --output-dir $(abspath $(BUILD_DIR)); \
	fi

# Step 2: Verilate and compile
build: generate
	@echo "Verilating SoC..."
	verilator $(VERILATOR_FLAGS) \
		$(GATEWARE) $(VEXRISCV) $(NPU_SRCS) \
		sim_main.cpp
	@echo "Compiling Verilator model..."
	$(MAKE) -C $(OBJ_DIR) -f V$(VERILOG_TOP).mk V$(VERILOG_TOP)

# Step 3: Run simulation
run: build
	@echo "Running SoC simulation..."
	./$(OBJ_DIR)/V$(VERILOG_TOP)

clean:
	rm -rf $(OBJ_DIR)
