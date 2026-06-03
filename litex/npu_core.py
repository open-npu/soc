"""
NPU Core LiteX Wrapper
Instantiates npu_top.v as a Wishbone peripheral with DMA master and IRQ.
"""

import os
from migen import *
from litex.soc.interconnect import wishbone
from litex.soc.interconnect.csr_eventmanager import EventManager, EventSourceLevel

# Path to NPU RTL sources
NPU_RTL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rtl/src"))
NPU_INC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../rtl/include"))


class NPUCore(Module):
    def __init__(self, platform):
        # --- Wishbone Slave (CPU -> NPU CSR, 12-bit byte addr = 4KB) ---
        # LiteX WB adr is word-addressed; NPU uses 12-bit byte address
        self.wb_slave = wb_slv = wishbone.Interface(data_width=32, adr_width=30)

        # --- Wishbone Master (NPU DMA -> main memory) ---
        self.wb_master = wb_mst = wishbone.Interface(data_width=32, adr_width=30)

        # --- IRQ via EventManager ---
        self.submodules.ev = EventManager()
        self.ev.done = EventSourceLevel()
        self.ev.finalize()

        # Internal IRQ wire from NPU
        irq_signal = Signal()
        self.comb += self.ev.done.trigger.eq(irq_signal)

        # --- Address conversion ---
        # NPU slave: LiteX sends 30-bit word addr, NPU expects 12-bit byte addr
        # byte_addr[11:0] = word_addr[9:0] << 2
        npu_slv_adr = Signal(12)
        self.comb += npu_slv_adr[2:12].eq(wb_slv.adr[:10])
        # bits [1:0] are always 0 (word-aligned)

        # NPU master: NPU outputs 32-bit byte addr, LiteX expects 30-bit word addr
        # byte_addr[31:2] -> word_addr[29:0]
        npu_mst_adr_byte = Signal(32)
        self.comb += wb_mst.adr.eq(npu_mst_adr_byte[2:])

        # --- Instantiate npu_top ---
        self.specials += Instance("npu_top",
            # Clock & Reset
            i_clk     = ClockSignal(),
            i_rst_n   = ~ResetSignal(),

            # Wishbone Slave (CSR)
            i_wb_slv_cyc_i = wb_slv.cyc,
            i_wb_slv_stb_i = wb_slv.stb,
            i_wb_slv_we_i  = wb_slv.we,
            i_wb_slv_adr_i = npu_slv_adr,
            i_wb_slv_dat_i = wb_slv.dat_w,
            i_wb_slv_sel_i = wb_slv.sel,
            o_wb_slv_dat_o = wb_slv.dat_r,
            o_wb_slv_ack_o = wb_slv.ack,

            # Wishbone Master (DMA)
            o_wb_mst_cyc_o = wb_mst.cyc,
            o_wb_mst_stb_o = wb_mst.stb,
            o_wb_mst_we_o  = wb_mst.we,
            o_wb_mst_adr_o = npu_mst_adr_byte,
            o_wb_mst_dat_o = wb_mst.dat_w,
            o_wb_mst_sel_o = wb_mst.sel,
            i_wb_mst_ack_i = wb_mst.ack,
            i_wb_mst_dat_i = wb_mst.dat_r,

            # IRQ
            o_irq_o = irq_signal,
        )

        # --- Add NPU RTL sources ---
        rtl_files = [
            "npu_pe.v",
            "npu_systolic.v",
            "npu_ppu.v",
            "npu_csr.v",
            "npu_sram.v",
            "npu_dma.v",
            "npu_dw_conv.v",
            "npu_ctrl.v",
            "npu_compute.v",
            "npu_top.v",
        ]
        for f in rtl_files:
            platform.add_source(os.path.join(NPU_RTL_DIR, f))

        # Add include directory for npu_defines.vh
        platform.add_verilog_include_path(NPU_INC_DIR)
