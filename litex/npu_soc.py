#!/usr/bin/env python3
"""
Open-NPU SoC: VexRiscv + NPU + RAM + UART
LiteX-based SoC for Verilator simulation.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from migen import *
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.soc import SoCRegion
from litex.soc.integration.builder import Builder
from litex.build.sim import SimPlatform
from litex.build.sim.config import SimConfig
from litex.build.generic_platform import Pins, Subsignal

from npu_core import NPUCore


# --- Simulation Platform IO ---
_sim_io = [
    ("sys_clk", 0, Pins(1)),
    ("sys_rst", 0, Pins(1)),
    ("serial", 0,
        Subsignal("source_valid", Pins(1)),
        Subsignal("source_ready", Pins(1)),
        Subsignal("source_data",  Pins(8)),
        Subsignal("sink_valid",   Pins(1)),
        Subsignal("sink_ready",   Pins(1)),
        Subsignal("sink_data",    Pins(8)),
    ),
]


# --- Sim CRG ---
class SimCRG(Module):
    def __init__(self, platform):
        self.clock_domains.cd_sys = ClockDomain()
        clk = platform.request("sys_clk")
        rst = platform.request("sys_rst")
        self.comb += [
            self.cd_sys.clk.eq(clk),
            self.cd_sys.rst.eq(rst),
        ]


# --- NPU SoC ---
class NPUSoC(SoCCore):
    def __init__(self, platform, **kwargs):
        SoCCore.__init__(self, platform,
            cpu_type                 = "vexriscv",
            cpu_variant              = "standard",
            clk_freq                 = int(50e6),
            integrated_rom_size      = 64 * 1024,
            integrated_sram_size     = 8 * 1024,
            integrated_main_ram_size = 128 * 1024 * 1024,
            ident                    = "Open-NPU SoC",
            uart_name                = "sim",
            **kwargs,
        )

        # CRG
        self.submodules.crg = SimCRG(platform)

        # NPU Peripheral
        self.submodules.npu = npu = NPUCore(platform)

        # NPU CSR slave at 0x8000_0000 (4KB)
        self.bus.add_slave("npu", npu.wb_slave,
            region=SoCRegion(origin=0x8000_0000, size=0x1000, cached=False))

        # NPU DMA master
        self.bus.add_master("npu_dma", master=npu.wb_master)

        # NPU IRQ
        self.irq.add("npu")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Open-NPU SoC")
    parser.add_argument("--build", action="store_true", help="Build SoC (generate Verilog)")
    parser.add_argument("--run",   action="store_true", help="Run Verilator simulation")
    parser.add_argument("--output-dir", default="build/npu_soc", help="Output directory")
    parser.add_argument("--trace", action="store_true", help="Enable VCD trace")
    args = parser.parse_args()

    sys_clk_freq = int(50e6)

    platform = SimPlatform("SIM_DEVICE", _sim_io)
    soc = NPUSoC(platform, sys_clk_freq=sys_clk_freq)

    sim_config = SimConfig(default_clk="sys_clk")
    sim_config.add_clocker("sys_clk", freq_hz=sys_clk_freq)
    sim_config.add_module("serial2console", "serial")

    builder = Builder(soc, output_dir=args.output_dir)

    if args.run:
        # Full sim: build Verilator binary and run
        builder.build(
            run        = True,
            sim_config = sim_config,
            trace      = args.trace,
        )
    else:
        # Generate Verilog only (no Verilator compile)
        builder.build(
            run        = False,
            sim_config = sim_config,
        )
        print(f"SoC Verilog generated in: {args.output_dir}/gateware/")


if __name__ == "__main__":
    main()
