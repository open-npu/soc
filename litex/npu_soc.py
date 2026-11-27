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
from litex.soc.interconnect import wishbone
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
    def __init__(self, platform, dma_bus_width=32, **kwargs):
        if dma_bus_width not in (32, 64, 128):
            raise ValueError(f"Unsupported dma_bus_width={dma_bus_width}; must be 32, 64, or 128")
        main_ram_size = 128 * 1024 * 1024

        SoCCore.__init__(self, platform,
            cpu_type                 = "vexriscv",
            cpu_variant              = "standard",
            clk_freq                 = int(50e6),
            integrated_rom_size      = 64 * 1024,
            integrated_sram_size     = 8 * 1024,
            integrated_main_ram_size = main_ram_size if dma_bus_width == 32 else 0,
            ident                    = "Open-NPU SoC",
            uart_name                = "sim",
            **kwargs,
        )

        # CRG
        self.submodules.crg = SimCRG(platform)

        # NPU Peripheral
        self.submodules.npu = npu = NPUCore(platform, dma_bus_width=dma_bus_width)

        # NPU CSR slave at 0x8000_0000 (4KB) on 32-bit CPU control bus
        self.bus.add_slave("npu", npu.wb_slave,
            region=SoCRegion(origin=0x8000_0000, size=0x1000, cached=False))

        if dma_bus_width == 32:
            # Classic shared 32-bit Wishbone interconnect
            self.bus.add_master("npu_dma", master=npu.wb_master)
        else:
            # Dual-tier bus: 32-bit CPU control bus + wide (64/128-bit) memory data bus.
            # CPU accesses main_ram through a 0-cycle combinational UpConverter, while
            # NPU DMA connects directly at full native width.
            word_shift = {64: 3, 128: 4}[dma_bus_width]
            wide_adr_w = 32 - word_shift

            cpu_ram_bus = wishbone.Interface(data_width=32, adr_width=30)
            cpu_wide_bus = wishbone.Interface(data_width=dma_bus_width, adr_width=wide_adr_w)
            wide_ram_bus = wishbone.Interface(data_width=dma_bus_width, adr_width=wide_adr_w)

            self.submodules.cpu_ram_up = wishbone.UpConverter(cpu_ram_bus, cpu_wide_bus)
            self.submodules.main_ram = wishbone.SRAM(
                main_ram_size, bus=wide_ram_bus, init=[], name="main_ram"
            )
            self.submodules.main_ram_arb = wishbone.Arbiter(
                [cpu_wide_bus, npu.wb_master], wide_ram_bus
            )
            self.bus.add_slave("main_ram", cpu_ram_bus,
                region=SoCRegion(origin=self.mem_map["main_ram"], size=main_ram_size, mode="rwx"))

        # NPU IRQ
        self.irq.add("npu")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Open-NPU SoC")
    parser.add_argument("--build", action="store_true", help="Build SoC (generate Verilog)")
    parser.add_argument("--run",   action="store_true", help="Run Verilator simulation")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--dma-bus-width", type=int, default=32, choices=[32, 64, 128],
                        help="NPU DMA / main_ram data bus width in bits (default: 32)")
    parser.add_argument("--trace", action="store_true", help="Enable VCD trace")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = (
            "build/npu_soc" if args.dma_bus_width == 32
            else f"build/npu_soc_w{args.dma_bus_width}"
        )

    sys_clk_freq = int(50e6)

    platform = SimPlatform("SIM_DEVICE", _sim_io)
    soc = NPUSoC(platform, sys_clk_freq=sys_clk_freq, dma_bus_width=args.dma_bus_width)

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
