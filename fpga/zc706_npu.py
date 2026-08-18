#!/usr/bin/env python3
"""
ZC706 FPGA target for Open-NPU (M0: BRAM-only bring-up).

SoC: VexRiscv + NPU + UART + BRAM main RAM (no DDR3 in M0).
Build (on a machine with Vivado):
    cd soc/fpga
    python3 zc706_npu.py --build --load   [--flash]

M1 adds LiteDRAM DDR3 (see xilinx_zc706.py TODO-M1) for full-model runs.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'litex'))

from migen import *
from litex.soc.integration.soc_core import SoCCore
from litex.soc.integration.builder import Builder
from litex.soc.cores.clock import S7MMCM, S7IDELAYCTRL

from xilinx_zc706 import Platform
from npu_core import NPUCore


# CRG: 200MHz LVDS in → MMCM → 100MHz sys ---------------------------------------------------------

class CRG(Module):
    def __init__(self, platform, sys_clk_freq=100e6):
        self.clock_domains.cd_sys = ClockDomain()
        clk200 = platform.request("clk200")
        self.submodules.mmcm = mmcm = S7MMCM(speedgrade=-2)
        mmcm.register_clkin(clk200, 200e6)
        mmcm.create_clkout(self.cd_sys, sys_clk_freq)
        platform.add_false_path_constraints(self.cd_sys.clk,
                                            platform.lookup_request("cpu_reset", loose=True))


# SoC ----------------------------------------------------------------------------------------------

class ZC706NPUSoC(SoCCore):
    def __init__(self, platform, sys_clk_freq=100e6, **kwargs):
        SoCCore.__init__(self, platform,
            cpu_type                 = "vexriscv",
            cpu_variant              = "standard",
            clk_freq                 = int(sys_clk_freq),
            integrated_rom_size      = 64 * 1024,     # LiteX BIOS
            integrated_sram_size     = 8 * 1024,
            # M0: BRAM main RAM — sized for single-layer / small tests.
            # 7z045 has ~19.6Mb BRAM; 512KB here is comfortable.
            integrated_main_ram_size = 512 * 1024,
            **kwargs)
        self.submodules.crg = CRG(platform, sys_clk_freq)
        self.submodules.npu = NPUCore(platform)
        self.bus.add_master(master=self.npu.wb_mst)


def main():
    from litex.build.parser import LiteXArgumentParser
    parser = LiteXArgumentParser(description="Open-NPU on ZC706 (M0 BRAM-only)")
    parser.add_argument("--sys-clk-freq", default=100e6, type=float)
    args = parser.parse_args()

    platform = Platform()
    soc = ZC706NPUSoC(platform, sys_clk_freq=args.sys_clk_freq)
    builder = Builder(soc, **parser.builder_argdict(args))
    builder.build(**parser.toolchain_argdict(args))

if __name__ == "__main__":
    main()
