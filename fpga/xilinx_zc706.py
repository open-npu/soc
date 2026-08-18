# ZC706 (XC7Z045) LiteX platform for Open-NPU bring-up.
#
# ⚠ PIN TABLE STATUS: values marked TODO-VERIFY must be checked against the
# ZC706 Master XDC (Xilinx UG954 / vivado board files) before building.
# Template format follows litex-boards digilent_zedboard (7-series, DDR3).

from litex.build.generic_platform import Pins, IOStandard, Subsignal
from litex.build.xilinx import Xilinx7SeriesPlatform

# IOs ----------------------------------------------------------------------------------------------

_io = [
    # ── M0 minimal set (clock + uart + reset + 1 led) ─────────────────
    # System clock: ZC706 has a fixed 200MHz LVDS oscillator (Y2/Y3 pair
    # historically used by litex-boards zc706 platform as "clk200").
    ("clk200", 0,
        Subsignal("p", Pins("TODO-VERIFY")),  # e.g. user clock osc P pin
        Subsignal("n", Pins("TODO-VERIFY")),
        IOStandard("LVDS_25"),                # bank voltage per XDC
    ),

    # UART: ZC706 PL UART is on the FMC/PMOD or via CP210x USB-UART bridge
    # (MIO in PS). For PL-fabric UART choose the PMOD or the PL-side USB
    # bridge pins per the master XDC.
    ("serial", 0,
        Subsignal("tx", Pins("TODO-VERIFY")),
        Subsignal("rx", Pins("TODO-VERIFY")),
        IOStandard("LVCMOS25"),
    ),

    # CPU reset button (PL side push button, e.g. SW or user_btn)
    ("cpu_reset", 0, Pins("TODO-VERIFY"), IOStandard("LVCMOS25")),

    # One status LED (heartbeat / done indicator)
    ("user_led", 0, Pins("TODO-VERIFY"), IOStandard("LVCMOS25")),

    # ── M1: DDR3 SODIMM (PL side, banks 33-35, 1GB) ───────────────────
    # Full addr/ba/dq/dqs/dm/ck/cke/cs/cas/ras/we/odt table — copy from
    # the ZC706 master XDC verbatim. LiteDRAM 7-series PHY:
    #   module = MT8JTF12864 (1GB SODIMM on ZC706)
    #   speedgrade = -3 (check device speed grade)
    # ("ddram", 0, ... TODO-M1 ...),
]

# Platform -----------------------------------------------------------------------------------------

class Platform(Xilinx7SeriesPlatform):
    default_clk_name   = "clk200"
    default_clk_period = 1e9 / 200e6

    def __init__(self):
        Xilinx7SeriesPlatform.__init__(self, "xc7z045ffg900-2", _io,
                                       toolchain="vivado")
