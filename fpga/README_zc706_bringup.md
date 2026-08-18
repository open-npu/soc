# Open-NPU ZC706 FPGA Bring-up 计划

目标板：Xilinx ZC706（XC7Z045-2FFG900，Kintex-7 fabric + 双核 A9 PS）

## 阶梯式 bring-up

### M0：BRAM-only 最小验证（先做）
- 配置：VexRiscv + NPU + UART + 512KB BRAM 主存，100MHz
- 只需 6 个引脚（clk200 P/N、UART TX/RX、cpu_reset、LED）——不碰 DDR3
- 跑通标准：串口上传一个单层 Conv 测试 → NPU 完成 → PERF 计数器读回与 sim 周期数**逐拍一致**
- 首次暴露真实时序问题（树形加法链 16×44b 单周期是头号观察对象；不收敛就插 1 拍流水，bit-exact 不受影响）

### M1：+DDR3 全模型
- 补齐 ZC706 SODIMM DDR3 完整引脚表（platform 文件 TODO-M1）+ LiteDRAM 7-series PHY
- 10/10 模型矩阵板上回归 + PERF 对拍 sim
- 真实 DDR 延迟下的 DMA/burst 收益实测（burst spike 的终审）

### M2：收尾
- IRQ 中断驱动模式、FreeRTOS 实跑（一期审计的两个半扎实项）
- 时序收敛到目标频率（记录 Fmax 报告）

## 需要用户提供的输入

1. **ZC706 Master XDC**（Xilinx 官网或本机 Vivado `data/boards/board_files/zc706/*/` 下）
   ——我需要从里面抄：200MHz 差分时钟引脚、PL 侧 UART 引脚（或确认走 PS MIO UART）、复位按钮、LED、以及 M1 的完整 DDR3 SODIMM 引脚表
2. **Vivado 安装机器**：本机无 Vivado；build（`python3 zc706_npu.py --build`）需在有 Vivado 的机器上跑（版本 ≥2020.1 均可，7-series）
3. 确认板卡供电/下载线（JTAG-USB）

## 文件

- `xilinx_zc706.py` — platform（引脚表，TODO-VERIFY 处待 XDC 核对）
- `zc706_npu.py` — target（M0 BRAM-only；CRG=200MHz LVDS→MMCM→100MHz sys）
- 复用：`soc/litex/npu_core.py`（platform 化 NPU 封装，无需改动）

## 与一期审计的衔接

板测不是一期宣称的必要条件（功能=bit-exact 仿真、性能=周期数均与时钟无关）；板测的价值全部落在二期项上：STA 替代验证、真实存储子系统、IRQ/FreeRTOS 实证。
