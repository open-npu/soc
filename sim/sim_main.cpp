// Minimal Verilator testbench for Open-NPU SoC
// Drives clock, captures UART output from BIOS boot

#include <verilated.h>
#include "Vsim.h"
#include <cstdio>
#include <cstdlib>

#ifdef VCD_TRACE
#include <verilated_vcd_c.h>
#endif

#define MAX_SIM_TIME 20000000000  // 20B cycles — model_e (31 layers) needs >5B
#define UART_TIMEOUT  3000000000  // Stop if no UART activity for 3B cycles

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

#ifdef VCD_TRACE
    Verilated::traceEverOn(true);
#endif

    Vsim* top = new Vsim;

#ifdef VCD_TRACE
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("sim.vcd");
    printf("[SoC Sim] VCD tracing enabled → sim.vcd\n");
    bool vcd_active = false;
    uint64_t vcd_start = 0;
    uint64_t vcd_duration = 0;
    const char* vcd_env = getenv("VCD_WINDOW");
    if (vcd_env) vcd_duration = strtoull(vcd_env, NULL, 10);
    else vcd_duration = 5000000;  // Default: 5M cycles window
#endif

    // Initialize signals
    top->sys_clk = 0;
    top->sys_rst = 1;  // Assert reset
    top->serial_sink_valid = 0;
    top->serial_sink_data = 0;
    top->serial_source_ready = 1;  // Always ready to receive UART data
    top->sim_trace = 0;

    // Hold reset for 10 clock cycles
    for (int i = 0; i < 20; i++) {
        top->sys_clk = !top->sys_clk;
        top->eval();
    }
    top->sys_rst = 0;  // Deassert reset

    uint64_t sim_time = 0;
    uint64_t last_uart_time = 0;
    int uart_bytes = 0;
    int dma_wr_count = 0;  // Track DMA writes for debug
    bool layer0_started = false;

    printf("[SoC Sim] Starting VexRiscv + NPU SoC simulation...\n");
    printf("[SoC Sim] Reset deasserted, waiting for BIOS UART output...\n");
    printf("---UART OUTPUT BEGIN---\n");

    while (sim_time < MAX_SIM_TIME && !Verilated::gotFinish()) {
        // Toggle clock
        top->sys_clk = !top->sys_clk;
        top->eval();
        sim_time++;

        // Check UART output on rising edge
        if (top->sys_clk) {
#ifdef VCD_TRACE
            // Start VCD when NPU becomes busy (layer 0 starts)
            // Detect by monitoring NPU interrupt or UART "Layers:" message
            if (!vcd_active && uart_bytes > 100) {
                // UART has printed the header, NPU should be starting soon
                vcd_active = true;
                vcd_start = sim_time;
                printf("[SoC Sim] VCD recording started at cycle %lu (window=%lu)\n",
                       sim_time / 2, vcd_duration / 2);
            }
            if (vcd_active) {
                tfp->dump(sim_time);
                if (sim_time - vcd_start > vcd_duration) {
                    vcd_active = false;
                    printf("[SoC Sim] VCD recording stopped at cycle %lu\n", sim_time / 2);
                }
            }
#endif
            if (top->serial_source_valid) {
                char c = (char)(top->serial_source_data & 0xFF);
                putchar(c);
                fflush(stdout);
                uart_bytes++;
                last_uart_time = sim_time;
            }

            // Early termination: if we've received UART data and then
            // nothing for UART_TIMEOUT cycles, BIOS has finished printing
            if (uart_bytes > 100 && (sim_time - last_uart_time) > UART_TIMEOUT) {
                break;
            }
        }
    }

    printf("\n---UART OUTPUT END---\n");
    printf("[SoC Sim] Simulation ended at cycle %lu, %d UART bytes received\n",
           sim_time / 2, uart_bytes);

#ifdef VCD_TRACE
    tfp->close();
    delete tfp;
#endif
    top->final();
    delete top;
    return (uart_bytes > 0) ? 0 : 1;
}
