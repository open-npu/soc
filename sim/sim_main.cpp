// Minimal Verilator testbench for Open-NPU SoC
// Drives clock, captures UART output from BIOS boot

#include <verilated.h>
#include "Vsim.h"
#include <cstdio>
#include <cstdlib>

#define MAX_SIM_TIME 100000000  // 100M cycles (~2s @ 50MHz)
#define UART_TIMEOUT  30000000  // Stop if no UART activity for this long

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vsim* top = new Vsim;

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

    top->final();
    delete top;
    return (uart_bytes > 0) ? 0 : 1;
}
