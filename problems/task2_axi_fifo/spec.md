# Task 2: AXI-Stream FIFO (Medium)

## Objective

Implement a 4-entry FIFO with correct AXI-Stream handshaking on both input and output interfaces.

## Interface

```verilog
module axi_fifo #(
    parameter DATA_W = 8
) (
    input  wire              clk,
    input  wire              rst,       // synchronous reset, active high

    // Slave (input) AXI-Stream interface
    input  wire              s_valid,
    output wire              s_ready,
    input  wire [DATA_W-1:0] s_data,

    // Master (output) AXI-Stream interface
    output wire              m_valid,
    input  wire              m_ready,
    output wire [DATA_W-1:0] m_data,

    // Status
    output wire              full,
    output wire              empty
);
```

## AXI-Stream Protocol Rules

1. A transfer occurs when both valid and ready are high on a clock edge
2. **Critical:** A master (sender) MUST NOT deassert s_valid once asserted until a transfer completes
3. s_ready may be deasserted at any time (backpressure is always legal from a slave)
4. Data must not be lost when the FIFO is full (s_ready goes low to prevent new data)
5. Data must not be lost when downstream stalls (m_ready=0)

## Scoring

- Compilation: 10%
- Functional correctness (basic fill/drain, simultaneous ops): 40%
- Protocol compliance (backpressure, no-drop on stall): 30%
- Parameterization (correct at DATA_W=8 and DATA_W=64): 20%

## Notes

- Depth is fixed at 4 entries
- Use a circular buffer with head/tail pointers
- The module must be fully synthesizable (no `initial` blocks in design, only in testbench)
