# Task 6: Dual-Read / Single-Write Register File (Medium)

## Objective

Implement the standard RISC-style register file: two independent asynchronous read
ports and one synchronous write port, with register 0 hard-wired to zero. This is
the core state element in every processor and AI-accelerator sequencer.

## Interface

```verilog
module register_file #(
    parameter ADDR_W = 5,    // 2^5 = 32 registers
    parameter DATA_W = 32
) (
    input  wire              clk,
    input  wire              we,           // write enable
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire [DATA_W-1:0] wr_data,
    input  wire [ADDR_W-1:0] rd_addr_a,   // read port A
    input  wire [ADDR_W-1:0] rd_addr_b,   // read port B
    output wire [DATA_W-1:0] rd_data_a,
    output wire [DATA_W-1:0] rd_data_b
);
```

## Specification

- **Synchronous write**: on the rising edge of `clk`, if `we` is high, write `wr_data` to `regs[wr_addr]`.
- **Asynchronous read**: `rd_data_a` and `rd_data_b` are combinational outputs of `regs[rd_addr_a]` and `regs[rd_addr_b]`.
- **Register 0 hardwired**: reading address 0 always returns `0`, regardless of any write. Writing to address 0 has **no effect**.
- **No reset port**: registers may contain `x` on startup (testbench initialises via writes).
- Both read ports are fully independent and may be read simultaneously.
- **Write-then-read hazard**: the testbench reads one cycle *after* writing, so no forwarding path is required.

## Scoring

- Correct compilation: 10%
- Passing simulation tests: 70%
- Formal verification (SymbiYosys, if available): 10%
- Area efficiency vs reference: 10%

## Notes

- Declare as `reg [DATA_W-1:0] regs [0:(1<<ADDR_W)-1];`
- Guard every write: `if (we && wr_addr != 0) regs[wr_addr] <= wr_data;`
- Guard every read: `assign rd_data_a = (rd_addr_a == 0) ? 0 : regs[rd_addr_a];`
- Using `initial` blocks to clear registers is acceptable and synthesizable in modern flows.
