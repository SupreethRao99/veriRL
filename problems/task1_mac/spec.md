# Task 1: Pipelined MAC Unit (Easy)

## Objective

Design a pipelined multiply-accumulate unit in synthesizable Verilog.

## Interface

```verilog
module mac_unit (
    input  wire        clk,
    input  wire        rst,      // synchronous reset, active high
    input  wire        en,       // enable — when low, hold accumulator
    input  wire        clear,    // synchronous clear of accumulator (takes effect after pipeline delay)
    input  wire signed [7:0]  a, // signed 8-bit input
    input  wire signed [7:0]  b, // signed 8-bit input
    output reg  signed [31:0] acc_out  // accumulated result
);
```

## Specification

- The pipeline must have **exactly 2 stages**: multiply in stage 1, accumulate in stage 2
- Output `acc_out` is valid **2 clock cycles** after inputs `a` and `b` are presented
- When `en` is low, the accumulator holds its value (no new accumulation)
- When `clear` is high, the accumulator resets to 0 (synchronous, respects pipeline latency)
- `rst` resets all pipeline registers and accumulator to 0

## Scoring

- Correct compilation: 10%
- Passing simulation test vectors: 60%
- Correct 2-stage pipeline structure (verified by synthesis register count): 20%
- Area efficiency vs reference: 10%

## Notes

- Use signed arithmetic throughout — `a` and `b` are signed
- The accumulator is 32-bit to prevent overflow across many accumulations
- Back-to-back inputs are valid — your pipeline must handle consecutive accumulations correctly
