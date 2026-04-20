# Task 8: Pipelined 4-Element INT8 Dot Product (Medium)

## Objective

Design a 2-stage pipelined unit that computes the dot product of two 4-element
signed INT8 vectors. Dot products are the innermost operation in attention score
computation (Q·Kᵀ) and fully-connected layers — latency and throughput here
directly dominate inference speed.

## Interface

```verilog
module dot_product_4 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [7:0] a0, a1, a2, a3,
    input  wire signed [7:0] b0, b1, b2, b3,
    output reg         valid_out,
    output reg  signed [17:0] result
);
```

## Specification

Computes `result = a0*b0 + a1*b1 + a2*b2 + a3*b3` over a **2-stage pipeline**:

- **Stage 1** (cycle N): Register 4 signed 16-bit partial products:
  `p0 = a0*b0`, `p1 = a1*b1`, `p2 = a2*b2`, `p3 = a3*b3`
- **Stage 2** (cycle N+1): Register the sum `p0+p1+p2+p3` into `result`; assert `valid_out`.

**Timing:**
- `valid_out` pulses high exactly **1 cycle** after each `valid_in` pulse.
- For back-to-back `valid_in` pulses, `valid_out` pulses back-to-back one cycle later.
- `rst` clears both pipeline stages and deasserts `valid_out`.

**Bit widths:**
- Products: signed 16-bit (`8×8 → 16`).
- Result: signed 18-bit (sum of four 16-bit products; max = 4 × 127² = 64 516 fits in 17 bits, but use 18 for signed range safety).

## Scoring

- Correct compilation: 10%
- Passing simulation tests: 60%
- Correct 2-stage pipeline structure (DFF count via synthesis): 20%
- Area efficiency vs reference: 10%

## Notes

- Declare products as `reg signed [15:0] p0, p1, p2, p3;` and a `reg stage1_valid`.
- Both stages update in the **same** `always @(posedge clk)` block.
- The result is `$signed(p0) + $signed(p1) + $signed(p2) + $signed(p3)` — all signed.
