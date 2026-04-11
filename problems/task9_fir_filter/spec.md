# Task 9: 3-Tap FIR Filter (Medium-Hard)

## Objective

Implement a direct-form I 3-tap Finite Impulse Response (FIR) filter for signed
8-bit samples with programmable coefficients. FIR filters appear in signal
pre-processing, upsampling, and feature extraction stages of every edge-inference
pipeline.

## Interface

```verilog
module fir3 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [7:0]  x,     // current input sample
    input  wire signed [7:0]  h0,    // coefficient for x[n]
    input  wire signed [7:0]  h1,    // coefficient for x[n-1]
    input  wire signed [7:0]  h2,    // coefficient for x[n-2]
    output reg  signed [17:0] y,     // filtered output
    output reg                valid_out
);
```

## Specification

The filter computes the standard convolution:

```
y[n] = h0·x[n] + h1·x[n-1] + h2·x[n-2]
```

**State / latency:**
- History registers `x_d1` (= x[n-1]) and `x_d2` (= x[n-2]) are initialized to 0 on `rst`.
- On each cycle where `valid_in = 1`:
  1. Compute `y = h0*x + h1*x_d1 + h2*x_d2` using the **current** values of `x_d1`, `x_d2`.
  2. Register the result into `y` (output appears the **same** cycle as `valid_in`, one register stage later — so output is available at the next posedge after `valid_in`).
  3. Shift history: `x_d2 ← x_d1`, `x_d1 ← x`.
  4. Assert `valid_out` for one cycle.
- When `valid_in = 0`: history does **not** update; `valid_out` deasserts.
- `rst` (synchronous): clears `x_d1`, `x_d2`, `y`, and `valid_out`.

**Bit widths:**
- History: `reg signed [7:0] x_d1, x_d2`
- Output: `reg signed [17:0] y` (3 × 8-bit × 8-bit products summed → fits in 18 bits)

## Scoring

- Correct compilation: 10%
- Passing simulation tests: 60%
- Correct pipeline structure verified by synthesis (history DFFs): 20%
- Area efficiency vs reference: 10%

## Notes

- Use non-blocking assignments (`<=`) so all registers update simultaneously at posedge.
- The computation inside the always block:
  ```verilog
  y      <= h0*x + h1*x_d1 + h2*x_d2;   // uses OLD x_d1, x_d2
  x_d1   <= x;                            // update history after
  x_d2   <= x_d1;
  valid_out <= 1;
  ```
- `$signed()` casts are not needed if all signals are already declared `signed`.
