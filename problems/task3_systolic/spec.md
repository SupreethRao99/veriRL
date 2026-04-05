# Task 3: 4×4 Weight-Stationary Systolic Array (Hard)

## Objective

Design a weight-stationary 4×4 systolic array that computes C = A × B for integer matrices.

## Interface

```verilog
module systolic_array (
    input  wire        clk,
    input  wire        rst,
    input  wire        load_weights,          // pulse high to latch weight matrix
    input  wire [63:0] weights_flat,          // 16 × 4-bit weights, row-major
    input  wire        start,                 // pulse high to begin computation
    input  wire [127:0] activations_flat,     // 16 × 8-bit activations, row-major
    output wire [255:0] outputs_flat,         // 16 × 16-bit results, row-major
    output wire        done                   // pulses high when all outputs valid
);
```

## Specification

### Architecture
- 4×4 grid of Processing Elements (PEs)
- Each PE holds one preloaded 4-bit weight
- Activations are skewed by row: row i begins accumulating i cycles after start
- PE[i][j] accumulates `weight[i][j] × activation[i]` for exactly **4 consecutive cycles**
- After 4 accumulations, `output[i][j] = 4 × weight[i][j] × activation[i]`

### Timing Requirement — Critical
- `done` must assert within **7 clock cycles** of the `start` pulse
- Row 0 accumulates cycles 0–3, row 1 cycles 1–4, row 2 cycles 2–5, row 3 cycles 3–6
- Row 3 finishes last (cycle 6); `done` asserts at posedge 7

### Processing Element
Each PE in row i, column j computes:
```
if (cycle >= i && cycle < i+4):
    acc[i][j] += weight[i][j] * activations[i]
```

### Data Layout
- `weights_flat[(i*4+j)*4 +: 4]` = weight for PE at row i, column j
- `activations_flat[i*8 +: 8]` = activation for row i (held stable during computation)
- `outputs_flat[(i*4+j)*16 +: 16]` = result `output[i][j]`

## Scoring
- Compilation: 5%
- Functional correctness (all outputs match reference): 50%
- Timing (done asserts within 7 cycles): 30%
- Area efficiency: 15%

## Hints
- Use a 3-bit cycle counter `cyc` (0–6) to gate each row's accumulation window
- Row i accumulates when `cyc >= i && cyc < i+4`
- Set `done_reg <= 1` when `cyc == 6` (row 3 completes its 4th accumulation)
- No shift-register delay lines needed — just gated enables per row
