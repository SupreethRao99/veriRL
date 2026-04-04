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
- Activations flow left-to-right across rows with correct diagonal skewing
- PE[i][j] receives activation[i] at cycle i+j (0-indexed from start pulse)
- Each PE accumulates: acc += weight × activation each cycle it receives valid data
- After 4 accumulations, PE[i][j] output is valid

### Timing Requirement — Critical
- `done` must assert within **7 clock cycles** of the `start` pulse
- This requires correct diagonal skewing of input activations
- The wavefront propagates diagonally: PE[0][0] fires at cycle 0, PE[1][1] at cycle 1, PE[3][3] at cycle 6

### Processing Element
Each PE computes:
```
acc += weight[i][j] * activation_in
activation_out = activation_in  // pass through to next column
```

### Data Layout
- `weights_flat[4*i+j * 4 +: 4]` = weight for PE at row i, column j
- `activations_flat[i*8 +: 8]` = activation for row i
- `outputs_flat[(i*4+j)*16 +: 16]` = result C[i][j]

## Scoring
- Compilation: 5%
- Functional correctness (all 16 outputs match numpy reference): 50%
- Timing (done asserts within 7 cycles): 30%
- Area efficiency: 15%

## Hints
- Implement skew delay lines using shift registers on each activation input
- Row i needs a shift register of depth i before the first PE
- The total latency is: skew_depth + 4 accumulation cycles = i + 4 cycles for row i
- done asserts when the last PE (PE[3][3]) has finished: at cycle 3 (skew) + 3 (propagation) = 6 cycles after start, so done at cycle 7
