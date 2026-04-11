# Task 4: Parameterized ReLU-Clip Unit (Easy)

## Objective

Design a fully combinational activation unit that applies ReLU (rectified linear unit)
followed by saturating cast to a narrower unsigned integer — a core building block in
every quantized neural-network inference pipeline.

## Interface

```verilog
module relu_clip #(
    parameter IN_W  = 8,   // input bit-width  (signed)
    parameter OUT_W = 4    // output bit-width (unsigned, <= IN_W)
) (
    input  wire signed [IN_W-1:0]  in_val,
    output wire        [OUT_W-1:0] out_val,
    output wire                    saturated
);
```

## Specification

The module performs two sequential operations on `in_val`:

1. **ReLU**: if `in_val < 0`, output zero.
2. **Saturating cast**: clamp to the unsigned `OUT_W`-bit range `[0, 2^OUT_W − 1]`.

Formally, let `MAX = 2^OUT_W − 1`:

| Condition                        | `out_val`             | `saturated` |
|----------------------------------|-----------------------|-------------|
| `in_val < 0`                     | `0`                   | `1`         |
| `0 ≤ in_val ≤ MAX`               | `in_val[OUT_W-1:0]`   | `0`         |
| `in_val > MAX`                   | `MAX`                 | `1`         |

- `saturated` is **1** whenever the output was clipped (either by ReLU or by the upper bound).
- The module is **fully combinational** — no clock or reset ports.
- Must be correct for any `OUT_W ≤ IN_W`.

## Scoring

- Correct compilation: 10%
- Passing simulation tests (IN_W=8, OUT_W=4): 75%
- Formal verification (SymbiYosys, if available): 10%
- Area efficiency vs reference: 5%

## Notes

- Use `localparam integer MAX_OUT = (1 << OUT_W) - 1;` to compute the saturation value.
- The sign bit of `in_val` is `in_val[IN_W-1]`.
- A negative signed value has a 1 in the MSB; masking it off gives you ReLU for free.
- Unsigned comparison with `MAX_OUT` handles the upper saturation.
