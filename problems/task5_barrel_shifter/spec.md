# Task 5: Parameterized Barrel Shifter (Easy)

## Objective

Design a fully combinational, parameterized barrel shifter supporting left shift,
logical right shift, and arithmetic right shift. Barrel shifters are a critical
component in every SIMD and vector processing unit.

## Interface

```verilog
module barrel_shifter #(
    parameter WIDTH = 8
) (
    input  wire [WIDTH-1:0]          data_in,
    input  wire [$clog2(WIDTH)-1:0]  shift_amt,   // 0 … WIDTH-1
    input  wire                      direction,    // 0 = left, 1 = right
    input  wire                      arithmetic,   // 0 = logical, 1 = arithmetic (right only)
    output wire [WIDTH-1:0]          data_out
);
```

## Specification

| `direction` | `arithmetic` | Operation                                  |
|-------------|--------------|---------------------------------------------|
| `0`         | `x` (ignored)| Logical left shift: `data_in << shift_amt`  |
| `1`         | `0`          | Logical right shift: `data_in >> shift_amt` |
| `1`         | `1`          | Arithmetic right shift: `$signed(data_in) >>> shift_amt` |

- Shift by 0 must return `data_in` unchanged.
- `arithmetic` is ignored when `direction == 0`.
- The module is **fully combinational** — no clock or reset ports.
- Must synthesize cleanly for WIDTH = 8 (testbench) and WIDTH = 16 (synthesis check).

## Scoring

- Correct compilation: 10%
- Passing simulation tests (WIDTH=8): 80%
- Area efficiency vs reference: 10%

## Notes

- Use Verilog's built-in shift operators: `<<`, `>>`, `>>>`.
- `>>>` on a `wire` does logical shift; cast to `$signed()` first for arithmetic behaviour.
- A simple ternary chain over `{direction, arithmetic}` is sufficient.
