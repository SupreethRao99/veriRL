# Task 10: IEEE 754 FP16 Adder (Hard)

## Objective

Implement a combinational IEEE 754 half-precision (FP16) floating-point adder.
FP16 arithmetic is the compute primitive of every modern GPU tensor core and
AI accelerator — getting it right is non-trivial and formally verifiable.

## Interface

```verilog
module fp16_adder (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] result
);
```

## FP16 Format (IEEE 754-2008)

```
Bit 15  : sign       (s)
Bits 14:10 : exponent (e, biased with bias=15)
Bits  9:0  : mantissa (m, implicit leading 1 for normal numbers)

Value = (-1)^s × 2^(e−15) × 1.m    for normal numbers (e = 1..30)
Value = 0                            for e = 0, m = 0  (zero)
```

## Scope (What You Must Handle)

| Case                   | Requirement                                  |
|------------------------|----------------------------------------------|
| Normal + Normal        | Correct result, normalized                   |
| x + 0  or  0 + x       | Return x                                     |
| x + (−x) (cancellation)| Return +0.0 (`16'h0000`)                    |
| Overflow to infinity   | Return `16'h7C00` (+Inf) or `16'hFC00` (−Inf)|
| NaN input (e=31,m≠0)   | Propagate: return `16'h7E00`                 |
| Infinity input (e=31,m=0)| Propagate or handle ∞±∞ as NaN             |

**Rounding**: truncate (round toward zero). Round-to-nearest is not required but earns full area score.

## Algorithm

```
1.  Extract fields: sign, exp (5-bit), mantissa (10-bit)
2.  Prepend implicit 1: full_m = {1, mantissa}  (11 bits; 0 for zero/subnormal)
3.  If |a| < |b|: swap so that |a| >= |b|
4.  Compute alignment shift d = exp_a − exp_b  (≥ 0 after swap)
5.  Shift full_m_b right by d (with 3 guard bits for rounding)
6.  If signs equal:  sum_m = full_m_a + shifted_m_b
    If signs differ: sum_m = full_m_a − shifted_m_b
7.  Normalize: count leading zeros in sum_m, left-shift, adjust exponent
8.  Handle exponent overflow → ±Inf
9.  Pack result: {sign_result, exp_result[4:0], sum_m[9:0]}
```

## Scoring

- Correct compilation: 5%
- Passing simulation tests (normal numbers + zero + special cases): 60%
- Formal verification (SymbiYosys, if available): 15%
- Area efficiency vs reference: 20%

## Useful Constants

```verilog
localparam BIAS    = 15;
localparam INF     = 16'h7C00;
localparam NEG_INF = 16'hFC00;
localparam QNAN    = 16'h7E00;
```

## Hint: Alignment and Normalization

```verilog
// Extended mantissa with guard bits
wire [13:0] m_b_shifted = {1'b1, man_b, 3'b0} >> d;   // 14 bits: 1 hidden + 10 + 3 guard

// After subtraction, find first 1 in result (clz):
// Use a priority encoder or a generate loop to count leading zeros
```
