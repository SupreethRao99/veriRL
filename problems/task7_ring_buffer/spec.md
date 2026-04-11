# Task 7: Parameterized Ring Buffer (Medium)

## Objective

Implement a power-of-2-depth ring buffer (circular FIFO) with push/pop interface,
full/empty flags, and an item count output. Ring buffers are the standard data
structure for KV-cache management and activation streaming in inference accelerators.

## Interface

```verilog
module ring_buffer #(
    parameter DEPTH  = 8,   // must be power of 2
    parameter DATA_W = 8
) (
    input  wire              clk,
    input  wire              rst,          // synchronous reset, active high
    input  wire              push,
    input  wire [DATA_W-1:0] push_data,
    input  wire              pop,
    output wire [DATA_W-1:0] pop_data,    // valid only when !empty
    output wire              full,
    output wire              empty,
    output wire [$clog2(DEPTH):0] count   // 0 … DEPTH (extra bit for full detection)
);
```

## Specification

- On `push` when **not full**: store `push_data` at tail, advance tail, increment count.
- On `pop` when **not empty**: advance head, decrement count.
- On **simultaneous push + pop** when **neither full nor empty**: both happen, count unchanged.
- On `push` when **full**: push is silently ignored (no data loss).
- On `pop` when **empty**: pop is silently ignored.
- `rst` (synchronous): clears head, tail, and count to 0.
- `pop_data` always reflects `mem[head]` combinatorially (valid only when `!empty`).
- `full` is true when `count == DEPTH`.
- `empty` is true when `count == 0`.

## Scoring

- Correct compilation: 10%
- Passing simulation tests: 70%
- Area efficiency vs reference: 20%

## Notes

- Wrap head and tail with modulo arithmetic or by masking to `$clog2(DEPTH)-1` bits.
- The count register needs one extra bit: `[$clog2(DEPTH):0]` to distinguish 0 from DEPTH.
- Simultaneous push + pop is the tricky case — handle it explicitly with `do_push`/`do_pop` wires.
