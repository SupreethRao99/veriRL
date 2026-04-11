// Formal properties for relu_clip (SymbiYosys, mode prove)
// Wraps the DUT and asserts invariants that hold for ALL input combinations.
`ifndef IN_W
  `define IN_W 8
`endif
`ifndef OUT_W
  `define OUT_W 4
`endif

module relu_clip_formal #(
    parameter IN_W  = `IN_W,
    parameter OUT_W = `OUT_W
);
    localparam integer MAX_OUT = (1 << OUT_W) - 1;

    // Symbolic inputs (smtbmc treats undriven wires as free variables)
    wire signed [IN_W-1:0]  in_val;
    wire        [OUT_W-1:0] out_val;
    wire                    saturated;

    relu_clip #(.IN_W(IN_W), .OUT_W(OUT_W)) dut (
        .in_val(in_val), .out_val(out_val), .saturated(saturated)
    );

`ifdef FORMAL
    // P1: Negative input always produces zero output
    always @(*) begin
        if ($signed(in_val) < 0)
            assert (out_val == {OUT_W{1'b0}});
    end

    // P2: Input in range [0, MAX_OUT] passes through unchanged
    always @(*) begin
        if ($signed(in_val) >= 0 && in_val <= MAX_OUT[IN_W-1:0])
            assert (out_val == in_val[OUT_W-1:0]);
    end

    // P3: Input above MAX_OUT clamps to MAX_OUT
    always @(*) begin
        if ($signed(in_val) > $signed(MAX_OUT[IN_W-1:0]))
            assert (out_val == MAX_OUT[OUT_W-1:0]);
    end

    // P4: saturated is high iff input was clipped by either ReLU or upper bound
    always @(*) begin
        if ($signed(in_val) < 0 || $signed(in_val) > $signed(MAX_OUT[IN_W-1:0]))
            assert (saturated == 1'b1);
        else
            assert (saturated == 1'b0);
    end

    // P5: out_val is always a valid OUT_W-bit unsigned value (no X/Z)
    always @(*) assert (!$isunknown(out_val));

    // P6: saturated is always defined
    always @(*) assert (!$isunknown(saturated));
`endif

endmodule
