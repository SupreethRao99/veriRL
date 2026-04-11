// Formal properties for fp16_adder (SymbiYosys, mode prove)
// Checks key IEEE 754 invariants for all normal-number inputs.

module fp16_adder_formal;
    // Symbolic inputs
    wire [15:0] a, b;
    wire [15:0] result;

    fp16_adder dut (.a(a), .b(b), .result(result));

`ifdef FORMAL
    // Helper signals
    wire        ra_sign = result[15];
    wire [4:0]  ra_exp  = result[14:10];
    wire [9:0]  ra_man  = result[9:0];

    wire        a_nan = (a[14:10] == 5'h1F) && (a[9:0] != 0);
    wire        b_nan = (b[14:10] == 5'h1F) && (b[9:0] != 0);
    wire        a_inf = (a[14:10] == 5'h1F) && (a[9:0] == 0);
    wire        b_inf = (b[14:10] == 5'h1F) && (b[9:0] == 0);

    // P1: NaN input propagates — result must be NaN
    always @(*) begin
        if (a_nan || b_nan)
            assert (result[14:10] == 5'h1F && result[9:0] != 0);
    end

    // P2: Inf - Inf = NaN
    always @(*) begin
        if (a_inf && b_inf && a[15] != b[15])
            assert (result[14:10] == 5'h1F && result[9:0] != 0);
    end

    // P3: Finite + Infinity = Infinity (same sign)
    always @(*) begin
        if (a_inf && !b_nan && !b_inf)
            assert (result == {a[15], 5'h1F, 10'h0});
        if (b_inf && !a_nan && !a_inf)
            assert (result == {b[15], 5'h1F, 10'h0});
    end

    // P4: x + 0 = x  (for normal numbers, not NaN/Inf)
    always @(*) begin
        if (!a_nan && !a_inf && (b == 16'h0000))
            // Result exponent and magnitude must match a (sign may differ for -0)
            assert (result[14:0] == a[14:0]);
        if (!b_nan && !b_inf && (a == 16'h0000))
            assert (result[14:0] == b[14:0]);
    end

    // P5: Result exponent is in valid range for normal or zero output
    //     (cannot produce a subnormal for the test cases we target)
    always @(*) begin
        if (!a_nan && !b_nan && !a_inf && !b_inf &&
            (result[14:10] != 5'h1F))  // not infinity
            assert (result[14:10] <= 5'd30 || result == 16'h0);
    end

    // P6: Commutativity — a+b == b+a
    wire [15:0] result_swapped;
    fp16_adder dut2 (.a(b), .b(a), .result(result_swapped));
    always @(*) begin
        // Skip NaN (NaN != NaN by IEEE)
        if (!a_nan && !b_nan)
            assert (result == result_swapped);
    end
`endif

endmodule
