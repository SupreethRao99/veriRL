`timescale 1ns/1ps
// Reference implementation: IEEE 754 FP16 adder (combinational)
// Handles: normal numbers, zero, infinity, NaN propagation.
// Rounding: truncate (round toward zero).
module fp16_adder (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] result
);
    // --- Field extraction ---
    wire        sa = a[15],      sb = b[15];
    wire [4:0]  ea = a[14:10],   eb = b[14:10];
    wire [9:0]  ma = a[9:0],     mb = b[9:0];

    // Special value detection
    wire a_inf  = (ea == 5'h1F) & (ma == 10'h0);
    wire b_inf  = (eb == 5'h1F) & (mb == 10'h0);
    wire a_nan  = (ea == 5'h1F) & (ma != 10'h0);
    wire b_nan  = (eb == 5'h1F) & (mb != 10'h0);
    wire a_zero = (ea == 5'h00) & (ma == 10'h0);
    wire b_zero = (eb == 5'h00) & (mb == 10'h0);

    // Full mantissa with implicit leading 1 (12 bits for alignment headroom)
    wire [10:0] fa = a_zero ? 11'h0 : {1'b1, ma};
    wire [10:0] fb = b_zero ? 11'h0 : {1'b1, mb};

    // --- Swap so that |a_op| >= |b_op| ---
    wire [15:0] mag_a = {1'b0, a[14:0]};
    wire [15:0] mag_b = {1'b0, b[14:0]};
    wire        swap  = (mag_b > mag_a);

    wire        s_big  = swap ? sb      : sa;
    wire [4:0]  e_big  = swap ? eb      : ea;
    wire [10:0] f_big  = swap ? fb      : fa;
    wire        s_sml  = swap ? sa      : sb;
    wire [4:0]  e_sml  = swap ? ea      : eb;
    wire [10:0] f_sml  = swap ? fa      : fb;

    // --- Alignment ---
    wire [4:0] diff = e_big - e_sml;
    wire [10:0] f_sml_sh = (diff >= 11) ? 11'h0 : (f_sml >> diff);

    // --- Addition / subtraction ---
    wire same_sign = (s_big == s_sml);
    wire [11:0] sum_raw = same_sign
        ? ({1'b0, f_big} + {1'b0, f_sml_sh})
        : ({1'b0, f_big} - {1'b0, f_sml_sh});

    // --- Normalization ---
    // Find leading-one position in 12-bit sum_raw
    reg [3:0] lz;
    always @(*) begin
        casez (sum_raw)
            12'b1???????????  : lz = 4'd0;
            12'b01??????????  : lz = 4'd1;
            12'b001?????????  : lz = 4'd2;
            12'b0001????????  : lz = 4'd3;
            12'b00001???????  : lz = 4'd4;
            12'b000001??????  : lz = 4'd5;
            12'b0000001?????  : lz = 4'd6;
            12'b00000001????  : lz = 4'd7;
            12'b000000001???  : lz = 4'd8;
            12'b0000000001??  : lz = 4'd9;
            12'b00000000001?  : lz = 4'd10;
            12'b000000000001  : lz = 4'd11;
            default           : lz = 4'd12; // zero
        endcase
    end

    // Normalize: shift so the leading 1 lands at bit[10], then sum_norm[9:0] is mantissa.
    // Overflow (lz==0, bit[11]=1): shift right by 1 → leading 1 moves from [11] to [10].
    // Normal (lz>=1): shift left by (lz-1) → leading 1 moves from [11-lz] to [10].
    wire [3:0] norm_shift = (lz == 0) ? 4'd0 : (lz - 4'd1);
    wire [11:0] sum_norm  = (lz == 0) ? (sum_raw >> 1) : (sum_raw << norm_shift);
    // Exponent: +1 for overflow, -(lz-1) for left-shift normalization.
    wire [5:0] exp_adj = (sum_raw[11])
        ? ({1'b0, e_big} + 6'd1)
        : ({1'b0, e_big} - {2'b0, lz} + 6'd1);

    wire result_zero = (sum_raw == 12'h0);
    wire exp_overflow = (exp_adj >= 6'd31) & ~result_zero;

    // --- Special case output mux ---
    wire [15:0] nan_out  = 16'h7E00;
    wire [15:0] inf_out  = {s_big, 5'h1F, 10'h0};
    wire [15:0] zero_out = 16'h0000;
    wire [15:0] norm_out = {s_big, exp_adj[4:0], sum_norm[9:0]};

    assign result =
        (a_nan | b_nan)                 ? nan_out  :
        (a_inf & b_inf & (sa != sb))    ? nan_out  :  // ∞ - ∞
        (a_inf | b_inf)                 ? inf_out  :
        (exp_overflow)                  ? inf_out  :
        (result_zero)                   ? zero_out :
        norm_out;

endmodule
