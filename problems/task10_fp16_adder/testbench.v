`timescale 1ns/1ps

module tb_fp16_adder;
    reg  [15:0] a, b;
    wire [15:0] result;

    integer pass_count = 0;
    integer fail_count = 0;

    fp16_adder dut (.a(a), .b(b), .result(result));

    // FP16 constants
    // Format: 1 sign | 5 exponent | 10 mantissa
    localparam FP16_0p0    = 16'h0000;  //  0.0
    localparam FP16_1p0    = 16'h3C00;  //  1.0
    localparam FP16_2p0    = 16'h4000;  //  2.0
    localparam FP16_3p0    = 16'h4200;  //  3.0
    localparam FP16_4p0    = 16'h4400;  //  4.0
    localparam FP16_5p0    = 16'h4500;  //  5.0
    localparam FP16_6p0    = 16'h4600;  //  6.0
    localparam FP16_0p5    = 16'h3800;  //  0.5
    localparam FP16_1p5    = 16'h3E00;  //  1.5
    localparam FP16_0p25   = 16'h3400;  //  0.25
    localparam FP16_1p75   = 16'h3F00;  //  1.75
    localparam FP16_2p5    = 16'h4100;  //  2.5
    localparam FP16_3p5    = 16'h4300;  //  3.5
    localparam FP16_N1p0   = 16'hBC00;  // -1.0
    localparam FP16_N2p0   = 16'hC000;  // -2.0
    localparam FP16_N0p5   = 16'hB800;  // -0.5
    localparam FP16_INF    = 16'h7C00;  // +Inf
    localparam FP16_NINF   = 16'hFC00;  // -Inf
    localparam FP16_QNAN   = 16'h7E00;  // canonical quiet NaN

    task check;
        input [15:0] expected;
        input [63:0] test_id;
        begin
            #1;
            if (result === expected) begin
                $display("PASS: test %0d — a=0x%04h b=0x%04h result=0x%04h",
                         test_id, a, b, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — a=0x%04h b=0x%04h got=0x%04h expected=0x%04h",
                         test_id, a, b, result, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Check NaN: any exponent=31 non-zero mantissa is acceptable
    task check_nan;
        input [63:0] test_id;
        begin
            #1;
            if (result[14:10] == 5'h1F && result[9:0] != 10'h0) begin
                $display("PASS: test %0d — NaN result=0x%04h (any NaN accepted)", test_id, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — expected NaN got=0x%04h (exp[14:10] should be 5'h1F with non-zero mantissa)",
                         test_id, result);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        // --- Basic same-sign additions ---
        a=FP16_1p0;  b=FP16_1p0;  check(FP16_2p0,  1);  // 1+1=2
        a=FP16_1p0;  b=FP16_0p5;  check(FP16_1p5,  2);  // 1+0.5=1.5
        a=FP16_1p5;  b=FP16_0p25; check(FP16_1p75, 3);  // 1.5+0.25=1.75
        a=FP16_2p0;  b=FP16_1p0;  check(FP16_3p0,  4);  // 2+1=3
        a=FP16_3p0;  b=FP16_3p0;  check(FP16_6p0,  5);  // 3+3=6 (carry-out → normalize)

        // --- Subtraction (opposite signs) ---
        a=FP16_1p0;  b=FP16_N1p0; check(FP16_0p0,  6);  // 1-1=0 (cancellation)
        a=FP16_2p0;  b=FP16_N1p0; check(FP16_1p0,  7);  // 2-1=1
        a=FP16_3p5;  b=FP16_N0p5; check(FP16_3p0,  8);  // 3.5-0.5=3 (normalization needed)
        a=FP16_2p5;  b=FP16_N1p0; check(FP16_1p5,  9);  // 2.5-1=1.5
        a=FP16_4p0;  b=FP16_N2p0; check(FP16_2p0, 10);  // 4-2=2

        // --- Zero operands ---
        a=FP16_0p0;  b=FP16_5p0;  check(FP16_5p0, 11);  // 0+5=5
        a=FP16_5p0;  b=FP16_0p0;  check(FP16_5p0, 12);  // 5+0=5
        a=FP16_0p0;  b=FP16_0p0;  check(FP16_0p0, 13);  // 0+0=0

        // --- Large exponent difference (alignment) ---
        // 4.0 (exp=17) + 0.25 (exp=13) — difference=4, shift 0.25 right by 4
        // 4.0 = 1.0000000000 × 2^2, 0.25 = 1.0000000000 × 2^-2
        // 0.25 aligned to exp=2: 1.0000000000 >> 4 = 0.0001000000
        // sum: 1.0000000000 + 0.0001000000 = 1.0001000000 × 2^2 = 4.25
        // 4.25 = 0_10001_0001000000 = 0x4440
        a=FP16_4p0;  b=FP16_0p25; check(16'h4440, 14);

        // --- Commutativity ---
        a=FP16_0p5;  b=FP16_1p0;  check(FP16_1p5,  15); // 0.5+1 same as 1+0.5

        // --- Negative + Negative ---
        a=FP16_N1p0; b=FP16_N1p0; check(FP16_N2p0, 16); // -1-1=-2

        // --- Special cases: Infinity ---
        a=FP16_1p0;  b=FP16_INF;  check(FP16_INF,  17); // finite+inf=inf
        a=FP16_INF;  b=FP16_1p0;  check(FP16_INF,  18); // inf+finite=inf
        a=FP16_NINF; b=FP16_1p0;  check(FP16_NINF, 19); // -inf+finite=-inf

        // --- Special case: inf - inf = NaN ---
        a=FP16_INF;  b=FP16_NINF;
        check_nan(20);

        // --- NaN propagation ---
        a=FP16_QNAN; b=FP16_1p0;
        check_nan(21);
        a=FP16_1p0;  b=FP16_QNAN;
        check_nan(22);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #10000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
