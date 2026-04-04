`timescale 1ns/1ps

module tb_mac_unit;
    reg        clk, rst, en, clear;
    reg signed [7:0]  a, b;
    wire signed [31:0] acc_out;

    integer pass_count = 0;
    integer fail_count = 0;

    mac_unit dut (
        .clk(clk), .rst(rst), .en(en), .clear(clear),
        .a(a), .b(b), .acc_out(acc_out)
    );

    always #5 clk = ~clk;

    task check;
        input signed [31:0] expected;
        input [63:0] test_id;
        begin
            if (acc_out === expected) begin
                $display("PASS: test %0d — acc_out=%0d (expected %0d)", test_id, acc_out, expected);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — acc_out=%0d (expected %0d)", test_id, acc_out, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        clk = 0; rst = 1; en = 0; clear = 0; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        rst = 0;

        // Test 1: single accumulation a=3, b=4
        en = 1; a = 8'sd3; b = 8'sd4;
        @(posedge clk);
        a = 0; b = 0; en = 0;
        @(posedge clk); @(posedge clk);
        check(32'sd12, 1);

        // Test 2: back-to-back accumulation
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd1; b = 8'sd1;
        @(posedge clk);
        a = 8'sd2; b = 8'sd2;
        @(posedge clk);
        a = 0; b = 0; en = 0;
        @(posedge clk);
        check(32'sd1, 2);
        @(posedge clk);
        check(32'sd5, 3);

        // Test 3: negative inputs
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = -8'sd5; b = 8'sd3;
        @(posedge clk);
        en = 0; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        check(-32'sd15, 4);

        // Test 4: enable hold
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd2; b = 8'sd2;
        @(posedge clk); en = 0; a = 8'sd9; b = 8'sd9;
        @(posedge clk);
        @(posedge clk);
        check(32'sd4, 5);

        // Test 5: clear signal
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd5; b = 8'sd5;
        @(posedge clk);
        a = 0; b = 0; en = 0; clear = 1;
        @(posedge clk);
        clear = 0;
        @(posedge clk); @(posedge clk);
        check(32'sd0, 6);

        // Test 6: rst clears everything
        en = 1; a = 8'sd7; b = 8'sd7;
        @(posedge clk);
        rst = 1; @(posedge clk); rst = 0;
        @(posedge clk);
        check(32'sd0, 7);

        // Test 7: Large positive numbers
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd127; b = 8'sd127;
        @(posedge clk);
        en = 0; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        check(32'sd16129, 8);

        // Test 8: Large negative numbers
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = -8'sd128; b = -8'sd128;
        @(posedge clk);
        en = 0; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        check(32'sd16384, 9);

        // Test 9: Mixed signs
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = -8'sd10; b = 8'sd10;
        @(posedge clk);
        en = 0; a = 0; b = 0;
        @(posedge clk); @(posedge clk);
        check(-32'sd100, 10);

        // Test 10: Accumulate multiple values
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd2; b = 8'sd3;
        @(posedge clk);
        a = 8'sd4; b = 8'sd5;
        @(posedge clk);
        a = 8'sd6; b = 8'sd7;
        @(posedge clk);
        en = 0;
        @(posedge clk);
        check(32'sd6, 11);  // first result: 2*3=6
        @(posedge clk);
        check(32'sd26, 12); // 6 + 4*5=20
        @(posedge clk);
        check(32'sd68, 13); // 26 + 6*7=42

        // Test 11: Clear in middle of computation
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd5; b = 8'sd5;
        @(posedge clk);
        a = 8'sd10; b = 8'sd10;
        @(posedge clk);
        a = 0; b = 0; clear = 1;
        @(posedge clk);
        clear = 0;
        @(posedge clk);
        check(32'sd0, 14);

        // Test 12: Alternating enable/disable
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd1; b = 8'sd1;
        @(posedge clk);
        en = 0; a = 8'sd2; b = 8'sd2;
        @(posedge clk);
        en = 1; a = 8'sd3; b = 8'sd3;
        @(posedge clk);
        en = 0; a = 0; b = 0;
        @(posedge clk);
        @(posedge clk);
        check(32'sd10, 15);

        // Test 13: Zero inputs
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd0; b = 8'sd0;
        @(posedge clk);
        en = 0;
        @(posedge clk); @(posedge clk);
        check(32'sd0, 16);

        // Test 14: One zero input
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd50; b = 8'sd0;
        @(posedge clk);
        en = 0;
        @(posedge clk); @(posedge clk);
        check(32'sd0, 17);

        // Test 15: Continuous accumulation
        rst = 1; @(posedge clk); rst = 0;
        en = 1;
        a = 8'sd1; b = 8'sd2; @(posedge clk);
        a = 8'sd2; b = 8'sd2; @(posedge clk);
        a = 8'sd3; b = 8'sd2; @(posedge clk);
        a = 8'sd4; b = 8'sd2; @(posedge clk);
        en = 0; a = 0; b = 0;
        @(posedge clk);
        check(32'sd2, 18);  // 1*2=2
        @(posedge clk);
        check(32'sd6, 19);  // 2 + 2*2=6
        @(posedge clk);
        check(32'sd12, 20); // 6 + 3*2=12
        @(posedge clk);
        check(32'sd20, 21); // 12 + 4*2=20

        // Test 16: Boundary cases
        rst = 1; @(posedge clk); rst = 0;
        en = 1; a = 8'sd1; b = 8'sd127;
        @(posedge clk);
        en = 0;
        @(posedge clk); @(posedge clk);
        check(32'sd127, 22);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #10000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
