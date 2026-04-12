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
        @(posedge clk); @(posedge clk); @(posedge clk);
        #1; rst = 0; @(posedge clk); #1;

        // Test 1: single accumulation a=3, b=4
        en = 1; a = 8'sd3; b = 8'sd4;
        @(posedge clk); #1; // Stage 1 captures 3,4
        en = 0; a = 0; b = 0;
        repeat(2) @(posedge clk); #1; // Stage 2 adds 12 to 0
        check(32'sd12, 1);

        // Test 2-3: back-to-back accumulation
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd1; b = 8'sd1;
        @(posedge clk); #1; // Stage 1 captures 1,1
        a = 8'sd2; b = 8'sd2;
        @(posedge clk); #1; // Stage 2 adds 1 to 0. Stage 1 captures 2,2.
        check(32'sd1, 2);
        en = 0; a = 0; b = 0;
        @(posedge clk); #1; // Stage 2 adds 4 to 1.
        check(32'sd5, 3);

        // Test 4: negative inputs
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = -8'sd5; b = 8'sd3;
        @(posedge clk); #1;
        en = 0; a = 0; b = 0;
        repeat(2) @(posedge clk); #1;
        check(-32'sd15, 4);

        // Test 5: enable hold
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd2; b = 8'sd2;
        @(posedge clk); #1;
        en = 0; a = 8'sd9; b = 8'sd9;
        repeat(2) @(posedge clk); #1;
        check(32'sd4, 5);

        // Test 6: clear signal
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd5; b = 8'sd5;
        @(posedge clk); #1; // s1 captures 25
        en = 0; a = 0; b = 0; clear = 1;
        @(posedge clk); #1; // s2 adds 25 to 0. s1 captures clear=1.
        @(posedge clk); #1; // s2 clears acc.
        check(32'sd0, 6);
        clear = 0;

        // Test 7: rst clears everything
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd7; b = 8'sd7;
        @(posedge clk); #1;
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        check(32'sd0, 7);

        // Test 8: Large positive numbers
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0;
        a = 0; b = 0; en = 0;
        repeat(2) @(posedge clk); #1;
        en = 1; a = 8'sd127; b = 8'sd127;
        @(posedge clk); #1;
        en = 0; a = 0; b = 0;
        repeat(2) @(posedge clk); #1;
        check(32'sd16129, 8);

        // Test 9: Large negative numbers
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = -8'sd128; b = -8'sd128;
        @(posedge clk); #1;
        en = 0; a = 0; b = 0;
        repeat(2) @(posedge clk); #1;
        check(32'sd16384, 9);

        // Test 10: Mixed signs
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = -8'sd10; b = 8'sd10;
        @(posedge clk); #1;
        en = 0; a = 0; b = 0;
        repeat(2) @(posedge clk); #1;
        check(-32'sd100, 10);

        // Test 11-13: Accumulate multiple values
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd2; b = 8'sd3;
        @(posedge clk); #1; // s1 captures 6
        a = 8'sd4; b = 8'sd5;
        @(posedge clk); #1; // s2 adds 6 to 0. s1 captures 20.
        check(32'sd6, 11);
        a = 8'sd6; b = 8'sd7;
        @(posedge clk); #1; // s2 adds 20 to 6. s1 captures 42.
        check(32'sd26, 12);
        en = 0; a = 0; b = 0;
        @(posedge clk); #1; // s2 adds 42 to 26.
        check(32'sd68, 13);

        // Test 14: Clear in middle of computation
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd5; b = 8'sd5;
        @(posedge clk); #1;
        a = 8'sd10; b = 8'sd10;
        @(posedge clk); #1;
        en = 0; a = 0; b = 0; clear = 1;
        @(posedge clk); #1;
        clear = 0;
        @(posedge clk); #1; @(posedge clk); #1;
        check(32'sd0, 14);

        // Test 15: Alternating enable/disable
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd1; b = 8'sd1;
        @(posedge clk); #1;
        en = 0; a = 8'sd2; b = 8'sd2;
        @(posedge clk); #1;
        en = 1; a = 8'sd3; b = 8'sd3;
        @(posedge clk); #1;
        en = 0; a = 0; b = 0;
        repeat(2) @(posedge clk); #1;
        check(32'sd10, 15);

        // Test 16: Zero inputs
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd0; b = 8'sd0;
        @(posedge clk); #1;
        en = 0;
        repeat(2) @(posedge clk); #1;
        check(32'sd0, 16);

        // Test 17: One zero input
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd50; b = 8'sd0;
        @(posedge clk); #1;
        en = 0;
        repeat(2) @(posedge clk); #1;
        check(32'sd0, 17);

        // Test 18-21: Continuous accumulation
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1;
        a = 8'sd1; b = 8'sd2;
        @(posedge clk); #1; // s1 = 2
        a = 8'sd2; b = 8'sd2;
        @(posedge clk); #1; // s2 = 2, s1 = 4
        check(32'sd2, 18);
        a = 8'sd3; b = 8'sd2;
        @(posedge clk); #1; // s2 = 6, s1 = 6
        check(32'sd6, 19);
        a = 8'sd4; b = 8'sd2;
        @(posedge clk); #1; // s2 = 12, s1 = 8
        check(32'sd12, 20);
        en = 0; a = 0; b = 0;
        @(posedge clk); #1; // s2 = 20
        check(32'sd20, 21);

        // Test 22: Boundary cases
        rst = 1; repeat(3) @(posedge clk); #1; rst = 0; @(posedge clk); #1;
        en = 1; a = 8'sd1; b = 8'sd127;
        @(posedge clk); #1;
        en = 0;
        repeat(2) @(posedge clk); #1;
        check(32'sd127, 22);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #10000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
