`timescale 1ns/1ps

module tb_dot_product_4;
    reg        clk, rst, valid_in;
    reg signed [7:0] a0, a1, a2, a3;
    reg signed [7:0] b0, b1, b2, b3;
    wire              valid_out;
    wire signed [17:0] result;

    integer pass_count = 0;
    integer fail_count = 0;

    dot_product_4 dut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .a0(a0), .a1(a1), .a2(a2), .a3(a3),
        .b0(b0), .b1(b1), .b2(b2), .b3(b3),
        .valid_out(valid_out), .result(result)
    );

    always #5 clk = ~clk;

    task check_result;
        input signed [17:0] expected;
        input [63:0] test_id;
        begin
            if (result === expected && valid_out === 1'b1) begin
                $display("PASS: test %0d — result=%0d valid_out=1", test_id, result);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — result=%0d valid_out=%0d expected=%0d",
                         test_id, result, valid_out, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_valid_low;
        input [63:0] test_id;
        begin
            if (valid_out === 1'b0) begin
                $display("PASS: test %0d — valid_out=0 (correct drain)", test_id);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — valid_out should be 0, got 1", test_id);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        clk=0; rst=1; valid_in=0;
        a0=0; a1=0; a2=0; a3=0;
        b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); @(posedge clk);
        rst=0; @(posedge clk);

        // Test 1: [1,1,1,1] · [1,1,1,1] = 4
        valid_in=1; a0=1; a1=1; a2=1; a3=1; b0=1; b1=1; b2=1; b3=1;
        @(posedge clk); valid_in=0; a0=0; a1=0; a2=0; a3=0; b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); #1;
        check_result(18'sd4, 1);

        // Test 2: valid_out deasserts after drain
        @(posedge clk); #1;
        check_valid_low(2);

        // Test 3: [1,2,3,4] · [5,6,7,8] = 5+12+21+32 = 70
        valid_in=1; a0=1; a1=2; a2=3; a3=4; b0=5; b1=6; b2=7; b3=8;
        @(posedge clk); valid_in=0; a0=0; a1=0; a2=0; a3=0; b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); #1;
        check_result(18'sd70, 3);

        // Test 4: negative inputs [-1,-2,-3,-4] · [1,1,1,1] = -10
        valid_in=1;
        a0=-8'sd1; a1=-8'sd2; a2=-8'sd3; a3=-8'sd4;
        b0= 8'sd1; b1= 8'sd1; b2= 8'sd1; b3= 8'sd1;
        @(posedge clk); valid_in=0; a0=0; a1=0; a2=0; a3=0; b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); #1;
        check_result(-18'sd10, 4);

        // Test 5: mixed signs [-1,2,-3,4] · [1,1,1,1] = -1+2-3+4 = 2
        valid_in=1;
        a0=-8'sd1; a1= 8'sd2; a2=-8'sd3; a3= 8'sd4;
        b0= 8'sd1; b1= 8'sd1; b2= 8'sd1; b3= 8'sd1;
        @(posedge clk); valid_in=0; a0=0; a1=0; a2=0; a3=0; b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); #1;
        check_result(18'sd2, 5);

        // Test 6: max values [127,127,127,127]·[127,127,127,127] = 4*16129 = 64516
        valid_in=1;
        a0=8'sd127; a1=8'sd127; a2=8'sd127; a3=8'sd127;
        b0=8'sd127; b1=8'sd127; b2=8'sd127; b3=8'sd127;
        @(posedge clk); valid_in=0; a0=0; a1=0; a2=0; a3=0; b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); #1;
        check_result(18'sd64516, 6);

        // Test 7: zero vectors
        valid_in=1; a0=0; a1=0; a2=0; a3=0; b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); valid_in=0;
        @(posedge clk); #1;
        check_result(18'sd0, 7);

        // Test 8+9: back-to-back valid_in — two consecutive inputs, two consecutive outputs
        valid_in=1;
        a0=1; a1=0; a2=0; a3=0; b0=3; b1=0; b2=0; b3=0;  // dot = 3
        @(posedge clk);
        a0=2; a1=0; a2=0; a3=0; b0=4; b1=0; b2=0; b3=0;  // dot = 8
        @(posedge clk); valid_in=0; a0=0; a1=0; a2=0; a3=0; b0=0; b1=0; b2=0; b3=0;
        @(posedge clk); #1;
        check_result(18'sd3, 8);  // first result
        @(posedge clk); #1;
        check_result(18'sd8, 9);  // second result

        // Test 10: rst clears valid_out
        valid_in=1; a0=5; b0=5; a1=0; b1=0; a2=0; b2=0; a3=0; b3=0;
        @(posedge clk); rst=1; valid_in=0;
        @(posedge clk); rst=0; #1;
        check_valid_low(10);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #100000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
