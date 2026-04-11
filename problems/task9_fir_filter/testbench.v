`timescale 1ns/1ps

module tb_fir3;
    reg        clk, rst, valid_in;
    reg signed [7:0] x, h0, h1, h2;
    wire signed [17:0] y;
    wire               valid_out;

    integer pass_count = 0;
    integer fail_count = 0;

    fir3 dut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x(x), .h0(h0), .h1(h1), .h2(h2),
        .y(y), .valid_out(valid_out)
    );

    always #5 clk = ~clk;

    task check_y;
        input signed [17:0] expected;
        input [63:0] test_id;
        begin
            if (y === expected && valid_out === 1'b1) begin
                $display("PASS: test %0d — y=%0d valid_out=1", test_id, y);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — y=%0d valid_out=%0d expected=%0d",
                         test_id, y, valid_out, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_valid_low;
        input [63:0] test_id;
        begin
            if (valid_out === 1'b0) begin
                $display("PASS: test %0d — valid_out=0 (correct)", test_id);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — valid_out should be 0", test_id);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Helper: drive one sample and capture output
    task drive_sample;
        input signed [7:0] sample;
        begin
            x = sample; valid_in = 1;
            @(posedge clk); valid_in = 0; x = 0;
            #1; // combinational settle after posedge
        end
    endtask

    initial begin
        clk=0; rst=1; valid_in=0; x=0; h0=0; h1=0; h2=0;
        @(posedge clk); @(posedge clk);
        rst=0; @(posedge clk);

        // ==========================================================
        // Test Group 1: h=[1,0,0] (pure delay — y[n] = x[n])
        // ==========================================================
        h0=8'sd1; h1=8'sd0; h2=8'sd0;

        // Feed impulse x=1, then zeros
        drive_sample(8'sd1);
        check_y(18'sd1, 1);   // y = 1*1 + 0*0 + 0*0 = 1

        drive_sample(8'sd0);
        check_y(18'sd0, 2);   // y = 1*0 + 0*1 + 0*0 = 0

        drive_sample(8'sd0);
        check_y(18'sd0, 3);   // y = 1*0 + 0*0 + 0*1 = 0

        // ==========================================================
        // Test Group 2: h=[1,1,1] — 3-sample moving average (×3)
        // ==========================================================
        rst=1; @(posedge clk); rst=0; @(posedge clk);
        h0=8'sd1; h1=8'sd1; h2=8'sd1;

        drive_sample(8'sd1);   // history: d1=0, d2=0
        check_y(18'sd1, 4);    // 1*1 + 1*0 + 1*0 = 1

        drive_sample(8'sd0);   // history: d1=1, d2=0
        check_y(18'sd1, 5);    // 1*0 + 1*1 + 1*0 = 1

        drive_sample(8'sd0);   // history: d1=0, d2=1
        check_y(18'sd1, 6);    // 1*0 + 1*0 + 1*1 = 1

        drive_sample(8'sd0);   // history: d1=0, d2=0
        check_y(18'sd0, 7);    // all zero

        // ==========================================================
        // Test Group 3: h=[1,2,3] — step response x=5 repeated
        // ==========================================================
        rst=1; @(posedge clk); rst=0; @(posedge clk);
        h0=8'sd1; h1=8'sd2; h2=8'sd3;

        drive_sample(8'sd5);   // d1=0, d2=0
        check_y(18'sd5, 8);    // 1*5 + 2*0 + 3*0 = 5

        drive_sample(8'sd5);   // d1=5, d2=0
        check_y(18'sd15, 9);   // 1*5 + 2*5 + 3*0 = 15

        drive_sample(8'sd5);   // d1=5, d2=5
        check_y(18'sd30, 10);  // 1*5 + 2*5 + 3*5 = 30

        drive_sample(8'sd5);   // steady state
        check_y(18'sd30, 11);  // 1*5 + 2*5 + 3*5 = 30

        // ==========================================================
        // Test Group 4: negative coefficients h=[-1,2,-1]
        // Laplacian-like kernel — zero sum, detects changes
        // ==========================================================
        rst=1; @(posedge clk); rst=0; @(posedge clk);
        h0=-8'sd1; h1=8'sd2; h2=-8'sd1;

        drive_sample(8'sd0);
        check_y(18'sd0, 12);   // all zero history

        drive_sample(8'sd4);   // d1=0, d2=0
        check_y(-18'sd4, 13);  // -1*4 + 2*0 + -1*0 = -4

        drive_sample(8'sd0);   // d1=4, d2=0
        check_y(18'sd8, 14);   // -1*0 + 2*4 + -1*0 = 8

        drive_sample(8'sd0);   // d1=0, d2=4
        check_y(-18'sd4, 15);  // -1*0 + 2*0 + -1*4 = -4

        // ==========================================================
        // Test Group 5: valid_in=0 should not update history or valid_out
        // ==========================================================
        rst=1; @(posedge clk); rst=0; @(posedge clk);
        h0=8'sd1; h1=8'sd1; h2=8'sd1;

        drive_sample(8'sd3);   // push 3 into history, d1=3
        check_y(18'sd3, 16);

        // One idle cycle (valid_in=0)
        valid_in=0; @(posedge clk); #1;
        check_valid_low(17);   // valid_out must deassert

        // Now send another sample — history should still see d1=3 from test 16
        drive_sample(8'sd1);   // d1=3, d2=0 (d2 was never set)
        check_y(18'sd4, 18);   // 1*1 + 1*3 + 1*0 = 4

        // ==========================================================
        // Test Group 6: rst clears history
        // ==========================================================
        h0=8'sd1; h1=8'sd1; h2=8'sd1;
        drive_sample(8'sd10);  // load 10 into history
        rst=1; @(posedge clk); rst=0; @(posedge clk);

        drive_sample(8'sd1);
        check_y(18'sd1, 19);   // history cleared: 1*1 + 1*0 + 1*0 = 1

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #100000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
