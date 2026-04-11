`timescale 1ns/1ps

module tb_relu_clip;
    parameter IN_W  = 8;
    parameter OUT_W = 4;
    // MAX_OUT = 2^4 - 1 = 15

    reg signed [IN_W-1:0]  in_val;
    wire       [OUT_W-1:0] out_val;
    wire                   saturated;

    integer pass_count = 0;
    integer fail_count = 0;

    relu_clip #(.IN_W(IN_W), .OUT_W(OUT_W)) dut (
        .in_val(in_val),
        .out_val(out_val),
        .saturated(saturated)
    );

    task check_out;
        input [OUT_W-1:0] expected_out;
        input              expected_sat;
        input [63:0]       test_id;
        begin
            #1; // let combinational settle
            if (out_val === expected_out && saturated === expected_sat) begin
                $display("PASS: test %0d — in=%0d out=%0d sat=%0d",
                         test_id, in_val, out_val, saturated);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — in=%0d got out=%0d sat=%0d expected out=%0d sat=%0d",
                         test_id, in_val, out_val, saturated, expected_out, expected_sat);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        // --- Negative inputs: ReLU clips to 0, saturated=1 ---
        in_val = -8'sd128; check_out(4'd0, 1, 1);   // most-negative INT8
        in_val = -8'sd1;   check_out(4'd0, 1, 2);   // -1
        in_val = -8'sd64;  check_out(4'd0, 1, 3);   // -64

        // --- Zero: no clipping ---
        in_val = 8'sd0;    check_out(4'd0, 0, 4);

        // --- Positive in-range: pass through ---
        in_val = 8'sd1;    check_out(4'd1,  0, 5);
        in_val = 8'sd7;    check_out(4'd7,  0, 6);
        in_val = 8'sd14;   check_out(4'd14, 0, 7);
        in_val = 8'sd15;   check_out(4'd15, 0, 8);  // boundary: exactly MAX_OUT

        // --- Positive above MAX_OUT: clamp to 15, saturated=1 ---
        in_val = 8'sd16;   check_out(4'd15, 1, 9);
        in_val = 8'sd20;   check_out(4'd15, 1, 10);
        in_val = 8'sd100;  check_out(4'd15, 1, 11);
        in_val = 8'sd127;  check_out(4'd15, 1, 12); // most-positive INT8

        // --- Edge: just below and at MAX_OUT ---
        in_val = 8'sd8;    check_out(4'd8,  0, 13);
        in_val = 8'sd12;   check_out(4'd12, 0, 14);
        in_val = 8'sd13;   check_out(4'd13, 0, 15);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #5000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
