`timescale 1ns/1ps

module tb_barrel_shifter;
    parameter WIDTH = 8;

    reg [WIDTH-1:0]         data_in;
    reg [$clog2(WIDTH)-1:0] shift_amt;
    reg                     direction;
    reg                     arithmetic;
    wire [WIDTH-1:0]        data_out;

    integer pass_count = 0;
    integer fail_count = 0;

    barrel_shifter #(.WIDTH(WIDTH)) dut (
        .data_in(data_in),
        .shift_amt(shift_amt),
        .direction(direction),
        .arithmetic(arithmetic),
        .data_out(data_out)
    );

    task check;
        input [WIDTH-1:0] expected;
        input [63:0]      test_id;
        begin
            #1;
            if (data_out === expected) begin
                $display("PASS: test %0d — in=0x%02h amt=%0d dir=%0d ari=%0d out=0x%02h",
                         test_id, data_in, shift_amt, direction, arithmetic, data_out);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — in=0x%02h amt=%0d dir=%0d ari=%0d got=0x%02h expected=0x%02h",
                         test_id, data_in, shift_amt, direction, arithmetic, data_out, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        // --- Left shifts ---
        data_in=8'h01; shift_amt=3'd1; direction=0; arithmetic=0; check(8'h02, 1); // 1<<1
        data_in=8'h01; shift_amt=3'd7; direction=0; arithmetic=0; check(8'h80, 2); // 1<<7
        data_in=8'hFF; shift_amt=3'd1; direction=0; arithmetic=0; check(8'hFE, 3); // 0xFF<<1
        data_in=8'hFF; shift_amt=3'd4; direction=0; arithmetic=0; check(8'hF0, 4); // 0xFF<<4
        data_in=8'hAA; shift_amt=3'd0; direction=0; arithmetic=0; check(8'hAA, 5); // shift 0 = no-op

        // --- Logical right shifts ---
        data_in=8'h80; shift_amt=3'd1; direction=1; arithmetic=0; check(8'h40, 6);  // MSB→0
        data_in=8'hFF; shift_amt=3'd4; direction=1; arithmetic=0; check(8'h0F, 7);  // upper nibble lost
        data_in=8'hF0; shift_amt=3'd4; direction=1; arithmetic=0; check(8'h0F, 8);
        data_in=8'h0F; shift_amt=3'd2; direction=1; arithmetic=0; check(8'h03, 9);
        data_in=8'hAA; shift_amt=3'd0; direction=1; arithmetic=0; check(8'hAA, 10); // shift 0

        // --- Arithmetic right shifts ---
        data_in=8'h80; shift_amt=3'd1; direction=1; arithmetic=1; check(8'hC0, 11); // sign extended
        data_in=8'hFF; shift_amt=3'd4; direction=1; arithmetic=1; check(8'hFF, 12); // all-ones
        data_in=8'h80; shift_amt=3'd7; direction=1; arithmetic=1; check(8'hFF, 13); // full extension
        data_in=8'h40; shift_amt=3'd1; direction=1; arithmetic=1; check(8'h20, 14); // positive: no extension
        data_in=8'h7F; shift_amt=3'd3; direction=1; arithmetic=1; check(8'h0F, 15); // positive arith == logical

        // --- arithmetic flag ignored on left shift ---
        data_in=8'h01; shift_amt=3'd2; direction=0; arithmetic=1; check(8'h04, 16);
        data_in=8'hF0; shift_amt=3'd2; direction=0; arithmetic=1; check(8'hC0, 17);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #5000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
