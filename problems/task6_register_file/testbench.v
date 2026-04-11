`timescale 1ns/1ps

module tb_register_file;
    parameter ADDR_W = 5;
    parameter DATA_W = 32;

    reg              clk, we;
    reg  [ADDR_W-1:0] wr_addr, rd_addr_a, rd_addr_b;
    reg  [DATA_W-1:0] wr_data;
    wire [DATA_W-1:0] rd_data_a, rd_data_b;

    integer pass_count = 0;
    integer fail_count = 0;

    register_file #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
        .clk(clk), .we(we),
        .wr_addr(wr_addr), .wr_data(wr_data),
        .rd_addr_a(rd_addr_a), .rd_addr_b(rd_addr_b),
        .rd_data_a(rd_data_a), .rd_data_b(rd_data_b)
    );

    always #5 clk = ~clk;

    task check_a;
        input [DATA_W-1:0] expected;
        input [63:0] test_id;
        begin
            if (rd_data_a === expected) begin
                $display("PASS: test %0d — rd_a[r%0d]=%0d", test_id, rd_addr_a, rd_data_a);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — rd_a[r%0d] got=%0d expected=%0d",
                         test_id, rd_addr_a, rd_data_a, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_b;
        input [DATA_W-1:0] expected;
        input [63:0] test_id;
        begin
            if (rd_data_b === expected) begin
                $display("PASS: test %0d — rd_b[r%0d]=%0d", test_id, rd_addr_b, rd_data_b);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — rd_b[r%0d] got=%0d expected=%0d",
                         test_id, rd_addr_b, rd_data_b, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        clk=0; we=0; wr_addr=0; wr_data=0; rd_addr_a=0; rd_addr_b=0;

        // Test 1: r0 always reads 0 (before any write)
        rd_addr_a = 5'd0; #1;
        check_a(32'd0, 1);

        // Test 2: write r1=42, read back
        we=1; wr_addr=5'd1; wr_data=32'd42;
        @(posedge clk); #1; we=0;
        rd_addr_a = 5'd1; #1;
        check_a(32'd42, 2);

        // Test 3: write r0=99 — must have no effect, r0 stays 0
        we=1; wr_addr=5'd0; wr_data=32'd99;
        @(posedge clk); #1; we=0;
        rd_addr_a = 5'd0; #1;
        check_a(32'd0, 3);

        // Test 4+5: simultaneous read of two different registers
        we=1; wr_addr=5'd2; wr_data=32'd100;
        @(posedge clk); #1;
        we=1; wr_addr=5'd3; wr_data=32'd200;
        @(posedge clk); #1; we=0;
        rd_addr_a=5'd2; rd_addr_b=5'd3; #1;
        check_a(32'd100, 4);
        check_b(32'd200, 5);

        // Test 6: read port B of r1
        rd_addr_b=5'd1; #1;
        check_b(32'd42, 6);

        // Test 7: overwrite r1=0, verify
        we=1; wr_addr=5'd1; wr_data=32'd0;
        @(posedge clk); #1; we=0;
        rd_addr_a=5'd1; #1;
        check_a(32'd0, 7);

        // Test 8: large register address (r31)
        we=1; wr_addr=5'd31; wr_data=32'hDEADBEEF;
        @(posedge clk); #1; we=0;
        rd_addr_a=5'd31; #1;
        check_a(32'hDEADBEEF, 8);

        // Test 9: write r0 again, still reads 0
        we=1; wr_addr=5'd0; wr_data=32'hFFFFFFFF;
        @(posedge clk); #1; we=0;
        rd_addr_a=5'd0; rd_addr_b=5'd0; #1;
        check_a(32'd0, 9);
        check_b(32'd0, 10);

        // Test 11: write MAX value to r5
        we=1; wr_addr=5'd5; wr_data=32'hFFFFFFFF;
        @(posedge clk); #1; we=0;
        rd_addr_a=5'd5; #1;
        check_a(32'hFFFFFFFF, 11);

        // Test 12: two independent ports see different registers simultaneously
        we=1; wr_addr=5'd10; wr_data=32'd1234;
        @(posedge clk); #1; we=0;
        rd_addr_a=5'd10; rd_addr_b=5'd31; #1;
        check_a(32'd1234, 12);
        check_b(32'hDEADBEEF, 13);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #50000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
