`timescale 1ns/1ps

module tb_axi_fifo;
    parameter DATA_W = 8;

    reg              clk, rst;
    reg              s_valid, m_ready;
    reg  [DATA_W-1:0] s_data;
    wire             s_ready, m_valid, full, empty;
    wire [DATA_W-1:0] m_data;

    integer pass_count = 0;
    integer fail_count = 0;

    axi_fifo #(.DATA_W(DATA_W)) dut (
        .clk(clk), .rst(rst),
        .s_valid(s_valid), .s_ready(s_ready), .s_data(s_data),
        .m_valid(m_valid), .m_ready(m_ready), .m_data(m_data),
        .full(full), .empty(empty)
    );

    always #5 clk = ~clk;

    task check_eq;
        input [DATA_W-1:0] got, expected;
        input [63:0] test_id;
        begin
            if (got === expected) begin
                $display("PASS: test %0d — got=%0d expected=%0d", test_id, got, expected);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — got=%0d expected=%0d", test_id, got, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_flag;
        input actual, expected;
        input [63:0] test_id;
        begin
            if (actual === expected) begin
                $display("PASS: flag test %0d", test_id);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: flag test %0d — got=%0d expected=%0d", test_id, actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    initial begin
        clk = 0; rst = 1; s_valid = 0; m_ready = 0; s_data = 0;
        @(posedge clk); @(posedge clk);
        rst = 0; @(posedge clk);

        // Test 1-2: empty on reset
        check_flag(empty, 1, 1);
        check_flag(full,  0, 2);

        // Test 3-4: push one item
        s_valid = 1; s_data = 8'hAA;
        @(posedge clk); s_valid = 0;
        @(posedge clk);
        check_flag(empty, 0, 3);
        check_flag(m_valid, 1, 4);

        // Test 5: pop one item
        m_ready = 1;
        @(posedge clk); m_ready = 0;
        @(posedge clk);
        check_flag(empty, 1, 5);

        // Test 6-7: fill to full
        s_valid = 1;
        s_data = 8'h01; @(posedge clk);
        s_data = 8'h02; @(posedge clk);
        s_data = 8'h03; @(posedge clk);
        s_data = 8'h04; @(posedge clk);
        s_valid = 0;
        @(posedge clk);
        check_flag(full, 1, 6);
        check_flag(s_ready, 0, 7);

        // Test 8-11: drain in order
        m_ready = 1;
        @(posedge clk); check_eq(m_data, 8'h01, 8);
        @(posedge clk); check_eq(m_data, 8'h02, 9);
        @(posedge clk); check_eq(m_data, 8'h03, 10);
        @(posedge clk); check_eq(m_data, 8'h04, 11);
        m_ready = 0;
        @(posedge clk);
        check_flag(empty, 1, 12);

        // Test 13-14: simultaneous enq+deq
        s_valid = 1; s_data = 8'hBB; @(posedge clk);
        s_data = 8'hCC; @(posedge clk);
        s_valid = 0; @(posedge clk);
        s_valid = 1; s_data = 8'hDD; m_ready = 1;
        @(posedge clk);
        s_valid = 0; m_ready = 0;
        @(posedge clk);
        check_flag(empty, 0, 13);
        check_flag(full, 0, 14);

        // Test 15-16: downstream stall
        rst = 1; @(posedge clk); rst = 0;
        s_valid = 1; s_data = 8'hEE; @(posedge clk); s_valid = 0;
        m_ready = 0;
        repeat(5) @(posedge clk);
        check_flag(m_valid, 1, 15);
        m_ready = 1; @(posedge clk);
        check_eq(m_data, 8'hEE, 16);
        m_ready = 0;

        // Test 17-20: multiple enqueue and dequeue cycles
        rst = 1; @(posedge clk); rst = 0;
        s_valid = 1;
        s_data = 8'h11; @(posedge clk);
        s_data = 8'h22; @(posedge clk);
        s_data = 8'h33; @(posedge clk);
        s_valid = 0; @(posedge clk);
        m_ready = 1;
        @(posedge clk); check_eq(m_data, 8'h11, 17);
        @(posedge clk); check_eq(m_data, 8'h22, 18);
        @(posedge clk); check_eq(m_data, 8'h33, 19);
        m_ready = 0;
        @(posedge clk);
        check_flag(empty, 1, 20);

        // Test 21-24: Fill, drain partially, refill
        rst = 1; @(posedge clk); rst = 0;
        s_valid = 1;
        s_data = 8'hAA; @(posedge clk);
        s_data = 8'hBB; @(posedge clk);
        s_data = 8'hCC; @(posedge clk);
        s_data = 8'hDD; @(posedge clk);
        s_valid = 0; @(posedge clk);
        m_ready = 1;
        @(posedge clk); check_eq(m_data, 8'hAA, 21);
        @(posedge clk); check_eq(m_data, 8'hBB, 22);
        m_ready = 0; s_valid = 1;
        @(posedge clk);
        s_data = 8'hEE; @(posedge clk);
        s_valid = 0;
        m_ready = 1;
        @(posedge clk); check_eq(m_data, 8'hCC, 23);
        @(posedge clk); check_eq(m_data, 8'hDD, 24);

        // Test 25-28: Rapid push/pop pattern
        rst = 1; @(posedge clk); rst = 0;
        s_valid = 1; m_ready = 0;
        s_data = 8'h55; @(posedge clk);
        s_data = 8'h66; m_ready = 1; @(posedge clk);
        check_eq(m_data, 8'h55, 25);
        s_data = 8'h77; @(posedge clk);
        check_eq(m_data, 8'h66, 26);
        s_data = 8'h88; @(posedge clk);
        check_eq(m_data, 8'h77, 27);
        s_valid = 0; @(posedge clk);
        check_eq(m_data, 8'h88, 28);

        // Test 29-30: Check s_ready during full
        rst = 1; @(posedge clk); rst = 0;
        s_valid = 1;
        s_data = 8'h01; @(posedge clk);
        s_data = 8'h02; @(posedge clk);
        s_data = 8'h03; @(posedge clk);
        s_data = 8'h04; @(posedge clk);
        @(posedge clk);
        check_flag(full, 1, 29);
        check_flag(s_ready, 0, 30);

        // Test 31-32: Empty after drain
        rst = 1; @(posedge clk); rst = 0;
        s_valid = 1;
        s_data = 8'hFF; @(posedge clk);
        s_valid = 0; @(posedge clk);
        m_ready = 1;
        @(posedge clk);
        @(posedge clk);
        check_flag(empty, 1, 31);
        check_flag(m_valid, 0, 32);

        // Test 33-36: Back to back transactions
        rst = 1; @(posedge clk); rst = 0;
        s_valid = 1; m_ready = 1;
        s_data = 8'h99; @(posedge clk);
        s_data = 8'hAA; @(posedge clk);
        s_data = 8'hBB; @(posedge clk);
        s_data = 8'hCC; @(posedge clk);
        s_data = 8'hDD; @(posedge clk);  // will fail to enqueue (full)
        check_eq(m_data, 8'h99, 33);
        s_valid = 0; m_ready = 0;
        @(posedge clk);
        check_flag(empty, 0, 34);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #20000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
