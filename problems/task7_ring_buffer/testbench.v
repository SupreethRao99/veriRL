`timescale 1ns/1ps

module tb_ring_buffer;
    parameter DEPTH  = 8;
    parameter DATA_W = 8;

    reg              clk, rst, push, pop;
    reg  [DATA_W-1:0] push_data;
    wire [DATA_W-1:0] pop_data;
    wire              full, empty;
    wire [$clog2(DEPTH):0] count;

    integer pass_count = 0;
    integer fail_count = 0;

    ring_buffer #(.DEPTH(DEPTH), .DATA_W(DATA_W)) dut (
        .clk(clk), .rst(rst),
        .push(push), .push_data(push_data),
        .pop(pop), .pop_data(pop_data),
        .full(full), .empty(empty), .count(count)
    );

    always #5 clk = ~clk;

    task check_flag;
        input actual, expected;
        input [63:0] test_id;
        input [127:0] name;
        begin
            if (actual === expected) begin
                $display("PASS: test %0d — %s=%0d", test_id, name, actual);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — %s got=%0d expected=%0d", test_id, name, actual, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_count;
        input [$clog2(DEPTH):0] expected;
        input [63:0] test_id;
        begin
            if (count === expected) begin
                $display("PASS: test %0d — count=%0d", test_id, count);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — count got=%0d expected=%0d", test_id, count, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    task check_pop;
        input [DATA_W-1:0] expected;
        input [63:0] test_id;
        begin
            if (pop_data === expected) begin
                $display("PASS: test %0d — pop_data=%0d", test_id, pop_data);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: test %0d — pop_data got=%0d expected=%0d", test_id, pop_data, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    integer i;

    initial begin
        clk=0; rst=1; push=0; pop=0; push_data=0;
        @(posedge clk); #1; @(posedge clk); #1;
        rst=0; @(posedge clk); #1;

        // Test 1-3: reset state
        check_flag(empty, 1, 1, "empty");
        check_flag(full,  0, 2, "full");
        check_count(0,       3);

        // Test 4: push one item
        push=1; push_data=8'hAA;
        @(posedge clk); #1; push=0; #1;
        check_flag(empty, 0, 4, "empty");
        check_count(1,       5);
        check_pop(8'hAA,     6);

        // Test 5: pop it back out
        pop=1; @(posedge clk); #1; pop=0; #1;
        check_flag(empty, 1, 7, "empty");
        check_count(0,       8);

        // Test 6: fill to DEPTH (8 items)
        push=1;
        for (i=1; i<=8; i=i+1) begin
            push_data = i[DATA_W-1:0];
            @(posedge clk); #1;
        end
        push=0; #1;
        check_flag(full, 1, 9, "full");
        check_count(8, 10);

        // Test 7: push when full — ignored
        push=1; push_data=8'hFF;
        @(posedge clk); #1; push=0; #1;
        check_flag(full, 1, 11, "full");
        check_count(8,      12);

        // Test 8: pop and verify FIFO order (items 1..8 in order)
        pop=1;
        for (i=1; i<=8; i=i+1) begin
            check_pop(i[DATA_W-1:0], 12+i); // tests 13..20
            @(posedge clk); #1;
        end
        pop=0; #1;
        check_flag(empty, 1, 21, "empty");

        // Test 9: simultaneous push+pop (count stays same)
        push=1; pop=0; push_data=8'h55;
        @(posedge clk); #1;
        // now count=1; do simultaneous push+pop
        push=1; pop=1; push_data=8'h66;
        @(posedge clk); #1; push=0; pop=0; #1;
        check_count(1, 22); // was 1, +1-1=1

        // Test 10: wrap-around: push 8, pop 4, push 4 more, verify order
        // Start fresh
        rst=1; @(posedge clk); #1; rst=0; @(posedge clk); #1;
        push=1;
        for (i=0; i<8; i=i+1) begin
            push_data = 8'h10 + i[7:0];
            @(posedge clk); #1;
        end
        push=0; #1;
        pop=1; #1;
        for (i=0; i<4; i=i+1) begin
            check_pop(8'h10 + i[7:0], 23+i); // tests 23..26
            @(posedge clk); #1;
        end
        pop=0; #1;
        push=1;
        for (i=0; i<4; i=i+1) begin
            push_data = 8'h20 + i[7:0];
            @(posedge clk); #1;
        end
        push=0; #1;
        pop=1; #1;
        for (i=4; i<8; i=i+1) begin
            check_pop(8'h10 + i[7:0], 27+i-4); // tests 27..30, expecting 0x14..0x17
            @(posedge clk); #1;
        end
        for (i=0; i<4; i=i+1) begin
            check_pop(8'h20 + i[7:0], 31+i); // tests 31..34
            @(posedge clk); #1;
        end
        pop=0; #1;
        check_flag(empty, 1, 35, "empty");

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #100000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
