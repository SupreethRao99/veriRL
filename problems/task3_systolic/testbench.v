`timescale 1ns/1ps

module tb_systolic;
    reg        clk, rst, load_weights, start;
    reg [63:0]  weights_flat;
    reg [127:0] activations_flat;
    wire [255:0] outputs_flat;
    wire         done;

    integer pass_count = 0;
    integer fail_count = 0;
    integer done_cycle;
    integer start_cycle;

    systolic_array dut (
        .clk(clk), .rst(rst),
        .load_weights(load_weights), .weights_flat(weights_flat),
        .start(start), .activations_flat(activations_flat),
        .outputs_flat(outputs_flat), .done(done)
    );

    always #5 clk = ~clk;

    integer cycle_counter;
    always @(posedge clk) begin
        if (start) cycle_counter <= 0;
        else if (cycle_counter < 30) cycle_counter <= cycle_counter + 1;
    end

    always @(posedge clk) begin
        if (done) done_cycle <= cycle_counter;
    end

    task check16;
        input [15:0] got, expected;
        input [63:0] test_id;
        begin
            if (got === expected) begin
                $display("PASS: output[%0d]=%0d (expected %0d)", test_id, got, expected);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: output[%0d]=%0d (expected %0d)", test_id, got, expected);
                fail_count = fail_count + 1;
            end
        end
    endtask

    integer i, j;

    initial begin
        clk = 0; rst = 1; load_weights = 0; start = 0;
        weights_flat = 0; activations_flat = 0; cycle_counter = 0; done_cycle = 99;

        @(posedge clk); @(posedge clk);
        #1; rst = 0; @(posedge clk); #1;

        // Test Case 1: Identity weights, all activations = 1
        // Expected: C[i][j] = 1 for all (all-ones 4x4)
        weights_flat = 64'h1111111111111111;
        load_weights = 1; @(posedge clk); #1; load_weights = 0; #1;

        activations_flat = 128'd0;
        activations_flat[7:0]   = 8'd1;
        activations_flat[15:8]  = 8'd1;
        activations_flat[23:16] = 8'd1;
        activations_flat[31:24] = 8'd1;

        start = 1; @(posedge clk); #1; start = 0; #1;

        begin: wait_loop1
            repeat(15) begin
                @(posedge clk); #1;
                if (done) disable wait_loop1;
            end
        end
        // Wait one more cycle so the done_cycle non-blocking capture settles
        @(posedge clk); #1;

        if (done_cycle <= 7) begin
            $display("PASS: timing — done at cycle %0d (<=7)", done_cycle);
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL: timing — done at cycle %0d (expected <=7)", done_cycle);
            fail_count = fail_count + 1;
        end

        $display("CYCLES: %0d", done_cycle);

        for (i = 0; i < 16; i = i + 1)
            check16(outputs_flat[i*16 +: 16], 16'd4, i);

        // Test Case 2: Zero weights
        weights_flat = 64'h0000000000000000;
        load_weights = 1; @(posedge clk); #1; load_weights = 0; #1;

        activations_flat = 128'd0;
        activations_flat[7:0]   = 8'd255;
        activations_flat[15:8]  = 8'd255;
        activations_flat[23:16] = 8'd255;
        activations_flat[31:24] = 8'd255;

        done_cycle = 99;
        start = 1; @(posedge clk); #1; start = 0; #1;
        repeat(12) @(posedge clk);
        #1;

        for (i = 0; i < 16; i = i + 1)
            check16(outputs_flat[i*16 +: 16], 16'd0, 16 + i);

        // Test Case 3: Known values (i+j+1) weights
        weights_flat = 64'h0;
        weights_flat[3:0]   = 4'd1; weights_flat[7:4]   = 4'd2;
        weights_flat[11:8]  = 4'd3; weights_flat[15:12] = 4'd4;
        weights_flat[19:16] = 4'd2; weights_flat[23:20] = 4'd3;
        weights_flat[27:24] = 4'd4; weights_flat[31:28] = 4'd5;
        weights_flat[35:32] = 4'd3; weights_flat[39:36] = 4'd4;
        weights_flat[43:40] = 4'd5; weights_flat[47:44] = 4'd6;
        weights_flat[51:48] = 4'd4; weights_flat[55:52] = 4'd5;
        weights_flat[59:56] = 4'd6; weights_flat[63:60] = 4'd7;
        load_weights = 1; @(posedge clk); #1; load_weights = 0; #1;

        activations_flat = 128'd0;
        activations_flat[7:0]   = 8'd1;
        activations_flat[15:8]  = 8'd2;
        activations_flat[23:16] = 8'd3;
        activations_flat[31:24] = 8'd4;

        done_cycle = 99;
        start = 1; @(posedge clk); #1; start = 0; #1;
        repeat(12) @(posedge clk);
        #1;

        check16(outputs_flat[0*16 +: 16],  16'd4,  32);
        check16(outputs_flat[1*16 +: 16],  16'd8,  33);
        check16(outputs_flat[2*16 +: 16],  16'd12, 34);
        check16(outputs_flat[3*16 +: 16],  16'd16, 35);
        check16(outputs_flat[4*16 +: 16],  16'd16, 36);
        check16(outputs_flat[5*16 +: 16],  16'd24, 37);
        check16(outputs_flat[6*16 +: 16],  16'd32, 38);
        check16(outputs_flat[7*16 +: 16],  16'd40, 39);
        check16(outputs_flat[8*16 +: 16],  16'd36, 40);
        check16(outputs_flat[9*16 +: 16],  16'd48, 41);
        check16(outputs_flat[10*16 +: 16], 16'd60, 42);
        check16(outputs_flat[11*16 +: 16], 16'd72, 43);
        check16(outputs_flat[12*16 +: 16], 16'd64, 44);
        check16(outputs_flat[13*16 +: 16], 16'd80, 45);
        check16(outputs_flat[14*16 +: 16], 16'd96, 46);
        check16(outputs_flat[15*16 +: 16], 16'd112,47);

        // Test Case 4: Powers of 2 weights (1,2,4,8)
        weights_flat = 64'h0;
        weights_flat[3:0]   = 4'd1; weights_flat[7:4]   = 4'd2;
        weights_flat[11:8]  = 4'd4; weights_flat[15:12] = 4'd8;
        weights_flat[19:16] = 4'd1; weights_flat[23:20] = 4'd2;
        weights_flat[27:24] = 4'd4; weights_flat[31:28] = 4'd8;
        weights_flat[35:32] = 4'd1; weights_flat[39:36] = 4'd2;
        weights_flat[43:40] = 4'd4; weights_flat[47:44] = 4'd8;
        weights_flat[51:48] = 4'd1; weights_flat[55:52] = 4'd2;
        weights_flat[59:56] = 4'd4; weights_flat[63:60] = 4'd8;
        load_weights = 1; @(posedge clk); #1; load_weights = 0; #1;

        activations_flat = 128'd0;
        activations_flat[7:0]   = 8'd10;
        activations_flat[15:8]  = 8'd20;
        activations_flat[23:16] = 8'd30;
        activations_flat[31:24] = 8'd40;

        done_cycle = 99;
        start = 1; @(posedge clk); #1; start = 0; #1;
        repeat(12) @(posedge clk);
        #1;

        check16(outputs_flat[0*16 +: 16],  16'd40,  48);
        check16(outputs_flat[1*16 +: 16],  16'd80,  49);
        check16(outputs_flat[2*16 +: 16],  16'd160, 50);
        check16(outputs_flat[3*16 +: 16],  16'd320, 51);

        // Test Case 5: Single column weights
        weights_flat = 64'h0;
        weights_flat[3:0]   = 4'd5; weights_flat[7:4]   = 4'd0;
        weights_flat[11:8]  = 4'd0; weights_flat[15:12] = 4'd0;
        weights_flat[19:16] = 4'd5; weights_flat[23:20] = 4'd0;
        weights_flat[27:24] = 4'd0; weights_flat[31:28] = 4'd0;
        weights_flat[35:32] = 4'd5; weights_flat[39:36] = 4'd0;
        weights_flat[43:40] = 4'd0; weights_flat[47:44] = 4'd0;
        weights_flat[51:48] = 4'd5; weights_flat[55:52] = 4'd0;
        weights_flat[59:56] = 4'd0; weights_flat[63:60] = 4'd0;
        load_weights = 1; @(posedge clk); #1; load_weights = 0; #1;

        activations_flat = 128'd0;
        activations_flat[7:0]   = 8'd2;
        activations_flat[15:8]  = 8'd3;
        activations_flat[23:16] = 8'd4;
        activations_flat[31:24] = 8'd5;

        done_cycle = 99;
        start = 1; @(posedge clk); #1; start = 0; #1;
        repeat(12) @(posedge clk);
        #1;

        check16(outputs_flat[0*16 +: 16],  16'd40, 52);
        check16(outputs_flat[4*16 +: 16],  16'd60, 53);
        check16(outputs_flat[8*16 +: 16],  16'd80, 54);
        check16(outputs_flat[12*16 +: 16], 16'd100, 55);

        // Test Case 6: All 8's weights
        weights_flat = 64'h8888888888888888;
        load_weights = 1; @(posedge clk); #1; load_weights = 0; #1;

        activations_flat = 128'd0;
        activations_flat[7:0]   = 8'd1;
        activations_flat[15:8]  = 8'd1;
        activations_flat[23:16] = 8'd1;
        activations_flat[31:24] = 8'd1;

        done_cycle = 99;
        start = 1; @(posedge clk); #1; start = 0; #1;
        repeat(12) @(posedge clk);
        #1;

        for (i = 0; i < 16; i = i + 1)
            check16(outputs_flat[i*16 +: 16], 16'd32, 56 + i);

        // Test Case 7: Diagonal weights only
        weights_flat = 64'h0;
        weights_flat[3:0]   = 4'd1; weights_flat[23:20] = 4'd2;
        weights_flat[43:40] = 4'd3; weights_flat[63:60] = 4'd4;
        load_weights = 1; @(posedge clk); #1; load_weights = 0; #1;

        activations_flat = 128'd0;
        activations_flat[7:0]   = 8'd10;
        activations_flat[15:8]  = 8'd10;
        activations_flat[23:16] = 8'd10;
        activations_flat[31:24] = 8'd10;

        done_cycle = 99;
        start = 1; @(posedge clk); #1; start = 0; #1;
        repeat(12) @(posedge clk);
        #1;

        check16(outputs_flat[0*16 +: 16],   16'd40, 72);
        check16(outputs_flat[5*16 +: 16],   16'd80, 73);
        check16(outputs_flat[10*16 +: 16],  16'd120, 74);
        check16(outputs_flat[15*16 +: 16],  16'd160, 75);

        $display("SUMMARY: %0d passed, %0d failed", pass_count, fail_count);
        $finish;
    end

    initial #100000 begin
        $display("FAIL: timeout");
        $finish;
    end

endmodule
