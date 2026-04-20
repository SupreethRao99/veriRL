// Reference implementation — correct 2-stage pipelined MAC
// Stage 1: multiply; Stage 2: accumulate
`timescale 1ns/1ps

module mac_unit (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire        clear,
    input  wire signed [7:0]  a,
    input  wire signed [7:0]  b,
    output reg  signed [31:0] acc_out
);

    // Stage 1 registers
    reg signed [15:0] product_s1;
    reg               en_s1;
    reg               clear_s1;

    // Stage 1
    always @(posedge clk) begin
        if (rst) begin
            product_s1 <= 16'sd0;
            en_s1      <= 1'b0;
            clear_s1   <= 1'b0;
        end else begin
            product_s1 <= a * b;
            en_s1      <= en;
            clear_s1   <= clear;
        end
    end

    // Stage 2 accumulator
    always @(posedge clk) begin
        if (rst) begin
            acc_out <= 32'sd0;
        end else if (clear_s1) begin
            acc_out <= 32'sd0;
        end else if (en_s1) begin
            acc_out <= acc_out + {{16{product_s1[15]}}, product_s1};
        end
        if (en_s1) $display("TIME=%0t en_s1=1 product_s1=%d acc_out_PRE=%d", $time, product_s1, acc_out);
    end

endmodule
