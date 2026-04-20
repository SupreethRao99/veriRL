`timescale 1ns/1ps
// Reference implementation: 2-stage pipelined INT8 dot product (4 elements)
module dot_product_4 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [7:0] a0, a1, a2, a3,
    input  wire signed [7:0] b0, b1, b2, b3,
    output reg         valid_out,
    output reg  signed [17:0] result
);
    // Stage 1: four 8×8 → 16-bit products
    reg signed [15:0] p0, p1, p2, p3;
    reg               s1_valid;

    always @(posedge clk) begin
        if (rst) begin
            p0 <= 0; p1 <= 0; p2 <= 0; p3 <= 0;
            s1_valid  <= 1'b0;
            result    <= 0;
            valid_out <= 1'b0;
        end else begin
            // Stage 1
            p0       <= a0 * b0;
            p1       <= a1 * b1;
            p2       <= a2 * b2;
            p3       <= a3 * b3;
            s1_valid <= valid_in;

            // Stage 2
            result    <= p0 + p1 + p2 + p3;
            valid_out <= s1_valid;
        end
    end

endmodule
