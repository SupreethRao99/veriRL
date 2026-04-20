`timescale 1ns/1ps
// Reference implementation: direct-form I 3-tap FIR filter
// y[n] = h0*x[n] + h1*x[n-1] + h2*x[n-2]
module fir3 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [7:0]  x,
    input  wire signed [7:0]  h0, h1, h2,
    output reg  signed [17:0] y,
    output reg                valid_out
);
    reg signed [7:0] x_d1, x_d2;   // delay line: x[n-1], x[n-2]

    always @(posedge clk) begin
        if (rst) begin
            x_d1      <= 8'sb0;
            x_d2      <= 8'sb0;
            y         <= 18'sb0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            // Compute output using CURRENT history (non-blocking: uses pre-edge values)
            y         <= h0*x + h1*x_d1 + h2*x_d2;
            // Shift delay line
            x_d1      <= x;
            x_d2      <= x_d1;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule
