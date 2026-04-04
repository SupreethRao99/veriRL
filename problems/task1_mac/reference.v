// Reference implementation — correct 2-stage pipelined MAC
// Stage 1: multiply; Stage 2: accumulate
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

    // Stage 2 accumulator
    always @(posedge clk) begin
        if (rst) begin
            product_s1 <= 0;
            en_s1      <= 0;
            clear_s1   <= 0;
        end else begin
            product_s1 <= a * b;
            en_s1      <= en;
            clear_s1   <= clear;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            acc_out <= 0;
        end else if (clear_s1) begin
            acc_out <= 0;
        end else if (en_s1) begin
            acc_out <= acc_out + product_s1;
        end
    end

endmodule
