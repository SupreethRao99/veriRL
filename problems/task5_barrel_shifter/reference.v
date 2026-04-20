// Reference implementation: parameterized barrel shifter
module barrel_shifter #(
    parameter WIDTH = 8
) (
    input  wire [WIDTH-1:0]          data_in,
    input  wire [$clog2(WIDTH)-1:0]  shift_amt,
    input  wire                      direction,   // 0=left, 1=right
    input  wire                      arithmetic,  // 0=logical, 1=arith (right only)
    output wire [WIDTH-1:0]          data_out
);
    wire [WIDTH-1:0] left_out  = data_in << shift_amt;
    wire [WIDTH-1:0] right_log = data_in >> shift_amt;
    wire [WIDTH-1:0] right_ari = $signed(data_in) >>> shift_amt;

    assign data_out = direction
        ? (arithmetic ? right_ari : right_log)
        : left_out;

endmodule
