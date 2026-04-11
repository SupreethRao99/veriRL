// Reference implementation: ReLU-Clip activation unit
// Fully combinational; correct for any OUT_W <= IN_W.
module relu_clip #(
    parameter IN_W  = 8,
    parameter OUT_W = 4
) (
    input  wire signed [IN_W-1:0]  in_val,
    output wire        [OUT_W-1:0] out_val,
    output wire                    saturated
);
    localparam integer MAX_OUT = (1 << OUT_W) - 1;

    // ReLU: negative values → 0
    wire neg    = in_val[IN_W-1];                         // sign bit
    wire [IN_W-1:0] relu_out = neg ? {IN_W{1'b0}} : in_val;

    // Saturating cast to OUT_W bits
    wire pos_clip = (relu_out > MAX_OUT[IN_W-1:0]);

    assign out_val   = pos_clip ? MAX_OUT[OUT_W-1:0] : relu_out[OUT_W-1:0];
    assign saturated = neg | pos_clip;

endmodule
