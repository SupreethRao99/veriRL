// Reference implementation: dual-read / single-write register file
// Register 0 is hardwired to 0.
module register_file #(
    parameter ADDR_W = 5,
    parameter DATA_W = 32
) (
    input  wire              clk,
    input  wire              we,
    input  wire [ADDR_W-1:0] wr_addr,
    input  wire [DATA_W-1:0] wr_data,
    input  wire [ADDR_W-1:0] rd_addr_a,
    input  wire [ADDR_W-1:0] rd_addr_b,
    output wire [DATA_W-1:0] rd_data_a,
    output wire [DATA_W-1:0] rd_data_b
);
    localparam DEPTH = 1 << ADDR_W;

    reg [DATA_W-1:0] regs [0:DEPTH-1];

    integer i;
    initial begin
        for (i = 0; i < DEPTH; i = i + 1)
            regs[i] = 0;
    end

    // Synchronous write — skip r0
    always @(posedge clk)
        if (we && (wr_addr != 0))
            regs[wr_addr] <= wr_data;

    // Asynchronous read — r0 hardwired
    assign rd_data_a = (rd_addr_a == 0) ? {DATA_W{1'b0}} : regs[rd_addr_a];
    assign rd_data_b = (rd_addr_b == 0) ? {DATA_W{1'b0}} : regs[rd_addr_b];

endmodule
