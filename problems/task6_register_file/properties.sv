// Formal properties for register_file (SymbiYosys, mode prove, depth 3)
`ifndef ADDR_W
  `define ADDR_W 5
`endif
`ifndef DATA_W
  `define DATA_W 32
`endif

module register_file_formal #(
    parameter ADDR_W = `ADDR_W,
    parameter DATA_W = `DATA_W
);
    reg clk = 0;
    always #5 clk = ~clk;

    // Symbolic inputs
    reg              we;
    reg [ADDR_W-1:0] wr_addr, rd_addr_a, rd_addr_b;
    reg [DATA_W-1:0] wr_data;
    wire [DATA_W-1:0] rd_data_a, rd_data_b;

    register_file #(.ADDR_W(ADDR_W), .DATA_W(DATA_W)) dut (
        .clk(clk), .we(we),
        .wr_addr(wr_addr), .wr_data(wr_data),
        .rd_addr_a(rd_addr_a), .rd_addr_b(rd_addr_b),
        .rd_data_a(rd_data_a), .rd_data_b(rd_data_b)
    );

`ifdef FORMAL
    // P1: Reading address 0 always returns 0 (combinational, checked every cycle)
    always @(*) begin
        if (rd_addr_a == {ADDR_W{1'b0}})
            assert (rd_data_a == {DATA_W{1'b0}});
        if (rd_addr_b == {ADDR_W{1'b0}})
            assert (rd_data_b == {DATA_W{1'b0}});
    end

    // P2: Write to address 0 must never corrupt r0 (checked on next cycle)
    // We use an auxiliary register to track if we tried to write r0
    reg wrote_r0 = 0;
    always @(posedge clk) begin
        if (we && wr_addr == 0) wrote_r0 <= 1;
        else                    wrote_r0 <= 0;
    end
    always @(*) begin
        if (wrote_r0) begin
            if (rd_addr_a == 0) assert (rd_data_a == 0);
            if (rd_addr_b == 0) assert (rd_data_b == 0);
        end
    end

    // P3: Write-then-read — value written at posedge appears combinatorially after
    // Auxiliary storage for last valid write
    reg [ADDR_W-1:0] last_wr_addr;
    reg [DATA_W-1:0] last_wr_data;
    reg              last_we;
    always @(posedge clk) begin
        last_wr_addr <= wr_addr;
        last_wr_data <= wr_data;
        last_we      <= we;
    end
    always @(*) begin
        if (last_we && last_wr_addr != 0 && rd_addr_a == last_wr_addr)
            assert (rd_data_a == last_wr_data);
        if (last_we && last_wr_addr != 0 && rd_addr_b == last_wr_addr)
            assert (rd_data_b == last_wr_data);
    end
`endif

endmodule
