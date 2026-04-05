// Reference implementation: weight-stationary 4x4 systolic array
//
// Architecture:
//   - Row i's activation is delayed by i cycles (diagonal skewing via row-gated enable)
//   - PE[i][j] accumulates exactly 4 times: at cycles i, i+1, i+2, i+3
//   - Row 3 finishes last (cycles 3..6), so done_reg fires at cyc==6
//   - done wire is high for one cycle at posedge 7 from start
//   - output[i][j] = 4 * activations[i] * weights[i][j]

module systolic_array (
    input  wire        clk,
    input  wire        rst,
    input  wire        load_weights,
    input  wire [63:0] weights_flat,
    input  wire        start,
    input  wire [127:0] activations_flat,
    output wire [255:0] outputs_flat,
    output wire        done
);

    reg [3:0]  weights [0:3][0:3];
    reg [15:0] acc     [0:3][0:3];
    reg [2:0]  cyc;
    reg        running;
    reg        done_reg;

    assign done = done_reg;

    genvar gi, gj;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : row_out
            for (gj = 0; gj < 4; gj = gj + 1) begin : col_out
                assign outputs_flat[(gi*4+gj)*16 +: 16] = acc[gi][gj];
            end
        end
    endgenerate

    // Weight loading
    integer li, lj;
    always @(posedge clk) begin
        if (load_weights)
            for (li = 0; li < 4; li = li + 1)
                for (lj = 0; lj < 4; lj = lj + 1)
                    weights[li][lj] <= weights_flat[(li*4+lj)*4 +: 4];
    end

    // Main compute block
    // Row i accumulates during cycles [i .. i+3] (4 times), gated by cyc.
    // Total active cycles: 0..6 (7 cycles), done_reg <= 1 at cyc==6.
    integer ci, cj;
    always @(posedge clk) begin
        if (rst) begin
            running  <= 0;
            done_reg <= 0;
            cyc      <= 0;
            for (ci = 0; ci < 4; ci = ci + 1)
                for (cj = 0; cj < 4; cj = cj + 1)
                    acc[ci][cj] <= 0;
        end else if (start) begin
            running  <= 1;
            done_reg <= 0;
            cyc      <= 0;
            for (ci = 0; ci < 4; ci = ci + 1)
                for (cj = 0; cj < 4; cj = cj + 1)
                    acc[ci][cj] <= 0;
        end else if (running) begin
            // Each row i fires when cyc >= i and cyc < i+4
            for (ci = 0; ci < 4; ci = ci + 1) begin
                if (cyc >= ci && cyc < ci + 4) begin
                    for (cj = 0; cj < 4; cj = cj + 1)
                        acc[ci][cj] <= acc[ci][cj]
                            + {{12{1'b0}}, weights[ci][cj]}
                              * activations_flat[ci*8 +: 8];
                end
            end
            cyc <= cyc + 1;
            if (cyc == 3'd6) begin
                done_reg <= 1;
                running  <= 0;
            end
        end else begin
            done_reg <= 0;
        end
    end

endmodule
