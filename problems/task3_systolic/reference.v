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

    // PE weight storage
    reg [3:0] weights [0:3][0:3];

    // Internal PE connections
    // act_in[i][j]: activation entering PE[i][j]
    reg [7:0] act_pipe [0:3][0:4];  // one extra for output (unused)

    // Accumulators
    reg [15:0] acc [0:3][0:3];
    reg [2:0]  cyc_count;     // counts 0..7 from start
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
        if (load_weights) begin
            for (li = 0; li < 4; li = li + 1)
                for (lj = 0; lj < 4; lj = lj + 1)
                    weights[li][lj] <= weights_flat[(li*4+lj)*4 +: 4];
        end
    end

    // Activation skew delay lines: row i gets i-cycle delay
    // act_pipe[i][0] = raw activation for row i
    // act_pipe[i][k] = activation delayed by k cycles
    integer si;
    always @(posedge clk) begin
        for (si = 0; si < 4; si = si + 1) begin
            if (rst) begin
                act_pipe[si][0] <= 0;
                act_pipe[si][1] <= 0;
                act_pipe[si][2] <= 0;
                act_pipe[si][3] <= 0;
            end else if (running) begin
                act_pipe[si][0] <= activations_flat[si*8 +: 8];
                act_pipe[si][1] <= act_pipe[si][0];
                act_pipe[si][2] <= act_pipe[si][1];
                act_pipe[si][3] <= act_pipe[si][2];
            end
        end
    end

    // Compute: each PE[i][j] uses act_pipe[i][j] as its skewed input
    integer ci, cj;
    always @(posedge clk) begin
        if (rst) begin
            running   <= 0;
            done_reg  <= 0;
            cyc_count <= 0;
            for (ci = 0; ci < 4; ci = ci + 1)
                for (cj = 0; cj < 4; cj = cj + 1)
                    acc[ci][cj] <= 0;
        end else if (start) begin
            running   <= 1;
            done_reg  <= 0;
            cyc_count <= 0;
            for (ci = 0; ci < 4; ci = ci + 1)
                for (cj = 0; cj < 4; cj = cj + 1)
                    acc[ci][cj] <= 0;
        end else if (running) begin
            cyc_count <= cyc_count + 1;
            for (ci = 0; ci < 4; ci = ci + 1) begin
                for (cj = 0; cj < 4; cj = cj + 1) begin
                    // PE[i][j] fires when its skewed activation arrives
                    // act_pipe[i][j] has j-cycle delay, PE in column j
                    // Valid from cycle ci onward for row ci
                    acc[ci][cj] <= acc[ci][cj] + weights[ci][cj] * act_pipe[ci][cj];
                end
            end
            if (cyc_count == 3'd6) begin
                done_reg <= 1;
                running  <= 0;
            end
        end else begin
            done_reg <= 0;
        end
    end

endmodule
