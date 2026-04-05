module systolic_array (
    input  wire         clk,
    input  wire         rst,
    input  wire         load_weights,
    input  wire [63:0]  weights_flat,
    input  wire         start,
    input  wire [127:0] activations_flat,
    output wire [255:0] outputs_flat,
    output wire         done
);

    reg [3:0]  weights [0:3][0:3];
    reg [15:0] acc     [0:3][0:3];
    reg [2:0]  cyc;
    reg        running;
    reg        done_reg;

    assign done = done_reg;

    // Pack accumulator output back to flat 1D wire array
    genvar gi, gj;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : row_out
            for (gj = 0; gj < 4; gj = gj + 1) begin : col_out
                assign outputs_flat[(gi*4+gj)*16 +: 16] = acc[gi][gj];
            end
        end
    endgenerate

    // Area Opt: Pre-calculate 4-bit * 8-bit products combinationally to strictly force 
    // lightweight 4x8 Multipliers across the synthesizer rather than bulky 16x8 or 16x16.
    wire [11:0] prod [0:3][0:3];
    genvar gr, gc;
    generate
        for (gr = 0; gr < 4; gr = gr + 1) begin : row_prod
            for (gc = 0; gc < 4; gc = gc + 1) begin : col_prod
                assign prod[gr][gc] = weights[gr][gc] * activations_flat[gr*8 +: 8];
            end
        end
    endgenerate

    // Load Weights synchronously
    integer li, lj;
    always @(posedge clk) begin
        if (load_weights) begin
            for (li = 0; li < 4; li = li + 1) begin
                for (lj = 0; lj < 4; lj = lj + 1) begin
                    weights[li][lj] <= weights_flat[(li*4+lj)*4 +: 4];
                end
            end
        end
    end

    // Use a unified internal cycle to allow doing "Cycle 0" logic during `start` high pulse
    wire [2:0] next_cyc = start ? 3'd0 : cyc;
    
    integer ci, cj;
    always @(posedge clk) begin
        if (rst) begin
            running  <= 0;
            done_reg <= 0;
            cyc      <= 0;
            for (ci = 0; ci < 4; ci = ci + 1) begin
                for (cj = 0; cj < 4; cj = cj + 1) begin
                    acc[ci][cj] <= 0;
                end
            end
            
        end else if (start || running) begin
            // Evaluate current row bounds (starts on `start` = 0)
            for (ci = 0; ci < 4; ci = ci + 1) begin
                if (next_cyc >= ci && next_cyc < ci + 4) begin
                    for (cj = 0; cj < 4; cj = cj + 1) begin
                        // If `start` is high, replace previous acc with 0 during the accumulation
                        acc[ci][cj] <= (start ? 16'd0 : acc[ci][cj]) + prod[ci][cj];
                    end
                end else if (start) begin
                    for (cj = 0; cj < 4; cj = cj + 1) begin
                        acc[ci][cj] <= 0;
                    end
                end
            end
            
            if (start) begin
                running  <= 1;
                done_reg <= 0;
                cyc      <= 1;
            end else begin
                cyc <= cyc + 1;
                // Assert completion synchronously hitting the 7th cycle accurately
                if (cyc == 3'd6) begin
                    done_reg <= 1;
                    running  <= 0;
                end
            end
            
        end else begin
            done_reg <= 0;
        end
    end

endmodule