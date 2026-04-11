`timescale 1ns/1ps
// Reference implementation: parameterized ring buffer (circular FIFO)
module ring_buffer #(
    parameter DEPTH  = 8,
    parameter DATA_W = 8
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              push,
    input  wire [DATA_W-1:0] push_data,
    input  wire              pop,
    output wire [DATA_W-1:0] pop_data,
    output wire              full,
    output wire              empty,
    output wire [$clog2(DEPTH):0] count
);
    localparam PTR_W = $clog2(DEPTH);

    reg [DATA_W-1:0] mem  [0:DEPTH-1];
    reg [PTR_W-1:0]  head, tail;
    reg [PTR_W:0]    cnt;

    assign full     = (cnt == DEPTH);
    assign empty    = (cnt == 0);
    assign count    = cnt;
    assign pop_data = mem[head];

    wire do_push = push & ~full;
    wire do_pop  = pop  & ~empty;

    always @(posedge clk) begin
        if (rst) begin
            head <= {PTR_W{1'b0}};
            tail <= {PTR_W{1'b0}};
            cnt  <= {(PTR_W+1){1'b0}};
        end else begin
            if (do_push) begin
                mem[tail] <= push_data;
                tail      <= (tail == DEPTH-1) ? {PTR_W{1'b0}} : tail + 1'b1;
            end
            if (do_pop)
                head <= (head == DEPTH-1) ? {PTR_W{1'b0}} : head + 1'b1;
            // Explicit case split avoids synthesis warning on subtraction
            if (do_push & ~do_pop)
                cnt <= cnt + 1'b1;
            else if (do_pop & ~do_push)
                cnt <= cnt - 1'b1;
        end
    end

endmodule
