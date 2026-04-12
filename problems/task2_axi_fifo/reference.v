`timescale 1ns/1ps

module axi_fifo #(
    parameter DATA_W = 8
) (
    input  wire              clk,
    input  wire              rst,
    input  wire              s_valid,
    output wire              s_ready,
    input  wire [DATA_W-1:0] s_data,
    output wire              m_valid,
    input  wire              m_ready,
    output wire [DATA_W-1:0] m_data,
    output wire              full,
    output wire              empty
);
    localparam DEPTH = 4;

    reg [DATA_W-1:0] mem [0:DEPTH-1];
    reg [1:0] head, tail;
    reg [2:0] count;  // 0..4

    assign full    = (count == DEPTH);
    assign empty   = (count == 0);
    assign s_ready = !full;
    assign m_valid = !empty;
    assign m_data  = mem[head];

    wire enq = s_valid && s_ready;
    wire deq = m_valid && m_ready;

    always @(posedge clk) begin
        if (rst) begin
            head  <= 0;
            tail  <= 0;
            count <= 0;
        end else begin
            if (enq) begin
                mem[tail] <= s_data;
                tail      <= tail + 1;
            end
            if (deq) begin
                head  <= head + 1;
            end
            case ({enq, deq})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end

endmodule
