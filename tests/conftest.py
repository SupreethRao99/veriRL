"""Shared pytest fixtures for VeriRL tests."""

import pytest
import shutil
from pathlib import Path

from verirl_env.server.verirl_env_environment import VerirlEnvironment
from verirl_env.server.evaluator import VerilogEvaluator
from verirl_env.models import VerirlAction

# Check if EDA tools are available
HAS_IVERILOG = shutil.which("iverilog") is not None
HAS_YOSYS = shutil.which("yosys") is not None
HAS_SBY = shutil.which("sby") is not None


@pytest.fixture
def evaluator():
    """Create a VerilogEvaluator instance."""
    return VerilogEvaluator()


@pytest.fixture
def environment():
    """Create a VerirlEnvironment instance."""
    return VerirlEnvironment()


@pytest.fixture
def mac_reference_verilog():
    """Reference implementation for mac_unit task."""
    return """
module mac_unit (
    input  wire        clk, rst, en, clear,
    input  wire signed [7:0]  a, b,
    output reg  signed [31:0] acc_out
);
    reg signed [15:0] product_s1;
    reg               en_s1, clear_s1;

    always @(posedge clk) begin
        if (rst) begin
            product_s1 <= 0;
            en_s1 <= 0;
            clear_s1 <= 0;
        end else begin
            product_s1 <= a * b;
            en_s1 <= en;
            clear_s1 <= clear;
        end
    end

    always @(posedge clk) begin
        if (rst) acc_out <= 0;
        else if (clear_s1) acc_out <= 0;
        else if (en_s1) acc_out <= acc_out + product_s1;
    end
endmodule
"""


@pytest.fixture
def axi_reference_verilog():
    """Reference implementation for axi_fifo task."""
    return """
module axi_fifo #(parameter DATA_W = 8) (
    input  wire              clk, rst,
    input  wire              s_valid,
    output wire              s_ready,
    input  wire [DATA_W-1:0] s_data,
    output wire              m_valid,
    input  wire              m_ready,
    output wire [DATA_W-1:0] m_data,
    output wire              full, empty
);
    localparam DEPTH = 4;
    reg [DATA_W-1:0] mem [0:DEPTH-1];
    reg [1:0] head, tail;
    reg [2:0] count;

    assign full = (count == DEPTH);
    assign empty = (count == 0);
    assign s_ready = !full;
    assign m_valid = !empty;
    assign m_data = mem[head];

    wire enq = s_valid && s_ready;
    wire deq = m_valid && m_ready;

    always @(posedge clk) begin
        if (rst) begin
            head <= 0;
            tail <= 0;
            count <= 0;
        end else begin
            if (enq) begin
                mem[tail] <= s_data;
                tail <= tail + 1;
            end
            if (deq) head <= head + 1;
            case ({enq, deq})
                2'b10: count <= count + 1;
                2'b01: count <= count - 1;
                default: count <= count;
            endcase
        end
    end
endmodule
"""


@pytest.fixture
def requires_iverilog(request):
    """Skip test if iverilog is not available."""
    if not HAS_IVERILOG:
        pytest.skip("iverilog not installed")


@pytest.fixture
def requires_yosys(request):
    """Skip test if yosys is not available."""
    if not HAS_YOSYS:
        pytest.skip("yosys not installed")


@pytest.fixture
def requires_eda_tools(request):
    """Skip test if EDA tools are not available."""
    if not HAS_IVERILOG or not HAS_YOSYS:
        pytest.skip("iverilog and/or yosys not installed")


@pytest.fixture
def requires_sby(request):
    """Skip test if SymbiYosys is not available."""
    if not HAS_SBY:
        pytest.skip("sby (SymbiYosys) not installed")


@pytest.fixture
def relu_clip_reference_verilog():
    """Reference implementation for relu_clip task."""
    return """
module relu_clip #(
    parameter IN_W  = 8,
    parameter OUT_W = 4
) (
    input  wire signed [IN_W-1:0]  in_val,
    output wire        [OUT_W-1:0] out_val,
    output wire                    saturated
);
    localparam integer MAX_OUT = (1 << OUT_W) - 1;
    wire neg    = in_val[IN_W-1];
    wire [IN_W-1:0] relu_out = neg ? {IN_W{1'b0}} : in_val;
    wire pos_clip = (relu_out > MAX_OUT[IN_W-1:0]);
    assign out_val   = pos_clip ? MAX_OUT[OUT_W-1:0] : relu_out[OUT_W-1:0];
    assign saturated = neg | pos_clip;
endmodule
"""


@pytest.fixture
def barrel_shifter_reference_verilog():
    """Reference implementation for barrel_shifter task."""
    return """
module barrel_shifter #(
    parameter WIDTH = 8
) (
    input  wire [WIDTH-1:0]          data_in,
    input  wire [$clog2(WIDTH)-1:0]  shift_amt,
    input  wire                      direction,
    input  wire                      arithmetic,
    output wire [WIDTH-1:0]          data_out
);
    wire [WIDTH-1:0] left_out  = data_in << shift_amt;
    wire [WIDTH-1:0] right_log = data_in >> shift_amt;
    wire [WIDTH-1:0] right_ari = $signed(data_in) >>> shift_amt;
    assign data_out = direction
        ? (arithmetic ? right_ari : right_log)
        : left_out;
endmodule
"""


@pytest.fixture
def register_file_reference_verilog():
    """Reference implementation for register_file task."""
    return """
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
        for (i = 0; i < DEPTH; i = i + 1) regs[i] = 0;
    end
    always @(posedge clk)
        if (we && (wr_addr != 0)) regs[wr_addr] <= wr_data;
    assign rd_data_a = (rd_addr_a == 0) ? {DATA_W{1'b0}} : regs[rd_addr_a];
    assign rd_data_b = (rd_addr_b == 0) ? {DATA_W{1'b0}} : regs[rd_addr_b];
endmodule
"""


@pytest.fixture
def ring_buffer_reference_verilog():
    """Reference implementation for ring_buffer task."""
    return """
`timescale 1ns/1ps
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
                tail <= (tail == DEPTH-1) ? {PTR_W{1'b0}} : tail + 1'b1;
            end
            if (do_pop)
                head <= (head == DEPTH-1) ? {PTR_W{1'b0}} : head + 1'b1;
            if (do_push & ~do_pop) cnt <= cnt + 1'b1;
            else if (do_pop & ~do_push) cnt <= cnt - 1'b1;
        end
    end
endmodule
"""


@pytest.fixture
def dot_product_reference_verilog():
    """Reference implementation for dot_product task."""
    return """
`timescale 1ns/1ps
module dot_product_4 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [7:0] a0, a1, a2, a3,
    input  wire signed [7:0] b0, b1, b2, b3,
    output reg         valid_out,
    output reg  signed [17:0] result
);
    reg signed [15:0] p0, p1, p2, p3;
    reg               s1_valid;
    always @(posedge clk) begin
        if (rst) begin
            p0 <= 0; p1 <= 0; p2 <= 0; p3 <= 0;
            s1_valid <= 1'b0; result <= 0; valid_out <= 1'b0;
        end else begin
            p0 <= a0 * b0; p1 <= a1 * b1;
            p2 <= a2 * b2; p3 <= a3 * b3;
            s1_valid  <= valid_in;
            result    <= p0 + p1 + p2 + p3;
            valid_out <= s1_valid;
        end
    end
endmodule
"""


@pytest.fixture
def fir_filter_reference_verilog():
    """Reference implementation for fir_filter task."""
    return """
`timescale 1ns/1ps
module fir3 (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [7:0]  x,
    input  wire signed [7:0]  h0, h1, h2,
    output reg  signed [17:0] y,
    output reg                valid_out
);
    reg signed [7:0] x_d1, x_d2;
    always @(posedge clk) begin
        if (rst) begin
            x_d1 <= 8'sb0; x_d2 <= 8'sb0;
            y <= 18'sb0; valid_out <= 1'b0;
        end else if (valid_in) begin
            y     <= h0*x + h1*x_d1 + h2*x_d2;
            x_d1  <= x; x_d2 <= x_d1;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end
endmodule
"""


@pytest.fixture
def fp16_adder_reference_verilog():
    """Reference implementation for fp16_adder task."""
    return """
`timescale 1ns/1ps
module fp16_adder (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] result
);
    wire sa = a[15], sb = b[15];
    wire [4:0] ea = a[14:10], eb = b[14:10];
    wire [9:0] ma = a[9:0],   mb = b[9:0];
    wire a_inf  = (ea == 5'h1F) & (ma == 10'h0);
    wire b_inf  = (eb == 5'h1F) & (mb == 10'h0);
    wire a_nan  = (ea == 5'h1F) & (ma != 10'h0);
    wire b_nan  = (eb == 5'h1F) & (mb != 10'h0);
    wire a_zero = (ea == 5'h00) & (ma == 10'h0);
    wire b_zero = (eb == 5'h00) & (mb == 10'h0);
    wire [10:0] fa = a_zero ? 11'h0 : {1'b1, ma};
    wire [10:0] fb = b_zero ? 11'h0 : {1'b1, mb};
    wire [15:0] mag_a = {1'b0, a[14:0]};
    wire [15:0] mag_b = {1'b0, b[14:0]};
    wire swap  = (mag_b > mag_a);
    wire        s_big  = swap ? sb      : sa;
    wire [4:0]  e_big  = swap ? eb      : ea;
    wire [10:0] f_big  = swap ? fb      : fa;
    wire        s_sml  = swap ? sa      : sb;
    wire [4:0]  e_sml  = swap ? ea      : eb;
    wire [10:0] f_sml  = swap ? fa      : fb;
    wire [4:0] diff = e_big - e_sml;
    wire [10:0] f_sml_sh = (diff >= 11) ? 11'h0 : (f_sml >> diff);
    wire same_sign = (s_big == s_sml);
    wire [11:0] sum_raw = same_sign
        ? ({1'b0, f_big} + {1'b0, f_sml_sh})
        : ({1'b0, f_big} - {1'b0, f_sml_sh});
    reg [3:0] lz;
    always @(*) begin
        casez (sum_raw)
            12'b1??????????? : lz = 4'd0;
            12'b01?????????? : lz = 4'd1;
            12'b001????????? : lz = 4'd2;
            12'b0001???????? : lz = 4'd3;
            12'b00001??????? : lz = 4'd4;
            12'b000001?????? : lz = 4'd5;
            12'b0000001????? : lz = 4'd6;
            12'b00000001???? : lz = 4'd7;
            12'b000000001??? : lz = 4'd8;
            12'b0000000001?? : lz = 4'd9;
            12'b00000000001? : lz = 4'd10;
            12'b000000000001 : lz = 4'd11;
            default          : lz = 4'd12;
        endcase
    end
    wire [3:0] norm_shift = (lz == 0) ? 4'd0 : (lz - 4'd1);
    wire [11:0] sum_norm  = (lz == 0) ? (sum_raw >> 1) : (sum_raw << norm_shift);
    wire [5:0] exp_adj = (sum_raw[11])
        ? ({1'b0, e_big} + 6'd1)
        : ({1'b0, e_big} - {2'b0, lz} + 6'd1);
    wire result_zero = (sum_raw == 12'h0);
    wire exp_overflow = (exp_adj >= 6'd31) & ~result_zero;
    wire [15:0] nan_out  = 16'h7E00;
    wire [15:0] inf_out  = {s_big, 5'h1F, 10'h0};
    wire [15:0] zero_out = 16'h0000;
    wire [15:0] norm_out = {s_big, exp_adj[4:0], sum_norm[9:0]};
    assign result =
        (a_nan | b_nan)              ? nan_out  :
        (a_inf & b_inf & (sa != sb)) ? nan_out  :
        (a_inf | b_inf)              ? inf_out  :
        (exp_overflow)               ? inf_out  :
        (result_zero)                ? zero_out :
        norm_out;
endmodule
"""


@pytest.fixture
def systolic_reference_verilog():
    """Reference implementation for systolic_array task.

    Architecture: row i accumulates for exactly 4 cycles starting at cycle i
    (diagonal skewing via row-gated enable, no shift registers).
    output[i][j] = 4 * activations[i] * weights[i][j]
    done fires at posedge 7 from start (done_cycle <= 7).
    """
    return """
module systolic_array (
    input  wire        clk, rst, load_weights, start,
    input  wire [63:0] weights_flat,
    input  wire [127:0] activations_flat,
    output wire [255:0] outputs_flat,
    output wire        done
);
    reg [3:0]  weights [0:3][0:3];
    reg [15:0] acc     [0:3][0:3];
    reg [2:0]  cyc;
    reg        running, done_reg;

    assign done = done_reg;

    genvar gi, gj;
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : row_out
            for (gj = 0; gj < 4; gj = gj + 1) begin : col_out
                assign outputs_flat[(gi*4+gj)*16 +: 16] = acc[gi][gj];
            end
        end
    endgenerate

    integer li, lj;
    always @(posedge clk) begin
        if (load_weights)
            for (li = 0; li < 4; li = li + 1)
                for (lj = 0; lj < 4; lj = lj + 1)
                    weights[li][lj] <= weights_flat[(li*4+lj)*4 +: 4];
    end

    integer ci, cj;
    always @(posedge clk) begin
        if (rst) begin
            running <= 0; done_reg <= 0; cyc <= 0;
            for (ci = 0; ci < 4; ci = ci + 1)
                for (cj = 0; cj < 4; cj = cj + 1)
                    acc[ci][cj] <= 0;
        end else if (start) begin
            running <= 1; done_reg <= 0; cyc <= 0;
            for (ci = 0; ci < 4; ci = ci + 1)
                for (cj = 0; cj < 4; cj = cj + 1)
                    acc[ci][cj] <= 0;
        end else if (running) begin
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
"""
