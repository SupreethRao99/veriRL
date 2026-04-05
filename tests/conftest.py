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
