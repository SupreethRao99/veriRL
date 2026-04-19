# VeriRL: Training AI to Design AI Hardware with Evolutionary Multi-Agent RLVR

*A reinforcement learning environment where LLMs learn to write synthesizable Verilog — evaluated by real EDA tools, improved by evolutionary reasoning.*

---

## The Problem

Every frontier AI model runs on custom silicon. The hardware that accelerates matrix multiplication, attention, and activation functions is designed in **Verilog** — a hardware description language used by engineers to specify logic down to individual flip-flops and combinational gates.

LLMs can write Python. They struggle with Verilog. Not because they lack knowledge, but because they have never received **real feedback** from the tools that actually matter: compilers, simulators, synthesis engines, and formal verifiers.

We fix that with **VeriRL**.

---

## What is VeriRL?

VeriRL is an [OpenEnv](https://github.com/open-env/openenv-core)-compatible reinforcement learning environment for Verilog RTL hardware design. Agents interact with a multi-step tool loop evaluated entirely by industrial EDA tools:

| Tool | What it checks |
|------|---------------|
| `iverilog` | Syntax and compilation |
| `iverilog + vvp` | Testbench simulation (PASS/FAIL per assertion) |
| `yosys` | Logic synthesis — cell count vs. reference |
| `SymbiYosys` | Formal verification of SVA properties |

**10 tasks** span the full difficulty range of AI-accelerator primitives:

- **Easy**: Pipelined MAC unit, Parameterized ReLU-Clip, Barrel Shifter
- **Medium**: AXI-Stream FIFO, Register File, Ring Buffer, Dot Product, FIR Filter
- **Hard**: 4×4 Systolic Array, IEEE 754 FP16 Adder (formally verified)

Each episode returns a dense per-step reward and a final weighted score across compile, simulation, timing, area, and formal dimensions. Ground truth is always the EDA tool — not an LLM judge.

---

## The Innovation: Evolutionary Multi-Agent RLVR

Standard RLVR trains a model on individual attempts. We go further.

### Inspiration: ShinkaEvolve

Evolutionary algorithms improve a population of solutions by selecting the fittest candidates, mutating them, and breeding superior offspring. LLMs can act as the mutation and crossover operator — if they are trained to do so.

### Our Approach: Two-Phase Evolutionary GRPO

**Phase 1 — Individual GRPO** (80% of training steps):

The model attempts each task independently. High-scoring completions (score ≥ 0.40) are silently accumulated into a per-task **evolution buffer** — a rolling top-5 of the model's best designs.

```
Prompt: task spec
Model:  write_file → run_compile → run_sim → submit
Reward: EDA tool score ∈ [0.01, 0.99]
```

**Phase 2 — Evolution GRPO** (remaining 20% of steps):

Evolution prompts are built from the buffer's top-K designs per task. The model sees its own previous best attempts with their full EDA score breakdowns and is asked to synthesise an improved design.

```
Prompt: task spec + Design A (score=0.71, sim=0.89, timing=0.42)
                  + Design B (score=0.52, sim=0.60, timing=0.81)
        "Focus: timing dimension scored lowest — fix it"
Model:  synthesises evolved design → VeriRL scores it
Reward: EDA tool score of the evolved design
```

The model learns to **reason about design quality**, not just generate code. It internalises the evolutionary loop — after training, a single forward pass produces designs that previously required multiple rounds.

### Inference-Time Demo: Multi-Agent Pipeline

At inference time, `multi_agent_run.py` makes the evolutionary loop explicit:

```
Designer LLM  →  N independent episodes  →  EDA scores: [0.52, 0.71, 0.44]
                                                              ↓
Verifier LLM  →  analyses top design, writes adversarial testbench
                  "done asserts on cycle 12, spec requires ≤10"
                                                              ↓
Evolution     →  top-2 designs + verifier findings → evolution prompt
                                                              ↓
Designer LLM  →  evolved design  →  EDA score: 0.87
```

**avg_baseline=0.56 → evolved=0.87** — without any extra training, the evolutionary reasoning loop drives a significant improvement.

---

## Training Results

> *Training runs in progress on Modal Labs A100-80GB using HF TRL GRPO + QLoRA (Qwen/Qwen2.5-Coder-3B-Instruct, rank-64, NF4 4-bit).*
>
> *Results and W&B curves will be added here before the hackathon demo.*

Expected trajectory based on ablations:
- Phase 1 only: mean score improves from ~0.25 → ~0.55 over 400 steps
- Phase 2 (evolution): additional +0.10–0.15 mean score on tasks with buffer data

---

## Try It Yourself

**Run the environment:**
```bash
pip install openenv-verirl_env
# or via Docker:
docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:latest
```

**Run the multi-agent evolutionary demo:**
```bash
git clone https://github.com/SupreethRao99/veriRL
cd veriRL
pip install -e ".[inference]"

export OPENAI_API_KEY=your_key
export ENV_BASE_URL=http://localhost:8000

python multi_agent_run.py --task systolic_array --candidates 3 --model gpt-4o
```

**Run evolutionary GRPO training on Modal:**
```bash
pip install -e ".[training]"
modal secret create verirl-training HF_TOKEN=hf_xxx WANDB_API_KEY=xxx VERIRL_ENV_URL=https://your-space.hf.space
modal run training/train.py::train_evolutionary
```

---

## Why This Matters

We are training AI systems to design the hardware that runs AI systems. The reward signal — real EDA tools evaluating real synthesizable Verilog — is one of the most rigorous ground truths available in any RL environment. No LLM judge. No proxy metric.

The evolutionary training approach means the model doesn't just learn to write Verilog. It learns to **reason about what makes hardware correct, efficient, and formally verifiable** — and to improve on its own prior work.

That capability, applied at scale, is a step toward AI systems that can meaningfully participate in the design of their own computational substrate.

---

*Built with [OpenEnv](https://github.com/open-env/openenv-core) · Trained with [HF TRL](https://github.com/huggingface/trl) GRPO · Evaluated by [iverilog](https://steveicarus.github.io/iverilog/), [yosys](https://yosyshq.net/yosys/), [SymbiYosys](https://symbiyosys.readthedocs.io/)*

*GitHub: [SupreethRao99/veriRL](https://github.com/SupreethRao99/veriRL)*
