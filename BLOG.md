# VeriRL: Training AI to Design AI Hardware with SFT + RLVR

*A two-phase training pipeline where LLMs learn to write synthesizable Verilog — first by imitation, then by trial-and-error with real EDA tools as the judge.*

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

Each episode returns a dense reward and a final weighted score across compile, simulation, timing, area, and formal dimensions. Ground truth is always the EDA tool — not an LLM judge.

---

## The Training Pipeline: SFT → RLVR

Training happens in two sequential phases, each building on the last.

### Phase 1 — SFT Warm-Start

Before the model sees any RL signal, we fine-tune it on real Verilog code using supervised learning. This gives the policy a strong prior: it learns what valid, synthesizable Verilog looks like before being asked to reason about correctness under feedback.

**Dataset**: [PyraNet-Verilog](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog) — 692K Verilog samples filtered to ~50K compile-clean examples.

**Stack**: [Unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer`, QLoRA (rank-16, NF4 4-bit), Qwen3-4B-Thinking-2507.

```
Input:  functional description ("Implement a pipelined MAC unit...")
Output: correct Verilog module
```

Each training example is formatted as a chat turn with a fixed system prompt establishing the designer persona. Samples with compilation errors are filtered out — the model only imitates working code.

The SFT checkpoint is pushed to HuggingFace Hub (`Supreeth/verirl-sft-qwen3-4b-thinking`) and serves as the starting point for Phase 2.

### Phase 2 — RLVR with GRPO

With a warm-started policy, we switch to reinforcement learning. The model now interacts with the VeriRL environment and receives reward from real EDA tools.

**Stack**: TRL `GRPOTrainer` + vLLM, QLoRA (rank-1, NF4), starting from the SFT checkpoint.

```
Prompt: task spec
Model:  write_file → run_compile → run_sim → submit
Reward: EDA tool score ∈ [0.01, 0.99]
```

The reward decomposes across dimensions — compile success, simulation pass rate, synthesis area, timing, and formal verification — giving the model a dense signal even for partially correct designs.

**Why rank-1 LoRA for RL?** Policy gradient updates carry roughly 1 bit of signal per episode. A rank-1 adapter has more than enough capacity to absorb this signal without the regularisation problems of higher-rank adapters in RL settings.

**Curriculum sampling** controls task difficulty across training:

| Difficulty | Tasks | Sample weight |
|-----------|-------|--------------|
| Easy | MAC, ReLU-Clip, Barrel Shifter | 40% |
| Medium | AXI-FIFO, Register File, Ring Buffer, Dot Product, FIR Filter | 40% |
| Hard | Systolic Array, FP16 Adder | 20% |

This prevents the model from getting stuck on tasks it cannot yet solve, while ensuring it sees hard tasks often enough to make progress.

**vLLM in server mode** (2-GPU setup): vLLM runs on GPU 1 with a dedicated 22 GB KV cache, while training runs on GPU 0. This lets us use a full 32K context window without the OOM risk of colocating both on one GPU.

---

## Infrastructure

Training runs on [Modal Labs](https://modal.com) with two separate functions:

```bash
modal run modal_infra.py::sft    # Phase 1 — H100, ~8h
modal run modal_infra.py::train  # Phase 2 — 2×L4, ~4h
```

Checkpoints are written to a persistent Modal volume and pushed to HuggingFace Hub after every save, so preemption restarts automatically resume from the last checkpoint.

---

## Training Results

> *Training runs in progress on Modal Labs using Unsloth SFT (H100) + TRL GRPO + vLLM (2×L4).*
>
> *Results and W&B curves will be added here before the hackathon demo.*

Expected trajectory:
- SFT baseline: mean score ~0.40–0.50 across tasks (strong prior from PyraNet-Verilog)
- RLVR fine-tuning: additional +0.10–0.20 mean score, with the largest gains on medium/hard tasks where simulation feedback is most informative

---

## Try It Yourself

**Run the environment:**
```bash
pip install openenv-verirl_env
# or via Docker:
docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:latest
```

**Run SFT warm-start on Modal:**
```bash
pip install -e ".[training]"
modal secret create verirl-training HF_TOKEN=hf_xxx WANDB_API_KEY=xxx VERIRL_ENV_URL=https://your-env.modal.run
modal run modal_infra.py::sft
```

**Run RLVR training on Modal (from SFT checkpoint):**
```bash
modal run modal_infra.py::train
```

**Run locally (smoke test):**
```bash
export VERIRL_ENV_URL=http://localhost:8000
python training/train.py --smoke
```

---

## Why This Matters

We are training AI systems to design the hardware that runs AI systems. The reward signal — real EDA tools evaluating real synthesizable Verilog — is one of the most rigorous ground truths available in any RL environment. No LLM judge. No proxy metric.

The two-phase approach matters because RL alone on a randomly-initialised policy is brittle: the model rarely produces valid Verilog by chance, so reward is sparse and training stalls. SFT pre-training solves the cold-start problem — the policy arrives at RL already capable of generating plausible designs, and GRPO refines it toward designs that are actually correct and efficient.

The result is a model that doesn't just write Verilog. It learns to **reason about what makes hardware correct, efficient, and formally verifiable** — using the same tools a human RTL engineer would reach for.

---

*Built with [OpenEnv](https://github.com/open-env/openenv-core) · SFT with [Unsloth](https://github.com/unslothai/unsloth) · RLVR with [HF TRL](https://github.com/huggingface/trl) GRPO + vLLM · Evaluated by [iverilog](https://steveicarus.github.io/iverilog/), [yosys](https://yosyshq.net/yosys/), [SymbiYosys](https://symbiyosys.readthedocs.io/) · Trained on [Modal Labs](https://modal.com)*

*GitHub: [SupreethRao99/veriRL](https://github.com/SupreethRao99/veriRL)*
