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

**Dataset**: [PyraNet-Verilog](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog) — 692K Verilog samples filtered to compile-clean examples.

**Stack**: [Unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer`, QLoRA (rank-16, NF4 4-bit), Qwen3-4B-Thinking-2507.

```
Input:  functional description ("Implement a pipelined MAC unit...")
Output: correct Verilog module
```

Each training example is formatted as a chat turn with a fixed system prompt establishing the designer persona. Samples with compilation errors are filtered out — the model only imitates working code.

The SFT checkpoint is pushed to HuggingFace Hub as both a LoRA adapter
(`Supreeth/verirl-sft-qwen3-4b-thinking`) and a merged bf16 copy
(`Supreeth/verirl-sft-qwen3-4b-thinking-merged`). The merged copy is what vLLM
loads during the GRPO phase — vLLM cannot load PEFT adapter repos directly.

### Phase 2 — RLVR with GRPO

With a warm-started policy, we switch to reinforcement learning. The model now interacts with the VeriRL environment and receives reward from real EDA tools.

**Stack**: TRL `GRPOTrainer` + vLLM, QLoRA (rank-1), starting from the merged SFT checkpoint.

```
Prompt: task spec
Model:  write_file → run_compile → run_sim → submit
Reward: EDA tool score ∈ [0.01, 0.99]
```

The reward decomposes across four components — tool-use discipline, compile success, simulation pass rate, and the final weighted EDA score — giving the model a dense signal even for partially correct designs.

**Why rank-1 LoRA for RL?** Policy gradient updates carry roughly 1 bit of signal per episode. A rank-1 adapter has more than enough capacity to absorb this signal without the regularisation problems of higher-rank adapters in RL settings.

**Curriculum sampling** controls task difficulty across training:

| Difficulty | Tasks | Sample weight |
|-----------|-------|--------------|
| Easy | MAC, ReLU-Clip, Barrel Shifter | 40% |
| Medium | AXI-FIFO, Register File, Ring Buffer, Dot Product, FIR Filter | 40% |
| Hard | Systolic Array, FP16 Adder | 20% |

This prevents the model from getting stuck on tasks it cannot yet solve, while ensuring it sees hard tasks often enough to make progress.

**vLLM GPU strategy** (auto-detected at runtime):
- **2 GPUs**: vLLM server on GPU 1 with a dedicated 22 GB KV cache; training on GPU 0. Full context window, no OOM risk.
- **1 GPU**: vLLM colocate mode on a single card. Context capped at 8192 to share VRAM with the training process.

---

## Code Architecture

The training stack is split into a **backend-agnostic core** and thin **infrastructure adapters**:

```
training/
├── config.py        # SFTConfig + TrainConfig dataclasses (YAML-backed)
├── curriculum.py    # Task difficulty buckets + system prompt
├── dataset.py       # Curriculum dataset builder (easy → medium → hard)
├── environment.py   # VerirlToolEnv — TRL environment_factory adapter
├── reward.py        # Four reward functions (tool, compile, sim, final)
├── runtime.py       # Shared utilities: vLLM startup, env health check,
│                    # checkpoint resolution — used by both adapters
├── sft.py           # SFT training loop (Unsloth, backend-agnostic)
├── trainer.py       # GRPO training loop (TRL + vLLM, backend-agnostic)
└── wandb_task_logging.py  # Per-task reward buffering for W&B
```

Infrastructure adapters contain only backend-specific glue:

```
infra/
├── hf_jobs.py       # CLI: submit SFT/GRPO jobs to HuggingFace Jobs
├── modal_env.py     # Modal: deploy VeriRL env server (CPU, persistent)
├── modal_infra.py   # Modal: sft() + train() functions
└── modal_merge.py   # Modal: merge SFT LoRA adapter and push merged bf16 to Hub
training/
├── hf_train_sft.py  # HF Jobs entry point (bootstraps repo, calls sft.run_sft)
└── hf_train_grpo.py # HF Jobs entry point (bootstraps repo, calls trainer.run_training)
```

`training/runtime.py` eliminates the duplication that previously existed between
the Modal and HF Jobs adapters — both call the same `wait_for_env_server`,
`start_vllm_server`, `build_vllm_kwargs`, and `resolve_resume_checkpoint` functions.

---

## Infrastructure

### HuggingFace Jobs (primary)

```bash
export VERIRL_ENV_URL=https://<username>-verirl-env.hf.space

# Phase 1 — SFT warm-start (1×A10G, ~8h)
python infra/hf_jobs.py sft

# Phase 2 — RLVR GRPO (2×A10G or 1×H200)
python infra/hf_jobs.py train

# Monitor
python infra/hf_jobs.py ps
python infra/hf_jobs.py logs <job-id>
```

### Modal Labs (alternative)

```bash
modal run infra/modal_infra.py::sft    # Phase 1 — H100, ~8h
modal run infra/modal_infra.py::train  # Phase 2 — 2×L4, ~4h
modal deploy infra/modal_env.py        # Deploy env server (CPU, persistent)
```

Checkpoints are written locally during the job and pushed to HuggingFace Hub
after every save step, so preemption restarts automatically resume from the last
checkpoint via `VERIRL_RESUME_FROM_CHECKPOINT=latest`.

---

## Training Results

> *Training runs in progress on HuggingFace Jobs and Modal Labs.*
>
> *Results and W&B curves will be added here after the training runs complete.*

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

**Run SFT warm-start on HF Jobs:**
```bash
pip install -e ".[sft]"
huggingface-cli login
python infra/hf_jobs.py sft
```

**Run RLVR training on HF Jobs (from SFT checkpoint):**
```bash
export VERIRL_ENV_URL=https://<username>-verirl-env.hf.space
python infra/hf_jobs.py train
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

*Built with [OpenEnv](https://github.com/open-env/openenv-core) · SFT with [Unsloth](https://github.com/unslothai/unsloth) · RLVR with [HF TRL](https://github.com/huggingface/trl) GRPO + vLLM · Evaluated by [iverilog](https://steveicarus.github.io/iverilog/), [yosys](https://yosyshq.net/yosys/), [SymbiYosys](https://symbiyosys.readthedocs.io/) · Trained on [HuggingFace Jobs](https://huggingface.co/docs/hub/jobs) and [Modal Labs](https://modal.com)*

*GitHub: [SupreethRao99/veriRL](https://github.com/SupreethRao99/veriRL)*
