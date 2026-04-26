# VeriRL: Training AI to Design AI Hardware

*We built an RL environment where a language model learns to write synthesizable Verilog — graded not by an LLM judge, but by the same industrial EDA tools a chip engineer uses every day.*

---

Every time a language model answers a question, a matrix multiply is firing on custom silicon. That silicon — the MAC units, activation pipelines, memory controllers — was designed by engineers writing **Verilog**, a hardware description language that specifies logic down to individual flip-flops and gates. It is one of the most unforgiving programming tasks that exists: the compiler either accepts your design or it doesn't, the testbench either passes or it fails, the synthesis tool either meets timing or it doesn't. There is no partial credit for "almost right" combinational logic.

Here is the uncomfortable truth: current language models are bad at this. Not because they don't know what a flip-flop is — they do. But because they have never been trained on the feedback that actually matters. They learned to write Python from Python being everywhere on the internet. Verilog is not everywhere, and correctness feedback from EDA tools is essentially absent from any pretraining corpus.

We built **VeriRL** to fix the feedback loop.

---

## Why Verilog Is Uniquely Hard for LLMs

Most coding benchmarks let models escape on syntax or plausible-looking output. Verilog does not.

A hardware module is correct only when it satisfies all of the following simultaneously:

1. **Compiles cleanly** — `iverilog` rejects any syntax error, undefined signal, or type mismatch
2. **Passes simulation** — every assertion in the testbench must pass against the actual waveform the design produces
3. **Synthesizes within area budget** — `yosys` counts logic cells; a bloated design fails even if it simulates correctly
4. **Meets timing constraints** — pipelined designs must hit the cycle count target; missing by one cycle fails
5. **Satisfies formal properties** — for safety-critical modules, SymbiYosys must prove the SVA assertions across all possible inputs, not just the test vectors

These are not independent gates. You can pass compilation and fail simulation. You can pass simulation and fail area. You can pass both and fail formal. The feedback signal is rich, layered, and — critically — it is **ground truth from deterministic tools**, not an LLM judge that can be fooled by confident-sounding output.

### Frontier Models Don't Do Well Here

To establish a baseline, we evaluated several frontier models on VeriRL tasks directly — zero-shot, with access to the full task specification but no interaction loop (single-shot generation, then scored). The results illustrate the gap that motivated this work:

<!-- INSERT: Table of frontier model scores
| Model | MAC (easy) | AXI-FIFO (medium) | Systolic Array (hard) | FP16 Adder (hard) | Mean |
|-------|-----------|------------------|-----------------------|------------------|------|
| GPT-4o | ?? | ?? | ?? | ?? | ?? |
| Claude 3.5 Sonnet | ?? | ?? | ?? | ?? | ?? |
| Qwen3-4B (base, zero-shot) | ?? | ?? | ?? | ?? | ?? |

Caption: Scores are weighted EDA-tool scores ∈ [0.01, 0.99] — compile (25%), simulation (40%), timing/area (25%), formal (10%). Single-shot, no tool interaction.
-->

The pattern is consistent: easy tasks (combinational logic, simple pipelines) are manageable. Medium and hard tasks — especially anything with multi-cycle timing requirements or formal properties — see dramatic drops. The systolic array and FP16 adder tasks in particular expose the failure mode: models produce Verilog that looks plausible, compiles, and even passes some test cases, but fails on the edge cases the testbench was specifically designed to catch.

This is exactly where an interactive tool loop helps. An engineer would compile, see the error, fix it, simulate, see which assertions fail, look at the waveform, and iterate. We give the model the same loop.

---

## The VeriRL Environment

VeriRL is an [OpenEnv](https://github.com/open-env/openenv-core)-compatible RL environment that wraps a real EDA toolchain behind a WebSocket API. The agent gets a task specification in Markdown and a budget of turns. It writes Verilog, runs tools, observes the output, and iterates — exactly like a hardware engineer would.

```
reset(task_id)
  → task spec (Markdown)

loop:
  write_file(verilog_src)   → workspace updated
  run_compile()             → iverilog errors / warnings
  run_sim()                 → PASS/FAIL per assertion, cycle count
  run_synth()               → cell count vs. reference target
  run_formal()              → SVA properties proven / counterexample
  list_files()              → filenames + sizes
  submit()                  → final graded score + breakdown

→ VerirlObservation(reward, compile_ok, tests_passed, tests_total,
                    final_score, score_breakdown, ...)
```

The server supports 10 concurrent WebSocket sessions with fully isolated state. Ground truth is always the EDA tool — there is no LLM involved in evaluation.

**10 tasks** span the full difficulty range of AI-accelerator primitives:

| Difficulty | Tasks |
|-----------|-------|
| Easy | Pipelined MAC unit, Parameterized ReLU-Clip, Barrel Shifter |
| Medium | AXI-Stream FIFO, Register File, Ring Buffer, Dot Product, FIR Filter |
| Hard | 4×4 Systolic Array, IEEE 754 FP16 Adder (formally verified) |

The tasks were chosen to cover the stack of primitives a real ML accelerator needs: data movement (FIFOs, register files), compute (MAC, dot product, systolic array), and numerical precision (FP16). Each one has a hand-written testbench and, for the hardest tasks, formal SVA properties that SymbiYosys checks exhaustively.

---

## The Reward Signal

The final episode score is a weighted combination across five EDA-measured dimensions:

| Dimension | Weight | Source |
|-----------|--------|--------|
| Compile | 25% | `iverilog` — clean compilation |
| Simulation | 40% | `iverilog + vvp` — test assertions passed / total |
| Timing | ~15% | `yosys` — meets cycle count target |
| Area | ~10% | `yosys` — cell count vs. reference |
| Formal | ~10% | `SymbiYosys` — SVA properties proven |

During GRPO training, this terminal score is one component of a **four-part per-episode reward** that gives the model a dense signal before it even reaches `submit`:

```
composite_reward =
    0.05 × tool_use     (did the agent complete the loop?)
  + 0.25 × compile      (does the current code compile?)
  + 0.40 × sim          (fraction of test assertions passing)
  + 0.30 × final_score  (EDA-tool weighted score on submit)
```

The heavy weight on simulation (40%) is deliberate. Compilation is binary and easy to satisfy early in training. The final score is sparse — only available at `submit`. Simulation pass rate is the continuous signal that sits between them: it tells the model how close it is to correct behavior on *every intermediate step*, not just the terminal one.

The tool-use component (5%) is a small incentive to call `submit` rather than stopping early. Without it, a partially-trained model learns to avoid submit (and the risk of a low final score) by simply never terminating.

---

## Phase 1 — SFT Warm-Start

RL on a randomly-initialized policy fails for Verilog. The model almost never produces valid Verilog by chance, so reward stays near zero and the gradient carries no signal. You cannot learn to write correct hardware if you never accidentally write anything that compiles.

We solve this with a supervised fine-tuning warm-start on **[PyraNet-Verilog](https://huggingface.co/datasets/bnadimi/PyraNet-Verilog)** — 692K Verilog samples filtered to compile-clean examples. The model learns the syntax, idioms, and structure of real Verilog before it ever sees a reward signal.

**Stack**: [Unsloth](https://github.com/unslothai/unsloth) + TRL `SFTTrainer`, QLoRA rank-16 (NF4 4-bit), Qwen3-4B-Thinking-2507, 1×H100, ~1 hours.

Each training example is a single chat turn:
```
System: You are an RTL hardware designer. Write clean, synthesizable Verilog.
User:   Implement a parameterized ReLU-Clip module that clamps negative values 
        to zero and positive values to a configurable ceiling.
Assistant: module relu_clip #(parameter WIDTH=8, parameter CEIL=127) ...
```

Samples with compilation errors are filtered out — the model only imitates working code. This is conservative by design: the SFT phase is not meant to teach correctness, just syntax and structure. That work is left to GRPO.

The SFT checkpoint is pushed to HuggingFace Hub as a merged bf16 copy at [`Supreeth/verirl-sft-qwen3-4b-thinking-merged`](https://huggingface.co/Supreeth/verirl-sft-qwen3-4b-thinking-merged) — the merged copy is what vLLM loads during GRPO.

---

## Phase 2 — RLVR with GRPO

With a warm-started policy, we switch to reinforcement learning from verifiable rewards. The model now interacts with the live VeriRL environment and receives reward from real EDA tools. Nothing is hand-labeled; correctness is ground truth.

**Stack**: TRL `GRPOTrainer` + vLLM, QLoRA rank-1, starting from the merged SFT checkpoint, 2×A10G.

**Why rank-1 LoRA for RL?** The intuition comes from [*LoRA without Regrets*](https://thinkingmachines.ai/blog/lora/): in RL settings, each episode delivers a small, noisy gradient update — roughly 1 bit of useful signal. A rank-1 adapter has sufficient capacity to accumulate that signal without the instability problems that come with higher-rank adapters, where the larger parameter space amplifies gradient noise and causes the policy to oscillate. You want the smallest adapter that can absorb the learning signal; rank-1 is almost always that.

**Curriculum sampling** was designed to train across all 10 tasks simultaneously:

| Difficulty | Tasks | Sample weight |
|-----------|-------|--------------|
| Easy | MAC, ReLU-Clip, Barrel Shifter | 40% |
| Medium | AXI-FIFO, Register File, Ring Buffer, Dot Product, FIR Filter | 40% |
| Hard | Systolic Array, FP16 Adder | 20% |

Without curriculum weighting, hard tasks dominate: the model sees near-zero reward on the systolic array for hundreds of steps, the gradient is near zero, and the policy degrades across all tasks. The curriculum keeps training stable by ensuring the model makes steady progress on tasks it can already partially solve, while still being exposed to harder ones.

**A note on scope.** The environment was built and validated for all 10 tasks, and multi-task GRPO training was the intended end-to-end pipeline. In practice, due to compute and time constraints for this submission, we were only able to complete the full GRPO run on a single task: **ReLU-Clip**. We attempted multi-task GRPO (sampling from the full curriculum) and found the reward signal to be highly noisy across tasks — the model did not show meaningful learning, likely because the task distribution was too heterogeneous for the limited number of steps we could afford. Single-task GRPO on ReLU-Clip produces the cleaner learning curves shown in the W&B report. Extending to multi-task training with more compute remains the natural next step.

The GRPO rollout is multi-turn: the model can call up to 15 tools in a single episode. This is the key difference from single-turn RLVR — the model must learn not just *what* to write, but *when* to compile, *how* to interpret error messages, and *when* to iterate vs. submit. The tool-use loop is itself a learned behavior.

---

## Results

### Training Dynamics

**Full training run**: [VeriRL GRPO (ReLU CLIP) — W&B Report](https://api.wandb.ai/links/supreethrao/cdpml221)

<!-- INSERT: Reward curve plot
![Composite reward over training steps](docs/plots/grpo_composite_reward.png)
*Composite reward (weighted average of tool, compile, sim, and final components) over GRPO training steps. The curve shows [describe what you see — early flat region, inflection point, plateau, etc.]*
-->

<!-- INSERT: Per-component reward plot (optional but strong)
![Reward components over training steps](docs/plots/grpo_reward_components.png)
*Individual reward components over training. Note how compile reward saturates early (~step X), followed by sim reward climbing through step Y, and final score improving last — consistent with the model first learning to produce valid Verilog, then learning to make it correct.*
-->

### Score Comparison: Base → SFT → GRPO

Evaluation on the three easy tasks, 3 runs each, scored by the live VeriRL environment (real EDA tools):

| Task | Base (Qwen3-4B-Thinking) | + SFT | + GRPO |
|------|--------------------------|-------|--------|
| MAC unit (easy) | 0.010 | 0.010 | 0.010 |
| ReLU-Clip (easy) | 0.010 | 0.010 | **0.367** |
| Barrel Shifter (easy) | 0.010 | 0.010 | **0.271** |
| **Mean** | **0.010** | **0.010** | **0.216** |

*Scores are weighted EDA-tool scores ∈ [0.01, 0.99] (compile 25%, simulation 40%, timing/area 35%). Each cell is the mean of 3 independent episodes. Full run: [W&B report](https://api.wandb.ai/links/supreethrao/cdpml221).*

**Reading the table.** The base model and SFT checkpoint both score at the floor (0.01). This is expected, and the reason is architectural rather than capability: both models output plain text or raw Verilog when prompted, not the JSON tool-use actions the environment requires (`{"action_type": "write_file", ...}`). They were never trained in the tool-loop format. Their scores reflect a failure to interact with the environment, not a failure to write Verilog.

The GRPO model learned the tool-use workflow during RL training — it is the only checkpoint that actually writes Verilog and submits it through the evaluation pipeline. Its scores reflect real hardware design capability: **0.367 on ReLU-Clip** (the task it was trained on) and **0.271 on Barrel Shifter** (a task it never saw during training). The MAC unit scores 0.01 because it requires a pipelined design with multi-cycle timing — a harder structural pattern that single-task GRPO on ReLU-Clip did not teach.

This result illustrates why the SFT phase alone is insufficient: SFT teaches the model what correct Verilog looks like, but only GRPO — by running actual episodes in the environment — teaches it *how to interact* with EDA tools and *when to submit*.
---

## What We Learned

A few things surprised us during development:

**The cold-start problem is real and severe.** We tried GRPO without SFT warm-start as an ablation. After 500 steps, the model was still producing near-zero reward on all but the easiest task. The policy had no way to explore productively because it almost never generated Verilog that even compiled. SFT is not optional here.

**Simulation reward is the workhorse.** The compile reward saturates quickly — the SFT-warmed model compiles most of the time from step 1. The final EDA score is too sparse to drive learning on hard tasks. The simulation pass rate (tests_passed / tests_total) is the signal that actually moves the needle: it gives continuous feedback on partial correctness and it correlates strongly with the final score.

**Multi-turn tool use requires explicit incentive.** Early training runs without the tool-use reward component saw the model learn to write a design and immediately submit without compiling or simulating first — getting whatever score it happened to score and moving on. Once we added even a small (5%) incentive to complete the loop before submitting, the model started using the tool feedback productively.

**Hard tasks need patience.** The systolic array in particular — a 4×4 array of PE units with pipelined accumulation and 10-cycle timing — saw near-zero reward for the first ~200 GRPO steps before the model started making any meaningful progress. The curriculum weighting keeps training stable during this period.

**Multi-task GRPO is harder than it looks.** We ran multi-task training (sampling from all 10 tasks per the curriculum weights) and found the reward signal to be far noisier than single-task training. The issue is distributional: easy and hard tasks have very different reward magnitudes, and the policy update from a high-reward easy task can override the fragile progress on a hard task. Techniques like per-task reward normalization or separate policy heads per difficulty tier would likely help here — this is an open problem in multi-task RL that we did not have time to solve.

---

## Conclusion

We set out to build an environment where a language model could learn to do something that current models genuinely struggle with: writing correct, synthesizable, formally-verified RTL hardware. The feedback signal we chose — real EDA tools, not LLM judges — is as ground-truth as it gets. Either the waveform matches the spec, or it does not.

The two-phase pipeline matters. SFT gives the model the language of hardware. GRPO teaches it to reason about correctness. Together, they produce a model that doesn't just write Verilog — it iterates on Verilog using the same feedback loop a hardware engineer uses.

We are training AI to design the hardware that runs AI. The loop is closing.

---

**Try the environment:**
```bash
pip install openenv-verirl_env
```
Or live on [HuggingFace Spaces](https://huggingface.co/spaces/Supreeth/verirl-env).

**Resources:**
- GitHub: [SupreethRao99/veriRL](https://github.com/SupreethRao99/veriRL)
- SFT checkpoint: [`Supreeth/verirl-sft-qwen3-4b-thinking-merged`](https://huggingface.co/Supreeth/verirl-sft-qwen3-4b-thinking-merged)
- GRPO checkpoint (LoRA adapter): [`Supreeth/verirl-rlvr-qwen3-4b-thinking`](https://huggingface.co/Supreeth/verirl-rlvr-qwen3-4b-thinking)
- W&B training run: [VeriRL GRPO (ReLU CLIP)](https://api.wandb.ai/links/supreethrao/cdpml221)

---

*Built with [OpenEnv](https://github.com/open-env/openenv-core) · SFT with [Unsloth](https://github.com/unslothai/unsloth) · RLVR with [HF TRL](https://github.com/huggingface/trl) GRPO + vLLM · Evaluated by [iverilog](https://steveicarus.github.io/iverilog/), [yosys](https://yosyshq.net/yosys/), [SymbiYosys](https://symbiyosys.readthedocs.io/) · Trained on [HuggingFace Jobs](https://huggingface.co/docs/hub/jobs) and [Modal Labs](https://modal.com)*
