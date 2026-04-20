"""
VeriRL RLVR Training Script — Modal Labs
=========================================
Runs Group Relative Policy Optimisation (GRPO) to fine-tune a code LLM on
all 10 VeriRL tasks. The reward signal comes directly from the VeriRL
environment hosted on Hugging Face Spaces (or any OpenEnv-compatible server).

Usage
-----
1. Deploy the VeriRL server to HF Spaces (or run locally):
       docker run -p 8000:8000 ghcr.io/SupreethRao99/veriRL:latest

2. Set secrets in Modal dashboard:
       modal secret create verirl-training \\
           HF_TOKEN=hf_xxx \\
           WANDB_API_KEY=wandb_xxx \\
           VERIRL_ENV_URL=https://your-space.hf.space

3. Run training:
       modal run training/train.py

4. (Optional) Run a quick smoke-test without GPU:
       modal run training/train.py::smoke_test

Architecture
------------
- Base model: Qwen/Qwen2.5-Coder-3B-Instruct  (swap to 7B/14B for more capacity)
- Algorithm:  GRPO (TRL GRPOTrainer)
- Reward:     VeriRL environment final_score  ∈ (0, 1)
- Curriculum: tasks sampled by difficulty weight (easy→medium→hard)
- Logging:    Weights & Biases
- Checkpoints saved to HF Hub every N steps
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import textwrap
from dataclasses import dataclass, field
from typing import Optional

import modal

# ---------------------------------------------------------------------------
# Modal app and image
# ---------------------------------------------------------------------------

app = modal.App("verirl-rlvr")

# Build a GPU image with all training dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core training stack
        "torch==2.3.1",
        "transformers==4.44.0",
        "trl==0.9.6",
        "peft==0.12.0",
        "accelerate==0.33.0",
        "bitsandbytes==0.43.3",
        # VeriRL client
        "openenv-verirl_env",
        "openai>=1.40.0",
        # Logging + misc
        "wandb>=0.17.0",
        "huggingface_hub>=0.24.0",
        "datasets>=2.20.0",
        "numpy",
        "python-dotenv",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

# Secrets injected from Modal dashboard
verirl_secrets = modal.Secret.from_name("verirl-training")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    # Model
    base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    hf_output_repo: str = "SupreethRao99/verirl-rlvr-qwen2.5-coder-3b"

    # GRPO hyper-parameters
    num_train_epochs: int = 3
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_generations: int = 6          # G in GRPO (completions per prompt)
    max_prompt_length: int = 1024
    max_completion_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    kl_coeff: float = 0.05            # β — KL penalty coefficient

    # Curriculum
    # Tasks sampled with these weights (sum need not equal 1)
    task_difficulty_weights: dict = field(default_factory=lambda: {
        "easy":   0.40,   # mac_unit, relu_clip, barrel_shifter
        "medium": 0.40,   # axi_fifo, register_file, ring_buffer, dot_product, fir_filter
        "hard":   0.20,   # systolic_array, fp16_adder
    })

    # Training infra
    max_steps: int = 500
    save_steps: int = 100
    logging_steps: int = 10
    warmup_ratio: float = 0.05
    push_to_hub: bool = True
    use_4bit: bool = True             # QLoRA for memory efficiency

    # Environment
    env_url: str = "http://localhost:8000"  # overridden from env var
    env_timeout_per_task: int = 120         # seconds per episode
    max_parallel_envs: int = 4             # concurrent VeriRL sessions


# ---------------------------------------------------------------------------
# Task curriculum
# ---------------------------------------------------------------------------

TASKS_BY_DIFFICULTY = {
    "easy":   ["mac_unit", "relu_clip", "barrel_shifter"],
    "medium": ["axi_fifo", "register_file", "ring_buffer", "dot_product", "fir_filter"],
    "hard":   ["systolic_array", "fp16_adder"],
}

ALL_TASKS = [t for tasks in TASKS_BY_DIFFICULTY.values() for t in tasks]


def sample_task(config: TrainConfig, rng: random.Random = random) -> str:
    """Sample a task ID according to difficulty weights."""
    difficulties = list(config.task_difficulty_weights.keys())
    weights = [config.task_difficulty_weights[d] for d in difficulties]
    chosen_difficulty = rng.choices(difficulties, weights=weights, k=1)[0]
    return rng.choice(TASKS_BY_DIFFICULTY[chosen_difficulty])


# ---------------------------------------------------------------------------
# System prompt and action parsing (mirrors inference.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert RTL hardware designer. Implement the given Verilog specification.

    WORKFLOW:
      1. write_file   — write a complete, synthesizable Verilog module
      2. run_compile  — check for syntax errors; fix and repeat if needed
      3. run_sim      — run the testbench; fix FAIL lines with write_file
      4. submit       — when tests pass or turns are low

    Available actions — respond with exactly ONE JSON object, no markdown:
      {"action_type": "write_file", "filename": "design.v", "verilog_src": "<full module>"}
      {"action_type": "run_compile"}
      {"action_type": "run_sim"}
      {"action_type": "run_synth"}
      {"action_type": "run_formal"}
      {"action_type": "submit"}

    Rules:
    - No `initial` blocks in the design (testbench only)
    - Use always @(posedge clk) for sequential, assign / always @(*) for combinational
    - Multi-file: use separate write_file calls with different filenames
""").strip()


def parse_action_from_text(text: str) -> dict:
    """Extract the first JSON object from an LLM completion."""
    text = text.strip()
    for marker in ("```json", "```"):
        if marker in text:
            text = text.split(marker)[1].split("```")[0].strip()
            break
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "submit", "message": "parse_error"}


# ---------------------------------------------------------------------------
# Episode runner — runs ONE complete VeriRL episode, returns final_score
# ---------------------------------------------------------------------------


async def run_episode(
    task_id: str,
    model_client,           # openai.AsyncOpenAI
    model_name: str,
    env_url: str,
    timeout: int,
) -> tuple[float, list[dict]]:
    """
    Run a single episode.

    The VeriRL client is synchronous; we wrap each blocking call in
    asyncio.to_thread so the event loop stays responsive during batched scoring.

    Returns:
        (final_score, conversation_messages)
        final_score ∈ (0, 1)
    """
    import time
    from verirl_env import VerirlAction, verirl_env  # type: ignore

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.01
    start = time.time()

    env = verirl_env(base_url=env_url)
    try:
        result = await asyncio.to_thread(env.reset, task_id=task_id)
        obs = result.observation
        messages.append({"role": "user", "content": _format_obs(obs)})

        for _ in range(100):  # guard loop
            if time.time() - start > timeout:
                await asyncio.to_thread(env.step, VerirlAction(action_type="submit"))
                break

            response = await model_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=2048,
            )
            assistant_text = response.choices[0].message.content or ""
            action_dict = parse_action_from_text(assistant_text)

            valid_fields = {"action_type", "verilog_src", "filename", "message"}
            action = VerirlAction(**{k: v for k, v in action_dict.items() if k in valid_fields})

            result = await asyncio.to_thread(env.step, action)
            obs = result.observation

            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": _format_obs(obs)})

            if result.done:
                final_score = max(0.01, min(0.99, float(obs.final_score or 0.01)))
                break

    finally:
        await asyncio.to_thread(env.close)

    return final_score, messages


def _format_obs(obs) -> str:
    parts = []
    if obs.task_spec:
        parts.append(f"TASK:\n{obs.task_spec}")
    if obs.tool_stdout:
        parts.append(f"OUTPUT:\n{obs.tool_stdout}")
    if obs.tool_stderr:
        parts.append(f"ERRORS:\n{obs.tool_stderr}")
    if getattr(obs, "current_files", None):
        summary = ", ".join(f"{n}({len(s)}B)" for n, s in sorted(obs.current_files.items()))
        parts.append(f"Files: {summary}")
    parts.append(
        f"compile={'OK' if obs.compile_ok else 'FAIL'} "
        f"tests={obs.tests_passed}/{obs.tests_total} "
        f"turn={obs.turn_number}/{obs.turn_number + obs.turns_remaining}"
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# GRPO reward function — called by TRL with a batch of completions
# ---------------------------------------------------------------------------


def build_reward_fn(config: TrainConfig):
    """
    Returns a reward function compatible with TRL GRPOTrainer.

    TRL calls reward_fn(prompts, completions, **kwargs) and expects
    a list of floats with the same length as completions.

    Each completion is treated as a single write_file → compile → sim → submit
    episode so that the GRPO reward matches what the agent would receive at
    inference time.
    """

    def score_one(completion: str, task_id: str) -> float:
        """Score a single completion synchronously against the VeriRL env."""
        from verirl_env import VerirlAction, verirl_env  # type: ignore

        env = verirl_env(base_url=config.env_url)
        score = 0.01
        try:
            env.reset(task_id=task_id)

            # Parse the completion as a write_file action
            action_dict = parse_action_from_text(completion)
            valid_fields = {"action_type", "verilog_src", "filename", "message"}
            action = VerirlAction(**{k: v for k, v in action_dict.items() if k in valid_fields})

            # If the model wrote Verilog, run the standard eval pipeline
            if action.action_type == "write_file" and action.verilog_src:
                env.step(action)
                env.step(VerirlAction(action_type="run_compile"))
                env.step(VerirlAction(action_type="run_sim"))

            result = env.step(VerirlAction(action_type="submit"))
            obs = result.observation
            score = max(0.01, min(0.99, float(obs.final_score or 0.01)))
        except Exception:
            score = 0.01
        finally:
            env.close()
        return score

    def reward_fn(prompts, completions, task_ids=None, **kwargs) -> list[float]:
        if task_ids is None:
            task_ids = [random.choice(ALL_TASKS)] * len(completions)
        # Run each episode in a thread pool for parallelism without async overhead
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=config.max_parallel_envs) as pool:
            futures = [pool.submit(score_one, c, t)
                       for c, t in zip(completions, task_ids)]
            return [f.result() for f in futures]

    return reward_fn


# ---------------------------------------------------------------------------
# Dataset builder — generates (prompt, task_id) pairs for GRPO
# ---------------------------------------------------------------------------


def build_dataset(config: TrainConfig, n_samples: int = 1000):
    """
    Build a HuggingFace Dataset of Verilog design prompts.
    Each sample contains the task spec as the user prompt.
    """
    from datasets import Dataset  # type: ignore
    from verirl_env import verirl_env  # type: ignore

    records = []
    rng = random.Random(42)

    # Fetch task specs synchronously (one episode per task, no inference needed)
    specs: dict[str, str] = {}
    for task_id in ALL_TASKS:
        env = verirl_env(base_url=config.env_url)
        try:
            result = env.reset(task_id=task_id)
            specs[task_id] = result.observation.task_spec or ""
        except Exception:
            specs[task_id] = f"Implement the {task_id} Verilog module."
        finally:
            env.close()

    for _ in range(n_samples):
        task_id = sample_task(config, rng)
        spec = specs.get(task_id, "")
        records.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"TASK SPECIFICATION:\n{spec}"},
            ],
            "task_id": task_id,
        })

    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Main training function — runs on Modal A100
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    gpu=modal.gpu.A100(count=1, size="80GB"),
    timeout=4 * 3600,          # 4 hours max
    secrets=[verirl_secrets],
    memory=65536,               # 64 GB RAM
)
def train():
    """Fine-tune Qwen2.5-Coder-3B-Instruct on VeriRL using GRPO."""
    import torch
    import wandb
    from huggingface_hub import login
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    # ── Authentication ────────────────────────────────────────────────────
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)

    config = TrainConfig(
        env_url=os.environ.get("VERIRL_ENV_URL", "http://localhost:8000"),
    )

    print(f"[VeriRL RLVR] Base model:  {config.base_model}")
    print(f"[VeriRL RLVR] Env URL:     {config.env_url}")
    print(f"[VeriRL RLVR] Output repo: {config.hf_output_repo}")
    print(f"[VeriRL RLVR] Tasks:       {ALL_TASKS}")

    # ── Model + tokenizer ─────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if config.use_4bit else None

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA config ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[VeriRL RLVR] Building dataset ...")
    dataset = build_dataset(config, n_samples=2000)
    print(f"[VeriRL RLVR] Dataset size: {len(dataset)}")

    # ── Reward function ───────────────────────────────────────────────────
    reward_fn = build_reward_fn(config)

    # ── GRPO training config ──────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir="/tmp/verirl-grpo-checkpoints",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        beta=config.kl_coeff,           # KL-penalty coefficient in TRL ≥ 0.9
        max_steps=config.max_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hf_output_repo,
        hub_token=hf_token,
        report_to="wandb" if wandb_key else "none",
        run_name="verirl-grpo-qwen2.5-coder-3b",
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[reward_fn],
        peft_config=lora_config,
    )

    print("[VeriRL RLVR] Starting GRPO training ...")
    trainer.train()

    print("[VeriRL RLVR] Training complete. Saving final model ...")
    trainer.save_model("/tmp/verirl-grpo-final")
    if config.push_to_hub:
        trainer.push_to_hub()
        print(f"[VeriRL RLVR] Model pushed to {config.hf_output_repo}")

    return {"status": "done", "output_repo": config.hf_output_repo}


# ---------------------------------------------------------------------------
# Smoke test — runs without a GPU to verify imports and env connectivity
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    timeout=300,
    secrets=[verirl_secrets],
)
def smoke_test():
    """
    Lightweight sanity check: verifies imports, env connectivity, and one
    episode loop without any model inference.
    """
    from verirl_env import VerirlAction, verirl_env  # type: ignore

    env_url = os.environ.get("VERIRL_ENV_URL", "http://localhost:8000")
    print(f"[smoke_test] Connecting to VeriRL at {env_url}")

    simple_verilog = textwrap.dedent("""
        module relu_clip #(parameter IN_W=8, parameter OUT_W=4) (
            input  wire signed [IN_W-1:0]  in_val,
            output wire        [OUT_W-1:0] out_val,
            output wire                    saturated
        );
            localparam integer MAX_OUT = (1 << OUT_W) - 1;
            wire neg = in_val[IN_W-1];
            wire [IN_W-1:0] relu_out = neg ? {IN_W{1'b0}} : in_val;
            wire pos_clip = (relu_out > MAX_OUT[IN_W-1:0]);
            assign out_val   = pos_clip ? MAX_OUT[OUT_W-1:0] : relu_out[OUT_W-1:0];
            assign saturated = neg | pos_clip;
        endmodule
    """).strip()

    env = verirl_env(base_url=env_url)
    try:
        result = env.reset(task_id="relu_clip")
        obs = result.observation
        print(f"[smoke_test] Task spec length: {len(obs.task_spec)}")

        env.step(VerirlAction(
            action_type="write_file",
            filename="design.v",
            verilog_src=simple_verilog,
        ))
        result = env.step(VerirlAction(action_type="run_compile"))
        print(f"[smoke_test] compile_ok={result.observation.compile_ok}")

        result = env.step(VerirlAction(action_type="run_sim"))
        obs = result.observation
        print(f"[smoke_test] sim: {obs.tests_passed}/{obs.tests_total} tests passed")

        result = env.step(VerirlAction(action_type="submit"))
        score = result.observation.final_score
        print(f"[smoke_test] final_score={score:.3f}")
    finally:
        env.close()
    print(f"[smoke_test] PASSED — relu_clip score={score:.3f}")
    return {"status": "ok", "relu_clip_score": float(score or 0)}


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VeriRL RLVR training")
    parser.add_argument("--smoke",  action="store_true", help="Run smoke test only")
    args = parser.parse_args()

    if args.smoke:
        with app.run():
            print(smoke_test.remote())
    else:
        with app.run():
            print(train.remote())
