"""Model loading, GRPO config assembly, and core training loop."""

from __future__ import annotations

import copy
import os

import torch
import wandb
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from training.config import TrainConfig
from training.curriculum import ALL_TASKS, build_evolution_prompt
from training.dataset import build_dataset, _load_specs_from_disk
from training.environment import make_env_class
from training.reward import reward_func, get_evolution_buffer


def setup_auth() -> tuple[str, str | None]:
    """Login to HuggingFace Hub and W&B. Returns (hf_token, wandb_key)."""
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    return hf_token, wandb_key


def load_model_and_tokenizer(config: TrainConfig, hf_token: str):
    """Load QLoRA model + tokenizer. Returns (model, tokenizer, lora_config)."""
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if config.use_4bit
        else None
    )

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

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, lora_config


def build_grpo_config(
    config: TrainConfig,
    hf_token: str,
    wandb_key: str | None,
    output_dir: str,
    **extra_kwargs,
) -> GRPOConfig:
    """Assemble a GRPOConfig from TrainConfig. Extra kwargs are forwarded as overrides."""
    return GRPOConfig(
        output_dir=output_dir,
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
        beta=config.kl_coeff,
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
        **extra_kwargs,
    )


def _build_evolution_dataset(
    evolution_buffer: dict,
    specs: dict[str, str],
    config: TrainConfig,
    min_samples: int = 100,
) -> Dataset | None:
    """
    Build a GRPO-compatible Dataset of evolution prompts from the collected
    top-K design buffer.

    Each prompt shows the model its own K best previous attempts for a task
    (with EDA scores) and asks it to synthesise an improved evolved design.
    The dataset is padded by repetition to at least `min_samples` rows.
    """
    records = []
    for task_id, candidates in evolution_buffer.items():
        if len(candidates) < 2:
            # Need at least 2 designs to build a meaningful evolution prompt
            continue
        top_k = candidates[: config.evolution_top_k]
        task_spec = specs.get(task_id, f"Implement the {task_id} Verilog module.")

        prompt_msgs = build_evolution_prompt(
            top_k_results=[
                {"code": code, "score": score, "score_breakdown": breakdown}
                for code, score, breakdown in top_k
            ],
            task_spec=task_spec,
        )
        records.append({"prompt": prompt_msgs, "task_id": task_id})

    if not records:
        return None

    # Pad by repetition so the evolution phase has enough data
    repeats = max(1, (min_samples + len(records) - 1) // len(records))
    records = (records * repeats)[:max(min_samples, len(records))]
    return Dataset.from_list(records)


def run_training(
    config: TrainConfig,
    grpo_config: GRPOConfig,
    hf_token: str,
    final_model_dir: str,
) -> dict:
    """
    Two-phase training loop:

    Phase 1 — Individual GRPO
        Standard GRPO on individual task prompts. The reward_func
        side-effect populates the evolution buffer with high-scoring designs.

    Phase 2 — Evolution GRPO  (skipped when evolution_phase_ratio == 0)
        GRPO on evolution prompts built from Phase 1's top-K designs.
        The model learns to synthesise improved designs by reasoning over
        multiple previous attempts and their EDA scores.
    """
    env_class = make_env_class(config.env_url)
    model, tokenizer, lora_config = load_model_and_tokenizer(config, hf_token)

    print(f"[VeriRL] base_model       = {config.base_model}")
    print(f"[VeriRL] env_url          = {config.env_url}")
    print(f"[VeriRL] output_repo      = {config.hf_output_repo}")
    print(f"[VeriRL] tasks            = {ALL_TASKS}")
    print(f"[VeriRL] evolution_ratio  = {config.evolution_phase_ratio}")

    # ── Phase 1: Individual GRPO ─────────────────────────────────────────────
    dataset = build_dataset(config, n_samples=config.dataset_n_samples)
    print(f"[VeriRL] Dataset: {len(dataset)} samples across {len(ALL_TASKS)} tasks")

    phase1_steps = max(
        1,
        int(config.max_steps * (1.0 - config.evolution_phase_ratio)),
    )
    phase1_config = copy.copy(grpo_config)
    phase1_config.max_steps = phase1_steps

    trainer = GRPOTrainer(
        model=model,
        args=phase1_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[reward_func],
        peft_config=lora_config,
        environment_factory=env_class,
    )

    print(f"[VeriRL] Phase 1: Individual GRPO — {phase1_steps}/{config.max_steps} steps")
    trainer.train()

    # ── Phase 2: Evolution GRPO ──────────────────────────────────────────────
    phase2_steps = config.max_steps - phase1_steps
    if config.evolution_phase_ratio > 0 and phase2_steps > 0:
        evolution_buffer = get_evolution_buffer()
        specs = _load_specs_from_disk()
        evolution_dataset = _build_evolution_dataset(evolution_buffer, specs, config)

        tasks_with_data = sum(
            1 for v in evolution_buffer.values() if len(v) >= 2
        )
        print(
            f"[VeriRL] Evolution buffer: {tasks_with_data}/{len(ALL_TASKS)} tasks "
            f"with ≥2 designs (score_threshold={config.evolution_score_threshold})"
        )

        if evolution_dataset and len(evolution_dataset) >= 10:
            phase2_config = copy.copy(grpo_config)
            phase2_config.max_steps = phase2_steps
            phase2_config.output_dir = grpo_config.output_dir + "/evolution"
            phase2_config.run_name = (
                (phase2_config.run_name or "verirl") + "-evolution"
            )
            # Disable hub push mid-training; final push happens at the end
            phase2_config.push_to_hub = False

            evo_trainer = GRPOTrainer(
                model=trainer.model,  # continue from Phase 1 weights
                args=phase2_config,
                train_dataset=evolution_dataset,
                tokenizer=tokenizer,
                reward_funcs=[reward_func],
                peft_config=None,  # LoRA already applied
                environment_factory=env_class,
            )
            print(
                f"[VeriRL] Phase 2: Evolution GRPO — {phase2_steps} steps, "
                f"{len(evolution_dataset)} evolution prompts"
            )
            evo_trainer.train()
            trainer = evo_trainer
        else:
            print(
                "[VeriRL] Phase 2: Skipped — insufficient evolution data "
                f"(tasks_with_data={tasks_with_data}, "
                f"dataset_size={len(evolution_dataset) if evolution_dataset else 0})"
            )
    else:
        print("[VeriRL] Phase 2: Disabled (evolution_phase_ratio=0)")

    # ── Save & push ──────────────────────────────────────────────────────────
    print("[VeriRL] Saving final model ...")
    trainer.save_model(final_model_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
        print(f"[VeriRL] Model pushed to {config.hf_output_repo}")

    return {"status": "done", "output_repo": config.hf_output_repo}
