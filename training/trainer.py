"""Model loading, GRPO config assembly, and core training loop."""

from __future__ import annotations

import os

import torch
import wandb
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from training.config import TrainConfig
from training.curriculum import ALL_TASKS
from training.dataset import build_dataset
from training.environment import make_env_class
from training.reward import reward_func


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


def run_training(
    config: TrainConfig,
    grpo_config: GRPOConfig,
    hf_token: str,
    final_model_dir: str,
) -> dict:
    """Core training loop shared by all training variants."""
    env_class = make_env_class(config.env_url)
    model, tokenizer, lora_config = load_model_and_tokenizer(config, hf_token)

    print(f"[VeriRL] base_model  = {config.base_model}")
    print(f"[VeriRL] env_url     = {config.env_url}")
    print(f"[VeriRL] output_repo = {config.hf_output_repo}")
    print(f"[VeriRL] tasks       = {ALL_TASKS}")

    print("[VeriRL] Building dataset ...")
    dataset = build_dataset(config, n_samples=2000)
    print(f"[VeriRL] Dataset: {len(dataset)} samples across {len(ALL_TASKS)} tasks")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_funcs=[reward_func],
        peft_config=lora_config,
        environment_factory=env_class,  # TRL manages the multi-turn tool loop
    )

    print("[VeriRL] Starting GRPO training ...")
    trainer.train()

    print("[VeriRL] Saving final model ...")
    trainer.save_model(final_model_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
        print(f"[VeriRL] Model pushed to {config.hf_output_repo}")

    return {"status": "done", "output_repo": config.hf_output_repo}
