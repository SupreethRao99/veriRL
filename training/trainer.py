"""Model loading, GRPO config assembly, and core training loop."""

from __future__ import annotations

import os

import torch
import wandb
from huggingface_hub import login

from training.config import TrainConfig
from training.wandb_task_logging import clear_task_rewards, flush_task_rewards


def _configure_wandb_defaults() -> None:
    """Set stable W&B names while allowing the caller to override them."""
    os.environ.setdefault("WANDB_PROJECT", "verirl-grpo")
    os.environ.setdefault("WANDB_RUN_NAME", "verirl-grpo-qwen3-4b-thinking")


def setup_auth() -> tuple[str, str | None]:
    """Login to HuggingFace Hub and W&B. Returns (hf_token, wandb_key)."""
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        _configure_wandb_defaults()
        wandb.login(key=wandb_key)
    return hf_token, wandb_key


def load_model_and_tokenizer(config: TrainConfig, hf_token: str):
    """Load the pre-merged SFT model directly in bf16.

    We load vllm_base_model (the already-merged full-weight repo) rather than
    the adapter repo to avoid the 4-bit merge rounding error: Unsloth saves the
    adapter on top of a quantized base, and merge_and_unload() on a 4-bit model
    degrades weight quality. The merged repo is clean bf16.

    GRPOTrainer wraps this with a fresh GRPO LoRA via peft_config=.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.chat_template_utils import qwen3_schema, qwen3_training_chat_template

    model = AutoModelForCausalLM.from_pretrained(
        config.vllm_base_model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.vllm_base_model, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The SFT-merged model's chat template diverges slightly from the stock Qwen3
    # template, causing two exact-string-match checks in TRL to fail:
    #   - add_response_schema → fixed by setting response_schema directly
    #   - get_training_chat_template → fixed by replacing the template with the
    #     prefix-preserving training variant (which TRL would have patched to anyway)
    tokenizer.response_schema = qwen3_schema
    tokenizer.chat_template = qwen3_training_chat_template

    return model, tokenizer


def build_grpo_config(
    config: TrainConfig,
    hf_token: str,
    wandb_key: str | None,
    output_dir: str,
    **extra_kwargs,
):
    """Assemble a GRPOConfig from TrainConfig. Extra kwargs are forwarded as overrides."""
    from trl import GRPOConfig

    return GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=int(config.max_steps * config.warmup_ratio),
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        beta=config.kl_coeff,
        max_steps=config.max_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        hub_strategy="checkpoint",
        logging_steps=1,
        reward_weights=config.reward_weights,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hf_output_repo,
        hub_token=hf_token,
        report_to="wandb" if wandb_key else "none",
        run_name=os.environ.get("WANDB_RUN_NAME", "verirl-grpo-qwen3-4b-thinking"),
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        use_liger_kernel=True,
        chat_template_kwargs={"enable_thinking": False},
        **extra_kwargs,
    )


def run_training(
    config: TrainConfig,
    grpo_config,
    hf_token: str,
    final_model_dir: str,
    resume_from_checkpoint: str | bool | None = None,
) -> dict:
    """Run agentic GRPO training."""
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOTrainer
    from training.curriculum import ALL_TASKS
    from training.dataset import build_dataset
    from training.environment import make_env_class
    from training.reward import (
        compile_reward,
        final_score_reward,
        sim_reward,
        tool_use_reward,
    )

    class WandbTaskRewardCallback(TrainerCallback):
        """Flush per-task reward means at the trainer's global step."""

        def on_step_end(self, args, state, control, **kwargs):
            flush_task_rewards(
                int(state.global_step),
                is_world_process_zero=getattr(state, "is_world_process_zero", True),
            )
            return control

    model, tokenizer = load_model_and_tokenizer(config, hf_token)

    print(f"[VeriRL] base_model  = {config.base_model}")
    print(f"[VeriRL] env_url     = {config.env_url}")
    print(f"[VeriRL] output_repo = {config.hf_output_repo}")
    print(f"[VeriRL] tasks       = {ALL_TASKS}")
    print(f"[VeriRL] reward_weights = {config.reward_weights}")

    dataset = build_dataset(config, n_samples=config.dataset_n_samples)
    print(f"[VeriRL] Dataset: {config.dataset_n_samples} curriculum samples across {len(ALL_TASKS)} tasks")

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,  # "all-linear"
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
    )

    VerirlToolEnv = make_env_class(config.env_url)
    clear_task_rewards()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            tool_use_reward,
            compile_reward,
            sim_reward,
            final_score_reward,
        ],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        environment_factory=VerirlToolEnv,
        callbacks=[WandbTaskRewardCallback()],
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("[VeriRL] Saving final model ...")
    trainer.save_model(final_model_dir)
    # GRPOConfig push_to_hub=True pushes on every save_steps checkpoint.
    # This final push catches the last step if it didn't land on a boundary.
    if config.push_to_hub:
        trainer.push_to_hub()
        print(f"[VeriRL] Adapter pushed to {config.hf_output_repo}")

    return {"status": "done", "output_repo": config.hf_output_repo}
