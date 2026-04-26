"""Model loading, GRPOConfig assembly, and the core GRPO training loop.

This module is backend-agnostic: it has no Modal or HF Jobs imports. Both
``modal_infra.py`` and ``training/hf_train_grpo.py`` call into it.

Requires the ``grpo`` extra (torch, transformers ≥ 5.x, trl ≥ 1.x, peft).
"""

from __future__ import annotations

import os

import torch
import wandb
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import qwen3_schema, qwen3_training_chat_template

from training.config import TrainConfig
from training.curriculum import ALL_TASKS
from training.dataset import build_dataset
from training.environment import make_env_class
from training.reward import compile_reward, final_score_reward, sim_reward, tool_use_reward
from training.wandb_task_logging import clear_task_rewards, flush_task_rewards


class WandbTaskRewardCallback(TrainerCallback):
    """Flush per-task reward means to W&B at the end of each trainer step.

    GRPOTrainer calls ``on_step_end`` after every gradient update. This
    callback drains the ``_PENDING`` buffer in ``wandb_task_logging`` and
    logs one mean-reward metric per task against the current global step.
    """

    def on_step_end(self, args, state, control, **kwargs):
        """Log buffered per-task rewards and return the unchanged control object."""
        flush_task_rewards(
            int(state.global_step),
            is_world_process_zero=getattr(state, "is_world_process_zero", True),
        )
        return control


def _configure_wandb_defaults() -> None:
    """Set stable W&B project and run-name defaults without overriding existing values."""
    os.environ.setdefault("WANDB_PROJECT", "verirl-grpo")
    os.environ.setdefault("WANDB_RUN_NAME", "verirl-grpo-qwen3-4b-thinking")


def setup_auth() -> tuple[str, str | None]:
    """Authenticate with HuggingFace Hub and optionally W&B.

    Reads ``HF_TOKEN`` (required) and ``WANDB_API_KEY`` (optional) from the
    environment. Logs into both services and returns the raw key values so
    callers can pass them on to ``build_grpo_config``.

    Returns:
        A ``(hf_token, wandb_key)`` tuple. ``wandb_key`` is ``None`` when
        ``WANDB_API_KEY`` is not set.

    Raises:
        KeyError: If ``HF_TOKEN`` is not set.
    """
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        _configure_wandb_defaults()
        wandb.login(key=wandb_key)
    return hf_token, wandb_key


def load_model_and_tokenizer(config: TrainConfig, hf_token: str):
    """Load the pre-merged SFT model in bf16 for GRPO fine-tuning.

    Loads ``config.vllm_base_model`` (the already-merged full-weight repo)
    rather than the adapter repo to avoid weight-quality degradation that
    occurs when calling ``merge_and_unload()`` on a 4-bit model.

    Also patches the tokenizer's chat template to the Qwen3 training variant
    that TRL expects. The SFT-merged checkpoint's template diverges slightly
    from the stock Qwen3 template, which breaks two exact-string checks in TRL:

    - ``add_response_schema`` — fixed by setting ``tokenizer.response_schema``
    - ``get_training_chat_template`` — fixed by replacing the template with the
      prefix-preserving training variant that TRL would have patched to anyway

    Args:
        config: Training configuration specifying ``vllm_base_model`` and the
            HuggingFace token for authenticated model downloads.
        hf_token: HuggingFace token passed to ``from_pretrained``.

    Returns:
        A ``(model, tokenizer)`` tuple ready to pass to ``GRPOTrainer``.
    """
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

    tokenizer.response_schema = qwen3_schema
    tokenizer.chat_template = qwen3_training_chat_template

    return model, tokenizer


def build_grpo_config(
    config: TrainConfig,
    hf_token: str,
    wandb_key: str | None,
    output_dir: str,
    **extra_kwargs,
) -> GRPOConfig:
    """Assemble a ``GRPOConfig`` from a ``TrainConfig``.

    Args:
        config: Training hyperparameters.
        hf_token: HuggingFace token for Hub pushes.
        wandb_key: W&B API key; if ``None``, reporting is disabled.
        output_dir: Local directory for checkpoints and the final model.
        **extra_kwargs: Forwarded verbatim to ``GRPOConfig`` (e.g. vLLM kwargs
            from ``build_vllm_kwargs``).

    Returns:
        A fully configured ``GRPOConfig`` instance.
    """
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
    grpo_config: GRPOConfig,
    hf_token: str,
    final_model_dir: str,
    resume_from_checkpoint: str | bool | None = None,
) -> dict:
    """Run the agentic GRPO training loop and save the final model.

    Builds the curriculum dataset, wires up the four reward functions, creates
    a ``GRPOTrainer`` with the VeriRL tool environment, and trains until
    ``grpo_config.max_steps`` is reached. Saves and optionally pushes the
    final adapter to the HuggingFace Hub.

    Args:
        config: Training hyperparameters (tasks, LoRA, reward weights, etc.).
        grpo_config: Fully assembled ``GRPOConfig``.
        hf_token: HuggingFace token for Hub pushes.
        final_model_dir: Directory to write the final model after training.
        resume_from_checkpoint: Checkpoint path, ``True`` (auto-detect), or
            ``None`` (fresh start). Passed directly to ``trainer.train()``.

    Returns:
        A dict with ``status`` and ``output_repo`` keys.
    """
    print(f"[VeriRL] base_model     = {config.base_model}")
    print(f"[VeriRL] env_url        = {config.env_url}")
    print(f"[VeriRL] output_repo    = {config.hf_output_repo}")
    print(f"[VeriRL] tasks          = {ALL_TASKS}")
    print(f"[VeriRL] reward_weights = {config.reward_weights}")

    model, tokenizer = load_model_and_tokenizer(config, hf_token)

    dataset = build_dataset(config, n_samples=config.dataset_n_samples)
    print(f"[VeriRL] Dataset: {config.dataset_n_samples} curriculum samples across {len(ALL_TASKS)} tasks")

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
    )

    VerirlToolEnv = make_env_class(config.env_url)
    clear_task_rewards()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[tool_use_reward, compile_reward, sim_reward, final_score_reward],
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
    if config.push_to_hub:
        trainer.push_to_hub()
        print(f"[VeriRL] Adapter pushed to {config.hf_output_repo}")

    return {"status": "done", "output_repo": config.hf_output_repo}
