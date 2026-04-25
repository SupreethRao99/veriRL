"""SFT warm-start on PyraNet-Verilog using Unsloth + TRL SFTTrainer.

Requires the ``sft`` extra (torch, transformers 4.x, trl 0.x, unsloth).
Note: unsloth is Linux/CUDA-only and therefore imported lazily inside
``run_sft`` rather than at module level so this file remains importable
on non-CUDA machines.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import wandb as wb
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are an expert RTL hardware designer. "
    "Write correct, synthesizable Verilog modules given a functional description."
)


def _format_example(example: dict, tokenizer) -> dict:
    """Format a single PyraNet-Verilog sample as a chat-template training string.

    Filters out examples that have no description, no code, or a compilation
    error in their metadata. Clean examples are formatted as a three-turn
    conversation (system → user → assistant) and rendered with the tokenizer's
    chat template so the model trains on the exact token sequence it generates.

    Args:
        example: A raw dataset row with ``description`` (JSON string) and ``code`` fields.
        tokenizer: The tokenizer whose ``apply_chat_template`` formats the turns.

    Returns:
        ``{"text": <formatted string>}`` on success, or ``{"text": None}`` to
        signal that the example should be filtered out.
    """
    try:
        meta = json.loads(example["description"])
        description = meta.get("description", "").strip()
        compile_status = meta.get("compile_status", "")
        if not description or (
            "error" in compile_status.lower()
            and "no error" not in compile_status.lower()
        ):
            return {"text": None}
    except (json.JSONDecodeError, TypeError):
        return {"text": None}

    code = (example.get("code") or "").strip()
    if not code:
        return {"text": None}

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": description},
        {"role": "assistant", "content": code},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def load_sft_dataset(tokenizer, max_samples: int | None = None):
    """Load and preprocess the PyraNet-Verilog dataset for SFT.

    Loads the ``bnadimi/PyraNet-Verilog`` HuggingFace dataset, optionally
    truncates it, formats each example as a chat turn, and filters out
    examples that fail the format check (no description, compile errors, etc.).

    Args:
        tokenizer: Tokenizer used to render chat templates.
        max_samples: If set, limits the dataset to this many rows before
            filtering. Useful for smoke runs and debugging.

    Returns:
        A filtered HuggingFace ``Dataset`` with a single ``text`` column.
    """
    ds = load_dataset("bnadimi/PyraNet-Verilog", split="train")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.map(
        lambda ex: _format_example(ex, tokenizer),
        remove_columns=ds.column_names,
    )
    return ds.filter(lambda ex: ex["text"] is not None)


def _latest_checkpoint(output_dir: str) -> str | None:
    """Return the path of the highest-numbered checkpoint in ``output_dir``, or None."""
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def run_sft(config, hf_token: str, wandb_key: str | None, output_dir: str) -> dict:
    """Run the SFT warm-start training loop and push results to HuggingFace Hub.

    Loads the base model with Unsloth (4-bit NF4 + LoRA), trains on the
    filtered PyraNet-Verilog dataset, and pushes two Hub repos:

    1. The LoRA adapter (``config.sft_output_repo``) for inspection.
    2. A merged bf16 copy (``config.sft_output_repo + "-merged"``) that vLLM
       can load directly without adapter-resolution overhead.

    Automatically resumes from the latest checkpoint in ``output_dir`` so
    preempted runs continue without manual intervention.

    Args:
        config: ``SFTConfig`` dataclass with all hyperparameters.
        hf_token: HuggingFace write token for Hub pushes.
        wandb_key: W&B API key; if ``None``, W&B logging is disabled.
        output_dir: Local directory for checkpoints and the final adapter.

    Returns:
        A dict with ``status``, ``output_repo``, and ``merged_repo`` keys.
    """
    # Unsloth is Linux/CUDA-only; import lazily so this module stays importable
    # on macOS and other non-CUDA environments.
    from unsloth import FastLanguageModel

    if wandb_key:
        wb.login(key=wandb_key)

    checkpoint = _latest_checkpoint(output_dir)
    if checkpoint:
        print(f"[SFT] Resuming from {checkpoint}")
    else:
        print(f"[SFT] Starting fresh — no checkpoint found in {output_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint or config.sft_base_model,
        max_seq_length=config.sft_max_seq_length,
        dtype=None,
        load_in_4bit=True,
        token=hf_token,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.sft_lora_r,
        lora_alpha=config.sft_lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_sft_dataset(tokenizer, max_samples=config.sft_max_samples)
    print(f"[SFT] Dataset: {len(dataset)} samples after filtering")
    print(f"[SFT] base_model  = {config.sft_base_model}")
    print(f"[SFT] output_repo = {config.sft_output_repo}")
    print(f"[SFT] max_steps   = {config.sft_max_steps}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=config.sft_per_device_batch_size,
            gradient_accumulation_steps=config.sft_gradient_accumulation_steps,
            num_train_epochs=config.sft_num_train_epochs,
            warmup_steps=config.sft_warmup_steps,
            max_steps=config.sft_max_steps,
            learning_rate=config.sft_learning_rate,
            logging_steps=10,
            save_steps=config.sft_save_steps,
            save_total_limit=3,
            push_to_hub=True,
            hub_model_id=config.sft_output_repo,
            hub_token=hf_token,
            hub_strategy="every_save",
            report_to="wandb" if wandb_key else "none",
            run_name="verirl-sft-qwen3-4b-thinking",
            bf16=True,
            torch_compile=True,
            dataset_text_field="text",
            max_seq_length=config.sft_max_seq_length,
            packing=True,
        ),
    )

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(output_dir)
    trainer.push_to_hub()

    # Push a merged (full-weight, no adapter) copy so vLLM can load it directly.
    # vLLM cannot load PEFT adapter repos; it needs a plain HF model.
    # Unsloth's push_to_hub_merged handles the 4-bit → bf16 dequantize path.
    merged_repo = config.sft_output_repo + "-merged"
    print(f"[SFT] Pushing merged model to {merged_repo} for vLLM ...")
    model.push_to_hub_merged(merged_repo, tokenizer, save_method="merged_16bit", token=hf_token)
    print(f"[SFT] Merged model pushed to {merged_repo}")

    return {
        "status": "done",
        "output_repo": config.sft_output_repo,
        "merged_repo": merged_repo,
    }
