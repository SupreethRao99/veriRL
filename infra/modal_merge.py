"""
Merge SFT LoRA into the base model and push to HuggingFace.

Uses unsloth's safe merged-save path to avoid the naive 4-bit upcast + LoRA
merge issue. The dequantization happens inside save_pretrained_merged, not via
a manual .to(dtype) call.

Usage
-----
  modal run modal_merge.py
"""

import modal

LORA_REPO    = "Supreeth/verirl-sft-qwen3-4b-thinking"
BASE_REPO    = "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit"
OUTPUT_REPO  = "Supreeth/verirl-sft-qwen3-4b-thinking-merged"
MERGED_DIR   = "/tmp/merged_model"
HF_CACHE_DIR = "/root/.cache/huggingface"

# ---------------------------------------------------------------------------
# Image — unsloth handles the 4-bit → bf16 dequant + merge path
# ---------------------------------------------------------------------------

merge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # transformers 5.x pulls torchao which requires torch>=2.7 — stay on 4.x
        "transformers>=4.51.0,<5.0.0",
        "torch",
        "unsloth",
        "peft",
        "huggingface_hub",
        "bitsandbytes",
        "accelerate",
        "safetensors",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({"TOKENIZERS_PARALLELISM": "false"})
)

# ---------------------------------------------------------------------------
# Volumes & secrets
# ---------------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
hf_secret    = modal.Secret.from_name("huggingface-secret")

app = modal.App("verirl-merge-lora")

# ---------------------------------------------------------------------------
# Merge function
# ---------------------------------------------------------------------------


@app.function(
    image=merge_image,
    gpu="L4",
    timeout=3600,
    secrets=[hf_secret],
    volumes={HF_CACHE_DIR: hf_cache_vol},
    memory=32768,
)
def merge_and_push() -> str:
    import os
    import torch
    from unsloth import FastLanguageModel

    hf_token = os.environ["HF_TOKEN"]

    print(f"[merge] Loading LoRA adapter:  {LORA_REPO}")
    print(f"[merge] Base model:            {BASE_REPO}")

    # Load the 4-bit base model + LoRA adapter via unsloth.
    # unsloth resolves the adapter config and pulls the correct base weights
    # automatically when you pass the adapter repo as model_name.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_REPO,      # adapter — unsloth auto-fetches the base
        max_seq_length=8192,
        load_in_4bit=True,         # keep 4-bit during loading to save VRAM
        token=hf_token,
    )

    print(f"[merge] Merging to bf16 and saving to {MERGED_DIR} ...")
    # save_pretrained_merged dequantizes each layer just-in-time and merges
    # the LoRA deltas before writing — never holds the full fp32 model in VRAM.
    # "merged_16bit" on an Ampere GPU (L4) writes bfloat16 weights.
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
    )

    # Verify the saved dtype is bf16
    from safetensors import safe_open
    import os as _os
    shard = next(
        f for f in _os.listdir(MERGED_DIR) if f.endswith(".safetensors")
    )
    with safe_open(f"{MERGED_DIR}/{shard}", framework="pt", device="cpu") as f:
        key = next(iter(f.keys()))
        saved_dtype = f.get_tensor(key).dtype
    print(f"[merge] Saved weight dtype: {saved_dtype}")
    assert saved_dtype == torch.bfloat16, (
        f"Expected bfloat16 but got {saved_dtype} — check unsloth version"
    )

    print(f"[merge] Pushing to {OUTPUT_REPO} ...")
    model.push_to_hub_merged(
        OUTPUT_REPO,
        tokenizer,
        save_method="merged_16bit",
        token=hf_token,
    )

    print(f"[merge] Done — merged model at https://huggingface.co/{OUTPUT_REPO}")
    return OUTPUT_REPO


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    result = merge_and_push.remote()
    print(f"Merged model pushed to: https://huggingface.co/{result}")
