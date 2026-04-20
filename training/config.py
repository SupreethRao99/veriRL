"""Training configuration dataclass with OmegaConf YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Path to the repo-root config file, relative to this file's location.
_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"


@dataclass
class TrainConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    base_model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    hf_output_repo: str = "SupreethRao99/verirl-rlvr-qwen2.5-coder-3b"
    use_4bit: bool = True               # QLoRA NF4 quantisation

    # ── GRPO hyper-parameters ──────────────────────────────────────────────
    num_train_epochs: int = 3
    learning_rate: float = 5e-6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 6            # G in GRPO — completions sampled per prompt
    max_prompt_length: int = 1024
    max_completion_length: int = 4096   # total tokens across all turns in one episode
    temperature: float = 0.8
    top_p: float = 0.95
    kl_coeff: float = 0.05             # β — KL penalty coefficient
    max_steps: int = 500
    save_steps: int = 100
    logging_steps: int = 10
    warmup_ratio: float = 0.05
    push_to_hub: bool = True

    # ── LoRA / PEFT ────────────────────────────────────────────────────────
    lora_r: int = 64
    lora_alpha: int = 128
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    # ── Curriculum — weights need not sum to 1 ─────────────────────────────
    task_difficulty_weights: dict = field(default_factory=lambda: {
        "easy":   0.40,   # mac_unit, relu_clip, barrel_shifter
        "medium": 0.40,   # axi_fifo, register_file, ring_buffer, dot_product, fir_filter
        "hard":   0.20,   # systolic_array, fp16_adder
    })

    # ── Evolutionary GRPO ─────────────────────────────────────────────────
    evolution_phase_ratio: float = 0.20    # fraction of max_steps for evolution phase (0 = disabled)
    evolution_top_k: int = 2               # top-K designs used to build each evolution prompt
    evolution_score_threshold: float = 0.40  # min episode score to enter the evolution buffer

    # ── Runtime ────────────────────────────────────────────────────────────
    env_url: str = "http://localhost:8000"  # overridden by VERIRL_ENV_URL secret
    dataset_n_samples: int = 2000

    # ------------------------------------------------------------------
    # YAML factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = _DEFAULT_CONFIG,
        **overrides,
    ) -> "TrainConfig":
        """Load a TrainConfig from *config.yaml* via OmegaConf.

        Any keyword arguments are applied on top of the YAML values, allowing
        per-run overrides (e.g. ``env_url``) without touching the shared file.

        Example::

            cfg = TrainConfig.from_yaml()
            cfg = TrainConfig.from_yaml("config.yaml", env_url="http://gpu:8000")
        """
        from omegaconf import OmegaConf  # lazy import — optional at import time

        raw = OmegaConf.load(path)
        t = raw.training

        kwargs: dict = {
            # Model
            "base_model":                    t.model.base_model,
            "hf_output_repo":                t.model.hf_output_repo,
            "use_4bit":                      bool(t.model.use_4bit),
            # GRPO
            "num_train_epochs":              int(t.grpo.num_train_epochs),
            "learning_rate":                 float(t.grpo.learning_rate),
            "per_device_train_batch_size":   int(t.grpo.per_device_train_batch_size),
            "gradient_accumulation_steps":   int(t.grpo.gradient_accumulation_steps),
            "num_generations":               int(t.grpo.num_generations),
            "max_prompt_length":             int(t.grpo.max_prompt_length),
            "max_completion_length":         int(t.grpo.max_completion_length),
            "temperature":                   float(t.grpo.temperature),
            "top_p":                         float(t.grpo.top_p),
            "kl_coeff":                      float(t.grpo.kl_coeff),
            "max_steps":                     int(t.grpo.max_steps),
            "save_steps":                    int(t.grpo.save_steps),
            "logging_steps":                 int(t.grpo.logging_steps),
            "warmup_ratio":                  float(t.grpo.warmup_ratio),
            "push_to_hub":                   bool(t.grpo.push_to_hub),
            # LoRA
            "lora_r":                        int(t.lora.r),
            "lora_alpha":                    int(t.lora.lora_alpha),
            "lora_target_modules":           list(OmegaConf.to_container(t.lora.target_modules)),
            "lora_dropout":                  float(t.lora.lora_dropout),
            "lora_bias":                     str(t.lora.bias),
            # Curriculum
            "task_difficulty_weights":       dict(OmegaConf.to_container(
                                                 t.curriculum.task_difficulty_weights
                                             )),
            # Evolutionary GRPO
            "evolution_phase_ratio":         float(t.evolution.phase_ratio),
            "evolution_top_k":               int(t.evolution.top_k),
            "evolution_score_threshold":     float(t.evolution.score_threshold),
            # Runtime
            "env_url":                       str(t.env_url),
            "dataset_n_samples":             int(t.dataset_n_samples),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
