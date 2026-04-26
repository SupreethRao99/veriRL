"""Training configuration dataclasses with OmegaConf YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


_DEFAULT_CONFIG = Path(__file__).parent.parent / "config.yaml"


@dataclass
class SFTConfig:
    """Hyperparameters for the Unsloth + TRL SFT warm-start phase.

    Defaults mirror the ``sft`` section of ``config.yaml``. Call
    ``SFTConfig.from_yaml()`` to load from disk; pass keyword overrides
    to adjust individual fields without editing the file.
    """

    sft_base_model: str = "Qwen/Qwen3-4B-Thinking-2507"
    sft_output_repo: str = "Supreeth/verirl-sft-qwen3-4b-thinking"

    sft_max_seq_length: int = 2048
    sft_lora_r: int = 16
    sft_lora_alpha: int = 32
    sft_per_device_batch_size: int = 2
    sft_gradient_accumulation_steps: int = 8
    sft_warmup_steps: int = 100
    sft_num_train_epochs: int = 1
    sft_max_steps: int = -1
    sft_learning_rate: float = 2e-4
    sft_save_steps: int = 500
    sft_max_samples: int = 100000

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG, **overrides) -> "SFTConfig":
        """Load SFT configuration from a YAML file.

        Args:
            path: Path to the config YAML (default: ``config.yaml`` at project root).
            **overrides: Field values that override what is in the file.

        Returns:
            A fully populated ``SFTConfig`` instance.
        """
        raw = OmegaConf.load(path)
        s = raw.sft
        kwargs: dict = {
            "sft_base_model":                  str(s.base_model),
            "sft_output_repo":                 str(s.output_repo),
            "sft_max_seq_length":              int(s.max_seq_length),
            "sft_lora_r":                      int(s.lora_r),
            "sft_lora_alpha":                  int(s.lora_alpha),
            "sft_per_device_batch_size":       int(s.per_device_batch_size),
            "sft_gradient_accumulation_steps": int(s.gradient_accumulation_steps),
            "sft_num_train_epochs":            int(s.num_train_epochs),
            "sft_warmup_steps":                int(s.warmup_steps),
            "sft_max_steps":                   int(s.max_steps),
            "sft_learning_rate":               float(s.learning_rate),
            "sft_save_steps":                  int(s.save_steps),
            "sft_max_samples":                 int(s.max_samples),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass
class TrainConfig:
    """Hyperparameters for the GRPO RLVR training phase.

    Defaults mirror the ``training`` section of ``config.yaml``. Call
    ``TrainConfig.from_yaml()`` to load from disk; pass keyword overrides
    to adjust individual fields without editing the file.

    The ``env_url`` field is not in the YAML and must be passed explicitly
    (either via ``from_yaml(env_url=...)`` or the ``VERIRL_ENV_URL`` env var).
    """

    # Model identifiers
    base_model: str = "Supreeth/verirl-sft-qwen3-4b-thinking"
    vllm_base_model: str = "Supreeth/verirl-sft-qwen3-4b-thinking-merged"
    hf_output_repo: str = "Supreeth/verirl-rlvr-qwen3-4b-thinking"
    use_4bit: bool = True

    # GRPO hyperparameters
    num_train_epochs: int = 3
    learning_rate: float = 5e-5
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    max_completion_length: int = 1024
    max_model_length: int = 8192
    temperature: float = 0.8
    top_p: float = 0.95
    kl_coeff: float = 0.05
    max_steps: int = 50
    save_steps: int = 20
    logging_steps: int = 1
    warmup_ratio: float = 0.05
    push_to_hub: bool = True
    reward_weights: list[float] = field(default_factory=lambda: [0.05, 0.25, 0.40, 0.30])

    # LoRA / PEFT
    lora_r: int = 1
    lora_alpha: int = 32
    lora_target_modules: str | list = "all-linear"
    lora_dropout: float = 0.0
    lora_bias: str = "none"

    # Curriculum
    task_difficulty_weights: dict = field(default_factory=lambda: {
        "easy":   1.0,
        "medium": 0.0,
        "hard":   0.0,
    })
    task_ids: list[str] = field(default_factory=lambda: ["mac_unit"])

    # Runtime
    env_url: str = "http://localhost:8000"
    dataset_n_samples: int = 2000

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG, **overrides) -> "TrainConfig":
        """Load GRPO training configuration from a YAML file.

        Args:
            path: Path to the config YAML (default: ``config.yaml`` at project root).
            **overrides: Field values that override what is in the file.
                Pass ``env_url=...`` here since it is not stored in the YAML.

        Returns:
            A fully populated ``TrainConfig`` instance.
        """
        raw = OmegaConf.load(path)
        t = raw.training
        kwargs: dict = {
            "base_model":                   t.model.base_model,
            "vllm_base_model":              t.model.vllm_base_model,
            "hf_output_repo":               t.model.hf_output_repo,
            "use_4bit":                     bool(t.model.use_4bit),
            "num_train_epochs":             int(t.grpo.num_train_epochs),
            "learning_rate":                float(t.grpo.learning_rate),
            "per_device_train_batch_size":  int(t.grpo.per_device_train_batch_size),
            "gradient_accumulation_steps":  int(t.grpo.gradient_accumulation_steps),
            "num_generations":              int(t.grpo.num_generations),
            "max_completion_length":        int(t.grpo.max_completion_length),
            "max_model_length":             int(t.grpo.max_model_length),
            "temperature":                  float(t.grpo.temperature),
            "top_p":                        float(t.grpo.top_p),
            "kl_coeff":                     float(t.grpo.kl_coeff),
            "max_steps":                    int(t.grpo.max_steps),
            "save_steps":                   int(t.grpo.save_steps),
            "logging_steps":                int(t.grpo.logging_steps),
            "warmup_ratio":                 float(t.grpo.warmup_ratio),
            "push_to_hub":                  bool(t.grpo.push_to_hub),
            "reward_weights":               list(t.grpo.reward_weights),
            "lora_r":                       int(t.lora.r),
            "lora_alpha":                   int(t.lora.lora_alpha),
            "lora_target_modules":          (
                str(t.lora.target_modules)
                if isinstance(t.lora.target_modules, str)
                else list(OmegaConf.to_container(t.lora.target_modules))
            ),
            "lora_dropout":                 float(t.lora.lora_dropout),
            "lora_bias":                    str(t.lora.bias),
            "task_difficulty_weights":      dict(OmegaConf.to_container(
                                                t.curriculum.task_difficulty_weights)),
            "task_ids":                     list(OmegaConf.to_container(t.curriculum.task_ids)),
            "env_url":                      str(t.env_url),
            "dataset_n_samples":            int(t.dataset_n_samples),
        }
        kwargs.update(overrides)
        return cls(**kwargs)
