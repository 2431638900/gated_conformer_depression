"""
Project configuration.

Paths can be overridden with environment variables so the code is not tied
to a particular machine:

  DAIC_WOZ_ROOT       raw DAIC-WOZ directory (participant folders + AVEC CSVs)
  MODMA_ROOT          raw MODMA audio_lanzhou_2015 directory
  HUBERT_MODEL        HuBERT checkpoint name or local path
                      (default: facebook/hubert-base-ls960)
  HUBERT_CACHE_DIR    Hugging Face / local cache for the backbone
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path | None = None) -> Path | None:
    value = os.environ.get(name, "").strip()
    if value:
        return Path(value).expanduser()
    return default


class Config:
    """Global configuration aligned with the paper (V10.12)."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Raw corpora (optional until preprocessing is run)
    DATA_DIR = _env_path("DAIC_WOZ_ROOT", PROJECT_ROOT / "data" / "daic_woz")
    MODMA_ROOT = _env_path("MODMA_ROOT", PROJECT_ROOT / "data" / "modma" / "audio_lanzhou_2015")

    # Processed segment directories
    OFFICIAL_SPLIT_DIR = PROJECT_ROOT / "data" / "official_split"
    LEGACY_SPLIT_DIR = PROJECT_ROOT / "data" / "augmented_stratified"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    AUGMENTED_DATA_DIR = PROJECT_ROOT / "data" / "augmented"

    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    LOG_DIR = PROJECT_ROOT / "logs"
    PLOT_DIR = PROJECT_ROOT / "plots"

    # Backbone
    PRETRAIN_MODEL_TYPE = "hubert"
    PRETRAIN_MODEL_NAME = os.environ.get("HUBERT_MODEL", "facebook/hubert-base-ls960").strip()
    PRETRAIN_CACHE_DIR = os.environ.get("HUBERT_CACHE_DIR", "").strip() or None
    PRETRAIN_FREEZE = True
    PRETRAIN_OUTPUT_DIM = 768
    PRETRAIN_LAYERS_TO_FINETUNE = 0

    WAV2VEC_MODEL = "facebook/wav2vec2-base"
    WAV2VEC_CACHE_DIR = PRETRAIN_CACHE_DIR
    WAV2VEC_FREEZE = True
    WAV2VEC_OUTPUT_DIM = 768

    # Preprocessing (transcript-based segmentation)
    VAD_AGGRESSIVENESS = 2
    VAD_FRAME_DURATION_MS = 30
    MIN_SPEECH_DURATION = 0.5
    PADDING_MS = 100
    MIN_SENTENCE_DURATION = 1.0
    MAX_SENTENCE_DURATION = 10.0
    SILENCE_THRESHOLD = 0.5
    SENTENCES_PER_SAMPLE = 5
    AUGMENTATION_FACTOR = 2
    MIN_SENTENCES_REQUIRED = 1

    # Audio
    SAMPLE_RATE = 16000
    AUDIO_DURATION = 30
    TARGET_LENGTH = SAMPLE_RATE * AUDIO_DURATION

    # Model (paper / V10.12)
    D_MODEL = 256
    NUM_CONFORMER_LAYERS = 6
    NUM_ATTENTION_HEADS = 8
    CONV_KERNEL_SIZE = 31
    FEED_FORWARD_DIM = 1024
    DROPOUT = 0.35
    DROP_PATH_RATE = 0.1
    NUM_CLASSES = 2

    # Training (paper / V10.12)
    BATCH_SIZE = 8
    NUM_EPOCHS = 80
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-3
    OPTIMIZER = "AdamW"
    BETAS = (0.9, 0.999)
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"
    WARMUP_EPOCHS = 8
    MIN_LR = 1e-6
    EARLY_STOPPING = True
    PATIENCE = 10
    USE_WEIGHTED_LOSS = True
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_GRAD_NORM = 1.0
    MIXUP_ALPHA = 0.2
    TIME_MASK_NUM = 4
    TIME_MASK_WIDTH = 35

    METRICS = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    SAVE_BEST_MODEL = True
    BEST_MODEL_METRIC = "f1"

    DEVICE = "cuda"
    NUM_WORKERS = 4 if os.name != "nt" else 0
    PIN_MEMORY = True

    RANDOM_SEED = 42
    DEBUG = False
    DEBUG_SAMPLES = 20

    @classmethod
    def get_split_dir(cls) -> Path:
        """Prefer the official AVEC split; fall back to a legacy folder name."""
        official_csv = cls.OFFICIAL_SPLIT_DIR / "train" / "augmented_train.csv"
        if official_csv.exists():
            return cls.OFFICIAL_SPLIT_DIR
        legacy_csv = cls.LEGACY_SPLIT_DIR / "train" / "augmented_train.csv"
        if legacy_csv.exists():
            return cls.LEGACY_SPLIT_DIR
        return cls.OFFICIAL_SPLIT_DIR

    @classmethod
    def resolve_audio_path(cls, audio_path: str) -> Path:
        """Resolve a manifest path that may still contain an old absolute location."""
        raw = Path(str(audio_path))
        if raw.exists():
            return raw
        name = raw.name
        split_dir = cls.get_split_dir()
        for split in ("train", "val", "dev", "test"):
            candidate = split_dir / split / name
            if candidate.exists():
                return candidate
        if raw.is_absolute():
            relative = cls.PROJECT_ROOT / "data" / "official_split" / raw.name
            if relative.exists():
                return relative
        return raw

    @classmethod
    def create_dirs(cls):
        for d in (
            cls.OFFICIAL_SPLIT_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.AUGMENTED_DATA_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.PLOT_DIR,
        ):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def display(cls):
        print("\n" + "=" * 60)
        print("Project configuration (paper / V10.12)")
        print("=" * 60)
        print(f"DAIC-WOZ root: {cls.DATA_DIR}")
        print(f"Official split: {cls.get_split_dir()}")
        print(f"HuBERT: {cls.PRETRAIN_MODEL_NAME}")
        print(f"Conformer: {cls.NUM_CONFORMER_LAYERS} layers, {cls.NUM_ATTENTION_HEADS} heads, d_model={cls.D_MODEL}")
        print(f"Train: lr={cls.LEARNING_RATE}, wd={cls.WEIGHT_DECAY}, epochs={cls.NUM_EPOCHS}, seed={cls.RANDOM_SEED}")
        print("=" * 60 + "\n")


config = Config()


if __name__ == "__main__":
    config.display()
    config.create_dirs()
    print("Configuration loaded.")
