"""
Central config for multilingual wav2vec2 fine-tuning: Hindi, Marathi, Bengali.

Everything is folder-wise / language-wise under RUNS_ROOT:

/kaggle/working/runs/
    hindi/
        checkpoints/
        logs/           <- tensorboard event files land here
        plots/
        vocab.json
        metrics.json
        test_predictions.csv
    marathi/
        ... same structure
    bengali/
        ... same structure
    comparison_summary.csv   <- cross-language WER/CER table, written by evaluate.py
"""

import os
import torch

# -------------------- Languages --------------------
# google/fleurs config name -> readable folder name
LANGUAGES = {
    "hi_in": "hindi",
    # "mr_in": "marathi",
    # "bn_in": "bengali",
}

DATASET_NAME = "google/fleurs"
SAMPLING_RATE = 16000
TEXT_COLUMN = "transcription"   # FLEURS' transcript field name

# -------------------- Base paths --------------------
WORK_DIR = "/kaggle/working"
CACHE_DIR = os.path.join(WORK_DIR, "hf_cache")
RUNS_ROOT = os.path.join(WORK_DIR, "runs")
COMPARISON_CSV = os.path.join(RUNS_ROOT, "comparison_summary.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RUNS_ROOT, exist_ok=True)


def lang_dirs(lang_name: str) -> dict:
    """All per-language paths, created on first call. This is what keeps
    results folder-wise / language-wise instead of dumped in one place."""
    root = os.path.join(RUNS_ROOT, lang_name)
    dirs = {
        "root": root,
        "checkpoints": os.path.join(root, "checkpoints"),
        "logs": os.path.join(root, "logs"),
        "plots": os.path.join(root, "plots"),
        "vocab": os.path.join(root, "vocab.json"),
        "predictions_csv": os.path.join(root, "test_predictions.csv"),
        "metrics_json": os.path.join(root, "metrics.json"),
        "log_history_json": os.path.join(root, "log_history.json"),
    }
    for key in ("root", "checkpoints", "logs", "plots"):
        os.makedirs(dirs[key], exist_ok=True)
    return dirs


# -------------------- Model --------------------
# facebook/wav2vec2-base is English-only pretraining -- unusable for Indic
# languages. XLSR-53 is pretrained across 53 languages incl. Hindi, and is
# the standard base checkpoint for exactly this kind of fine-tuning.
MODEL_CHECKPOINT = "facebook/wav2vec2-large-xlsr-53"
FREEZE_FEATURE_ENCODER = True

# -------------------- Training --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 1 # it is 15 actual 
PER_DEVICE_TRAIN_BATCH_SIZE = 4        # XLSR-53 (~315M params) is much bigger than base
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4         # effective batch = 16
LEARNING_RATE = 3e-4
WARMUP_STEPS = 40. # it is 300 in actual 
WEIGHT_DECAY = 0.005
FP16 = torch.cuda.is_available()
SAVE_TOTAL_LIMIT = 2
LOGGING_STEPS = 25
EVAL_STEPS = 100
SAVE_STEPS = 100
GRADIENT_CHECKPOINTING = True

TRAIN_SUBSET_SIZE = 300    # set e.g. 300 for a smoke test before a real run
EVAL_SUBSET_SIZE = 100     # set e.g. 100 for a smoke test before a real run

# -------------------- Experiment tracking --------------------
# "tensorboard", "wandb", "both", or "none"
LOGGING_BACKEND = "both"
WANDB_PROJECT = "wav2vec2-indic-asr"
WANDB_ENTITY = None   # set to your W&B username/team; None uses your default

SEED = 42
