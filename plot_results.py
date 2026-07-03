"""
Reads each language's log_history.json (saved by train.py) and produces:
- per-language: training loss curve, eval WER/CER curve
- combined: final WER/CER bar chart comparing all three languages

Run after training + evaluate.py have completed for the languages you want:
    python plot_results.py
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

import config


def _load_log_history(lang_name: str):
    dirs = config.lang_dirs(lang_name)
    path = dirs["log_history_json"]
    if not os.path.exists(path):
        print(f"[{lang_name}] no log_history.json found yet, skipping.")
        return None, dirs
    with open(path) as f:
        history = json.load(f)
    return pd.DataFrame(history), dirs


def plot_loss_curve(lang_name: str):
    df, dirs = _load_log_history(lang_name)
    if df is None:
        return
    train_rows = df[df["loss"].notna()] if "loss" in df.columns else pd.DataFrame()
    if train_rows.empty:
        print(f"[{lang_name}] no training loss rows found.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(train_rows["step"], train_rows["loss"], color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("Training loss (CTC)")
    plt.title(f"{lang_name.capitalize()} — Training Loss")
    plt.grid(alpha=0.3)
    out_path = os.path.join(dirs["plots"], "train_loss.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{lang_name}] saved {out_path}")


def plot_eval_curve(lang_name: str):
    df, dirs = _load_log_history(lang_name)
    if df is None:
        return
    eval_rows = df[df.get("eval_wer").notna()] if "eval_wer" in df.columns else pd.DataFrame()
    if eval_rows.empty:
        print(f"[{lang_name}] no eval WER rows found.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eval_rows["step"], eval_rows["eval_wer"], color="tab:red", label="WER")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("WER", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    if "eval_cer" in eval_rows.columns:
        ax2 = ax1.twinx()
        ax2.plot(eval_rows["step"], eval_rows["eval_cer"], color="tab:green", label="CER")
        ax2.set_ylabel("CER", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")

    plt.title(f"{lang_name.capitalize()} — Eval WER / CER")
    fig.tight_layout()
    out_path = os.path.join(dirs["plots"], "eval_wer_cer.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[{lang_name}] saved {out_path}")


def plot_language_comparison():
    if not os.path.exists(config.COMPARISON_CSV):
        print("No comparison_summary.csv yet -- run evaluate.py for all languages first.")
        return

    df = pd.read_csv(config.COMPARISON_CSV)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df["test_wer"], width, label="WER", color="tab:red")
    ax.bar([i + width/2 for i in x], df["test_cer"], width, label="CER", color="tab:green")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["language"])
    ax.set_ylabel("Error rate")
    ax.set_title("Test WER / CER across languages")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    out_path = os.path.join(config.RUNS_ROOT, "language_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined comparison -> {out_path}")


if __name__ == "__main__":
    for lang_cfg, lang_name in config.LANGUAGES.items():
        plot_loss_curve(lang_name)
        plot_eval_curve(lang_name)
    plot_language_comparison()
