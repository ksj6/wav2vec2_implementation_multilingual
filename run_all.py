"""
Full end-to-end run across Hindi, Marathi, and Bengali, sequentially.

This is the single entrypoint for "just run the whole thing":
    !python run_all.py

Given Kaggle's session limits, you can also comment out languages here to
split the work across multiple sessions -- each language is fully
self-contained (own folder, own checkpoints, own vocab), so partial runs
are safe.
"""

import config
from train import run_training
from evaluation import run_evaluation
from plot_results import plot_loss_curve, plot_eval_curve, plot_language_comparison


def main():
    results = {}

    for lang_config, lang_name in config.LANGUAGES.items():
        try:
            _, _, _, metrics = run_training(lang_config, lang_name)
            test_metrics = run_evaluation(lang_config, lang_name)
            plot_loss_curve(lang_name)
            plot_eval_curve(lang_name)
            results[lang_name] = test_metrics
        except Exception as e:
            print(f"[{lang_name}] FAILED: {e}")
            results[lang_name] = {"error": str(e)}

    plot_language_comparison()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for lang_name, res in results.items():
        print(f"{lang_name:10s} -> {res}")
    print(f"\nFull comparison table: {config.COMPARISON_CSV}")
    print(f"Combined chart: {config.RUNS_ROOT}/language_comparison.png")


if __name__ == "__main__":
    main()
