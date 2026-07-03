"""
Trains one language at a time. Everything -- checkpoints, logs, plots,
vocab, metrics -- lands under config.lang_dirs(lang_name).

Run for a single language:
    from train import run_training
    run_training("hi_in", "hindi")

For all three languages, use run_all.py instead (handles W&B run
open/close per language correctly).
"""

import os
import json
import numpy as np
import evaluate as hf_evaluate
from transformers import Trainer, TrainingArguments, set_seed

import config
from data_prep import get_prepared_splits
from dataset import DataCollatorCTCWithPadding
from model import build_model

wer_metric = hf_evaluate.load("wer")
cer_metric = hf_evaluate.load("cer")


def make_compute_metrics(processor):
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }
    return compute_metrics


def _report_to_list():
    backend = config.LOGGING_BACKEND
    if backend == "both":
        return ["tensorboard", "wandb"]
    if backend == "none":
        return []
    return [backend]


def run_training(lang_config: str, lang_name: str):
    set_seed(config.SEED)
    dirs = config.lang_dirs(lang_name)
    report_to = _report_to_list()

    if "wandb" in report_to:
        import wandb
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=f"wav2vec2-xlsr-{lang_name}",
            group=lang_name,
            tags=[lang_name, "wav2vec2-xlsr-53", "ctc"],
            reinit=True,
        )

    print(f"\n{'='*60}\n  TRAINING: {lang_name.upper()} ({lang_config})\n{'='*60}")

    print("[1/4] Preparing data ...")
    train_ds, eval_ds, test_ds, processor = get_prepared_splits(lang_config, lang_name)

    print("[2/4] Building model ...")
    model = build_model(processor)

    print("[3/4] Setting up trainer ...")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    training_args = TrainingArguments(
        output_dir=dirs["checkpoints"],
        group_by_length=True,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        eval_strategy="steps",
        num_train_epochs=config.NUM_EPOCHS,
        fp16=config.FP16,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        logging_steps=config.LOGGING_STEPS,
        logging_dir=dirs["logs"],              # <-- TensorBoard reads from here
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=report_to,
        run_name=f"wav2vec2-xlsr-{lang_name}",   # W&B run name
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=make_compute_metrics(processor),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,
    )

    print("[4/4] Training ...")
    train_result = trainer.train()

    final_dir = os.path.join(dirs["checkpoints"], "final_best_model")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    eval_metrics = trainer.evaluate()

    metrics = {
        "language": lang_name,
        "lang_config": lang_config,
        "train": train_result.metrics,
        "eval": eval_metrics,
    }
    with open(dirs["metrics_json"], "w") as f:
        json.dump(metrics, f, indent=2)

    # Full step-by-step loss/WER history -- this is what plot_results.py reads
    with open(dirs["log_history_json"], "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"[{lang_name}] done. Best model: {final_dir}")
    print(f"[{lang_name}] eval metrics: {eval_metrics}")

    if "wandb" in report_to:
        import wandb
        wandb.finish()

    return trainer, processor, test_ds, metrics


if __name__ == "__main__":
    # Single-language smoke test -- for all three, use run_all.py
    run_training("hi_in", "hindi")
