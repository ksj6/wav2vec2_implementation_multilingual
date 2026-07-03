"""
Final test-set evaluation for one language, writing:
- WER/CER printed + saved into that language's metrics.json
- predictions vs references CSV in that language's folder
- a row appended to the global comparison_summary.csv across all languages
"""

import os
import csv
import json
import torch
import evaluation as hf_evaluate
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

import config
from data_prep import get_prepared_splits
from dataset import DataCollatorCTCWithPadding

wer_metric = hf_evaluate.load("wer")
cer_metric = hf_evaluate.load("cer")


def load_finetuned(lang_name: str):
    dirs = config.lang_dirs(lang_name)
    model_dir = os.path.join(dirs["checkpoints"], "final_best_model")
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(config.DEVICE)
    model.eval()
    return model, processor, dirs


@torch.no_grad()
def _greedy_decode(model, processor, input_values):
    logits = model(input_values.to(config.DEVICE)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)


def run_evaluation(lang_config: str, lang_name: str, batch_size: int = 4):
    model, processor, dirs = load_finetuned(lang_name)
    _, _, test_ds, _ = get_prepared_splits(lang_config, lang_name)

    collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    loader = DataLoader(test_ds, batch_size=batch_size, collate_fn=collator)

    all_preds, all_refs = [], []
    for batch in loader:
        preds = _greedy_decode(model, processor, batch["input_values"])
        labels = batch["labels"].clone()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        refs = processor.batch_decode(labels, group_tokens=False)
        all_preds.extend(preds)
        all_refs.extend(refs)

    wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    cer = cer_metric.compute(predictions=all_preds, references=all_refs)

    print(f"[{lang_name}] Test WER: {wer:.4f}  |  Test CER: {cer:.4f}")

    with open(dirs["predictions_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reference", "prediction"])
        for ref, pred in zip(all_refs, all_preds):
            writer.writerow([ref, pred])

    # merge into that language's metrics.json if it exists
    result = {"language": lang_name, "test_wer": wer, "test_cer": cer, "n_test_examples": len(all_refs)}
    if os.path.exists(dirs["metrics_json"]):
        with open(dirs["metrics_json"]) as f:
            existing = json.load(f)
        existing["test"] = result
        result = existing
    with open(dirs["metrics_json"], "w") as f:
        json.dump(result, f, indent=2)

    _append_to_comparison_csv(lang_name, wer, cer, len(all_refs))
    return {"wer": wer, "cer": cer}


def _append_to_comparison_csv(lang_name, wer, cer, n_examples):
    file_exists = os.path.exists(config.COMPARISON_CSV)
    with open(config.COMPARISON_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["language", "test_wer", "test_cer", "n_test_examples"])
        writer.writerow([lang_name, f"{wer:.4f}", f"{cer:.4f}", n_examples])
    print(f"Comparison row appended -> {config.COMPARISON_CSV}")


if __name__ == "__main__":
    for lang_cfg, lang_name in config.LANGUAGES.items():
        run_evaluation(lang_cfg, lang_name)
