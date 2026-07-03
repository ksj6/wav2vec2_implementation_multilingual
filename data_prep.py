"""
Loads a single FLEURS language config, builds that language's own character
vocab (scripts differ completely between Hindi/Marathi/Bengali, so a shared
vocab would be wrong), and returns preprocessed splits.

Called per-language from train.py / run_all.py, not run standalone for all
three at once.
"""

import json
import re
from datasets import load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

import config

# Common punctuation across Latin + Devanagari (।) + Bengali scripts
CHARS_TO_IGNORE_REGEX = r"[\,\?\.\!\-\;\:\"“”‘’।॥\(\)]"


def load_raw_splits(lang_config: str):
    print(f"Loading {config.DATASET_NAME}/{lang_config} ...")
    train = load_dataset(
        config.DATASET_NAME, lang_config, split="train",
        cache_dir=config.CACHE_DIR, trust_remote_code=True,
    )
    eval_ = load_dataset(
        config.DATASET_NAME, lang_config, split="validation",
        cache_dir=config.CACHE_DIR, trust_remote_code=True,
    )
    test = load_dataset(
        config.DATASET_NAME, lang_config, split="test",
        cache_dir=config.CACHE_DIR, trust_remote_code=True,
    )

    if config.TRAIN_SUBSET_SIZE:
        train = train.select(range(min(config.TRAIN_SUBSET_SIZE, len(train))))
    if config.EVAL_SUBSET_SIZE:
        eval_ = eval_.select(range(min(config.EVAL_SUBSET_SIZE, len(eval_))))

    train = train.cast_column("audio", Audio(sampling_rate=config.SAMPLING_RATE))
    eval_ = eval_.cast_column("audio", Audio(sampling_rate=config.SAMPLING_RATE))
    test = test.cast_column("audio", Audio(sampling_rate=config.SAMPLING_RATE))

    return train, eval_, test


def clean_text(batch):
    text = batch[config.TEXT_COLUMN]
    text = re.sub(CHARS_TO_IGNORE_REGEX, "", text).lower().strip()
    batch["text"] = text
    return batch


def build_vocab(train, eval_, vocab_path: str):
    print(f"Building character vocab -> {vocab_path}")

    def extract_chars(batch):
        all_text = " ".join(batch["text"])
        return {"vocab": [list(set(all_text))]}

    train_vocab = train.map(extract_chars, batched=True, batch_size=-1, remove_columns=train.column_names)
    eval_vocab = eval_.map(extract_chars, batched=True, batch_size=-1, remove_columns=eval_.column_names)

    vocab_set = set(train_vocab["vocab"][0]) | set(eval_vocab["vocab"][0])
    vocab_list = sorted(vocab_set)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict.get(" ", len(vocab_dict))
    if " " in vocab_dict:
        del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    print(f"Vocab size: {len(vocab_dict)}")
    return vocab_dict


def build_processor(vocab_path: str):
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=config.SAMPLING_RATE,
        padding_value=0.0, do_normalize=True, return_attention_mask=True,
    )
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def get_prepared_splits(lang_config: str, lang_name: str):
    """Main entrypoint. lang_config e.g. 'hi_in', lang_name e.g. 'hindi'."""
    dirs = config.lang_dirs(lang_name)

    train, eval_, test = load_raw_splits(lang_config)
    keep_cols = lambda ds: ds.remove_columns([c for c in ds.column_names if c not in (config.TEXT_COLUMN, "audio")])
    train, eval_, test = keep_cols(train), keep_cols(eval_), keep_cols(test)

    train = train.map(clean_text)
    eval_ = eval_.map(clean_text)
    test = test.map(clean_text)

    build_vocab(train, eval_, dirs["vocab"])
    processor = build_processor(dirs["vocab"])

    train = train.map(lambda b: prepare_dataset(b, processor), remove_columns=train.column_names, num_proc=2)
    eval_ = eval_.map(lambda b: prepare_dataset(b, processor), remove_columns=eval_.column_names, num_proc=2)
    test = test.map(lambda b: prepare_dataset(b, processor), remove_columns=test.column_names, num_proc=2)

    return train, eval_, test, processor


if __name__ == "__main__":
    # Smoke test on Hindi only
    train, eval_, test, processor = get_prepared_splits("hi_in", "hindi")
    print(f"train={len(train)} eval={len(eval_)} test={len(test)}")
