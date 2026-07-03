"""
Loads facebook/wav2vec2-large-xlsr-53 and attaches a CTC head sized to
whichever language's vocab you pass in. Each language gets its own model
instance -- vocab sizes differ (Devanagari vs Bengali script), so the CTC
head can't be shared.
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import config


def build_model(processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    model = Wav2Vec2ForCTC.from_pretrained(
        config.MODEL_CHECKPOINT,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,   # required since XLSR-53's original head size != ours
    )

    if config.FREEZE_FEATURE_ENCODER:
        model.freeze_feature_encoder()

    if config.GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    model.to(config.DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model loaded: {n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    return model
