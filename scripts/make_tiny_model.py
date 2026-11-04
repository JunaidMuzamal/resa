"""Build a tiny randomly-initialised BERT classifier + tokenizer for the offline smoke test.

It is trained only to exercise the code paths; it says nothing about ReSA's accuracy.
    python scripts/make_tiny_model.py --out smoke/tiny-model
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="smoke/tiny-model")
    parser.add_argument("--corpus", default="external/intellisa-hub/replication/datasets/oracle")
    args = parser.parse_args()
    out = Path(args.out)
    if (out / "config.json").exists():
        print(f"{out} exists")
        return
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

    texts = []
    for path in Path(args.corpus).rglob("*"):
        if path.is_file() and path.suffix in (".pp", ".rb", ".yml", ".yaml"):
            texts.append(path.read_text(encoding="utf-8", errors="replace"))
        if len(texts) > 600:
            break
    texts += ["stage: verify | tech: puppet | smell: Hard-coded secret", "stage: recover | tech: chef | smell: Admin by default"]
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=4000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    config = BertConfig(vocab_size=fast.vocab_size, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                        max_position_embeddings=256, num_labels=2, id2label={0: "FP", 1: "TP"}, label2id={"FP": 0, "TP": 1})
    model = BertForSequenceClassification(config)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    fast.save_pretrained(out)
    print(f"tiny model written to {out}")


if __name__ == "__main__":
    main()
