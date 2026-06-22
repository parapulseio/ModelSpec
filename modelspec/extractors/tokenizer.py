"""Tokenizer extractor — reads tokenizer_config.json / tokenizer.json.

Emits tokenizer type, vocab size and chat-template presence. tokenizer.json can
be large but it is metadata (already fetched), and we only read a few keys.
See docs/extractors.md.
"""

from __future__ import annotations

import json

from modelspec.extractors.base import ExtractionSource, ExtractorResult, FieldClaim

# tokenizers model.type -> normalized tokenizer type label.
_TYPE_MAP = {
    "BPE": "BPE",
    "Unigram": "Unigram",
    "WordPiece": "WordPiece",
    "WordLevel": "WordLevel",
}


class TokenizerExtractor:
    name = "tokenizer"

    def can_handle(self, source: ExtractionSource) -> bool:
        return source.has("tokenizer_config.json") or source.has("tokenizer.json")

    def extract(self, source: ExtractionSource) -> ExtractorResult:
        claims: list[FieldClaim] = []
        passthrough: dict = {}

        # tokenizer_config.json: chat template + class hint.
        if source.has("tokenizer_config.json"):
            cfg = json.loads(
                source.path("tokenizer_config.json").read_text(encoding="utf-8")
            )
            has_template = bool(cfg.get("chat_template"))
            claims.append(
                FieldClaim("tokenizer.chat_template_present", has_template, "config", "high")
            )
            if "model_max_length" in cfg:
                passthrough["model_max_length"] = cfg["model_max_length"]
            tok_class = cfg.get("tokenizer_class")
            if tok_class:
                passthrough["tokenizer_class"] = tok_class

        # tokenizer.json: authoritative type + vocab size.
        if source.has("tokenizer.json"):
            tok = json.loads(source.path("tokenizer.json").read_text(encoding="utf-8"))
            model = tok.get("model") or {}
            mtype = model.get("type")
            if mtype:
                claims.append(
                    FieldClaim("tokenizer.type", _TYPE_MAP.get(mtype, mtype), "config", "high")
                )
            vocab = model.get("vocab")
            added = tok.get("added_tokens") or []
            if isinstance(vocab, dict):
                claims.append(
                    FieldClaim("tokenizer.vocab_size", len(vocab) + len(added), "config", "high")
                )
            elif isinstance(vocab, list):  # Unigram stores vocab as a list of pairs
                claims.append(
                    FieldClaim("tokenizer.vocab_size", len(vocab) + len(added), "config", "high")
                )

        return ExtractorResult(claims=claims, passthrough=passthrough)
