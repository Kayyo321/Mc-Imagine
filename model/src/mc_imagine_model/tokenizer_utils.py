"""
Shared prompt-tokenization helper.

Both the training pipeline (`data/dataset.py`) and the export pipeline
(`export/export_onnx.py`, golden-vector generation) need to turn a text prompt into the exact
`prompt_tokens: int32[1, max_tokens]` tensor the mod's Java `PromptTokenizer` must reproduce
bit-for-bit (see docs/model-spec.md "Tokenizer Format"). This module is the single place that
wraps the HuggingFace `tokenizers` runtime around `tokenizer.json` so both call sites agree.

`tokenizer.json` is vendored from `sentence-transformers/all-MiniLM-L6-v2` (BERT-family WordPiece,
uncased, [PAD]=0, [CLS]=101, [SEP]=102) at `model/checkpoints/all-MiniLM-L6-v2/tokenizer.json` — the
exact same file that ships inside the packaged `.mcim`.
"""

import os
from functools import lru_cache
from typing import List

import numpy as np
from tokenizers import Tokenizer

MAX_PROMPT_TOKENS = 128
PAD_ID = 0
CLS_ID = 101
SEP_ID = 102

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../model/src/mc_imagine_model
_MODEL_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))  # .../model
_CHECKPOINT_DIR = os.path.join(_MODEL_ROOT, "checkpoints", "all-MiniLM-L6-v2")
TOKENIZER_JSON_PATH = os.path.join(_CHECKPOINT_DIR, "tokenizer.json")


@lru_cache(maxsize=1)
def get_tokenizer(max_tokens: int = MAX_PROMPT_TOKENS) -> Tokenizer:
    """Loads (and caches) the vendored WordPiece tokenizer, configured to pin the spec exactly:
    truncate and pad to `max_tokens`, pad id 0. `[CLS]`/`[SEP]` are added automatically by the
    post-processor baked into `tokenizer.json` (verified: encode("") -> [101, 102, 0, 0, ...])."""
    if not os.path.exists(TOKENIZER_JSON_PATH):
        raise FileNotFoundError(
            f"tokenizer.json not found at {TOKENIZER_JSON_PATH} — run the checkpoint vendoring step "
            "(download sentence-transformers/all-MiniLM-L6-v2 into model/checkpoints/)."
        )
    tok = Tokenizer.from_file(TOKENIZER_JSON_PATH)
    tok.enable_truncation(max_length=max_tokens)
    tok.enable_padding(pad_id=PAD_ID, pad_token="[PAD]", length=max_tokens)
    return tok


def tokenize(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> List[int]:
    """Returns a fixed-length list of `max_tokens` token ids: [CLS] token... [SEP] padded with 0."""
    tok = get_tokenizer(max_tokens)
    return tok.encode(prompt).ids


def tokenize_batch(prompts: List[str], max_tokens: int = MAX_PROMPT_TOKENS) -> np.ndarray:
    """Returns int32[len(prompts), max_tokens]."""
    tok = get_tokenizer(max_tokens)
    encodings = tok.encode_batch(prompts)
    return np.array([e.ids for e in encodings], dtype=np.int32)
