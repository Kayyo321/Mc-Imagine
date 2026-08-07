"""
Lightweight text/prompt encoder.

Per docs/model-spec.md "Foundation Architecture", this wraps a frozen, pretrained MiniLM-class
sentence encoder — the architecture family distributed as `sentence-transformers/all-MiniLM-L6-v2`
(6 transformer layers, 384-dim, ~22M params) — rather than training a tokenizer + transformer from
scratch. The checkpoint is vendored under `model/checkpoints/all-MiniLM-L6-v2/` (downloaded once
from HuggingFace Hub; see that directory) so training and export are reproducible offline
afterward.

Why a hand-rolled BERT re-implementation instead of `transformers.AutoModel` directly: the encoder
has to run *inside* the traced/scripted ONNX export graph (docs/poc-plan.md Phase 6 — "run the
frozen MiniLM-class encoder on prompt_tokens ... inside the traced/scripted graph"), so we want a
minimal, fully-understood set of vanilla `torch.nn`/tensor ops with no HF-internals surprises
(caching objects, `ModelOutput` dataclasses, attention-implementation dispatch) that could trip up
`torch.onnx.export`. This module's submodule names (`embeddings`, `encoder.layer.i.attention.self.*`,
etc.) are deliberately structured to mirror the checkpoint's `model.safetensors` key names exactly,
so weights load via a plain `load_state_dict(strict=True)` with no manual key remapping — see
`load_pretrained_minilm` at the bottom, and `tests/test_model.py`'s parity check against the actual
`transformers` library output.
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../mc_imagine_model
_MODEL_ROOT = os.path.dirname(os.path.dirname(_PACKAGE_DIR))  # .../model
DEFAULT_CHECKPOINT_DIR = os.path.join(_MODEL_ROOT, "checkpoints", "all-MiniLM-L6-v2")

HIDDEN_SIZE = 384
NUM_LAYERS = 6
NUM_HEADS = 12
INTERMEDIATE_SIZE = 1536
MAX_POSITION_EMBEDDINGS = 512
VOCAB_SIZE = 30522
TYPE_VOCAB_SIZE = 2
LAYER_NORM_EPS = 1e-12


class BertEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE, padding_idx=0)
        self.position_embeddings = nn.Embedding(MAX_POSITION_EMBEDDINGS, HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(TYPE_VOCAB_SIZE, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        self.register_buffer(
            "position_ids", torch.arange(MAX_POSITION_EMBEDDINGS).unsqueeze(0), persistent=False
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_len]
        token_type_ids = torch.zeros_like(input_ids)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(embeddings)


class BertSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = NUM_HEADS
        self.head_dim = HIDDEN_SIZE // NUM_HEADS
        self.query = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.key = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.value = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.query(hidden_states))
        k = self._split_heads(self.key(hidden_states))
        v = self._split_heads(self.value(hidden_states))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores + additive_mask
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        b, _, s, _ = context.shape
        context = context.permute(0, 2, 1, 3).reshape(b, s, HIDDEN_SIZE)
        return context


class BertSelfOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BertAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self = BertSelfAttention()
        self.output = BertSelfOutput()

    def forward(self, hidden_states: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        self_out = self.self(hidden_states, additive_mask)
        return self.output(self_out, hidden_states)


class BertIntermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        self.LayerNorm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden_states) + input_tensor)


class BertLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.output = BertOutput()

    def forward(self, hidden_states: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(hidden_states, additive_mask)
        inter = self.intermediate(attn_out)
        return self.output(inter, attn_out)


class BertEncoderStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayer() for _ in range(NUM_LAYERS)])

    def forward(self, hidden_states: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layer:
            hidden_states = layer(hidden_states, additive_mask)
        return hidden_states


class MiniLMBackbone(nn.Module):
    """Module hierarchy (`embeddings`, `encoder.layer.i....`) deliberately mirrors the vendored
    `sentence-transformers/all-MiniLM-L6-v2` checkpoint's `model.safetensors` key names."""

    def __init__(self) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings()
        self.encoder = BertEncoderStack()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # additive_mask: [B, 1, 1, S], 0 where attended, large-negative where masked out.
        additive_mask = (1.0 - attention_mask[:, None, None, :].to(torch.float32)) * -10000.0
        hidden_states = self.embeddings(input_ids)
        return self.encoder(hidden_states, additive_mask)


class PromptEncoder(nn.Module):
    """
    Encodes text prompt tokens into a single 384-dim, L2-normalized, mean-pooled embedding —
    exactly the `sentence-transformers/all-MiniLM-L6-v2` recipe (mean pooling over the attention
    mask, then L2 normalization). Frozen: `requires_grad_(False)` and permanently in `eval()` mode,
    so the seed/prompt -> terrain mapping is the only thing training actually learns (see
    docs/model-spec.md "Foundation Architecture").

    `attention_mask` is derived as `tokens != 0` inside forward — valid because `[PAD] == 0` in the
    BERT-family WordPiece scheme this project's tokenizer.json uses (see docs/model-spec.md
    "Tokenizer Format") — so callers (including the exported ONNX graph) never need to pass a mask
    separately.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None, embed_dim: int = HIDDEN_SIZE,
                allow_random: bool = False) -> None:
        """
        Args:
            allow_random: if the pretrained checkpoint is missing, `False` (the default) raises —
                `model/checkpoints/` is gitignored, so a fresh clone is one forgotten `bootstrap`
                away from training a model that converges and has learned nothing about text
                (docs/remote-training-readiness-plan.md §2.E2). Pass `True` (train.py's
                `--allow-random-text-encoder`) to explicitly accept a randomly-initialized,
                non-semantic encoder instead — e.g. for an architecture-only smoke test.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.backbone = MiniLMBackbone()
        self._loaded_from = None
        if checkpoint_dir is None:
            checkpoint_dir = DEFAULT_CHECKPOINT_DIR
        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
            load_pretrained_minilm(self.backbone, checkpoint_dir)
            self._loaded_from = checkpoint_dir
        elif allow_random:
            # Documented fallback per docs/poc-plan.md, now opt-in only: a same-shape
            # randomly-initialized encoder of identical architecture, still frozen after init (so
            # seed-based determinism still holds). This changes what "training" actually learns —
            # flagged loudly, and only reachable when explicitly requested.
            print(
                "[PromptEncoder] WARNING: pretrained MiniLM checkpoint not found at "
                f"{checkpoint_dir}. --allow-random-text-encoder was passed, so falling back to a "
                "randomly-initialized, frozen encoder of the same architecture. Text semantics "
                "will NOT be meaningful — this is a degraded configuration, not the intended one."
            )
        else:
            raise RuntimeError(
                f"PromptEncoder: no pretrained MiniLM checkpoint found at {checkpoint_dir!r}. "
                "Training would converge with every prompt producing the same terrain "
                "(docs/remote-training-readiness-plan.md §2.E2). Run "
                "`python -m mc_imagine_model.scripts.bootstrap`, or pass "
                "--allow-random-text-encoder to accept a degraded, non-semantic encoder "
                "explicitly."
            )
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "PromptEncoder":
        # Always frozen/eval, regardless of the outer model's train()/eval() calls.
        return super().train(False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.Tensor): int64/int32 tensor of prompt token ids, shape [B, S].

        Returns:
            torch.Tensor: L2-normalized mean-pooled embeddings, shape [B, embed_dim].
        """
        tokens = tokens.to(torch.int64)
        attention_mask = (tokens != 0).to(torch.float32)
        hidden_states = self.backbone(tokens, attention_mask)  # [B, S, H]
        mask = attention_mask.unsqueeze(-1)  # [B, S, 1]
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts
        return F.normalize(pooled, p=2, dim=-1)


def load_pretrained_minilm(backbone: MiniLMBackbone, checkpoint_dir: str) -> None:
    """Loads `model.safetensors` from `checkpoint_dir` directly into `backbone`'s state dict. The
    submodule hierarchy above was written specifically so this works with `strict=True` and no key
    remapping (see module docstring)."""
    from safetensors.torch import load_file

    state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    # `position_ids` is a non-persistent buffer we already initialize identically (arange), and the
    # checkpoint may or may not include it depending on transformers version — anything else
    # missing/unexpected is a real problem.
    missing = [m for m in missing if not m.endswith("position_ids")]
    # `pooler.dense.*` is BERT's [CLS]-token pooler head — unused here since PromptEncoder does
    # mean pooling over all tokens instead (the sentence-transformers recipe), so it's expected to
    # be present in the checkpoint but never loaded into `backbone`.
    unexpected = [u for u in unexpected if not u.endswith("position_ids") and not u.startswith("pooler.")]
    if missing or unexpected:
        raise RuntimeError(
            f"MiniLM checkpoint load mismatch — missing={missing}, unexpected={unexpected}. "
            "The hand-rolled BertEmbeddings/BertLayer module hierarchy in text_encoder.py must "
            "mirror sentence-transformers/all-MiniLM-L6-v2's state_dict keys exactly."
        )
