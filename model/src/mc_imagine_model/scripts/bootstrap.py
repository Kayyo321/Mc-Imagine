"""
Downloads and vendors the frozen `sentence-transformers/all-MiniLM-L6-v2` prompt encoder into
`model/checkpoints/all-MiniLM-L6-v2/`.

Why this exists: `model/checkpoints/` is gitignored (90 MB of weights do not belong in git), so a
fresh clone has no encoder at all. `model/text_encoder.py` degrades *silently-but-loudly* in that
case — it falls back to a randomly-initialized frozen encoder, which still trains and still exports,
it just learns nothing about text. Day 1 fetched the checkpoint ad-hoc with no recorded command,
which is exactly the undocumented manual step docs/phase2-plan.md §0 lists as a handoff blocker.

Files fetched are precisely the ones the rest of the pipeline opens:

  model.safetensors        -> model/text_encoder.py::load_pretrained_minilm (safetensors.torch.load_file)
  tokenizer.json           -> tokenizer_utils.py::get_tokenizer, and shipped *inside* every .mcim
  vocab.txt                -> WordPiece vocabulary backing tokenizer.json
  config.json              -> architecture record (hidden size / layers / vocab) for provenance
  tokenizer_config.json    -> lowercasing + max-length settings that must match tokenizer.json
  special_tokens_map.json  -> [PAD]/[CLS]/[SEP] mapping the tokenizer contract depends on
  sentence_bert_config.json-> the sentence-transformers pooling/max-seq-length record

Usage:
    python -m mc_imagine_model.scripts.bootstrap
    python -m mc_imagine_model.scripts.bootstrap --force        # re-download even if present
    python -m mc_imagine_model.scripts.bootstrap --verify-only  # just check what is on disk
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import List

REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Required: the pipeline cannot run correctly without these.
REQUIRED_FILES: List[str] = [
    "model.safetensors",
    "tokenizer.json",
    "vocab.txt",
    "config.json",
]
# Optional: metadata the pipeline reads for provenance/consistency but can survive without.
OPTIONAL_FILES: List[str] = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentence_bert_config.json",
]

# A few state-dict entries whose presence and shape prove the download is the model we expect,
# checked without importing the model package (so bootstrap works even mid-refactor of the net).
EXPECTED_TENSORS = {
    "embeddings.word_embeddings.weight": (30522, 384),
    "embeddings.position_embeddings.weight": (512, 384),
    "encoder.layer.5.output.dense.weight": (384, 1536),
}


def default_checkpoint_dir() -> str:
    """`model/checkpoints/all-MiniLM-L6-v2`, resolved from this file so cwd never matters.

    Deliberately computed the same way `model/text_encoder.py::DEFAULT_CHECKPOINT_DIR` computes it
    (…/mc_imagine_model/<pkg dir> -> …/model), rather than importing it — bootstrap must be runnable
    before torch is even importable.
    """
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # …/mc_imagine_model
    model_root = os.path.dirname(os.path.dirname(pkg_dir))                # …/model
    return os.path.join(model_root, "checkpoints", "all-MiniLM-L6-v2")


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n} B"


def is_complete(dest_dir: str) -> bool:
    return all(os.path.isfile(os.path.join(dest_dir, f)) for f in REQUIRED_FILES)


def download(dest_dir: str, revision: str, force: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - environment problem, not a code path
        raise SystemExit(
            "huggingface_hub is not installed — run `pip install -r model/requirements.txt` first."
        ) from exc

    os.makedirs(dest_dir, exist_ok=True)
    all_files = REQUIRED_FILES + OPTIONAL_FILES
    for i, filename in enumerate(all_files, start=1):
        target = os.path.join(dest_dir, filename)
        if os.path.isfile(target) and not force:
            print(f"  [{i}/{len(all_files)}] {filename:<26} already present ({_human(os.path.getsize(target))}), skipping")
            continue
        print(f"  [{i}/{len(all_files)}] {filename:<26} downloading…", flush=True)
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID, filename=filename, revision=revision, force_download=force
            )
        except Exception as exc:
            if filename in OPTIONAL_FILES:
                print(f"      optional file unavailable ({type(exc).__name__}), continuing")
                continue
            raise SystemExit(
                f"Failed to download required file {filename!r} from {REPO_ID}: {exc}\n"
                "Check network access / HF availability and re-run."
            ) from exc
        # Copy out of the HF cache into the vendored directory so the repo is self-contained and
        # works offline afterwards (the cache uses symlinks/blobs we do not want to depend on).
        if os.path.abspath(cached) != os.path.abspath(target):
            shutil.copyfile(cached, target)
        print(f"      -> {target} ({_human(os.path.getsize(target))})")


def verify(dest_dir: str) -> None:
    """Fails loudly if the vendored checkpoint is not actually loadable by the training pipeline."""
    print("Verifying checkpoint…")

    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(dest_dir, f))]
    if missing:
        raise SystemExit(f"FAILED: missing required file(s) in {dest_dir}: {missing}")

    # 1. model.safetensors actually loads, and holds the tensors text_encoder.py expects.
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise SystemExit("safetensors is not installed — run `pip install -r model/requirements.txt`.") from exc

    st_path = os.path.join(dest_dir, "model.safetensors")
    state_dict = load_file(st_path)
    print(f"  model.safetensors     loads: {len(state_dict)} tensors")
    for key, shape in EXPECTED_TENSORS.items():
        if key not in state_dict:
            raise SystemExit(
                f"FAILED: {st_path} has no tensor {key!r} — this does not look like "
                f"{REPO_ID}. Re-run with --force."
            )
        actual = tuple(state_dict[key].shape)
        if actual != shape:
            raise SystemExit(f"FAILED: tensor {key!r} has shape {actual}, expected {shape}.")
        print(f"    {key:<44} {actual} OK")

    # 2. tokenizer.json parses, and carries the [PAD]=0/[CLS]=101/[SEP]=102 contract the exported
    #    graph and the Java side both assume (docs/model-spec.md "Tokenizer Format").
    tok_path = os.path.join(dest_dir, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok = json.load(f)
    vocab = tok.get("model", {}).get("vocab", {})
    print(f"  tokenizer.json        parses: {len(vocab)} vocab entries")
    for token, expected_id in (("[PAD]", 0), ("[CLS]", 101), ("[SEP]", 102)):
        got = vocab.get(token)
        if got != expected_id:
            raise SystemExit(
                f"FAILED: tokenizer.json maps {token} -> {got}, expected {expected_id}. The "
                "tokenizer contract in docs/model-spec.md assumes the BERT-family WordPiece ids."
            )
    print("    [PAD]=0 [CLS]=101 [SEP]=102 OK")

    # 3. Best-effort end-to-end check through the project's own tokenizer wrapper. Optional because
    #    it depends on the `tokenizers` runtime, which bootstrap itself does not require.
    try:
        from tokenizers import Tokenizer

        t = Tokenizer.from_file(tok_path)
        ids = t.encode("towering snow-capped peaks").ids
        print(f"  tokenizers runtime    encodes sample prompt -> {len(ids)} ids: {ids[:8]}…")
    except ImportError:
        print("  (tokenizers not installed — skipped live encode check)")

    print("Verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and vendor the MiniLM prompt encoder into model/checkpoints/."
    )
    parser.add_argument("--dest", type=str, default=None,
                        help="Destination directory (default: model/checkpoints/all-MiniLM-L6-v2)")
    parser.add_argument("--revision", type=str, default="main",
                        help="HuggingFace revision/branch/commit to fetch (default: main)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download every file even if it is already present")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip downloading; only verify what is already on disk")
    args = parser.parse_args()

    dest_dir = args.dest or default_checkpoint_dir()
    t0 = time.time()
    print(f"MiniLM bootstrap — repo={REPO_ID} revision={args.revision}")
    print(f"Destination: {dest_dir}")

    if args.verify_only:
        verify(dest_dir)
        return

    if is_complete(dest_dir) and not args.force:
        print("All required files already present — skipping download (use --force to re-fetch).")
    else:
        download(dest_dir, revision=args.revision, force=args.force)

    verify(dest_dir)
    total = sum(
        os.path.getsize(os.path.join(dest_dir, f))
        for f in os.listdir(dest_dir)
        if os.path.isfile(os.path.join(dest_dir, f))
    )
    print(f"Done in {time.time() - t0:.1f}s — {dest_dir} ({_human(total)} on disk).")


if __name__ == "__main__":
    sys.exit(main())
