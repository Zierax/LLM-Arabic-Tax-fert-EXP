"""Core logic for the Arabic Fertility Tax experiment.

This module deliberately keeps three things separate:

1. **Intrinsic tokenizer metrics** (CPU, deterministic): fertility, chars/token,
   bytes/token, unknown rate, and document-level bootstrap intervals.
2. **Cost arithmetic**: token inflation and, only when real prices are filled,
   dollars per million words. The equal-price cost index is not a market claim.
3. **TTFT**: a model/provider measurement, not a tokenizer-only measurement.
   The same prompt windows are sent to every provider; regressions are fitted
   on provider-reported input tokens.

No GPU is required for the tokenizer and corpus arms. See docs/METHODOLOGY.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and text utilities
# ---------------------------------------------------------------------------

ARABIC_RANGES: tuple[tuple[int, int], ...] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)

# Harakat, tanwin, shadda, sukun, superscript alef, Quranic marks.
DIACRITIC_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
WHITESPACE_RE = re.compile(r"\s+")

ARATOKEN_TRANSLATION = str.maketrans(
    {
        # Alif variants -> bare alif (AraToken's unified configuration).
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        # Arabic / Extended Arabic-Indic digits -> Western digits.
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        # Arabic punctuation -> Latin punctuation.
        "؟": "?",
        "؛": ";",
        "،": ",",
        "؍": "/",
        "۔": ".",
        "٪": "%",
        "٫": ".",
        "٬": ",",
        "﴿": ")",
        "﴾": "(",
        # Tatweel / kashida.
        "ـ": "",
    }
)

VIEWS = ("raw", "nfkc", "aratoken_norm")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_whitespace(text: str) -> str:
    """Collapse all Unicode whitespace runs to one ASCII space.

    This is applied when the corpus is built so every tokenizer sees the same
    document boundaries and no tokenizer gets an advantage from newlines.
    """
    return WHITESPACE_RE.sub(" ", text).strip()


def strip_diacritics(text: str) -> str:
    return DIACRITIC_RE.sub("", text)


def _lowercase_ascii_latin(text: str) -> str:
    return "".join(ch.lower() if ch.isascii() and ch.isalpha() else ch for ch in text)


def normalize_aratoken(text: str) -> str:
    """AraToken's published normalization pipeline.

    Order follows the tokenizer_config.json in mariklolik/AraToken:
    NFKC -> remove diacritics -> character replacements -> lowercase Latin.
    It maps alif variants to bare alif and removes tatweel.
    """
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text)
    text = text.translate(ARATOKEN_TRANSLATION)
    return _lowercase_ascii_latin(text)


def normalize_view(text: str, view: str) -> str:
    if view == "raw":
        return text
    if view == "nfkc":
        return unicodedata.normalize("NFKC", text)
    if view == "aratoken_norm":
        return normalize_aratoken(text)
    raise ValueError(f"unknown view: {view!r}; expected one of {VIEWS}")


def count_words(text: str) -> int:
    return len(text.split())


def is_arabic_letter(ch: str) -> bool:
    if ch == "\u0640":  # tatweel is a modifier, not a letter for our ratio
        return False
    cp = ord(ch)
    if not any(lo <= cp <= hi for lo, hi in ARABIC_RANGES):
        return False
    return unicodedata.category(ch).startswith("L")


def count_arabic_letters(text: str) -> int:
    return sum(1 for ch in text if is_arabic_letter(ch))


def count_nonspace_chars(text: str) -> int:
    """Non-space characters, excluding Arabic diacritics.

    Excluding diacritics keeps the Arabic-letter ratio comparable between
    undiacritized and fully vocalized text. The diacritic load is reported
    separately by diacritic_ratio().
    """
    return sum(
        1
        for ch in text
        if not ch.isspace() and not DIACRITIC_RE.fullmatch(ch)
    )


def arabic_letter_ratio(text: str) -> float:
    denom = count_nonspace_chars(text)
    return count_arabic_letters(text) / denom if denom else 0.0


def diacritic_ratio(text: str) -> float:
    letters = count_arabic_letters(text)
    return len(DIACRITIC_RE.findall(text)) / letters if letters else 0.0


def dedup_key(text: str) -> str:
    """Normalize aggressively enough to catch near-duplicates across strata."""
    text = unicodedata.normalize("NFKC", text)
    text = strip_diacritics(text)
    text = text.translate(ARATOKEN_TRANSLATION)
    return WHITESPACE_RE.sub(" ", text).strip().lower()


# ---------------------------------------------------------------------------
# Config and files
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {config_path}")
    cfg["_config_path"] = str(config_path)
    cfg["_base_dir"] = str(config_path.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return Path(cfg["_base_dir"]) / path


def config_path(cfg: dict[str, Any], key: str, default: str) -> Path:
    return resolve_path(cfg, str(cfg.get("project", {}).get(key, default)))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    tmp.replace(path)


def _atomic_write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            yield obj


# ---------------------------------------------------------------------------
# Corpus loading / summarising
# ---------------------------------------------------------------------------


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for obj in read_jsonl(path):
        text = normalize_whitespace(str(obj.get("text") or ""))
        if not text:
            continue
        doc_id = str(obj.get("doc_id") or f"doc-{len(docs):05d}")
        if doc_id in seen_ids:
            raise ValueError(f"duplicate doc_id in {path}: {doc_id}")
        seen_ids.add(doc_id)
        obj["doc_id"] = doc_id
        obj["text"] = text
        obj.setdefault("stratum", "unknown")
        obj["n_words"] = count_words(text)
        obj["n_chars"] = len(text)
        obj["n_bytes"] = len(text.encode("utf-8"))
        obj["arabic_letter_ratio"] = arabic_letter_ratio(text)
        obj["diacritic_ratio"] = diacritic_ratio(text)
        obj["sha256"] = sha256_text(text)
        docs.append(obj)
    if not docs:
        raise ValueError(f"no usable documents in {path}")
    return docs


def summarize_corpus(docs: list[dict[str, Any]]) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = {}
    for doc in docs:
        name = str(doc.get("stratum", "unknown"))
        s = by_stratum.setdefault(
            name,
            {
                "name": name,
                "docs": 0,
                "words": 0,
                "chars": 0,
                "bytes": 0,
                "arabic_letters": 0,
                "nonspace_chars": 0,
                "diacritics": 0,
                "licenses": set(),
            },
        )
        text = doc["text"]
        s["docs"] += 1
        s["words"] += count_words(text)
        s["chars"] += len(text)
        s["bytes"] += len(text.encode("utf-8"))
        s["arabic_letters"] += count_arabic_letters(text)
        s["nonspace_chars"] += count_nonspace_chars(text)
        s["diacritics"] += len(DIACRITIC_RE.findall(text))
        if doc.get("license"):
            s["licenses"].add(str(doc["license"]))

    strata: list[dict[str, Any]] = []
    for name, s in by_stratum.items():
        strata.append(
            {
                "name": name,
                "docs": s["docs"],
                "words": s["words"],
                "chars": s["chars"],
                "bytes": s["bytes"],
                "arabic_letter_ratio": (
                    s["arabic_letters"] / s["nonspace_chars"] if s["nonspace_chars"] else 0.0
                ),
                "diacritic_ratio": (
                    s["diacritics"] / s["arabic_letters"] if s["arabic_letters"] else 0.0
                ),
                "licenses": sorted(s["licenses"]),
            }
        )

    total_words = sum(s["words"] for s in strata)
    total_chars = sum(s["chars"] for s in strata)
    total_bytes = sum(s["bytes"] for s in strata)
    arabic_letters = sum(v["arabic_letters"] for v in by_stratum.values())
    nonspace_chars = sum(v["nonspace_chars"] for v in by_stratum.values())
    diacritics = sum(v["diacritics"] for v in by_stratum.values())
    return {
        "total_docs": len(docs),
        "total_words": total_words,
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "arabic_letter_ratio": arabic_letters / nonspace_chars if nonspace_chars else 0.0,
        "diacritic_ratio": diacritics / arabic_letters if arabic_letters else 0.0,
        "strata": strata,
    }


def validate_corpus(
    path: str | Path,
    *,
    expected_words: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    docs = load_corpus(path)
    summary = summarize_corpus(docs)
    if expected_words is not None and summary["total_words"] != expected_words:
        raise ValueError(
            f"corpus has {summary['total_words']} words; expected {expected_words}"
        )
    if cfg is not None:
        expected = {s["name"]: int(s["words"]) for s in cfg["corpus"]["strata"]}
        actual = {s["name"]: int(s["words"]) for s in summary["strata"]}
        for name, want in expected.items():
            got = actual.get(name, 0)
            if got != want:
                raise ValueError(f"stratum {name!r}: {got} words; expected {want}")
    seen_hashes: set[str] = set()
    for doc in docs:
        h = doc["sha256"]
        if h in seen_hashes:
            raise ValueError(f"duplicate text in corpus: {doc['doc_id']}")
        seen_hashes.add(h)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------


def _iter_source(stratum: dict[str, Any], cfg: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield raw texts from one Hugging Face dataset source."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": stratum["split"],
        "streaming": bool(cfg["corpus"].get("streaming", True)),
    }
    revision = stratum.get("revision")
    if revision:
        kwargs["revision"] = revision
    ds = load_dataset(
        stratum["source"],
        name=stratum.get("config"),
        **kwargs,
    )
    buffer_size = int(cfg["corpus"].get("shuffle_buffer", 0) or 0)
    if buffer_size:
        try:
            ds = ds.shuffle(seed=int(cfg["project"].get("seed", 20261025)), buffer_size=buffer_size)
        except Exception as exc:  # pragma: no cover - depends on dataset class
            log.warning("shuffle unavailable for %s: %s", stratum["source"], exc)

    text_field = stratum["text_field"]
    dialect_field = stratum.get("dialect_field")
    dialect_regex = stratum.get("dialect_regex")
    for row in ds:
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            continue
        meta: dict[str, Any] = {}
        if dialect_field:
            label = str(row.get(dialect_field, ""))
            if dialect_regex and not re.search(dialect_regex, label):
                continue
            meta["dialect"] = label
        yield text, meta


def _resolve_dataset_revision(source: str, revision: str | None) -> str | None:
    """Return the pinned revision, or resolve `main` to a commit SHA.

    A manifest that says `revision: null` cannot be reproduced after the
    upstream dataset moves. Resolving the SHA up front makes the first build
    reproducible even when the config has not been pinned yet.
    """
    if revision:
        return str(revision)
    if "/" not in source:
        # Local/synthetic source (or a non-HF dataset name); nothing to resolve.
        return None
    try:
        from huggingface_hub import dataset_info

        return str(dataset_info(source).sha)
    except Exception as exc:  # pragma: no cover - network/auth dependent
        log.warning("could not resolve revision for %s: %s", source, exc)
        return None


def build_corpus(
    cfg: dict[str, Any],
    *,
    out_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    corpus_cfg = cfg["corpus"]
    strata_cfg = corpus_cfg["strata"]
    total_expected = int(corpus_cfg["total_words"])
    declared = sum(int(s["words"]) for s in strata_cfg)
    if declared != total_expected:
        raise ValueError(
            f"stratum budgets sum to {declared}, but corpus.total_words is {total_expected}"
        )

    out = resolve_path(cfg, out_path or cfg["project"]["corpus_path"])
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; pass --force to rebuild")

    seed = int(cfg["project"].get("seed", 20261025))
    max_scan_rows = int(corpus_cfg.get("max_scan_rows", 2_000_000))
    global_min_arabic = float(corpus_cfg.get("min_arabic_ratio", 0.60))
    all_docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifest_strata: list[dict[str, Any]] = []

    for stratum in strata_cfg:
        name = str(stratum["name"])
        requested_revision = stratum.get("revision")
        resolved_revision = _resolve_dataset_revision(
            str(stratum["source"]), requested_revision
        )
        stratum = {**stratum, "revision": resolved_revision}
        need = int(stratum["words"])
        got = 0
        docs_for: list[dict[str, Any]] = []
        stats = {
            "scanned": 0,
            "skipped_short": 0,
            "skipped_duplicate": 0,
            "skipped_low_arabic": 0,
            "skipped_diacritics": 0,
            "skipped_low_unique": 0,
        }
        min_words = int(stratum.get("min_words", 1))
        max_words = int(stratum.get("max_words", 10_000))
        min_unique = int(stratum.get("min_unique_words", 0))
        min_arabic = float(stratum.get("min_arabic_ratio", global_min_arabic))
        min_diac = float(stratum.get("min_diacritic_ratio", 0.0))
        max_diac = float(stratum.get("max_diacritic_ratio", 1.0))

        log.info("building stratum %s: %d words", name, need)
        for raw_text, meta in _iter_source(stratum, cfg):
            stats["scanned"] += 1
            if stats["scanned"] > max_scan_rows:
                raise RuntimeError(
                    f"stratum {name}: scanned more than {max_scan_rows} rows without "
                    f"reaching {need} words; relax filters or raise max_scan_rows"
                )
            text = normalize_whitespace(raw_text)
            words = text.split()
            if len(words) < min_words:
                stats["skipped_short"] += 1
                continue
            if len(words) > max_words:
                words = words[:max_words]
                text = " ".join(words)
            if len(set(w.lower() for w in words)) < min_unique:
                stats["skipped_low_unique"] += 1
                continue
            if arabic_letter_ratio(text) < min_arabic:
                stats["skipped_low_arabic"] += 1
                continue
            dr = diacritic_ratio(text)
            if dr < min_diac or dr > max_diac:
                stats["skipped_diacritics"] += 1
                continue
            key = dedup_key(text)
            if key in seen:
                stats["skipped_duplicate"] += 1
                continue
            seen.add(key)

            take = min(len(words), need - got)
            if take <= 0:
                break
            if take < len(words):
                text = " ".join(words[:take])
            doc = {
                "doc_id": f"{name}-{len(docs_for):05d}",
                "stratum": name,
                "source": stratum["source"],
                "source_config": stratum.get("config"),
                "source_split": stratum.get("split"),
                "source_revision": stratum.get("revision"),
                "license": stratum.get("license"),
                "dialect": meta.get("dialect"),
                "text": text,
                "n_words": count_words(text),
                "n_chars": len(text),
                "n_bytes": len(text.encode("utf-8")),
                "arabic_letter_ratio": arabic_letter_ratio(text),
                "diacritic_ratio": diacritic_ratio(text),
                "sha256": sha256_text(text),
            }
            docs_for.append(doc)
            got += doc["n_words"]
            if len(docs_for) % 25 == 0:
                log.info("  %s: %d/%d words (%d docs)", name, got, need, len(docs_for))
            if got >= need:
                break

        if got != need:
            raise RuntimeError(
                f"stratum {name}: only {got}/{need} words after scanning "
                f"{stats['scanned']} rows. Stats: {stats}"
            )
        all_docs.extend(docs_for)
        manifest_strata.append(
            {
                "name": name,
                "words": got,
                "docs": len(docs_for),
                "source": stratum["source"],
                "config": stratum.get("config"),
                "split": stratum.get("split"),
                "revision": stratum.get("revision"),
                "revision_requested": requested_revision,
                "license": stratum.get("license"),
                **stats,
            }
        )
        log.info("  %s: done (%d words, %d docs)", name, got, len(docs_for))

    write_jsonl(out, all_docs)
    summary = summarize_corpus(all_docs)
    summary_by_name = {s["name"]: s for s in summary["strata"]}
    for row in manifest_strata:
        row.update(
            {
                "chars": summary_by_name[row["name"]]["chars"],
                "arabic_letter_ratio": summary_by_name[row["name"]]["arabic_letter_ratio"],
                "diacritic_ratio": summary_by_name[row["name"]]["diacritic_ratio"],
            }
        )
    manifest = {
        "project": cfg["project"].get("name"),
        "created_at": now_iso(),
        "config_path": cfg["_config_path"],
        "config_sha256": sha256_file(cfg["_config_path"]),
        "seed": seed,
        "total_words": summary["total_words"],
        "total_docs": summary["total_docs"],
        "file": str(out),
        "file_sha256": sha256_file(out),
        "strata": manifest_strata,
        "summary": summary,
        "warnings": [
            "The built corpus is not redistributed. Tashkeela is GPL-2.0 data.",
            "Exact 50,000-word budget is enforced; the last document in a stratum "
            "may be shorter than min_words to hit the budget exactly.",
        ],
    }
    manifest_path = out.with_name(out.stem + "_manifest.json")
    _atomic_write_json(manifest_path, manifest)
    log.info("wrote %s (%d docs, %d words)", out, summary["total_docs"], summary["total_words"])
    log.info("wrote %s", manifest_path)
    return manifest


# ---------------------------------------------------------------------------
# Tokenizer adapters
# ---------------------------------------------------------------------------


class TokenizerUnavailable(RuntimeError):
    """The tokenizer cannot be loaded (gated, missing key, unsupported kind)."""


class BaseAdapter:
    name: str
    spec: dict[str, Any]
    version: str
    is_remote = False

    def count(self, text: str, view: str) -> dict[str, int]:
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        return {}


class HFAdapter(BaseAdapter):
    def __init__(self, name: str, spec: dict[str, Any]):
        from transformers import AutoTokenizer

        self.name = name
        self.spec = spec
        repo = str(spec["repo"])
        revision = spec.get("revision")
        kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": False,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True, **kwargs)
        except Exception as exc:
            lowered = str(exc).lower()
            if any(token in lowered for token in ("gated", "401", "403", "restricted")):
                raise
            # A few research tokenizers ship only a slow implementation.
            log.warning("%s: fast tokenizer failed (%s); retrying slow", name, exc)
            self.tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=False, **kwargs)
        self.version = (
            f"hf:{repo}@{revision or 'main'}:{self.tokenizer.__class__.__name__}"
        )

    def count(self, text: str, view: str) -> dict[str, int]:
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
        except TypeError:
            encoded = self.tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        unk = sum(1 for token_id in ids if token_id == unk_id) if unk_id is not None else 0
        return {"tokens": len(ids), "unk": unk}

    def info(self) -> dict[str, Any]:
        max_len = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(max_len, int) and max_len > 10**9:
            max_len = None
        return {
            "kind": "hf",
            "repo": self.spec.get("repo"),
            "revision": self.spec.get("revision"),
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "is_fast": bool(getattr(self.tokenizer, "is_fast", False)),
            "vocab_size": len(self.tokenizer),
            "model_max_length": max_len,
            "version": self.version,
        }


class SentencePieceAdapter(BaseAdapter):
    def __init__(self, name: str, spec: dict[str, Any]):
        from huggingface_hub import hf_hub_download
        import sentencepiece as spm

        self.name = name
        self.spec = spec
        repo = str(spec["repo"])
        revision = spec.get("revision")
        filename = str(spec["filename"])
        path = hf_hub_download(repo_id=repo, filename=filename, revision=revision)
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.version = (
            f"sp:{repo}@{revision or 'main'}:{filename}:{sha256_file(path)[:12]}"
        )

    def count(self, text: str, view: str) -> dict[str, int]:
        # `raw` means no *external* normalization. The SentencePiece model's
        # own normalizer still applies, exactly as it does when the model is
        # deployed. The published AraToken pipeline is applied by the
        # `aratoken_norm` view in run_tokenization(), not here, so every
        # tokenizer sees the same pre-tokenization text in each view.
        pieces = self.sp.encode(text, out_type=str)
        unk_id = self.sp.unk_id()
        unk = sum(1 for piece in pieces if self.sp.piece_to_id(piece) == unk_id)
        return {"tokens": len(pieces), "unk": unk}

    def info(self) -> dict[str, Any]:
        return {
            "kind": "sentencepiece",
            "repo": self.spec.get("repo"),
            "revision": self.spec.get("revision"),
            "filename": self.spec.get("filename"),
            "vocab_size": self.sp.get_piece_size(),
            "published_normalization": self.spec.get("published_normalization"),
            "version": self.version,
        }


class TiktokenAdapter(BaseAdapter):
    def __init__(self, name: str, spec: dict[str, Any]):
        import tiktoken

        self.name = name
        self.spec = spec
        self.encoding_name = str(spec["encoding"])
        self.enc = tiktoken.get_encoding(self.encoding_name)
        self.version = f"tiktoken:{self.encoding_name}:{self.enc.n_vocab}"

    def count(self, text: str, view: str) -> dict[str, int]:
        ids = self.enc.encode(text, disallowed_special=())
        return {"tokens": len(ids), "unk": 0}

    def info(self) -> dict[str, Any]:
        return {
            "kind": "tiktoken",
            "encoding": self.encoding_name,
            "vocab_size": self.enc.n_vocab,
            "version": self.version,
        }


class GeminiCountAdapter(BaseAdapter):
    is_remote = True

    def __init__(self, name: str, spec: dict[str, Any]):
        from providers import GeminiClient

        self.name = name
        self.spec = spec
        self.client = GeminiClient(spec)
        self.version = f"gemini-count:{self.client.model}"

    def count(self, text: str, view: str) -> dict[str, int]:
        return {"tokens": self.client.count_tokens(text), "unk": 0}

    def info(self) -> dict[str, Any]:
        return {
            "kind": "gemini_count",
            "model": self.client.model,
            "endpoint": f"{self.client.base_url}/models/{self.client.model}:countTokens",
            "version": self.version,
            "note": "provider-reported count; tokenizer is not public",
        }


def build_adapter(name: str, spec: dict[str, Any]) -> BaseAdapter:
    kind = spec.get("kind")
    try:
        if kind == "hf":
            return HFAdapter(name, spec)
        if kind == "sentencepiece":
            return SentencePieceAdapter(name, spec)
        if kind == "tiktoken":
            return TiktokenAdapter(name, spec)
        if kind == "gemini_count":
            return GeminiCountAdapter(name, spec)
        raise TokenizerUnavailable(f"unknown tokenizer kind: {kind!r}")
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if any(token in lowered for token in ("gated", "401", "403", "access")):
            message += (
                " | Hint: request access on the Hugging Face model page, run "
                "`huggingface-cli login`, and set HF_TOKEN."
            )
        raise TokenizerUnavailable(f"{name}: {message}") from exc


def build_adapters(
    cfg: dict[str, Any],
    names: list[str] | None = None,
    *,
    strict: bool = False,
) -> tuple[dict[str, BaseAdapter], dict[str, dict[str, Any]]]:
    if names is None:
        names = list(cfg["analysis"]["tokenizers"])
    adapters: dict[str, BaseAdapter] = {}
    statuses: dict[str, dict[str, Any]] = {}
    for name in names:
        spec = cfg.get("tokenizers", {}).get(name)
        if not spec:
            raise KeyError(f"unknown tokenizer: {name}")
        try:
            adapter = build_adapter(name, spec)
            adapters[name] = adapter
            statuses[name] = {
                "status": "ok",
                "class": spec.get("class"),
                "optional": bool(spec.get("optional", False)),
                **adapter.info(),
            }
            log.info("tokenizer %s: ready (%s)", name, adapter.version)
        except Exception as exc:
            statuses[name] = {
                "status": "unavailable",
                "class": spec.get("class"),
                "kind": spec.get("kind"),
                "optional": bool(spec.get("optional", False)),
                "error": str(exc),
            }
            if strict:
                raise
            log.warning("tokenizer %s unavailable: %s", name, exc)
    return adapters, statuses


# ---------------------------------------------------------------------------
# Count cache
# ---------------------------------------------------------------------------


CACHE_SCHEMA_VERSION = "v2"


class CountCache:
    """Append-only JSONL cache keyed by tokenizer, view, doc and adapter version.

    Bump CACHE_SCHEMA_VERSION whenever the meaning of a cached count changes
    (normalization order, tokenizer wrapper, word counting, ...). Old rows stay
    on disk but are ignored, which is safer than silently mixing schemas.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for row in read_jsonl(self.path):
                key = row.get("key")
                if key:
                    self.rows[str(key)] = row
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")

    @staticmethod
    def key(
        tokenizer: str,
        view: str,
        doc: dict[str, Any],
        adapter_version: str,
    ) -> str:
        payload = "\x1f".join(
            [
                CACHE_SCHEMA_VERSION,
                tokenizer,
                view,
                str(doc["doc_id"]),
                str(doc["sha256"]),
                adapter_version,
            ]
        )
        return sha256_text(payload)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def put(self, row: dict[str, Any]) -> None:
        self.rows[str(row["key"])] = row
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CountCache":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def seed_int(key: str) -> int:
    return int(sha256_text(key)[:16], 16)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def bootstrap_micro_ci(
    pairs: list[tuple[int, int]],
    *,
    n_boot: int,
    ci: float,
    seed: int,
) -> list[float | None]:
    """Document-level bootstrap CI for sum(tokens) / sum(words)."""
    if len(pairs) < 2 or n_boot <= 0:
        return [None, None]
    rng = random.Random(seed)
    n = len(pairs)
    stats: list[float] = []
    for _ in range(n_boot):
        total_tokens = 0
        total_words = 0
        for _ in range(n):
            tokens, words = pairs[rng.randrange(n)]
            total_tokens += tokens
            total_words += words
        if total_words:
            stats.append(total_tokens / total_words)
    if not stats:
        return [None, None]
    alpha = (1.0 - ci) / 2.0
    return [percentile(stats, alpha * 100.0), percentile(stats, (1.0 - alpha) * 100.0)]


def paired_bootstrap(
    pairs: list[tuple[int, int, int]],
    *,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    """Paired document bootstrap for candidate vs reference.

    pairs are (candidate_tokens, reference_tokens, words). The ratio is
    candidate_tokens / reference_tokens (equal to the fertility ratio because
    both sides use the same word denominator). The difference is in
    tokens-per-word units.
    """
    if not pairs:
        return {}
    total_a = sum(a for a, _, _ in pairs)
    total_r = sum(r for _, r, _ in pairs)
    total_w = sum(w for _, _, w in pairs)
    observed_ratio = total_a / total_r if total_r else None
    observed_diff = (total_a - total_r) / total_w if total_w else None
    ratio_ci: list[float | None] = [None, None]
    diff_ci: list[float | None] = [None, None]
    if len(pairs) >= 2 and n_boot > 0:
        rng = random.Random(seed)
        n = len(pairs)
        ratios: list[float] = []
        diffs: list[float] = []
        for _ in range(n_boot):
            a_sum = 0
            r_sum = 0
            w_sum = 0
            for _ in range(n):
                a, r, w = pairs[rng.randrange(n)]
                a_sum += a
                r_sum += r
                w_sum += w
            if r_sum:
                ratios.append(a_sum / r_sum)
            if w_sum:
                diffs.append((a_sum - r_sum) / w_sum)
        alpha = (1.0 - ci) / 2.0
        if ratios:
            ratio_ci = [
                percentile(ratios, alpha * 100.0),
                percentile(ratios, (1.0 - alpha) * 100.0),
            ]
        if diffs:
            diff_ci = [
                percentile(diffs, alpha * 100.0),
                percentile(diffs, (1.0 - alpha) * 100.0),
            ]
    return {
        "n_pairs": len(pairs),
        "ratio": observed_ratio,
        "ratio_ci": ratio_ci,
        "diff_tokens_per_word": observed_diff,
        "diff_ci": diff_ci,
    }


def aggregate_rows(
    rows: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    tokenizer: str,
    view: str,
    class_label: str,
    context_window: int | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    analysis = cfg.get("analysis", {})
    n_boot = int(analysis.get("bootstrap", {}).get("n_boot", 2000))
    ci = float(analysis.get("bootstrap", {}).get("ci", 0.95))
    target_words = sum(int(d["n_words"]) for d in docs)

    strata_order: list[str] = []
    for doc in docs:
        name = str(doc["stratum"])
        if name not in strata_order:
            strata_order.append(name)

    groups: list[tuple[str, list[dict[str, Any]]]] = [("__all__", rows)]
    for name in strata_order:
        group = [r for r in rows if r["stratum"] == name]
        if group:
            groups.append((name, group))

    out: list[dict[str, Any]] = []
    for stratum, group in groups:
        tokens = sum(int(r["tokens"]) for r in group)
        words = sum(int(r["words"]) for r in group)
        chars = sum(int(r["chars"]) for r in group)
        nbytes = sum(int(r["bytes"]) for r in group)
        unk = sum(int(r.get("unk", 0)) for r in group)
        if words <= 0 or tokens <= 0:
            continue
        fertility = tokens / words
        pairs = [(int(r["tokens"]), int(r["words"])) for r in group]
        ci_values = bootstrap_micro_ci(
            pairs,
            n_boot=n_boot,
            ci=ci,
            seed=seed_int(f"{tokenizer}|{view}|{stratum}"),
        )
        expected_docs = (
            len(docs)
            if stratum == "__all__"
            else sum(1 for d in docs if str(d["stratum"]) == stratum)
        )
        out.append(
            {
                "tokenizer": tokenizer,
                "view": view,
                "scope": "all" if stratum == "__all__" else "stratum",
                "stratum": stratum,
                "tokenizer_class": class_label,
                "docs": len(group),
                "expected_docs": expected_docs,
                "coverage": len(group) / expected_docs if expected_docs else None,
                "words": words,
                "tokens": tokens,
                "chars": chars,
                "bytes": nbytes,
                "unk": unk,
                "fertility": fertility,
                "fertility_ci": ci_values,
                "chars_per_token": chars / tokens,
                "bytes_per_token": nbytes / tokens,
                "unk_rate": unk / tokens,
                "tokens_for_target_words": fertility * target_words,
                "target_words": target_words,
                "context_window": context_window,
                "context_words_equivalent": (
                    context_window / fertility if context_window else None
                ),
            }
        )
    return out


def build_cost_rows(cfg: dict[str, Any], metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pricing = cfg.get("pricing", {}) or {}
    analysis = cfg.get("analysis", {})
    ref_name = str(analysis.get("reference_tokenizer", "aratoken"))
    headline_view = str(analysis.get("headline_view", "raw"))
    illustrative_price = float(analysis.get("illustrative_price_usd_per_1m", 0.50))
    ref_metric = next(
        (
            m
            for m in metrics
            if m["tokenizer"] == ref_name
            and m["view"] == headline_view
            and m["scope"] == "all"
        ),
        None,
    )
    ref_price_raw = (pricing.get(ref_name, {}) or {}).get("input_usd_per_1m")
    ref_price = float(ref_price_raw) if ref_price_raw not in (None, "") else None

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric["view"] != headline_view or metric["scope"] != "all":
            continue
        price_info = pricing.get(metric["tokenizer"], {}) or {}
        price_raw = price_info.get("input_usd_per_1m")
        price = float(price_raw) if price_raw not in (None, "") else None
        fertility = float(metric["fertility"])
        inflation = (
            fertility / float(ref_metric["fertility"])
            if ref_metric and ref_metric["fertility"]
            else None
        )
        cost = fertility * price if price is not None else None
        cost_ratio = None
        if price is not None and ref_metric and ref_price is not None:
            ref_cost = float(ref_metric["fertility"]) * ref_price
            if ref_cost:
                cost_ratio = cost / ref_cost
        rows.append(
            {
                "tokenizer": metric["tokenizer"],
                "tokenizer_class": metric.get("tokenizer_class"),
                "view": headline_view,
                "reference": ref_name,
                "fertility": fertility,
                "tokens_for_target_words": metric["tokens_for_target_words"],
                "token_inflation_vs_reference": inflation,
                "illustrative_usd_per_1m_words": fertility * illustrative_price,
                "illustrative_price_usd_per_1m": illustrative_price,
                "input_usd_per_1m": price,
                "cost_usd_per_1m_words": cost,
                "cost_ratio_vs_reference": cost_ratio,
                "price_as_of": price_info.get("as_of"),
                "price_source": price_info.get("source"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Tokenization runner
# ---------------------------------------------------------------------------


def refresh_cost(result: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Recompute the cost table with the current config prices.

    `tokenization.json` stores the cost rows from tokenization time. Prices are
    usually filled after tokenization, so `bench.py report` must not reuse the
    stale rows. The view/reference are taken from the stored run metadata so a
    later config edit cannot silently change which metrics are priced.
    """
    meta = result.get("meta", {})
    analysis = dict(cfg.get("analysis", {}))
    analysis["headline_view"] = meta.get(
        "headline_view", analysis.get("headline_view", "raw")
    )
    analysis["reference_tokenizer"] = meta.get(
        "reference_tokenizer", analysis.get("reference_tokenizer", "aratoken")
    )
    if "illustrative_price_usd_per_1m" in meta:
        analysis["illustrative_price_usd_per_1m"] = meta[
            "illustrative_price_usd_per_1m"
        ]
    result["cost"] = build_cost_rows(
        {**cfg, "analysis": analysis}, result.get("metrics", [])
    )
    return result


def run_tokenization(
    cfg: dict[str, Any],
    docs: list[dict[str, Any]],
    tokenizer_names: list[str],
    views: list[str],
    out_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    limit: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    for view in views:
        if view not in VIEWS:
            raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    if limit is not None:
        docs = docs[: int(limit)]
    if not docs:
        raise ValueError("no documents to tokenize")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adapters, statuses = build_adapters(cfg, tokenizer_names, strict=False)
    if not adapters:
        raise RuntimeError(
            "no tokenizers available; inspect the status table and the error messages"
        )

    cache = CountCache(out / "cache" / "token_counts.jsonl")
    metrics: list[dict[str, Any]] = []
    rows_by: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []

    def make_row(
        item: tuple[int, dict[str, Any], str, str],
        adapter: BaseAdapter,
        view: str,
    ) -> dict[str, Any]:
        _idx, doc, text, key = item
        count = adapter.count(text, view)
        return {
            "key": key,
            "tokenizer": adapter.name,
            "view": view,
            "doc_id": doc["doc_id"],
            "stratum": doc["stratum"],
            "text_sha": doc["sha256"],
            "adapter_version": adapter.version,
            "tokens": int(count["tokens"]),
            "unk": int(count.get("unk", 0)),
            "words": count_words(text),
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
            "created_at": now_iso(),
        }

    try:
        for name, adapter in adapters.items():
            spec = cfg["tokenizers"][name]
            context_window = spec.get("context_window")
            context_window = int(context_window) if context_window else None
            for view in views:
                texts = [normalize_view(doc["text"], view) for doc in docs]
                rows_by_idx: dict[int, dict[str, Any]] = {}
                missing: list[tuple[int, dict[str, Any], str, str]] = []
                for idx, (doc, text) in enumerate(zip(docs, texts)):
                    key = CountCache.key(name, view, doc, adapter.version)
                    cached = cache.get(key)
                    if cached is not None:
                        rows_by_idx[idx] = cached
                    else:
                        missing.append((idx, doc, text, key))

                log.info(
                    "tokenizing %s / %s: %d cached, %d to run",
                    name,
                    view,
                    len(rows_by_idx),
                    len(missing),
                )
                if missing:
                    if adapter.is_remote:
                        max_workers = int(
                            cfg.get("analysis", {}).get("api_workers", 4) or 4
                        )
                        from tqdm import tqdm

                        with ThreadPoolExecutor(max_workers=max_workers) as pool:
                            futures = {
                                pool.submit(make_row, item, adapter, view): item
                                for item in missing
                            }
                            for future in tqdm(
                                as_completed(futures),
                                total=len(futures),
                                desc=f"{name}/{view}",
                                unit="doc",
                            ):
                                item = futures[future]
                                try:
                                    row = future.result()
                                except Exception as exc:
                                    errors.append(
                                        {
                                            "tokenizer": name,
                                            "view": view,
                                            "doc_id": item[1]["doc_id"],
                                            "error": str(exc),
                                        }
                                    )
                                    if strict:
                                        raise
                                    log.error("count failed for %s: %s", item[1]["doc_id"], exc)
                                    continue
                                cache.put(row)
                                rows_by_idx[item[0]] = row
                    else:
                        from tqdm import tqdm

                        for item in tqdm(
                            missing, desc=f"{name}/{view}", unit="doc"
                        ):
                            try:
                                row = make_row(item, adapter, view)
                            except Exception as exc:
                                errors.append(
                                    {
                                        "tokenizer": name,
                                        "view": view,
                                        "doc_id": item[1]["doc_id"],
                                        "error": str(exc),
                                    }
                                )
                                if strict:
                                    raise
                                log.error("count failed for %s: %s", item[1]["doc_id"], exc)
                                continue
                            cache.put(row)
                            rows_by_idx[item[0]] = row

                if strict and len(rows_by_idx) != len(docs):
                    missing_ids = [
                        doc["doc_id"] for idx, doc in enumerate(docs) if idx not in rows_by_idx
                    ]
                    raise RuntimeError(
                        f"{name}/{view}: missing counts for {len(missing_ids)} docs: "
                        f"{missing_ids[:5]}"
                    )
                rows = [rows_by_idx[idx] for idx in sorted(rows_by_idx)]
                if not rows:
                    log.error("%s/%s: no rows, skipping metrics", name, view)
                    continue
                rows_by[(name, view)] = {row["doc_id"]: row for row in rows}
                metrics.extend(
                    aggregate_rows(
                        rows,
                        docs,
                        cfg,
                        tokenizer=name,
                        view=view,
                        class_label=str(spec.get("class", "unknown")),
                        context_window=context_window,
                    )
                )
    finally:
        cache.close()

    # Paired comparisons against the reference tokenizer on identical docs.
    paired: list[dict[str, Any]] = []
    ref_name = str(cfg.get("analysis", {}).get("reference_tokenizer", "aratoken"))
    strata_order: list[str] = []
    for doc in docs:
        if str(doc["stratum"]) not in strata_order:
            strata_order.append(str(doc["stratum"]))
    if ref_name in adapters:
        for tok in adapters:
            if tok == ref_name:
                continue
            for view in views:
                a_map = rows_by.get((tok, view))
                r_map = rows_by.get((ref_name, view))
                if not a_map or not r_map:
                    continue
                common_ids = [d["doc_id"] for d in docs if d["doc_id"] in a_map and d["doc_id"] in r_map]
                for stratum in ["__all__"] + strata_order:
                    ids = (
                        common_ids
                        if stratum == "__all__"
                        else [i for i in common_ids if a_map[i]["stratum"] == stratum]
                    )
                    if not ids:
                        continue
                    pairs = [
                        (
                            int(a_map[i]["tokens"]),
                            int(r_map[i]["tokens"]),
                            int(a_map[i]["words"]),
                        )
                        for i in ids
                    ]
                    stats = paired_bootstrap(
                        pairs,
                        n_boot=int(
                            cfg.get("analysis", {})
                            .get("bootstrap", {})
                            .get("n_boot", 2000)
                        ),
                        ci=float(
                            cfg.get("analysis", {})
                            .get("bootstrap", {})
                            .get("ci", 0.95)
                        ),
                        seed=seed_int(f"paired|{tok}|{ref_name}|{view}|{stratum}"),
                    )
                    if stats:
                        paired.append(
                            {
                                "tokenizer": tok,
                                "reference": ref_name,
                                "view": view,
                                "scope": "all" if stratum == "__all__" else "stratum",
                                "stratum": stratum,
                                **stats,
                            }
                        )
    else:
        log.warning("reference tokenizer %r unavailable; paired comparisons skipped", ref_name)

    # View ablation as a paired comparison on identical documents. This makes
    # H5 checkable: diff < 0 means the target view reduced tokens, and the CI
    # is a document-level bootstrap interval for that difference.
    view_paired: list[dict[str, Any]] = []
    base_view = "raw"
    target_view = "aratoken_norm"
    if base_view in views and target_view in views:
        for tok in adapters:
            a_map = rows_by.get((tok, target_view))
            b_map = rows_by.get((tok, base_view))
            if not a_map or not b_map:
                continue
            common_ids = [
                d["doc_id"]
                for d in docs
                if d["doc_id"] in a_map and d["doc_id"] in b_map
            ]
            for stratum in ["__all__"] + strata_order:
                ids = (
                    common_ids
                    if stratum == "__all__"
                    else [i for i in common_ids if a_map[i]["stratum"] == stratum]
                )
                if not ids:
                    continue
                pairs = [
                    (
                        int(a_map[i]["tokens"]),
                        int(b_map[i]["tokens"]),
                        int(a_map[i]["words"]),
                    )
                    for i in ids
                ]
                stats = paired_bootstrap(
                    pairs,
                    n_boot=int(
                        cfg.get("analysis", {})
                        .get("bootstrap", {})
                        .get("n_boot", 2000)
                    ),
                    ci=float(
                        cfg.get("analysis", {}).get("bootstrap", {}).get("ci", 0.95)
                    ),
                    seed=seed_int(f"view|{tok}|{target_view}|{base_view}|{stratum}"),
                )
                if stats:
                    view_paired.append(
                        {
                            "tokenizer": tok,
                            "from_view": base_view,
                            "to_view": target_view,
                            "scope": "all" if stratum == "__all__" else "stratum",
                            "stratum": stratum,
                            **stats,
                        }
                    )

    summary = summarize_corpus(docs)
    result: dict[str, Any] = {
        "meta": {
            "created_at": now_iso(),
            "input": str(input_path) if input_path else None,
            "n_docs": len(docs),
            "n_words": summary["total_words"],
            "views": views,
            "tokenizers": tokenizer_names,
            "headline_view": str(
                cfg.get("analysis", {}).get("headline_view", "raw")
            ),
            "primary_tokenizers": list(
                cfg.get("analysis", {}).get("primary_tokenizers", [])
            ),
            "secondary_tokenizers": list(
                cfg.get("analysis", {}).get("secondary_tokenizers", [])
            ),
            "illustrative_price_usd_per_1m": float(
                cfg.get("analysis", {}).get("illustrative_price_usd_per_1m", 0.50)
            ),
            "seed": int(cfg.get("project", {}).get("seed", 20261025)),
            "config_path": cfg.get("_config_path"),
            "config_sha256": (
                sha256_file(cfg["_config_path"]) if cfg.get("_config_path") else None
            ),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "versions": package_versions(),
            "reference_tokenizer": ref_name,
        },
        "corpus": summary,
        "tokenizers": statuses,
        "metrics": metrics,
        "paired": paired,
        "view_paired": view_paired,
        "cost": build_cost_rows(cfg, metrics),
        "errors": errors,
    }
    _atomic_write_json(out / "tokenization.json", result)
    _atomic_write_text(out / "TOKENIZATION.md", render_tokenization_md(result, cfg))
    log.info("wrote %s and %s", out / "tokenization.json", out / "TOKENIZATION.md")
    return result


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any, nd: int = 3, dash: str = "—") -> str:
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.{nd}f}"
    return str(value)


def _fmt_ci(ci: Any, nd: int = 3) -> str:
    if not ci or len(ci) < 2 or ci[0] is None or ci[1] is None:
        return "—"
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(cell(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def _metric_index(result: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (m["tokenizer"], m["view"], m["stratum"]): m for m in result.get("metrics", [])
    }


def _ordered_tokenizers(cfg: dict[str, Any], result: dict[str, Any]) -> list[str]:
    analysis = cfg.get("analysis", {})
    names: list[str] = []
    for name in list(analysis.get("primary_tokenizers", [])) + list(
        analysis.get("secondary_tokenizers", [])
    ):
        status = result.get("tokenizers", {}).get(name, {})
        if status.get("status") == "ok" and name not in names:
            names.append(name)
    if not names:
        names = [
            n
            for n, status in result.get("tokenizers", {}).items()
            if status.get("status") == "ok"
        ]
    return names


def _md_tokenization_sections(result: dict[str, Any], cfg: dict[str, Any]) -> str:
    corpus = result.get("corpus", {})
    meta = result.get("meta", {})
    index = _metric_index(result)
    tokenizers = _ordered_tokenizers(cfg, result)
    analysis = cfg.get("analysis", {})
    headline_view = str(meta.get("headline_view", analysis.get("headline_view", "raw")))
    ref_name = str(
        meta.get("reference_tokenizer", analysis.get("reference_tokenizer", "aratoken"))
    )
    total_words = int(corpus.get("total_words", 0))
    parts: list[str] = []

    parts.append("## 1) تكوين الكوربس\n")
    strata_rows: list[list[Any]] = []
    for s in corpus.get("strata", []):
        strata_rows.append(
            [
                s["name"],
                _fmt(s["docs"]),
                _fmt(s["words"]),
                f"{100 * s['arabic_letter_ratio']:.1f}%",
                f"{100 * s['diacritic_ratio']:.1f}%",
                ", ".join(s.get("licenses", [])) or "—",
            ]
        )
    strata_rows.append(
        [
            "**الإجمالي**",
            f"**{_fmt(corpus.get('total_docs'))}**",
            f"**{_fmt(corpus.get('total_words'))}**",
            f"**{100 * corpus.get('arabic_letter_ratio', 0):.1f}%**",
            f"**{100 * corpus.get('diacritic_ratio', 0):.1f}%**",
            "",
        ]
    )
    parts.append(
        _md_table(
            ["Stratum", "Docs", "Words", "Arabic letters", "Diacritics", "License"],
            strata_rows,
        )
    )
    target_words = int(cfg.get("corpus", {}).get("total_words", 50000))
    size_note = (
        " هذا هو حجم التشغيل الكامل المستهدف."
        if total_words == target_words
        else f" هذا تشغيل جزئي (الهدف {_fmt(target_words)} كلمة)؛ لا تعرضه كرقم رئيسي."
    )
    parts.append(
        f"\n> الكوربس المستخدم في هذا التقرير: **{_fmt(total_words)} كلمة** "
        "whitespace. نسبة الحروف العربية ونسبة التشكيل محسوبتان على النص نفسه، "
        f"وليستا تقديراً.{size_note}\n"
    )

    parts.append(f"## 2) النتيجة الأساسية — view = `{headline_view}`\n")
    headline_rows: list[list[Any]] = []
    for name in tokenizers:
        m = index.get((name, headline_view, "__all__"))
        if not m:
            continue
        inflation = None
        ref_m = index.get((ref_name, headline_view, "__all__"))
        if ref_m and ref_m.get("fertility"):
            inflation = m["fertility"] / ref_m["fertility"]
        headline_rows.append(
            [
                name,
                m.get("tokenizer_class", "—"),
                (
                    f"{m['docs']}/{m['expected_docs']}"
                    if m.get("expected_docs") and m["docs"] < m["expected_docs"]
                    else _fmt(m.get("docs"))
                ),
                f"{m['fertility']:.3f}",
                _fmt_ci(m.get("fertility_ci")),
                f"{m['chars_per_token']:.2f}",
                _fmt(int(round(m["tokens_for_target_words"]))),
                f"{inflation:.2f}×" if inflation else "—",
            ]
        )
    parts.append(
        _md_table(
            [
                "Tokenizer",
                "Class",
                "Docs",
                "Fertility",
                "95% CI",
                "Chars/token",
                f"Tokens / {_fmt(total_words)} words",
                f"× {ref_name}",
            ],
            headline_rows,
        )
    )
    if not headline_rows:
        parts.append(
            f"\n> لا توجد نتائج للـ view الأساسي `{headline_view}` في هذا التشغيل. "
            "تحقق من `analysis.views` أو `--views`.\n"
        )
    parts.append(
        "\n**Fertility** = tokens ÷ whitespace words. **CI** = 95% document-level "
        "bootstrap interval (not a model-confidence interval). "
        f"**× {ref_name}** is the token ratio relative to the reference tokenizer "
        "(below 1× means fewer tokens); it is not a price and does not include provider pricing.\n"
    )
    parts.append(
        "\n> **Docs:** لو ظهر `X/Y`، فالـ tokenizer غطّى X مستنداً فقط من Y "
        "(مثلاً `--limit` أو فشل جزئي). في هذه الحالة `Tokens / target words` "
        "إسقاط خطي وليس قياساً كاملاً، والـ CI مبني على المستندات المتاحة فقط.\n"
    )

    parts.append("## 3) حسب الطبقة (per stratum)\n")
    strata_names = [s["name"] for s in corpus.get("strata", [])]
    per_rows: list[list[Any]] = []
    for name in tokenizers:
        row: list[Any] = [name]
        for stratum in strata_names:
            m = index.get((name, headline_view, stratum))
            row.append(f"{m['fertility']:.3f}" if m else "—")
        per_rows.append(row)
    parts.append(_md_table(["Tokenizer"] + strata_names, per_rows))
    parts.append(
        "\n> الطبقات ليست متساوية في الحجم: الفصحى 27,500 كلمة، المصري 10,000، "
        "الخليجي 5,000، المشكّل 7,500. المقارنة داخل كل طبقة، وليس بين الطبقات.\n"
    )
    parts.append(
        "\n> **ملاحظة عن التشكيل:** بعض الـ tokenizers (مثل AraBERT/CAMeLBERT) "
        "تزيل التشكيل داخلياً، لذلك `raw` على الطبقة المشكّلة يظهر قريباً من 1.0. "
        "AraToken لا يزيل التشكيل داخلياً، لذلك `raw` عنده مرتفع، و`aratoken_norm` "
        "هو الـ view الموحّد للمقارنة مع رقم الورقة. لكن التطبيع يحذف التشكيل، "
        "فاستخدم `raw` لقياس تكلفة الإدخال المشكّل نفسه. هذا سلوك الـ tokenizer، "
        "وليس تطبيعاً خارجياً.\n"
    )

    parts.append("## 4) تأثير التطبيع (view ablation)\n")
    view_names = list(meta.get("views", []))
    view_index: dict[tuple[str, str], dict[str, Any]] = {}
    for vp in result.get("view_paired", []):
        if vp.get("scope") == "all":
            view_index[(vp["tokenizer"], vp["from_view"], vp["to_view"])] = vp
    view_rows: list[list[Any]] = []
    for name in tokenizers:
        row = [name]
        for view in view_names:
            m = index.get((name, view, "__all__"))
            row.append(f"{m['fertility']:.3f}" if m else "—")
        vp = view_index.get((name, "raw", "aratoken_norm"))
        if vp:
            row.append(
                f"{vp['diff_tokens_per_word']:+.3f}"
                if vp.get("diff_tokens_per_word") is not None
                else "—"
            )
            row.append(_fmt_ci(vp.get("diff_ci")))
            row.append(
                f"{vp['ratio']:.3f}×" if vp.get("ratio") is not None else "—"
            )
        else:
            row.extend(["—", "—", "—"])
        view_rows.append(row)
    parts.append(
        _md_table(
            ["Tokenizer"] + [f"`{v}`" for v in view_names]
            + ["Δ norm−raw", "95% CI للفرق", "Ratio"],
            view_rows,
        )
    )
    parts.append(
        "\n> `raw` = النص كما هو. `nfkc` = توحيد Unicode فقط. `aratoken_norm` = "
        "خط أنابيب AraToken (NFKC + حذف التشكيل + توحيد الألف + تطبيع الأرقام/الترقيم). "
        "الـ view يُطبَّق قبل الـ tokenizer؛ بعض الـ tokenizers تطبّع داخلياً أيضاً.\n"
        "\n> **Δ** = `aratoken_norm` − `raw` بوحدة tokens/word، محسوب paired على نفس "
        "المستندات. القيمة السالبة تعني أن التطبيع قلّل الـ tokens. الـ CI هو "
        "document-level bootstrap للفرق، وليس CI حول نسبة. `Ratio` = tokens بعد "
        "التطبيع ÷ tokens قبلها.\n"
    )

    parts.append(f"## 5) الفرق paired مقابل `{ref_name}`\n")
    paired_views = [headline_view]
    if (
        "aratoken_norm" in meta.get("views", [])
        and headline_view != "aratoken_norm"
    ):
        paired_views.append("aratoken_norm")
    paired_tables: list[tuple[str, list[list[Any]]]] = []
    for view in paired_views:
        rows: list[list[Any]] = []
        for p in result.get("paired", []):
            if p.get("view") != view or p.get("scope") != "all":
                continue
            if p.get("tokenizer") not in tokenizers:
                continue
            rows.append(
                [
                    p["tokenizer"],
                    f"{p['ratio']:.3f}×" if p.get("ratio") is not None else "—",
                    _fmt_ci(p.get("ratio_ci")),
                    f"{p['diff_tokens_per_word']:+.3f}"
                    if p.get("diff_tokens_per_word") is not None
                    else "—",
                    _fmt_ci(p.get("diff_ci")),
                    _fmt(p.get("n_pairs")),
                ]
            )
        if rows:
            paired_tables.append((view, rows))
    if paired_tables:
        for view, rows in paired_tables:
            if view != headline_view:
                parts.append(f"### view = `{view}`\n")
            parts.append(
                _md_table(
                    [
                        "Tokenizer",
                        "Ratio vs reference",
                        "95% CI",
                        "Δ tokens/word",
                        "95% CI",
                        "n docs",
                    ],
                    rows,
                )
            )
            parts.append("")
        parts.append(
            "\n> الـ bootstrap paired يعيد أخذ نفس المستندات للطرفين، لذلك الـ CI "
            "أضيق من مقارنة عشوائية. النسبة = fertility للـ tokenizer ÷ fertility "
            "للمرجع.\n"
        )
    else:
        parts.append("لا توجد مقارنات paired متاحة (المرجع غير متاح أو لا توجد صفوف).\n")

    parts.append(f"## 6) التكلفة (cost audit) — view = `{headline_view}`\n")
    cost_rows: list[list[Any]] = []
    for c in result.get("cost", []):
        if c["tokenizer"] not in tokenizers:
            continue
        cost_rows.append(
            [
                c["tokenizer"],
                f"${c['input_usd_per_1m']:.4f}" if c.get("input_usd_per_1m") is not None else "—",
                f"${c['cost_usd_per_1m_words']:,.2f}"
                if c.get("cost_usd_per_1m_words") is not None
                else "—",
                f"{c['cost_ratio_vs_reference']:.2f}×"
                if c.get("cost_ratio_vs_reference") is not None
                else "—",
                str(c.get("price_as_of") or "—"),
            ]
        )
    parts.append(
        _md_table(
            ["Tokenizer", "Input $/1M tokens", "$/1M words", "Cost ratio vs ref", "Price as of"],
            cost_rows,
        )
    )
    if not cost_rows:
        parts.append(
            f"\n> لا توجد صفوف تكلفة: الـ view الأساسي `{headline_view}` غير موجود "
            "في هذا التشغيل، أو لا توجد metrics له.\n"
        )
    parts.append(
        "\n> التكلفة = fertility × سعر المليون token. الحساب يستخدم الـ view الأساسي "
        f"`{headline_view}` حتى تبقى الأرقام متسقة مع الجدول الأساسي. "
        "الأسعار الفعلية فارغة افتراضياً لأنها تتغير؛ املأ `pricing` في `config.yaml` "
        "من صفحة المزوّد يوم التشغيل. AraToken ليس له API frontier، لذلك مقارنة تكلفة "
        "الـ API معه هي **counterfactual** وليست سعراً حقيقياً.\n"
    )

    parts.append("## 7) حالة الـ tokenizers\n")
    status_rows: list[list[Any]] = []
    for name, s in result.get("tokenizers", {}).items():
        status_rows.append(
            [
                name,
                s.get("status"),
                s.get("class", "—"),
                s.get("kind", "—"),
                (s.get("error") or "—")[:220],
            ]
        )
    parts.append(_md_table(["Tokenizer", "Status", "Class", "Kind", "Note / error"], status_rows))
    return "\n\n".join(parts)


def render_tokenization_md(result: dict[str, Any], cfg: dict[str, Any]) -> str:
    meta = result.get("meta", {})
    header = (
        "# تقرير التقطيع — Arabic Fertility Tax\n\n"
        f"- **تاريخ التشغيل:** {meta.get('created_at', '—')}\n"
        f"- **المدخل:** `{meta.get('input') or '—'}`\n"
        f"- **المستندات:** {_fmt(meta.get('n_docs'))} — **الكلمات:** {_fmt(meta.get('n_words'))}\n"
        f"- **الـ views:** {', '.join(f'`{v}`' for v in meta.get('views', []))}\n"
        f"- **المرجع:** `{meta.get('reference_tokenizer', '—')}`\n\n"
        "> **اقرأ قبل الأرقام:** fertility هنا مقياس tokenizer فقط. "
        "تكلفة الـ API تعتمد على سعر المزوّد، و TTFT يعتمد على الموديل والشبكة. "
        "لا تخلط الثلاثة. التفاصيل في `docs/METHODOLOGY.md`.\n\n"
    )
    return header + _md_tokenization_sections(result, cfg) + "\n"


# ---------------------------------------------------------------------------
# TTFT arm
# ---------------------------------------------------------------------------


def build_windows(
    docs: list[dict[str, Any]],
    *,
    stratum: str,
    lengths: list[int],
    reps: int,
    seed: int,
) -> list[dict[str, Any]]:
    stratum_docs = [d for d in docs if str(d["stratum"]) == stratum]
    if not stratum_docs:
        raise ValueError(f"TTFT stratum {stratum!r} not found in corpus")
    words: list[str] = []
    for doc in stratum_docs:
        words.extend(doc["text"].split())
    if not words:
        raise ValueError(f"TTFT stratum {stratum!r} is empty")
    max_length = max(lengths)
    if len(words) < max_length:
        raise ValueError(
            f"TTFT needs at least {max_length} words in stratum {stratum!r}; "
            f"found {len(words)}"
        )

    rng = random.Random(seed)
    seen: set[str] = set()
    windows: list[dict[str, Any]] = []
    for length in lengths:
        for rep in range(reps):
            text = ""
            sha = ""
            for _attempt in range(200):
                start = rng.randrange(0, max(1, len(words) - length + 1))
                text = " ".join(words[start : start + length])
                sha = sha256_text(text)
                if sha not in seen:
                    break
            else:
                raise RuntimeError(
                    f"could not build a unique {length}-word window after 200 attempts; "
                    "reduce reps or add more corpus text"
                )
            seen.add(sha)
            windows.append(
                {
                    "length_words": length,
                    "rep": rep,
                    "prompt_id": f"{length}w-r{rep}",
                    "text": text,
                    "sha256": sha,
                }
            )
    return windows


def _estimate_input_tokens(
    estimator: BaseAdapter | None,
    system_prompt: str,
    text: str,
) -> int | None:
    if estimator is None:
        return None
    try:
        return int(estimator.count(f"{system_prompt}\n{text}", "raw")["tokens"])
    except Exception as exc:  # pragma: no cover - estimator is best-effort
        log.debug("input-token estimate failed: %s", exc)
        return None


def run_ttft(
    cfg: dict[str, Any],
    docs: list[dict[str, Any]],
    *,
    out_dir: str | Path,
    provider_names: list[str] | None = None,
    lengths: list[int] | None = None,
    reps: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from providers import ProviderError, build_provider

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ttft_cfg = cfg.get("ttft", {})
    stratum = str(ttft_cfg.get("stratum", "msa"))
    lengths = [int(x) for x in (lengths or ttft_cfg.get("lengths_words", [200, 800, 3200]))]
    reps = int(reps if reps is not None else ttft_cfg.get("reps", 5))
    warmup = int(ttft_cfg.get("warmup", 1))
    max_tokens = int(ttft_cfg.get("max_output_tokens", 1))
    system_prompt = str(ttft_cfg.get("system_prompt", "Reply with the single word OK."))
    seed = int(cfg.get("project", {}).get("seed", 20261025))

    all_providers = list(ttft_cfg.get("providers", []))
    if provider_names:
        # Explicit CLI selection wins over `enabled`, so a one-off run does not
        # require editing config.yaml. The `all` command still uses `enabled`.
        wanted = set(provider_names)
        providers = [p for p in all_providers if p.get("name") in wanted]
        missing = wanted - {str(p.get("name")) for p in providers}
        if missing:
            known = [str(p.get("name")) for p in all_providers]
            log.warning("unknown TTFT provider(s): %s; known: %s", sorted(missing), known)
        if not providers:
            raise RuntimeError(
                f"No TTFT providers matched --providers {sorted(wanted)}. "
                f"Known providers: {[str(p.get('name')) for p in all_providers]}"
            )
    else:
        providers = [p for p in all_providers if p.get("enabled", False)]

    windows = build_windows(
        docs, stratum=stratum, lengths=lengths, reps=reps, seed=seed
    )
    if dry_run:
        plan = {
            "stratum": stratum,
            "lengths_words": lengths,
            "reps": reps,
            "windows": len(windows),
            "warmup": warmup,
            "providers": [
                {
                    "name": p.get("name"),
                    "kind": p.get("kind"),
                    "model": p.get("model"),
                    "base_url": p.get("base_url"),
                }
                for p in providers
            ],
        }
        if not providers:
            plan["warning"] = (
                "No enabled TTFT providers; the plan is valid but an actual run "
                "will fail until you enable one in config.yaml."
            )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    if not providers:
        # Fail loudly instead of silently overwriting a previous ttft.json with
        # an empty summary. `bench.py all` handles this case separately.
        raise RuntimeError(
            "No enabled TTFT providers. Set `enabled: true` for at least one "
            "provider in config.yaml, or pass --providers <name>."
        )

    estimator_name = ttft_cfg.get("input_token_estimator")
    estimator: BaseAdapter | None = None
    if estimator_name:
        try:
            est_adapters, _ = build_adapters(cfg, [str(estimator_name)], strict=False)
            estimator = est_adapters.get(str(estimator_name))
        except Exception as exc:
            log.warning("input-token estimator %r unavailable: %s", estimator_name, exc)

    raw_path = out / "ttft_raw.jsonl"
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{seed_int(now_iso()) & 0xFFFF:04x}"
    raw_fh = open(raw_path, "a", encoding="utf-8", newline="\n")

    def write_row(row: dict[str, Any]) -> None:
        raw_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        raw_fh.flush()

    try:
        for spec in providers:
            name = str(spec.get("name") or spec.get("model"))
            try:
                client = build_provider(spec)
            except Exception as exc:
                write_row(
                    {
                        "run_id": run_id,
                        "timestamp": now_iso(),
                        "provider": name,
                        "kind": spec.get("kind"),
                        "model": spec.get("model"),
                        "base_url": spec.get("base_url"),
                        "region": spec.get("region"),
                        "status": "error",
                        "error": f"provider unavailable: {exc}",
                        "warmup": False,
                    }
                )
                log.error("TTFT provider %s unavailable: %s", name, exc)
                continue

            order = windows.copy()
            random.Random(seed_int(f"ttft|{name}|{run_id}")).shuffle(order)

            if warmup > 0:
                warm = windows[0]
                try:
                    warm_result = client.stream_chat(
                        warm["text"],
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    write_row(
                        {
                            "run_id": run_id,
                            "timestamp": now_iso(),
                            "provider": name,
                            "kind": client.kind,
                            "model": spec.get("model"),
                            "base_url": spec.get("base_url"),
                            "region": spec.get("region"),
                            "length_words": warm["length_words"],
                            "rep": -1,
                            "prompt_id": "warmup",
                            "prompt_sha": warm["sha256"],
                            "prompt_chars": len(warm["text"]) + len(system_prompt),
                            "status": "ok",
                            "warmup": True,
                            "ttft_ms": warm_result["ttft_ms"],
                            "first_event_ms": warm_result.get("first_event_ms"),
                            "first_any_delta_ms": warm_result.get("first_any_delta_ms"),
                            "total_ms": warm_result.get("total_ms"),
                            "input_tokens": warm_result.get("input_tokens"),
                            "output_tokens": warm_result.get("output_tokens"),
                            "cached_tokens": warm_result.get("cached_tokens"),
                            "model_version": warm_result.get("model_version"),
                            "error": None,
                        }
                    )
                except Exception as exc:
                    write_row(
                        {
                            "run_id": run_id,
                            "timestamp": now_iso(),
                            "provider": name,
                            "kind": spec.get("kind"),
                            "model": spec.get("model"),
                            "base_url": spec.get("base_url"),
                            "region": spec.get("region"),
                            "length_words": warm["length_words"],
                            "rep": -1,
                            "prompt_id": "warmup",
                            "prompt_sha": warm["sha256"],
                            "status": "error",
                            "warmup": True,
                            "error": str(exc)[:800],
                        }
                    )
                    log.warning("TTFT warmup failed for %s: %s", name, exc)

            consecutive_errors = 0
            for window in order:
                try:
                    result = client.stream_chat(
                        window["text"],
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=0.0,
                    )
                    input_tokens = result.get("input_tokens")
                    input_est = None
                    source = "reported" if input_tokens is not None else None
                    if input_tokens is None:
                        input_est = _estimate_input_tokens(
                            estimator, system_prompt, window["text"]
                        )
                        source = "estimated" if input_est is not None else None
                    write_row(
                        {
                            "run_id": run_id,
                            "timestamp": now_iso(),
                            "provider": name,
                            "kind": client.kind,
                            "model": spec.get("model"),
                            "model_version": result.get("model_version"),
                            "base_url": spec.get("base_url"),
                            "region": spec.get("region"),
                            "length_words": window["length_words"],
                            "rep": window["rep"],
                            "prompt_id": window["prompt_id"],
                            "prompt_sha": window["sha256"],
                            "prompt_chars": len(window["text"]) + len(system_prompt),
                            "status": "ok",
                            "warmup": False,
                            "ttft_ms": result["ttft_ms"],
                            "first_event_ms": result.get("first_event_ms"),
                            "first_any_delta_ms": result.get("first_any_delta_ms"),
                            "total_ms": result.get("total_ms"),
                            "input_tokens": input_tokens,
                            "input_tokens_est": input_est,
                            "input_tokens_source": source,
                            "output_tokens": result.get("output_tokens"),
                            "cached_tokens": result.get("cached_tokens"),
                            "request_id": result.get("request_id"),
                            "error": None,
                        }
                    )
                    consecutive_errors = 0
                    log.info(
                        "TTFT %s %dw r%d: %.1f ms (%s tokens)",
                        name,
                        window["length_words"],
                        window["rep"],
                        result["ttft_ms"],
                        input_tokens if input_tokens is not None else input_est,
                    )
                except Exception as exc:
                    consecutive_errors += 1
                    write_row(
                        {
                            "run_id": run_id,
                            "timestamp": now_iso(),
                            "provider": name,
                            "kind": spec.get("kind"),
                            "model": spec.get("model"),
                            "base_url": spec.get("base_url"),
                            "region": spec.get("region"),
                            "length_words": window["length_words"],
                            "rep": window["rep"],
                            "prompt_id": window["prompt_id"],
                            "prompt_sha": window["sha256"],
                            "status": "error",
                            "warmup": False,
                            "error": str(exc)[:800],
                        }
                    )
                    log.error(
                        "TTFT %s %dw r%d failed: %s",
                        name,
                        window["length_words"],
                        window["rep"],
                        exc,
                    )
                    if consecutive_errors >= int(ttft_cfg.get("max_consecutive_errors", 3)):
                        log.error("TTFT %s: too many consecutive errors; stopping", name)
                        break
    finally:
        raw_fh.close()

    summary = summarize_ttft(raw_path, cfg, run_id=run_id)
    # Record what actually ran, not what config.yaml says now. CLI overrides
    # (--lengths/--reps) and later config edits must not rewrite history.
    summary["protocol"].update(
        {
            "stratum": stratum,
            "lengths_words": lengths,
            "reps": reps,
            "warmup": warmup,
            "max_output_tokens": max_tokens,
            "system_prompt": system_prompt,
        }
    )
    summary["providers"] = [
        {
            key: p.get(key)
            for key in ("name", "kind", "model", "base_url", "region", "notes")
        }
        for p in providers
    ]
    _atomic_write_json(out / "ttft.json", summary)
    _atomic_write_text(out / "TTFT.md", render_ttft_md(summary))
    log.info("wrote %s and %s", out / "ttft.json", out / "TTFT.md")
    return summary


def ols_fit(x: list[float], y: list[float]) -> dict[str, float] | None:
    n = len(x)
    if n < 2 or n != len(y):
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    if sxx == 0:
        return None
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (intercept + slope * xi)) ** 2 for xi, yi in zip(x, y))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "slope_ms_per_token": slope,
        "intercept_ms": intercept,
        "ttft_at_1k_ms": intercept + slope * 1000.0,
        "r2": r2,
        "n": float(n),
    }


def bootstrap_ols(
    x: list[float],
    y: list[float],
    *,
    n_boot: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    observed = ols_fit(x, y)
    if observed is None:
        return {}
    slopes: list[float] = []
    intercepts: list[float] = []
    at1k: list[float] = []
    if n_boot > 0:
        rng = random.Random(seed)
        n = len(x)
        for _ in range(n_boot):
            idx = [rng.randrange(n) for _ in range(n)]
            fit = ols_fit([x[i] for i in idx], [y[i] for i in idx])
            if fit is None:
                continue
            slopes.append(fit["slope_ms_per_token"])
            intercepts.append(fit["intercept_ms"])
            at1k.append(fit["ttft_at_1k_ms"])
    alpha = (1.0 - ci) / 2.0
    return {
        **observed,
        "slope_ci": [percentile(slopes, alpha * 100), percentile(slopes, (1 - alpha) * 100)],
        "intercept_ci": [
            percentile(intercepts, alpha * 100),
            percentile(intercepts, (1 - alpha) * 100),
        ],
        "ttft_at_1k_ci": [percentile(at1k, alpha * 100), percentile(at1k, (1 - alpha) * 100)],
        "n_boot": len(slopes),
    }


def summarize_ttft(
    raw_path: str | Path,
    cfg: dict[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    rows = list(read_jsonl(raw_path))
    if run_id is not None:
        # ttft_raw.jsonl is append-only across runs. Summarise one run at a
        # time so an old provider/model version cannot leak into the fit.
        rows = [row for row in rows if row.get("run_id") == run_id]
    ttft_cfg = cfg.get("ttft", {})
    groups_map: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    provider_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    errors: list[dict[str, Any]] = []

    for row in rows:
        if row.get("status") != "ok":
            errors.append(
                {
                    "provider": row.get("provider"),
                    "length_words": row.get("length_words"),
                    "rep": row.get("rep"),
                    "error": row.get("error"),
                }
            )
            continue
        if row.get("warmup"):
            continue
        key = (
            str(row.get("provider")),
            str(row.get("model")),
            int(row.get("length_words", 0)),
        )
        groups_map.setdefault(key, []).append(row)
        provider_rows.setdefault(key[:2], []).append(row)

    groups: list[dict[str, Any]] = []
    for (provider, model, length), group in sorted(groups_map.items()):
        ttfts = [float(r["ttft_ms"]) for r in group if r.get("ttft_ms") is not None]
        input_tokens = [
            float(r["input_tokens"])
            for r in group
            if r.get("input_tokens") is not None
        ]
        if not input_tokens:
            input_tokens = [
                float(r["input_tokens_est"])
                for r in group
                if r.get("input_tokens_est") is not None
            ]
        groups.append(
            {
                "provider": provider,
                "model": model,
                "length_words": length,
                "n": len(group),
                "median_ttft_ms": percentile(ttfts, 50),
                "p95_ttft_ms": percentile(ttfts, 95),
                "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
                "median_input_tokens": percentile(input_tokens, 50),
                "min_input_tokens": min(input_tokens) if input_tokens else None,
                "max_input_tokens": max(input_tokens) if input_tokens else None,
            }
        )

    analysis = cfg.get("analysis", {})
    n_boot = int(analysis.get("bootstrap", {}).get("n_boot", 2000))
    ci = float(analysis.get("bootstrap", {}).get("ci", 0.95))
    fits: list[dict[str, Any]] = []
    for (provider, model), group in sorted(provider_rows.items()):
        x: list[float] = []
        y: list[float] = []
        sources: set[str] = set()
        for row in group:
            if row.get("ttft_ms") is None:
                continue
            token_value = row.get("input_tokens")
            source = "reported"
            if token_value is None:
                token_value = row.get("input_tokens_est")
                source = "estimated"
            if token_value is None:
                continue
            x.append(float(token_value))
            y.append(float(row["ttft_ms"]))
            sources.add(source)
        if len(x) < 5 or len(set(x)) < 3:
            continue
        fit = bootstrap_ols(
            x,
            y,
            n_boot=n_boot,
            ci=ci,
            seed=seed_int(f"ttft-fit|{provider}|{model}"),
        )
        if fit:
            fits.append(
                {
                    "provider": provider,
                    "model": model,
                    "n": len(x),
                    "input_token_source": "+".join(sorted(sources)),
                    **fit,
                }
            )

    return {
        "generated_at": now_iso(),
        "run_id": run_id,
        "protocol": {
            "stratum": ttft_cfg.get("stratum", "msa"),
            "lengths_words": [int(x) for x in ttft_cfg.get("lengths_words", [])],
            "reps": int(ttft_cfg.get("reps", 5)),
            "warmup": int(ttft_cfg.get("warmup", 1)),
            "max_output_tokens": int(ttft_cfg.get("max_output_tokens", 1)),
            "system_prompt": ttft_cfg.get("system_prompt"),
            "note": (
                "Sequential streaming requests, one visible output token, "
                "warmup excluded, identical prompt windows across providers."
            ),
        },
        "groups": groups,
        "fits": fits,
        "errors": errors,
        "n_raw_rows": len(rows),
    }


def render_ttft_md(summary: dict[str, Any]) -> str:
    groups = summary.get("groups", [])
    fits = summary.get("fits", [])
    errors = summary.get("errors", [])
    if not groups:
        return (
            "# TTFT — لم يتم تشغيله\n\n"
            "لا توجد صفوف TTFT ناجحة. فعّل مزوّداً واحداً على الأقل في `config.yaml` "
            "أو شغّل `python bench.py ttft --dry-run` للتحقق من الخطة.\n\n"
            + (
                "**أخطاء:**\n\n" + "\n".join(f"- {e}" for e in errors[:10])
                if errors
                else ""
            )
        )
    group_rows = [
        [
            g["provider"],
            g["model"],
            _fmt(g["length_words"]),
            _fmt(g["n"]),
            f"{g['median_ttft_ms']:.1f}" if g["median_ttft_ms"] is not None else "—",
            f"{g['p95_ttft_ms']:.1f}" if g["p95_ttft_ms"] is not None else "—",
            _fmt(int(round(g["median_input_tokens"]))) if g["median_input_tokens"] else "—",
        ]
        for g in groups
    ]
    fit_rows = [
        [
            f["provider"],
            f["model"],
            f"{f['slope_ms_per_token']:.3f}",
            _fmt_ci(f.get("slope_ci"), 3),
            f"{f['intercept_ms']:.1f}",
            f"{f['ttft_at_1k_ms']:.1f}",
            _fmt_ci(f.get("ttft_at_1k_ci"), 1),
            f"{f['r2']:.3f}",
            str(int(f["n"])),
            f.get("input_token_source", "—"),
        ]
        for f in fits
    ]
    parts = [
        "# TTFT — زمن أول token\n",
        "## إعدادات التشغيل\n",
    ]
    protocol = summary.get("protocol", {})
    run_id = summary.get("run_id")
    parts.append(
        "\n".join(
            [
                f"- **Run ID:** `{run_id}`" if run_id else "- **Run ID:** —",
                f"- **Stratum:** `{protocol.get('stratum', '—')}`",
                "- **Lengths (words):** "
                + ", ".join(str(x) for x in protocol.get("lengths_words", [])),
                f"- **Reps per length:** {protocol.get('reps', '—')}",
                f"- **Warmup requests per provider:** {protocol.get('warmup', '—')}",
                f"- **Max output tokens:** {protocol.get('max_output_tokens', '—')}",
                f"- **Generated at:** {summary.get('generated_at', '—')}",
            ]
        )
    )
    provider_meta = summary.get("providers", [])
    if provider_meta:
        parts.append("\n")
        parts.append(
            _md_table(
                ["Provider", "Kind", "Model", "Base URL", "Region"],
                [
                    [
                        p.get("name", "—"),
                        p.get("kind", "—"),
                        p.get("model", "—"),
                        p.get("base_url") or "—",
                        p.get("region") or "—",
                    ]
                    for p in provider_meta
                ],
            )
        )
    parts.append(
        "\n> `model_version` لكل استجابة مسجّل في `ttft_raw.jsonl`؛ لو المزوّد "
        "لا يرجعه يكون `—`.\n"
    )
    parts.append("\n## المتوسطات حسب الطول\n")
    parts.append(
        _md_table(
            [
                "Provider",
                "Model",
                "Words",
                "n",
                "Median TTFT (ms)",
                "p95 TTFT (ms)",
                "Median input tokens",
            ],
            group_rows,
        )
    )
    parts.append("\n## الانحدار الخطي: TTFT ~ input tokens\n")
    if fit_rows:
        parts.append(
            _md_table(
                [
                    "Provider",
                    "Model",
                    "Slope (ms/token)",
                    "95% CI",
                    "Intercept (ms)",
                    "TTFT @1k (ms)",
                    "95% CI",
                    "R²",
                    "n",
                    "Token source",
                ],
                fit_rows,
            )
        )
    else:
        parts.append(
            "لا يوجد انحدار كافٍ (يلزم 5 مشاهدات و3 أطوال مختلفة على الأقل)."
        )
    parts.append(
        "\n> **تحذير تفسيري:** TTFT هنا يقيس الموديل + الشبكة + queueing + "
        "معالجة الـ prompt. هو ليس «تأثير tokenizer» معزولاً. المقارنة بين "
        "مزوّدين مختلفين ليست مقارنة tokenizer فقط. استخدم الـ slope مع "
        "عدد الـ tokens الفعلي، ولا تفترض أن كل الفرق سببه التقطيع.\n"
    )
    if errors:
        parts.append("\n## أخطاء\n")
        parts.append("\n".join(f"- `{e.get('provider')}` / {e.get('length_words')}w: {e.get('error')}" for e in errors[:20]))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------


def _executive_summary(
    tok: dict[str, Any] | None,
    ttft: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> str:
    lines: list[str] = ["## الملخص التنفيذي\n"]
    if tok:
        index = _metric_index(tok)
        analysis = cfg.get("analysis", {})
        meta = tok.get("meta", {})
        headline_view = str(meta.get("headline_view", analysis.get("headline_view", "raw")))
        ref = str(
            meta.get("reference_tokenizer", analysis.get("reference_tokenizer", "aratoken"))
        )
        all_available = [
            (name, index[(name, headline_view, "__all__")])
            for name, status in tok.get("tokenizers", {}).items()
            if status.get("status") == "ok"
            and (name, headline_view, "__all__") in index
        ]
        if all_available:
            # BERT/WordPiece encoders are baselines, not LLM tokenizers. Keep
            # them out of the headline ranking so a low WordPiece count is not
            # mistaken for a cheap generative-model tokenizer.
            generative = [
                (name, metric)
                for name, metric in all_available
                if "bert-baseline"
                not in str(metric.get("tokenizer_class", ""))
            ]
            ranked = generative or all_available
            ranked.sort(key=lambda item: item[1]["fertility"])
            best_name, best = ranked[0]
            worst_name, worst = ranked[-1]
            scope = (
                f"tokenizers الـ LLM المتاحة في `{headline_view}`"
                if generative
                else f"كل الـ tokenizers المتاحة في `{headline_view}`"
            )
            lines.append(
                f"- **أقل fertility بين {scope}:** `{best_name}` عند "
                f"**{best['fertility']:.3f}** tokens/word "
                f"(95% CI {_fmt_ci(best.get('fertility_ci'))})."
            )
            lines.append(
                f"- **أعلى fertility بين {scope}:** `{worst_name}` عند "
                f"**{worst['fertility']:.3f}** tokens/word "
                f"(95% CI {_fmt_ci(worst.get('fertility_ci'))})."
            )
            if headline_view != "aratoken_norm":
                norm_metrics = [
                    (name, index[(name, "aratoken_norm", "__all__")])
                    for name, _ in ranked
                    if (name, "aratoken_norm", "__all__") in index
                ]
                if norm_metrics:
                    norm_metrics.sort(key=lambda item: item[1]["fertility"])
                    nb_name, nb = norm_metrics[0]
                    nw_name, nw = norm_metrics[-1]
                    lines.append(
                        f"- **بعد `aratoken_norm` (view موحّد للمقارنة):** أقل fertility "
                        f"`{nb_name}` عند **{nb['fertility']:.3f}**، وأعلى `{nw_name}` "
                        f"عند **{nw['fertility']:.3f}**. الترتيب يختلف عن `{headline_view}` "
                        "لأن التطبيع وبعض الـ tokenizers يحذفون التشكيل؛ استخدم `raw` "
                        "لقياس الإدخال المشكّل."
                    )
            baselines = [
                (name, metric)
                for name, metric in all_available
                if "bert-baseline" in str(metric.get("tokenizer_class", ""))
            ]
            if baselines:
                baselines.sort(key=lambda item: item[1]["fertility"])
                baseline_text = "، ".join(
                    f"`{name}` {metric['fertility']:.3f}"
                    for name, metric in baselines
                )
                lines.append(
                    f"- **خط الأساس (BERT WordPiece) في `{headline_view}`:** {baseline_text}. "
                    "هذه tokenizers تشفيرية وليست توليدية؛ fertility المنخفضة "
                    "عندها لا تعني تكلفة API أقل."
                )
            if (ref, headline_view, "__all__") in index:
                ref_m = index[(ref, headline_view, "__all__")]
                ratios = [
                    (name, metric["fertility"] / ref_m["fertility"])
                    for name, metric in ranked
                    if name != ref and ref_m.get("fertility")
                ]
                if ratios:
                    lo = min(r for _, r in ratios)
                    hi = max(r for _, r in ratios)
                    lines.append(
                        f"- **النسبة مقابل `{ref}` في `{headline_view}` بين tokenizers الـ LLM:** "
                        f"من **{lo:.2f}×** إلى **{hi:.2f}×** "
                        "في نفس الكوربس (أقل من 1× يعني tokens أقل من المرجع؛ وليست سعراً)."
                    )
        lines.append(
            f"- الكوربس: **{_fmt(tok.get('corpus', {}).get('total_words'))}** كلمة، "
            f"**{_fmt(tok.get('corpus', {}).get('total_docs'))}** مستند، "
            f"بنسبة حروف عربية **{100 * tok.get('corpus', {}).get('arabic_letter_ratio', 0):.1f}%**."
        )
    else:
        lines.append("- لم يتم تشغيل arm التقطيع بعد.")

    if ttft and ttft.get("fits"):
        lines.append(
            f"- **TTFT:** {len(ttft['fits'])} موديل/مزوّد لديه انحدار كافٍ. "
            "الـ slope بالمللي ثانية لكل token يظهر في جدول TTFT."
        )
    else:
        lines.append(
            "- **TTFT:** لا توجد نتائج كافية بعد. شغّل `python bench.py ttft` "
            "بعد تفعيل مزوّد واحد على الأقل."
        )
    total_words = (tok or {}).get("corpus", {}).get("total_words")
    target_words = int(cfg.get("corpus", {}).get("total_words", 50000))
    if total_words:
        if total_words == target_words:
            scale_note = (
                f"هذا التقرير يقيسها على **{_fmt(total_words)}** كلمة، "
                "وهي كل الكوربس المستهدف، بنفس الـ views وبنفس البروتوكول."
            )
        else:
            scale_note = (
                f"هذا التقرير يقيسها على **{_fmt(total_words)}** كلمة فقط "
                f"(الهدف {_fmt(target_words)})، بنفس الـ views وبنفس البروتوكول؛ "
                "لا تعمّمها كرقم كامل."
            )
    else:
        scale_note = "لا توجد نتائج تقطيع في هذا التشغيل."
    lines.append(
        "\n> **الخلاصة لا تُكتب قبل القياس.** الأرقام المنشورة في الأدبيات "
        "(مثل 1.199 لـ AraToken أو 3.0 لـ Llama) ليست أرقام هذا الكوربس. "
        f"{scale_note}\n"
    )
    return "\n".join(lines)


def render_report(
    tok: dict[str, Any] | None,
    ttft: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> str:
    parts: list[str] = [
        "# تقرير Experiment 1 — ضريبة التقطيع العربية (Arabic Fertility Tax)\n",
        f"**تاريخ التقرير:** {now_iso()}  ",
        f"**المشروع:** `{cfg.get('project', {}).get('name', 'arabic-fertility-tax')}`\n",
        "> **اقرأ قبل الأرقام:** هذا التقرير يفصل ثلاثة أشياء: "
        "(1) fertility/compression للـ tokenizer، (2) تكلفة API حسب أسعار حقيقية، "
        "(3) TTFT على مستوى الموديل/المزوّد. لا تخلط بينها. "
        "التفاصيل الكاملة في `docs/METHODOLOGY.md`، والأدبيات في `docs/RESEARCH.md`.\n",
        _executive_summary(tok, ttft, cfg),
    ]
    if tok:
        parts.append(_md_tokenization_sections(tok, cfg))
    else:
        parts.append("## 1) التقطيع\n\nلم يتم تشغيل arm التقطيع بعد.\n")
    if ttft:
        ttft_md = render_ttft_md(ttft)
        ttft_md = ttft_md.replace("# TTFT —", "## 8) TTFT —", 1)
        parts.append(ttft_md)
    else:
        parts.append("## 8) TTFT\n\nلم يتم تشغيل arm الـ TTFT بعد.\n")

    parts.append(
        "## 9) القيود وحدود التفسير\n\n"
        "- **Fertility ليست جودة.** التقطيع الأقل ليس بالضرورة الأفضل للفهم أو التوليد؛ "
        "الأدبيات (مثل Morphemes Without Borders, LREC 2026) تحذر من الربط المباشر بين "
        "التوافق المورفولوجي والأداء.\n"
        "- **AraBERT/CAMeLBERT خط أساس تشفيري.** هما WordPiece BERT tokenizers وليسا "
        "LLM tokenizers؛ fertility المنخفضة عندهما لا تعني تكلفة API أقل ولا جودة أعلى. "
        "استخدمهما كمرجع لغوي، لا كبديل عن tokenizer موديل توليدي.\n"
        "- **تكلفة الـ API ليست ضريبة tokenizer فقط.** السعر لكل token يختلف بين المزوّدين، "
        "وهناك خصومات caching/batch، وفواتير output منفصلة. AraToken ليس له API frontier؛ "
        "مقارنة تكلفته بأي API هي **counterfactual**.\n"
        "- **TTFT ليس تأثير tokenizer معزولاً.** يشمل الشبكة، queueing، معالجة الـ prompt، "
        "وأحياناً thinking. الـ slope مع عدد الـ tokens هو أقرب شيء لتأثير الحجم، وليس "
        "سبباً خالصاً للتقطيع.\n"
        "- **الكوربس غير متوازن عمداً.** الفصحى 27,500 كلمة، المصري 10,000، الخليجي 5,000، "
        "المشكّل 7,500. لا تعمّم النتائج على «العربية» ككل من دون ذكر الطبقة.\n"
        "- **مصادر البيانات مختلفة.** Wikipedia (CC-BY-SA)، dataflare (MIT)، Tashkeela "
        "(GPL-2.0). الكوربس المبني لا يُعاد توزيعه. الـ smoke fixture أصلي (CC0) وليس "
        "جزءاً من الـ 50k.\n"
        "- **قيود الوصول:** Llama 4 gated؛ Gemini tokenizer غير عام (countTokens فقط)؛ "
        "Falcon Arabic tokenizer غير متاح للتحميل العام؛ T100 غير معروف بالضبط.\n"
        "- **الـ CIs وصفية.** هي bootstrap على مستوى المستندات، وليست اختبارات فرض "
        "مع تصحيح multiplicity. لا تقرأ «تداخل الـ CI» كأنه «لا فرق».\n"
    )
    parts.append(
        "## 10) الاستنساخ\n\n"
        "```bash\n"
        "python bench.py doctor --deep\n"
        "python bench.py corpus\n"
        "python bench.py tokenize --tokenizers o200k,qwen3,aratoken,aratoken_lep,arabert,camelbert\n"
        "python bench.py report\n"
        "# اختياري: python bench.py ttft\n"
        "```\n\n"
        "كل الأوامر تقبل `--config config.yaml` **قبل** اسم الأمر. "
        "النتائج الخام في `results/`، والكاش في `results/cache/`. "
        "بصمة الكوربس ومراجعات المصادر في `data/corpus_manifest.json`.\n"
    )
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Doctor
# ---------------------------------------------------------------------------


def package_versions() -> dict[str, str | None]:
    names = [
        "datasets",
        "transformers",
        "sentencepiece",
        "tiktoken",
        "PyYAML",
        "requests",
        "huggingface_hub",
        "tqdm",
        "torch",
        "vllm",
    ]
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_doctor(cfg: dict[str, Any], *, deep: bool = False) -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    py_ok = sys.version_info >= (3, 10)
    add(
        "python",
        "ok" if py_ok else "fail",
        f"{sys.version.split()[0]} ({platform.platform()})",
    )

    versions = package_versions()
    for name in ["datasets", "transformers", "sentencepiece", "tiktoken", "PyYAML", "requests", "huggingface_hub", "tqdm"]:
        version = versions.get(name)
        add(
            f"package:{name}",
            "ok" if version else "warn",
            version or "not installed",
        )
    for name in ["torch", "vllm"]:
        version = versions.get(name)
        add(
            f"optional:{name}",
            "ok" if version else "warn",
            version or "not installed (only needed for local GPU serving)",
        )

    for env_name in [
        "HF_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "VLLM_API_KEY",
    ]:
        present = bool(os.environ.get(env_name))
        add(
            f"env:{env_name}",
            "ok" if present else "warn",
            "set" if present else "not set (only needed for the relevant arm)",
        )

    try:
        gpu_out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        )
        gpus = [line.strip() for line in gpu_out.splitlines() if line.strip()]
        if gpus:
            add("gpu", "ok", "; ".join(gpus))
            joined = " ".join(gpus).lower()
            if "t100" in joined:
                add(
                    "gpu:type",
                    "warn",
                    "T100 is not a standard NVIDIA product name; verify with nvidia-smi "
                    "and record the exact model (T10 16GB? T4? V100?).",
                )
            elif "t10" in joined or "t4" in joined or "v100" in joined:
                add(
                    "gpu:type",
                    "warn",
                    "T10/T4/V100-class hardware: no BF16. Use FP16 or 4-bit; "
                    "FlashAttention-2 may be unavailable. Verify before vLLM.",
                )
        else:
            add("gpu", "warn", "nvidia-smi returned no GPUs")
    except FileNotFoundError:
        add("gpu", "warn", "nvidia-smi not found (fine for CPU-only tokenizer arms)")
    except Exception as exc:
        add("gpu", "warn", f"nvidia-smi check failed: {exc}")

    try:
        usage = shutil.disk_usage(cfg["_base_dir"])
        free_gb = usage.free / (1024**3)
        add(
            "disk",
            "ok" if free_gb >= 10 else "warn",
            f"{free_gb:.1f} GB free in {cfg['_base_dir']}",
        )
    except Exception as exc:
        add("disk", "warn", f"disk check failed: {exc}")

    corpus = resolve_path(cfg, cfg["project"]["corpus_path"])
    smoke = resolve_path(cfg, cfg["project"]["smoke_path"])
    add("corpus", "ok" if corpus.exists() else "warn", str(corpus))
    add("smoke", "ok" if smoke.exists() else "warn", str(smoke))

    pricing = cfg.get("pricing", {}) or {}
    filled = sum(
        1
        for value in pricing.values()
        if isinstance(value, dict) and value.get("input_usd_per_1m") not in (None, "")
    )
    add(
        "pricing",
        "ok" if filled else "warn",
        f"{filled}/{len(pricing)} tokenizer prices filled; fill from provider pages "
        "on the run date for dollar comparisons.",
    )

    if deep:
        names = list(cfg.get("analysis", {}).get("tokenizers", []))
        adapters, statuses = build_adapters(cfg, names, strict=False)
        for name, status in statuses.items():
            add(
                f"tokenizer:{name}",
                "ok" if status.get("status") == "ok" else "warn",
                status.get("error") or status.get("version") or status.get("status", ""),
            )
        add("tokenizers:available", "ok" if adapters else "warn", f"{len(adapters)} ready")

    critical_fail = any(c["status"] == "fail" for c in checks)
    return {
        "generated_at": now_iso(),
        "checks": checks,
        "critical_fail": critical_fail,
    }


def format_doctor(report: dict[str, Any]) -> str:
    marks = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    lines = [f"[{marks.get(c['status'], '??')}] {c['name']}: {c['detail']}" for c in report["checks"]]
    if report.get("critical_fail"):
        lines.append("\nCritical failure: fix the FAIL items before running the benchmark.")
    return "\n".join(lines)




