#!/usr/bin/env python3
"""Command line entry point for the Arabic Fertility Tax experiment.

Examples
--------
    python bench.py doctor --deep
    python bench.py corpus
    python bench.py tokenize
    python bench.py ttft --dry-run
    python bench.py report
    python bench.py all --skip-ttft

`--config` is a global option and must come before the subcommand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import core


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Windows consoles often default to cp1252; Arabic log lines would crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _resolve(cfg: dict[str, Any], value: str | None, default_key: str) -> Path:
    if value:
        return core.resolve_path(cfg, value)
    return core.resolve_path(cfg, cfg["project"][default_key])


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def cmd_doctor(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    report = core.run_doctor(cfg, deep=args.deep)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(core.format_doctor(report))
    return 1 if report.get("critical_fail") else 0


def cmd_corpus(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    if args.smoke:
        path = core.resolve_path(cfg, cfg["project"]["smoke_path"])
        result = core.validate_corpus(path)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"\nSmoke fixture OK: {path} (sha256 {result['sha256'][:16]}…)")
        return 0
    manifest = core.build_corpus(
        cfg,
        out_path=args.out,
        force=args.force,
    )
    print(
        f"Corpus OK: {manifest['total_words']:,} words, "
        f"{manifest['total_docs']:,} docs"
    )
    print(f"File: {manifest['file']}")
    print(f"SHA256: {manifest['file_sha256']}")
    for row in manifest["strata"]:
        print(
            f"  {row['name']}: {row['words']:,} words, {row['docs']:,} docs, "
            f"Arabic {100 * row['arabic_letter_ratio']:.1f}%, "
            f"diacritics {100 * row['diacritic_ratio']:.1f}%"
        )
    return 0


def cmd_tokenize(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    input_path = _resolve(cfg, args.input, "corpus_path")
    if not input_path.exists():
        print(
            f"Input not found: {input_path}\n"
            "Build it first with `python bench.py corpus`, or pass "
            "`--input data/smoke.jsonl --out-dir results/smoke`.",
            file=sys.stderr,
        )
        return 2
    docs = core.load_corpus(input_path)
    tokenizers = _parse_csv(args.tokenizers) or list(cfg["analysis"]["tokenizers"])
    views = _parse_csv(args.views) or list(cfg["analysis"]["views"])
    out_dir = core.resolve_path(cfg, args.out_dir) if args.out_dir else core.resolve_path(
        cfg, cfg["project"]["results_dir"]
    )
    result = core.run_tokenization(
        cfg,
        docs,
        tokenizers,
        views,
        out_dir,
        input_path=input_path,
        limit=args.limit,
        strict=not args.allow_partial,
    )
    print(
        f"Tokenization OK: {result['meta']['n_docs']:,} docs, "
        f"{result['meta']['n_words']:,} words, {len(result['metrics'])} metric rows"
    )
    print(f"Wrote: {out_dir / 'tokenization.json'}")
    print(f"Wrote: {out_dir / 'TOKENIZATION.md'}")
    unavailable = [
        name
        for name, status in result["tokenizers"].items()
        if status.get("status") != "ok"
    ]
    if unavailable:
        print("Unavailable: " + ", ".join(unavailable))
    return 0


def cmd_ttft(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    input_path = _resolve(cfg, args.input, "corpus_path")
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 2
    docs = core.load_corpus(input_path)
    out_dir = core.resolve_path(cfg, args.out_dir) if args.out_dir else core.resolve_path(
        cfg, cfg["project"]["results_dir"]
    )
    try:
        summary = core.run_ttft(
            cfg,
            docs,
            out_dir=out_dir,
            provider_names=_parse_csv(args.providers),
            lengths=_parse_ints(args.lengths),
            reps=args.reps,
            dry_run=args.dry_run,
        )
    except RuntimeError as exc:
        print(f"TTFT failed: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0
    print(
        f"TTFT OK: {len(summary.get('groups', []))} groups, "
        f"{len(summary.get('fits', []))} fits, "
        f"{len(summary.get('errors', []))} errors"
    )
    print(f"Wrote: {out_dir / 'ttft.json'}")
    print(f"Wrote: {out_dir / 'TTFT.md'}")
    if summary.get("errors") and not summary.get("groups"):
        print(
            "TTFT failed: every request errored; inspect ttft_raw.jsonl",
            file=sys.stderr,
        )
        return 3
    return 0


def cmd_report(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    out_dir = core.resolve_path(cfg, args.out_dir) if args.out_dir else core.resolve_path(
        cfg, cfg["project"]["results_dir"]
    )
    tok_path = (
        core.resolve_path(cfg, args.tokenization)
        if args.tokenization
        else out_dir / "tokenization.json"
    )
    ttft_path = (
        core.resolve_path(cfg, args.ttft)
        if args.ttft
        else out_dir / "ttft.json"
    )
    if not tok_path.exists():
        print(f"Tokenization results not found: {tok_path}", file=sys.stderr)
        return 2
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    tok = core.refresh_cost(tok, cfg)
    ttft = json.loads(ttft_path.read_text(encoding="utf-8")) if ttft_path.exists() else None
    markdown = core.render_report(tok, ttft, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    core._atomic_write_text(out_dir / "REPORT.md", markdown)
    core._atomic_write_json(
        out_dir / "report.json",
        {"generated_at": core.now_iso(), "tokenization": tok, "ttft": ttft},
    )
    print(f"Wrote: {out_dir / 'REPORT.md'}")
    print(f"Wrote: {out_dir / 'report.json'}")
    return 0


def cmd_all(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    out_dir = core.resolve_path(cfg, args.out_dir) if args.out_dir else core.resolve_path(
        cfg, cfg["project"]["results_dir"]
    )
    corpus_path = core.resolve_path(cfg, cfg["project"]["corpus_path"])
    if not args.skip_corpus:
        if not corpus_path.exists() or args.force_corpus:
            manifest = core.build_corpus(cfg, out_path=corpus_path, force=args.force_corpus)
            print(
                f"Corpus: {manifest['total_words']:,} words, "
                f"{manifest['total_docs']:,} docs"
            )
        else:
            print(f"Corpus exists: {corpus_path} (use --force-corpus to rebuild)")
    if not corpus_path.exists():
        print("Corpus missing; cannot continue", file=sys.stderr)
        return 2

    docs = core.load_corpus(corpus_path)
    result = core.run_tokenization(
        cfg,
        docs,
        list(cfg["analysis"]["tokenizers"]),
        list(cfg["analysis"]["views"]),
        out_dir,
        input_path=corpus_path,
        strict=not args.allow_partial,
    )
    print(
        f"Tokenization: {len(result['metrics'])} metric rows -> "
        f"{out_dir / 'tokenization.json'}"
    )

    ttft_ran = False
    if not args.skip_ttft:
        enabled = [
            p for p in cfg.get("ttft", {}).get("providers", []) if p.get("enabled")
        ]
        if not enabled:
            print("TTFT skipped: no enabled providers in config.yaml")
        else:
            summary = core.run_ttft(cfg, docs, out_dir=out_dir)
            ttft_ran = True
            print(
                f"TTFT: {len(summary.get('groups', []))} groups, "
                f"{len(summary.get('fits', []))} fits, "
                f"{len(summary.get('errors', []))} errors"
            )
    else:
        print("TTFT skipped")

    tok_path = out_dir / "tokenization.json"
    ttft_path = out_dir / "ttft.json"
    tok = json.loads(tok_path.read_text(encoding="utf-8"))
    # Only include TTFT in the combined report when this run produced it.
    # Otherwise a stale ttft.json from an earlier run would silently leak into
    # the new report. `bench.py report` can still combine files explicitly.
    if not ttft_ran and ttft_path.exists():
        print(f"Note: {ttft_path} exists but was not produced by this run; not included")
    ttft = (
        json.loads(ttft_path.read_text(encoding="utf-8"))
        if ttft_ran and ttft_path.exists()
        else None
    )
    markdown = core.render_report(tok, ttft, cfg)
    core._atomic_write_text(out_dir / "REPORT.md", markdown)
    core._atomic_write_json(
        out_dir / "report.json",
        {"generated_at": core.now_iso(), "tokenization": tok, "ttft": ttft},
    )
    print(f"Report: {out_dir / 'REPORT.md'}")
    if ttft_ran and ttft and ttft.get("errors") and not ttft.get("groups"):
        print(
            "TTFT failed: every request errored; inspect ttft_raw.jsonl",
            file=sys.stderr,
        )
        return 3
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description="Arabic Fertility Tax: tokenizer, cost and TTFT benchmark.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: %(default)s)",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="Check Python, packages, keys, GPU and tokenizers")
    p.add_argument("--deep", action="store_true", help="Try loading every tokenizer")
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("corpus", help="Build or validate the corpus")
    p.add_argument("--smoke", action="store_true", help="Validate data/smoke.jsonl only")
    p.add_argument("--force", action="store_true", help="Overwrite an existing corpus")
    p.add_argument("--out", default=None, help="Output JSONL path")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("tokenize", help="Run intrinsic tokenizer metrics")
    p.add_argument("--input", default=None, help="Input JSONL (default: corpus)")
    p.add_argument("--tokenizers", default=None, help="Comma-separated names")
    p.add_argument("--views", default=None, help="Comma-separated views: raw,nfkc,aratoken_norm")
    p.add_argument("--out-dir", default=None, help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Only the first N docs")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Continue if a document fails (default: fail fast)",
    )
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("ttft", help="Run the streaming TTFT benchmark")
    p.add_argument("--input", default=None, help="Input JSONL (default: corpus)")
    p.add_argument("--providers", default=None, help="Comma-separated provider names")
    p.add_argument("--lengths", default=None, help="Comma-separated word lengths")
    p.add_argument("--reps", type=int, default=None, help="Repetitions per length")
    p.add_argument("--out-dir", default=None, help="Output directory")
    p.add_argument("--dry-run", action="store_true", help="Print the plan; no API calls")
    p.set_defaults(func=cmd_ttft)

    p = sub.add_parser("report", help="Combine tokenization and TTFT into REPORT.md")
    p.add_argument("--out-dir", default=None, help="Directory with tokenization.json")
    p.add_argument("--tokenization", default=None, help="Explicit tokenization.json")
    p.add_argument("--ttft", default=None, help="Explicit ttft.json (optional)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("all", help="Corpus -> tokenize -> TTFT -> report")
    p.add_argument("--skip-corpus", action="store_true")
    p.add_argument("--force-corpus", action="store_true")
    p.add_argument("--skip-ttft", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--out-dir", default=None)
    p.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    try:
        cfg = core.load_config(args.config)
        return int(args.func(args, cfg))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        if args.verbose:
            logging.exception("command failed")
        else:
            logging.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
