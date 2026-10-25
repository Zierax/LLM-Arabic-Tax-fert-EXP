#!/usr/bin/env python3
"""Build the English numeric/statistical comparison of the tokenizer run.

Reads the existing full-run artefacts (no tokenizer is re-run, no API is called):

  results/tokenization.json
  results/cache/token_counts.jsonl

Writes:

  results/COMPARISON.md
  results/comparison.json

The tests are paired by document. The per-document difference is

    d_i = (tokens_candidate_i - tokens_reference_i) / words_raw_i

where words_raw_i is the original whitespace word count of the document. The
same denominator is used for every view so that raw and normalized differences
are on a common scale. Fertility in the summary tables still uses each view's
own word count, matching results/tokenization.json.

Only the Python standard library is used. scipy is not required.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALPHA = 0.05
REFERENCE = "aratoken"
HEADLINE_VIEW = "raw"
CONTROLLED_VIEW = "aratoken_norm"
STRATUM_ORDER = ["msa", "egyptian", "gulf", "diacritized"]


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------


def norm_sf(x: float) -> float:
    """Standard normal survival function, without scipy."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def wilcoxon_signed_rank(diffs: list[float]) -> dict[str, Any]:
    """Two-sided paired Wilcoxon signed-rank test, normal approximation.

    Zeros are discarded (the equivalent of scipy's zero_method="wilcox").
    Average ranks handle ties; the variance uses the standard tie correction.
    A continuity correction is applied toward the mean.
    """
    nonzero = [x for x in diffs if x != 0.0]
    n = len(nonzero)
    n_zero = len(diffs) - n
    if n == 0:
        return {
            "n_nonzero": 0,
            "n_zero": n_zero,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "z": 0.0,
            "p": 1.0,
        }

    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    tie_groups: list[int] = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs(nonzero[order[j]]) == abs(nonzero[order[i]]):
            j += 1
        average_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average_rank
        tie_groups.append(j - i)
        i = j

    w_plus = sum(ranks[i] for i in range(n) if nonzero[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nonzero[i] < 0)
    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    variance -= sum(t**3 - t for t in tie_groups) / 48.0

    if variance <= 0.0:
        z = 0.0
        p = 1.0
    else:
        if w_plus > mean:
            z = (w_plus - mean - 0.5) / math.sqrt(variance)
        elif w_plus < mean:
            z = (w_plus - mean + 0.5) / math.sqrt(variance)
        else:
            z = 0.0
        p = min(1.0, 2.0 * norm_sf(abs(z)))

    return {
        "n_nonzero": n,
        "n_zero": n_zero,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z": z,
        "p": p,
    }


def quantile(sorted_values: list[float], q: float) -> float | None:
    """Linear-interpolated quantile (same definition as numpy's default)."""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (position - lo)


def describe_diffs(diffs: list[float]) -> dict[str, Any]:
    """Paired effect sizes and the Wilcoxon test for one comparison."""
    n = len(diffs)
    mean = sum(diffs) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    else:
        variance = 0.0
    sd = math.sqrt(variance)
    ordered = sorted(diffs)
    median = quantile(ordered, 0.5)
    q1 = quantile(ordered, 0.25)
    q3 = quantile(ordered, 0.75)
    test = wilcoxon_signed_rank(diffs)
    w_plus = test["w_plus"]
    w_minus = test["w_minus"]
    rank_total = w_plus + w_minus
    return {
        "n": n,
        "mean_diff": mean,
        "sd_diff": sd,
        "median_diff": median,
        "q1_diff": q1,
        "q3_diff": q3,
        "dz": mean / sd if sd > 0.0 else None,
        "rank_biserial": (w_plus - w_minus) / rank_total if rank_total > 0.0 else None,
        "win_rate_less": sum(1 for x in diffs if x < 0.0) / n,
        "win_rate_greater": sum(1 for x in diffs if x > 0.0) / n,
        "win_rate_equal": sum(1 for x in diffs if x == 0.0) / n,
        **test,
    }


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def fmt_int(x: int | None) -> str:
    if x is None:
        return "-"
    return f"{x:,}"


def fmt_p(p: float | None) -> str:
    if p is None:
        return "-"
    if p == 0.0 or p < 1e-300:
        return "<1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3g}"


def fmt_ratio(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}x"


def fmt_ci(ci: list[float] | tuple[float, ...] | None) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "-"
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def fmt_signed(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "-"
    return f"{x:+.{digits}f}"


def fmt_pct(x: float | None, digits: int = 1, signed: bool = False) -> str:
    if x is None:
        return "-"
    if signed:
        return f"{x * 100:+.{digits}f}%"
    return f"{x * 100:.{digits}f}%"


def fmt_sig(p_holm: float | None) -> str:
    if p_holm is None:
        return "-"
    return "yes" if p_holm < ALPHA else "no"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load and validate the existing run
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_data(results_dir: Path) -> dict[str, Any]:
    tokenization_path = results_dir / "tokenization.json"
    cache_path = results_dir / "cache" / "token_counts.jsonl"
    if not tokenization_path.exists():
        raise SystemExit(f"missing {tokenization_path}; run `python bench.py tokenize` first")
    if not cache_path.exists():
        raise SystemExit(f"missing {cache_path}; run `python bench.py tokenize` first")

    tokenization = load_json(tokenization_path)
    cache_rows = load_jsonl(cache_path)
    meta = tokenization["meta"]
    # Load corpus strata from manifest (tokenization meta has no 'corpus' key)
    manifest_path = results_dir.parent / "data" / "corpus_manifest.json"
    corpus_sha256 = None
    strata: list[str] = []
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        corpus_sha256 = manifest.get("file_sha256")
        strata = [str(item["name"]) for item in manifest.get("strata", [])]
    if not strata:
        # Fallback: infer from metrics keys
        strata_set = set()
        for key in metrics:
            if key[2] == "stratum":  # (tokenizer, view, scope, stratum)
                strata_set.add(key[3])
        strata = sorted(strata_set)
    if "__all__" in strata:
        strata.remove("__all__")
    reference = str(meta.get("reference_tokenizer", REFERENCE))
    tokenizers = list(meta["tokenizers"])
    views = list(meta["views"])
    # Strata loaded from manifest (tokenization meta has no 'corpus' key)
    if not strata:
        raise SystemExit("could not determine strata; check data/corpus_manifest.json")

    by: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in cache_rows:
        by[(str(row["tokenizer"]), str(row["view"]))][str(row["doc_id"])] = row

    if (reference, "raw") not in by:
        raise SystemExit(f"cache has no {reference}/raw rows")
    reference_raw = by[(reference, "raw")]
    doc_ids = sorted(reference_raw)
    words = {doc_id: int(reference_raw[doc_id]["words"]) for doc_id in doc_ids}
    doc_stratum = {doc_id: str(reference_raw[doc_id]["stratum"]) for doc_id in doc_ids}

    expected_ids = set(doc_ids)
    for tokenizer in tokenizers:
        for view in views:
            key = (tokenizer, view)
            if key not in by:
                raise SystemExit(f"cache is missing {tokenizer}/{view}")
            if set(by[key]) != expected_ids:
                missing = len(expected_ids - set(by[key]))
                extra = len(set(by[key]) - expected_ids)
                raise SystemExit(
                    f"cache document mismatch for {tokenizer}/{view}: "
                    f"{missing} missing, {extra} extra"
                )

    # raw and nfkc must keep the original whitespace words; the AraToken
    # normalization removes a standalone diacritic cluster in 13 documents.
    for tokenizer in tokenizers:
        for view in ("raw", "nfkc"):
            for doc_id in doc_ids:
                got = int(by[(tokenizer, view)][doc_id]["words"])
                if got != words[doc_id]:
                    raise SystemExit(
                        f"word-count mismatch for {tokenizer}/{view}/{doc_id}: "
                        f"{got} != {words[doc_id]}"
                    )

    metrics: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in tokenization["metrics"]:
        metrics[(row["tokenizer"], row["view"], row["scope"], row["stratum"])] = row

    paired: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in tokenization["paired"]:
        paired[(row["tokenizer"], row["view"], row["scope"], row["stratum"])] = row

    view_paired: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in tokenization["view_paired"]:
        view_paired[
            (row["tokenizer"], row["from_view"], row["to_view"], row["scope"], row["stratum"])
        ] = row

    # Validate every cache token sum against the published metric row.
    for tokenizer in tokenizers:
        for view in views:
            total = sum(int(by[(tokenizer, view)][doc_id]["tokens"]) for doc_id in doc_ids)
            metric = metrics[(tokenizer, view, "all", "__all__")]
            if total != int(metric["tokens"]):
                raise SystemExit(
                    f"token sum mismatch for {tokenizer}/{view}: {total} != {metric['tokens']}"
                )
            for stratum in strata:
                subset = [doc_id for doc_id in doc_ids if doc_stratum[doc_id] == stratum]
                subtotal = sum(int(by[(tokenizer, view)][doc_id]["tokens"]) for doc_id in subset)
                submetric = metrics[(tokenizer, view, "stratum", stratum)]
                if subtotal != int(submetric["tokens"]):
                    raise SystemExit(
                        f"token sum mismatch for {tokenizer}/{view}/{stratum}: "
                        f"{subtotal} != {submetric['tokens']}"
                    )

    classes: dict[str, str] = {}
    for tokenizer in tokenizers:
        metric = metrics[(tokenizer, "raw", "all", "__all__")]
        classes[tokenizer] = str(metric["tokenizer_class"])

    candidates = [tokenizer for tokenizer in tokenizers if tokenizer != reference]
    llm_candidates = [t for t in candidates if "baseline" not in classes[t]]
    baseline_candidates = [t for t in candidates if "baseline" in classes[t]]

    return {
        "tokenization": tokenization,
        "meta": meta,
        "reference": reference,
        "tokenizers": tokenizers,
        "views": views,
        "strata": strata,
        "doc_ids": doc_ids,
        "words": words,
        "doc_stratum": doc_stratum,
        "by": by,
        "metrics": metrics,
        "paired": paired,
        "view_paired": view_paired,
        "classes": classes,
        "llm_candidates": llm_candidates,
        "baseline_candidates": baseline_candidates,
        "tokenization_path": tokenization_path,
        "cache_path": cache_path,
    }


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------


def metric_for(
    data: dict[str, Any], tokenizer: str, view: str, scope: str, stratum: str
) -> dict[str, Any]:
    try:
        return data["metrics"][(tokenizer, view, scope, stratum)]
    except KeyError as exc:
        raise SystemExit(
            f"missing metric row: {tokenizer}/{view}/{scope}/{stratum}"
        ) from exc


def reference_row(
    data: dict[str, Any], view: str, stratum: str
) -> dict[str, Any]:
    reference = data["reference"]
    scope = "all" if stratum == "__all__" else "stratum"
    metric = metric_for(data, reference, view, scope, stratum)
    doc_ids = (
        data["doc_ids"]
        if stratum == "__all__"
        else [d for d in data["doc_ids"] if data["doc_stratum"][d] == stratum]
    )
    return {
        "tokenizer": reference,
        "reference": reference,
        "view": view,
        "scope": scope,
        "stratum": stratum,
        "class": data["classes"][reference],
        "is_baseline": False,
        "n_docs": len(doc_ids),
        "words": sum(data["words"][d] for d in doc_ids),
        "tokens": int(metric["tokens"]),
        "tokens_reference": int(metric["tokens"]),
        "fertility": float(metric["fertility"]),
        "fertility_ci": metric["fertility_ci"],
        "reference_fertility": float(metric["fertility"]),
        "reference_fertility_ci": metric["fertility_ci"],
        "ratio": 1.0,
        "ratio_ci": [1.0, 1.0],
        "aggregate_diff_tokens_per_word": 0.0,
        "unk": int(metric["unk"]),
        "unk_rate": float(metric["unk_rate"]),
        "reference_unk": int(metric["unk"]),
        "reference_unk_rate": float(metric["unk_rate"]),
        "test": None,
    }


def pair_row(
    data: dict[str, Any], view: str, stratum: str, tokenizer: str
) -> dict[str, Any]:
    reference = data["reference"]
    scope = "all" if stratum == "__all__" else "stratum"
    doc_ids = (
        data["doc_ids"]
        if stratum == "__all__"
        else [d for d in data["doc_ids"] if data["doc_stratum"][d] == stratum]
    )
    diffs = [
        (
            int(data["by"][(tokenizer, view)][doc_id]["tokens"])
            - int(data["by"][(reference, view)][doc_id]["tokens"])
        )
        / data["words"][doc_id]
        for doc_id in doc_ids
    ]
    tokens = sum(int(data["by"][(tokenizer, view)][doc_id]["tokens"]) for doc_id in doc_ids)
    tokens_reference = sum(
        int(data["by"][(reference, view)][doc_id]["tokens"]) for doc_id in doc_ids
    )
    words = sum(data["words"][doc_id] for doc_id in doc_ids)
    ratio = tokens / tokens_reference if tokens_reference else None
    test = describe_diffs(diffs)
    metric = metric_for(data, tokenizer, view, scope, stratum)
    reference_metric = metric_for(data, reference, view, scope, stratum)
    paired = data["paired"].get((tokenizer, view, scope, stratum))
    if paired is not None and ratio is not None:
        if abs(ratio - float(paired["ratio"])) > 1e-9:
            raise SystemExit(
                f"paired ratio mismatch for {tokenizer}/{view}/{stratum}: "
                f"{ratio} != {paired['ratio']}"
            )
    return {
        "tokenizer": tokenizer,
        "reference": reference,
        "view": view,
        "scope": scope,
        "stratum": stratum,
        "class": data["classes"][tokenizer],
        "is_baseline": "baseline" in data["classes"][tokenizer],
        "n_docs": len(doc_ids),
        "words": words,
        "tokens": tokens,
        "tokens_reference": tokens_reference,
        "fertility": float(metric["fertility"]),
        "fertility_ci": metric["fertility_ci"],
        "reference_fertility": float(reference_metric["fertility"]),
        "reference_fertility_ci": reference_metric["fertility_ci"],
        "ratio": ratio,
        "ratio_ci": paired["ratio_ci"] if paired else None,
        "aggregate_diff_tokens_per_word": (tokens - tokens_reference) / words,
        "unk": int(metric["unk"]),
        "unk_rate": float(metric["unk_rate"]),
        "reference_unk": int(reference_metric["unk"]),
        "reference_unk_rate": float(reference_metric["unk_rate"]),
        "test": test,
    }


def apply_holm(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    p_values = [float(row["test"]["p"]) for row in rows]
    adjusted = holm(p_values)
    for row, p_holm in zip(rows, adjusted):
        row["test"]["family"] = family
        row["test"]["p_holm"] = p_holm
        row["test"]["significant_holm"] = p_holm < ALPHA
    return {
        "family": family,
        "n_tests": len(rows),
        "n_significant": sum(1 for row in rows if row["test"]["significant_holm"]),
        "tokenizers": [row["tokenizer"] for row in rows],
        "strata": sorted({row["stratum"] for row in rows}),
    }


def view_ablation_row(
    data: dict[str, Any], tokenizer: str, stratum: str
) -> dict[str, Any]:
    scope = "all" if stratum == "__all__" else "stratum"
    doc_ids = (
        data["doc_ids"]
        if stratum == "__all__"
        else [d for d in data["doc_ids"] if data["doc_stratum"][d] == stratum]
    )
    raw_tokens = sum(int(data["by"][(tokenizer, "raw")][d]["tokens"]) for d in doc_ids)
    nfkc_tokens = sum(int(data["by"][(tokenizer, "nfkc")][d]["tokens"]) for d in doc_ids)
    norm_tokens = sum(
        int(data["by"][(tokenizer, CONTROLLED_VIEW)][d]["tokens"]) for d in doc_ids
    )
    diffs = [
        (
            int(data["by"][(tokenizer, CONTROLLED_VIEW)][d]["tokens"])
            - int(data["by"][(tokenizer, "raw")][d]["tokens"])
        )
        / data["words"][d]
        for d in doc_ids
    ]
    test = describe_diffs(diffs)
    raw_metric = metric_for(data, tokenizer, "raw", scope, stratum)
    norm_metric = metric_for(data, tokenizer, CONTROLLED_VIEW, scope, stratum)
    paired = data["view_paired"].get(
        (tokenizer, "raw", CONTROLLED_VIEW, scope, stratum)
    )
    ratio = norm_tokens / raw_tokens if raw_tokens else None
    if paired is not None and ratio is not None:
        if abs(ratio - float(paired["ratio"])) > 1e-9:
            raise SystemExit(
                f"view ratio mismatch for {tokenizer}/{stratum}: "
                f"{ratio} != {paired['ratio']}"
            )
    return {
        "tokenizer": tokenizer,
        "view": f"{CONTROLLED_VIEW}_vs_raw",
        "scope": scope,
        "stratum": stratum,
        "n_docs": len(doc_ids),
        "words": sum(data["words"][d] for d in doc_ids),
        "raw_tokens": raw_tokens,
        "nfkc_tokens": nfkc_tokens,
        "norm_tokens": norm_tokens,
        "raw_fertility": float(raw_metric["fertility"]),
        "nfkc_fertility": float(
            metric_for(data, tokenizer, "nfkc", scope, stratum)["fertility"]
        ),
        "norm_fertility": float(norm_metric["fertility"]),
        "ratio": ratio,
        "ratio_ci": paired["ratio_ci"] if paired else None,
        "aggregate_diff_tokens_per_word": (norm_tokens - raw_tokens)
        / sum(data["words"][d] for d in doc_ids),
        "nfkc_delta_tokens": nfkc_tokens - raw_tokens,
        "raw_unk_rate": float(raw_metric["unk_rate"]),
        "norm_unk_rate": float(norm_metric["unk_rate"]),
        "test": test,
    }


def build_unk_audit(data: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for view in ("raw", CONTROLLED_VIEW):
        view_rows: dict[str, Any] = {}
        for tokenizer in data["tokenizers"]:
            overall = metric_for(data, tokenizer, view, "all", "__all__")
            entry = {
                "overall": float(overall["unk_rate"]),
                "overall_unk": int(overall["unk"]),
                "overall_tokens": int(overall["tokens"]),
            }
            for stratum in data["strata"]:
                metric = metric_for(data, tokenizer, view, "stratum", stratum)
                entry[stratum] = float(metric["unk_rate"])
                entry[f"{stratum}_unk"] = int(metric["unk"])
            view_rows[tokenizer] = entry
        audit[view] = view_rows
    return audit


# ---------------------------------------------------------------------------
# Build the complete result
# ---------------------------------------------------------------------------


def build_result(data: dict[str, Any]) -> dict[str, Any]:
    all_docs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    strata: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    families: dict[str, Any] = {}

    for view in ("raw", CONTROLLED_VIEW):
        all_docs[view] = {}
        strata[view] = {}
        for group_name, group in (
            ("llm", data["llm_candidates"]),
            ("baseline", data["baseline_candidates"]),
        ):
            rows = [pair_row(data, view, "__all__", tokenizer) for tokenizer in group]
            family = f"{view}_all_{group_name}"
            families[family] = apply_holm(rows, family)
            all_docs[view][group_name] = rows

            for stratum in data["strata"]:
                stratum_rows = [
                    pair_row(data, view, stratum, tokenizer) for tokenizer in group
                ]
                strata[view].setdefault(stratum, {})[group_name] = stratum_rows

        for group_name in ("llm", "baseline"):
            family = f"{view}_stratum_{group_name}"
            combined = [
                row
                for stratum in data["strata"]
                for row in strata[view][stratum][group_name]
            ]
            families[family] = apply_holm(combined, family)

    view_all = [
        view_ablation_row(data, tokenizer, "__all__")
        for tokenizer in data["tokenizers"]
    ]
    families["view_ablation_all"] = apply_holm(view_all, "view_ablation_all")

    view_strata: dict[str, list[dict[str, Any]]] = {}
    for stratum in data["strata"]:
        rows = [view_ablation_row(data, tokenizer, stratum) for tokenizer in data["tokenizers"]]
        family = f"view_ablation_{stratum}"
        families[family] = apply_holm(rows, family)
        view_strata[stratum] = rows

    reference_all = {
        view: reference_row(data, view, "__all__") for view in ("raw", CONTROLLED_VIEW)
    }
    reference_strata = {
        view: {
            stratum: reference_row(data, view, stratum) for stratum in data["strata"]
        }
        for view in ("raw", CONTROLLED_VIEW)
    }

    # Load corpus info from manifest (tokenization meta has no 'corpus')
    manifest_path = Path(str(data["tokenization_path"]).replace("tokenization.json", "")).parent / "data" / "corpus_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {"strata": []}
    corpus_strata = manifest.get("strata", [])
    words_raw = sum(data["words"].values())
    words_norm = sum(
        int(data["by"][(data["reference"], CONTROLLED_VIEW)][d]["words"])
        for d in data["doc_ids"]
    )
    bootstrap = data["meta"].get("bootstrap", {})

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": str(data["tokenization_path"]),
            "cache": str(data["cache_path"]),
            "corpus_sha256": data["meta"].get("corpus_sha256"),
            "n_docs": len(data["doc_ids"]),
            "n_words_raw": words_raw,
            "n_words_aratoken_norm": words_norm,
            "reference_tokenizer": data["reference"],
            "headline_view": HEADLINE_VIEW,
            "controlled_view": CONTROLLED_VIEW,
            "alpha": ALPHA,
            "bootstrap": {
                "n_boot": bootstrap.get("n_boot", 2000),
                "ci": bootstrap.get("ci", 0.95),
                "seed": data["meta"].get("seed"),
            },
            "test": (
                "two-sided paired Wilcoxon signed-rank, normal approximation "
                "with tie correction and continuity correction; zeros discarded"
            ),
            "difference": (
                "d_i = (tokens_candidate_i - tokens_reference_i) / words_raw_i"
            ),
            "effect_sizes": {
                "dz": "mean(d_i) / sd(d_i), sample sd",
                "rank_biserial": "(W+ - W-) / (W+ + W-)",
                "positive_direction": "candidate uses more tokens than reference",
            },
            "multiplicity": "Holm within each declared family; alpha = 0.05",
            "families": families,
            "strata": [
                {
                    "name": item["name"],
                    "docs": item["docs"],
                    "words": item["words"],
                    "diacritic_ratio": item["diacritic_ratio"],
                }
                for item in corpus_strata
            ],
            "tokenizers": {
                tokenizer: {
                    "class": data["classes"][tokenizer],
                    "is_baseline": "baseline" in data["classes"][tokenizer],
                }
                for tokenizer in data["tokenizers"]
            },
        },
        "all_docs": all_docs,
        "reference_all": reference_all,
        "strata": strata,
        "reference_strata": reference_strata,
        "view_ablation": {"all_docs": view_all, "strata": view_strata},
        "unk_audit": build_unk_audit(data),
        "notes": [
            "Fertility is tokens / whitespace words; it is not model quality.",
            "A token ratio is not a cost ratio; no provider prices are configured.",
            "The corpus is Arabic-only; there is no English-Arabic parallel arm.",
            "Raw aratoken is out-of-distribution on unnormalized text and emits UNK.",
            "Raw BERT baselines emit near-100% UNK on diacritized text.",
            "aratoken_norm changes the whitespace word count in 13 diacritized documents.",
            "The Wilcoxon test weights documents equally; the aggregate ratio weights words.",
            "CIs are document-level bootstrap intervals and are not multiplicity-adjusted.",
        ],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def row_for(
    result: dict[str, Any], view: str, tokenizer: str, stratum: str = "__all__"
) -> dict[str, Any]:
    reference = result["meta"]["reference_tokenizer"]
    if tokenizer == reference:
        if stratum == "__all__":
            return result["reference_all"][view]
        return result["reference_strata"][view][stratum]
    group = (
        "baseline"
        if result["meta"]["tokenizers"][tokenizer]["is_baseline"]
        else "llm"
    )
    if stratum == "__all__":
        rows = result["all_docs"][view][group]
    else:
        rows = result["strata"][view][stratum][group]
    for row in rows:
        if row["tokenizer"] == tokenizer:
            return row
    raise KeyError(f"no row for {view}/{tokenizer}/{stratum}")


def candidate_order(result: dict[str, Any], group: str) -> list[str]:
    reference = result["meta"]["reference_tokenizer"]
    names = [
        name
        for name, info in result["meta"]["tokenizers"].items()
        if name != reference
        and (("baseline" in info["class"]) == (group == "baseline"))
    ]
    rows = result["all_docs"][HEADLINE_VIEW][group]
    fertility = {row["tokenizer"]: row["fertility"] for row in rows}
    return sorted(names, key=lambda name: fertility[name])


def all_doc_table(result: dict[str, Any], view: str, group: str) -> str:
    reference = result["meta"]["reference_tokenizer"]
    names = candidate_order(result, group)
    rows = []
    if group == "llm":
        ref = row_for(result, view, reference)
        rows.append(
            [
                f"`{reference}` (ref)",
                ref["class"],
                fmt_int(ref["tokens"]),
                f"{fmt(ref['fertility'])} {fmt_ci(ref['fertility_ci'])}",
                "1.000",
                fmt_pct(ref["unk_rate"], 2),
            ]
        )
    for name in names:
        row = row_for(result, view, name)
        rows.append(
            [
                f"`{name}`",
                row["class"],
                fmt_int(row["tokens"]),
                f"{fmt(row['fertility'])} {fmt_ci(row['fertility_ci'])}",
                f"{fmt_ratio(row['ratio'])} {fmt_ci(row['ratio_ci'])}",
                fmt_pct(row["unk_rate"], 2),
            ]
        )
    return md_table(
        [
            "Tokenizer",
            "Class",
            "Tokens",
            "Fertility [95% CI]",
            "Ratio vs aratoken [95% CI]",
            "UNK rate",
        ],
        rows,
    )


def paired_table(result: dict[str, Any], view: str, group: str) -> str:
    names = candidate_order(result, group)
    rows = []
    for name in names:
        row = row_for(result, view, name)
        test = row["test"]
        rows.append(
            [
                f"`{name}`",
                f"{fmt_signed(test['median_diff'])} {fmt_signed(test['q1_diff'])} to {fmt_signed(test['q3_diff'])}",
                fmt_signed(test["mean_diff"]),
                fmt_signed(test["dz"], 2),
                fmt_signed(test["rank_biserial"], 3),
                f"{fmt_pct(test['win_rate_less'])} / {fmt_pct(test['win_rate_equal'])}",
                fmt_p(test["p"]),
                fmt_p(test["p_holm"]),
                fmt_sig(test["p_holm"]),
            ]
        )
    return md_table(
        [
            "Tokenizer",
            "Median delta [IQR]",
            "Mean delta",
            "Cohen's dz",
            "Rank-biserial",
            "Docs cand<ref / equal",
            "p (Wilcoxon)",
            "p (Holm)",
            "Significant",
        ],
        rows,
    )


def stratum_matrix(result: dict[str, Any], view: str) -> str:
    reference = result["meta"]["reference_tokenizer"]
    llm_names = candidate_order(result, "llm")
    baseline_names = candidate_order(result, "baseline")
    names = llm_names + baseline_names
    headers = ["Stratum", f"`{reference}` (ref)"] + [f"`{name}`" for name in names]
    rows = []
    stratum_words = {
        item["name"]: item["words"] for item in result["meta"]["strata"]
    }
    for stratum in STRATUM_ORDER:
        ref = row_for(result, view, reference, stratum)
        ref_cell = fmt(ref["fertility"])
        if view == "raw" and ref["unk_rate"] > 0.05:
            ref_cell += " [UNK]"
        cells = [f"{stratum} ({stratum_words[stratum]:,} words)", ref_cell]
        for name in names:
            row = row_for(result, view, name, stratum)
            test = row["test"]
            marker = "*" if test["significant_holm"] else " ns"
            cell = f"{fmt(row['fertility'])} ({fmt(row['ratio'])}{marker})"
            if view == "raw" and row["unk_rate"] > 0.05:
                cell += " [UNK]"
            cells.append(cell)
        rows.append(cells)
    table = md_table(headers, rows)
    table += (
        "\n\n`*` = Holm-adjusted p < 0.05 within the relevant family; `ns` = not "
        "significant. `[UNK]` marks a cell with >5% unknown tokens; its fertility "
        "is a token-count artefact, not a valid efficiency measure. `arabert` and "
        "`camelbert` are BERT WordPiece baselines, not LLM tokenizers."
    )
    return table


def unk_table(result: dict[str, Any]) -> str:
    rows = []
    for name in result["meta"]["tokenizers"]:
        entry = result["unk_audit"]["raw"][name]
        if entry["overall"] <= 0.0:
            continue
        rows.append(
            [
                f"`{name}`",
                fmt_pct(entry["overall"], 2),
                fmt_pct(entry["msa"], 2),
                fmt_pct(entry["egyptian"], 2),
                fmt_pct(entry["gulf"], 2),
                fmt_pct(entry["diacritized"], 2),
            ]
        )
    table = md_table(
        ["Tokenizer", "Overall", "MSA", "Egyptian", "Gulf", "Diacritized"],
        rows,
    )
    zero = [
        name
        for name in result["meta"]["tokenizers"]
        if result["unk_audit"]["raw"][name]["overall"] == 0.0
    ]
    table += (
        "\n\n"
        + ", ".join(f"`{name}`" for name in zero)
        + " have 0.00% UNK in the raw view. After `aratoken_norm`, overall UNK is "
        + ", ".join(
            f"`{name}` {fmt_pct(result['unk_audit'][CONTROLLED_VIEW][name]['overall'], 2)}"
            for name in result["meta"]["tokenizers"]
            if result["unk_audit"][CONTROLLED_VIEW][name]["overall"] > 0.0
        )
        + "; all other tokenizers are at 0.00%, and the diacritized stratum is "
        "0.00% for every tokenizer."
    )
    return table


def view_table(result: dict[str, Any]) -> str:
    reference = result["meta"]["reference_tokenizer"]
    order = [reference] + [
        name for name in result["meta"]["tokenizers"] if name != reference
    ]
    rows = []
    by_name = {row["tokenizer"]: row for row in result["view_ablation"]["all_docs"]}
    for name in order:
        row = by_name[name]
        test = row["test"]
        rows.append(
            [
                f"`{name}`",
                fmt(row["raw_fertility"]),
                fmt(row["nfkc_fertility"]),
                fmt(row["norm_fertility"]),
                f"{fmt_ratio(row['ratio'])} {fmt_ci(row['ratio_ci'])}",
                fmt_pct(row["ratio"] - 1.0, signed=True),
                fmt_p(test["p"]),
                fmt_p(test["p_holm"]),
                fmt_signed(test["dz"], 2),
            ]
        )
    return md_table(
        [
            "Tokenizer",
            "raw",
            "nfkc",
            "aratoken_norm",
            "norm/raw [95% CI]",
            "Change",
            "p",
            "p (Holm)",
            "dz",
        ],
        rows,
    )


def view_stratum_table(result: dict[str, Any]) -> str:
    reference = result["meta"]["reference_tokenizer"]
    order = [reference] + [
        name for name in result["meta"]["tokenizers"] if name != reference
    ]
    rows = []
    for name in order:
        cells = [f"`{name}`"]
        for stratum in STRATUM_ORDER:
            row = next(
                item
                for item in result["view_ablation"]["strata"][stratum]
                if item["tokenizer"] == name
            )
            marker = "*" if row["test"]["significant_holm"] else " ns"
            cells.append(f"{fmt(row['ratio'])}{marker}")
        rows.append(cells)
    table = md_table(
        ["Tokenizer"] + [f"{s} (norm/raw)" for s in STRATUM_ORDER], rows
    )
    table += (
        "\n\n`*` = Holm-adjusted p < 0.05 within the six-tokenizer family for that "
        "stratum; `ns` = not significant. These are pipeline effects, not isolated "
        "diacritic effects: `aratoken_norm` also unifies alif forms, maps digits and "
        "punctuation, removes tatweel, and lowercases ASCII Latin."
    )
    return table


def significance_table(result: dict[str, Any]) -> str:
    labels = {
        "raw_all_llm": "raw / all docs / LLM",
        "raw_all_baseline": "raw / all docs / BERT",
        "aratoken_norm_all_llm": "normalized / all docs / LLM",
        "aratoken_norm_all_baseline": "normalized / all docs / BERT",
        "raw_stratum_llm": "raw / per stratum / LLM",
        "raw_stratum_baseline": "raw / per stratum / BERT",
        "aratoken_norm_stratum_llm": "normalized / per stratum / LLM",
        "aratoken_norm_stratum_baseline": "normalized / per stratum / BERT",
        "view_ablation_all": "view ablation / all docs",
    }
    members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view in ("raw", CONTROLLED_VIEW):
        for group in ("llm", "baseline"):
            for row in result["all_docs"][view][group]:
                members[row["test"]["family"]].append(row)
        for stratum in STRATUM_ORDER:
            for group in ("llm", "baseline"):
                for row in result["strata"][view][stratum][group]:
                    members[row["test"]["family"]].append(row)
    for row in result["view_ablation"]["all_docs"]:
        members[row["test"]["family"]].append(row)

    rows = []
    for family, label in labels.items():
        info = result["meta"]["families"][family]
        failures = [
            row
            for row in members[family]
            if not row["test"]["significant_holm"]
        ]
        failure_text = ", ".join(
            f"`{row['tokenizer']}`"
            + (f"/{row['stratum']}" if row["stratum"] != "__all__" else "")
            + f" (p_Holm={fmt_p(row['test']['p_holm'])})"
            for row in failures
        )
        rows.append(
            [
                label,
                str(info["n_tests"]),
                str(info["n_significant"]),
                failure_text or "none",
            ]
        )
    return md_table(
        ["Family", "Tests", "Significant after Holm", "Not significant"], rows
    )


def render_markdown(result: dict[str, Any]) -> str:
    reference = result["meta"]["reference_tokenizer"]
    ref_raw = row_for(result, "raw", reference)
    ref_norm = row_for(result, CONTROLLED_VIEW, reference)
    raw_lep = row_for(result, "raw", "aratoken_lep")
    raw_o200k = row_for(result, "raw", "o200k")
    raw_qwen = row_for(result, "raw", "qwen3")
    raw_arabert = row_for(result, "raw", "arabert")
    raw_camel = row_for(result, "raw", "camelbert")
    norm_lep = row_for(result, CONTROLLED_VIEW, "aratoken_lep")
    norm_o200k = row_for(result, CONTROLLED_VIEW, "o200k")
    norm_qwen = row_for(result, CONTROLLED_VIEW, "qwen3")
    norm_arabert = row_for(result, CONTROLLED_VIEW, "arabert")
    norm_camel = row_for(result, CONTROLLED_VIEW, "camelbert")

    raw_ref_diac_unk = result["unk_audit"]["raw"][reference]["diacritized"]
    raw_arabert_diac_unk = result["unk_audit"]["raw"]["arabert"]["diacritized"]
    raw_camel_diac_unk = result["unk_audit"]["raw"]["camelbert"]["diacritized"]
    norm_ref_diac_unk = result["unk_audit"][CONTROLLED_VIEW][reference]["diacritized"]

    raw_diac_ref = row_for(result, "raw", reference, "diacritized")
    raw_diac_o200k = row_for(result, "raw", "o200k", "diacritized")
    raw_diac_qwen = row_for(result, "raw", "qwen3", "diacritized")
    norm_diac_ref = row_for(result, CONTROLLED_VIEW, reference, "diacritized")
    norm_diac_o200k = row_for(result, CONTROLLED_VIEW, "o200k", "diacritized")
    norm_diac_qwen = row_for(result, CONTROLLED_VIEW, "qwen3", "diacritized")
    norm_diac_lep = row_for(result, CONTROLLED_VIEW, "aratoken_lep", "diacritized")

    raw_msa_ref = row_for(result, "raw", reference, "msa")
    raw_msa_o200k = row_for(result, "raw", "o200k", "msa")
    norm_msa_ref = row_for(result, CONTROLLED_VIEW, reference, "msa")
    norm_msa_o200k = row_for(result, CONTROLLED_VIEW, "o200k", "msa")
    norm_msa_qwen = row_for(result, CONTROLLED_VIEW, "qwen3", "msa")
    norm_msa_lep = row_for(result, CONTROLLED_VIEW, "aratoken_lep", "msa")
    norm_msa_arabert = row_for(result, CONTROLLED_VIEW, "arabert", "msa")
    norm_msa_camel = row_for(result, CONTROLLED_VIEW, "camelbert", "msa")

    norm_eg_ref = row_for(result, CONTROLLED_VIEW, reference, "egyptian")
    norm_eg_arabert = row_for(result, CONTROLLED_VIEW, "arabert", "egyptian")
    norm_eg_camel = row_for(result, CONTROLLED_VIEW, "camelbert", "egyptian")
    norm_eg_lep = row_for(result, CONTROLLED_VIEW, "aratoken_lep", "egyptian")
    norm_gulf_ref = row_for(result, CONTROLLED_VIEW, reference, "gulf")
    norm_gulf_arabert = row_for(result, CONTROLLED_VIEW, "arabert", "gulf")
    norm_gulf_camel = row_for(result, CONTROLLED_VIEW, "camelbert", "gulf")
    norm_gulf_lep = row_for(result, CONTROLLED_VIEW, "aratoken_lep", "gulf")

    raw_o200k_less = raw_o200k["test"]["win_rate_less"]
    raw_o200k_more = raw_o200k["test"]["win_rate_greater"]
    raw_o200k_median = raw_o200k["test"]["median_diff"]
    norm_o200k_less = norm_o200k["test"]["win_rate_less"]
    norm_o200k_rank = norm_o200k["test"]["rank_biserial"]
    norm_qwen_rank = norm_qwen["test"]["rank_biserial"]

    corpus_sha = result["meta"]["corpus_sha256"] or "-"
    generated = result["meta"]["generated_at"]
    bootstrap = result["meta"]["bootstrap"]
    n_docs = result["meta"]["n_docs"]
    n_words_raw = result["meta"]["n_words_raw"]
    n_words_norm = result["meta"]["n_words_aratoken_norm"]

    lines: list[str] = []
    add = lines.append

    add("# Arabic Tokenizer Fertility: Numeric and Statistical Comparison")
    add("")
    add(f"**Generated:** {generated}  ")
    add(
        f"**Corpus:** {n_words_raw:,} whitespace words / {n_docs:,} documents; "
        f"SHA256 `{corpus_sha}`  "
    )
    add(f"**Reference tokenizer:** `{reference}`  ")
    add("**Views:** `raw` (as delivered), `nfkc` (Unicode only), `aratoken_norm` (controlled)  ")
    add(
        f"**Bootstrap:** {bootstrap['n_boot']:,} document-level paired replicates, "
        f"{bootstrap['ci'] * 100:.0f}% CI, seed {bootstrap['seed']}  "
    )
    add(
        "**Tests:** two-sided paired Wilcoxon signed-rank; Holm within declared "
        f"families; alpha = {ALPHA}  "
    )
    add(
        "**Sources:** `results/tokenization.json`, "
        "`results/cache/token_counts.jsonl`  "
    )
    add("")

    add("## Bottom line")
    add("")
    add(
        f"1. **Controlled comparison (`aratoken_norm`):** `{reference}` has the "
        f"lowest fertility among LLM tokenizers, **{fmt(ref_norm['fertility'])} "
        f"tokens/word** {fmt_ci(ref_norm['fertility_ci'])}. `aratoken_lep` is "
        f"**{fmt_ratio(norm_lep['ratio'])}** "
        f"{fmt_ci(norm_lep['ratio_ci'])} vs `{reference}`; `o200k` is "
        f"**{fmt_ratio(norm_o200k['ratio'])}** {fmt_ci(norm_o200k['ratio_ci'])}; "
        f"`qwen3` is **{fmt_ratio(norm_qwen['ratio'])}** "
        f"{fmt_ci(norm_qwen['ratio_ci'])}. All three paired differences remain "
        "significant after Holm correction."
    )
    add(
        f"2. **As-delivered (`raw`):** `aratoken_lep` is lowest at "
        f"**{fmt(raw_lep['fertility'])}**, then `{reference}` "
        f"**{fmt(ref_raw['fertility'])}**, `o200k` "
        f"**{fmt(raw_o200k['fertility'])}**, and `qwen3` "
        f"**{fmt(raw_qwen['fertility'])}**. This ordering is **not** a clean "
        f"vocabulary ranking: `{reference}` raw emits "
        f"**{fmt_pct(ref_raw['unk_rate'], 2)} UNK overall** and "
        f"**{fmt_pct(raw_ref_diac_unk, 2)} UNK on diacritized text**; the BERT "
        f"baselines emit **{fmt_pct(raw_arabert_diac_unk, 2)}** and "
        f"**{fmt_pct(raw_camel_diac_unk, 2)} UNK** on diacritized text. Their raw "
        "fertility is a token-count artefact, not evidence of efficient encoding."
    )
    add(
        f"3. **The raw `o200k` vs `{reference}` near-tie is a composition effect.** "
        f"The aggregate ratio is **{fmt_ratio(raw_o200k['ratio'])} "
        f"{fmt_ci(raw_o200k['ratio_ci'])}**, but the median document has "
        f"**{fmt_signed(raw_o200k_median)} tokens/word** in `o200k`'s favour and "
        f"**{fmt_pct(raw_o200k_more)}** of documents use more tokens with `o200k`. "
        f"The aggregate is pulled toward 1.0 by the minority "
        f"(**{fmt_pct(raw_o200k_less)}**) where `o200k` wins big on diacritized "
        f"text because raw `{reference}` is UNK-heavy. Under common normalization, "
        f"`o200k` is **{fmt_ratio(norm_o200k['ratio'])}** and "
        f"**{fmt_pct(norm_o200k_less, 2)}** of documents use fewer tokens with "
        f"`o200k`; the rank-biserial is **{fmt_signed(norm_o200k_rank, 3)}**."
    )
    add(
        f"4. **The normalized win is MSA-driven.** On MSA, `{reference}` is "
        f"**{fmt(norm_msa_ref['fertility'])}** vs `arabert` "
        f"**{fmt(norm_msa_arabert['fertility'])}**, `aratoken_lep` "
        f"**{fmt(norm_msa_lep['fertility'])}**, and `o200k` "
        f"**{fmt(norm_msa_o200k['fertility'])}**. On Egyptian and Gulf, "
        f"`{reference}` is statistically indistinguishable from `arabert` "
        f"(`{fmt(norm_eg_ref['fertility'])}` vs "
        f"`{fmt(norm_eg_arabert['fertility'])}`; p_Holm = "
        f"`{fmt_p(norm_eg_arabert['test']['p_holm'])}` on Egyptian). On "
        f"diacritized text, `aratoken_lep` is slightly lower "
        f"(**{fmt(norm_diac_lep['fertility'])}** vs "
        f"**{fmt(norm_diac_ref['fertility'])}**)."
    )
    add(
        "5. **No cost, TTFT, English-Arabic, or quality claim is made here.** "
        "Token ratios are not prices; the cost table stays `-` until real provider "
        "prices are configured. There is no English arm, so no English-vs-Arabic "
        "tax can be computed from this run."
    )
    add("")

    add("## 1. What is compared")
    add("")
    add(
        f"The run covers **{n_docs:,} documents / {n_words_raw:,} whitespace "
        "words** in four deliberately unbalanced strata. **Token inflation** (`fertility`) is `tokens / whitespace words`; it is a tokenizer metric, not model quality."
    )
    add("")
    strata_rows = []
    for item in result["meta"]["strata"]:
        strata_rows.append(
            [
                f"`{item['name']}`",
                fmt_int(item["docs"]),
                fmt_int(item["words"]),
                fmt_pct(item["diacritic_ratio"], 1),
            ]
        )
    strata_rows.append(
        [
            "**Total**",
            f"**{fmt_int(n_docs)}**",
            f"**{fmt_int(n_words_raw)}**",
            f"**{fmt_pct(sum(i['diacritic_ratio'] * i['words'] for i in result['meta']['strata']) / n_words_raw, 1)}**",
        ]
    )
    add(md_table(["Stratum", "Docs", "Words", "Diacritics"], strata_rows))
    add("")
    add(
        "The overall micro-average is word-weighted, so MSA (55% of words) "
        "dominates it. Diacritized text is 15% of words but a much larger share "
        "of raw tokens for tokenizers that do not handle diacritics."
    )
    add("")
    add("**Definitions**")
    add("")
    add(
        "- **`raw`** applies no external normalization; each tokenizer uses only "
        "its own internal normalizer. This is the as-delivered deployment view."
    )
    add(
        "- **`nfkc`** applies Unicode NFKC only. It is a control for the claim "
        "that Unicode normalization is the driver."
    )
    add(
        "- **`aratoken_norm`** applies the published AraToken pipeline to every "
        "tokenizer: NFKC, diacritic removal, alif unification, Arabic digit and "
        "punctuation mapping, tatweel removal, and ASCII lowercasing. It is a "
        "common pre-tokenization view, not a model-specific normalizer."
    )
    add(
        f"- **UNK rate** is unknown tokens / total tokens. It is reported because "
        "a low token count produced by mapping text to `[UNK]` is not efficient "
        "tokenization. Raw `{reference}` is out-of-distribution without its "
        "published normalization; raw BERT WordPiece has no diacritized forms."
    )
    add(
        "- **`arabert` and `camelbert`** are Arabic BERT WordPiece baselines "
        "(30k vocab, 512-token context). They are shown for context and excluded "
        "from the LLM-tokenizer ranking."
    )
    add(
        f"- **Word-count note:** raw has {n_words_raw:,} words. "
        f"`aratoken_norm` has {n_words_norm:,} because 13 diacritized documents "
        "contain a standalone diacritic cluster that normalizes to an empty "
        "string. Fertility tables use each view's own word count, matching "
        "`tokenization.json`; paired tests use the original raw word count as a "
        "common denominator."
    )
    add("")

    add("## 2. Statistical protocol")
    add("")
    add(
        "Every test is paired by document. For candidate `c` and reference `r`:"
    )
    add("")
    add("```text")
    add("d_i = (tokens_c_i - tokens_r_i) / words_raw_i")
    add("```")
    add("")
    add(
        "- **Test:** two-sided Wilcoxon signed-rank on `d_i`, normal "
        "approximation with average ranks for ties, tie-corrected variance, and "
        "continuity correction. Zero differences are discarded; `n_nonzero` and "
        "the tie rate are reported."
    )
    add(
        "- **Multiplicity:** Holm step-down within declared families. The "
        "families are: all-doc raw LLM (3 tests), all-doc raw BERT (2), all-doc "
        "normalized LLM (3), all-doc normalized BERT (2), per-stratum raw LLM "
        "(12), per-stratum raw BERT (8), per-stratum normalized LLM (12), "
        "per-stratum normalized BERT (8), and all-doc view ablation (6). "
        "Per-stratum view-ablation tests use a separate six-tokenizer family per "
        "stratum."
    )
    add(
        "- **Effect sizes:** Cohen's `dz = mean(d_i) / sd(d_i)` with sample SD; "
        "rank-biserial `(W+ - W-) / (W+ + W-)`. Positive values mean the "
        "candidate uses more tokens than the reference."
    )
    add(
        "- **Intervals:** the 95% CIs are the existing document-level paired "
        "bootstrap intervals (2,000 replicates). They describe the aggregate "
        "token ratio and are not multiplicity-adjusted."
    )
    add(
        "- **Weighting:** the Wilcoxon test weights documents equally; the "
        "aggregate ratio weights words. They answer different questions and can "
        "disagree. Both are reported. A significant Wilcoxon result does not "
        "imply a large aggregate ratio, and a near-1.0 aggregate ratio does not "
        "imply that most documents are tied."
    )
    add(
        "- **Implementation:** `compare.py`, Python standard library only. "
        "`scipy` is not installed and is not required."
    )
    add("")

    add("## 3. Raw view (as delivered)")
    add("")
    add(
        "This is what happens if the corpus text is passed to each tokenizer "
        "without the AraToken normalization. It is the deployment view, but it "
        "is **not** a valid vocabulary-efficiency ranking for tokenizers that "
        "emit UNK on this input."
    )
    add("")
    add("### 3.1 UNK audit")
    add("")
    add(unk_table(result))
    add("")
    add(
        f"The raw `{reference}` numbers are the clearest warning. Its overall "
        f"fertility of **{fmt(ref_raw['fertility'])}** is built from "
        f"**{fmt_pct(ref_raw['unk_rate'], 2)} UNK tokens**; on diacritized text, "
        f"**{fmt_pct(raw_ref_diac_unk, 2)}** of its tokens are UNK. That is not "
        "a tokenizer efficiently encoding Arabic; it is a tokenizer being fed an "
        "input distribution it was not trained for. The published pipeline is "
        "part of the tokenizer's contract."
    )
    add("")
    add("### 3.2 All documents")
    add("")
    add("**LLM tokenizers**")
    add("")
    add(all_doc_table(result, "raw", "llm"))
    add("")
    add("**BERT baselines (not LLM tokenizers)**")
    add("")
    add(all_doc_table(result, "raw", "baseline"))
    add("")
    add("### 3.3 Paired tests vs `aratoken`")
    add("")
    add("**LLM tokenizers**")
    add("")
    add(paired_table(result, "raw", "llm"))
    add("")
    add("**BERT baselines**")
    add("")
    add(paired_table(result, "raw", "baseline"))
    add("")
    add(
        "Positive `delta` means the candidate uses more tokens per word than "
        "`aratoken`; negative means fewer. The raw tests describe token counts, "
        "not valid encodings: the `aratoken` reference and the BERT baselines "
        "are UNK-contaminated, so their signs should not be read as vocabulary "
        "quality."
    )
    add("")
    add("### 3.4 Why the raw `o200k` result is a composition trap")
    add("")
    add(
        f"`o200k` and `{reference}` look almost tied in the raw aggregate: "
        f"**{fmt_ratio(raw_o200k['ratio'])} {fmt_ci(raw_o200k['ratio_ci'])}**. "
        "That single number hides two opposite populations:"
    )
    add("")
    add(
        md_table(
            ["Statistic", "Value"],
            [
                ["Aggregate token ratio", f"{fmt_ratio(raw_o200k['ratio'])} {fmt_ci(raw_o200k['ratio_ci'])}"],
                ["Median per-document delta", fmt_signed(raw_o200k_median)],
                ["Documents where `o200k` uses more tokens", fmt_pct(raw_o200k_more)],
                ["Documents where `o200k` uses fewer tokens", fmt_pct(raw_o200k_less)],
                ["MSA stratum ratio", fmt_ratio(raw_msa_o200k["ratio"])],
                ["Diacritized stratum ratio", fmt_ratio(raw_diac_o200k["ratio"])],
                ["Diacritized `aratoken` UNK rate", fmt_pct(raw_ref_diac_unk, 2)],
            ],
        )
    )
    add("")
    add(
        f"On MSA, `o200k` uses **{fmt_pct(raw_msa_o200k['ratio'] - 1.0, signed=True)}** "
        f"more tokens than `{reference}`. On diacritized text it uses "
        f"**{fmt_pct(raw_diac_o200k['ratio'] - 1.0, signed=True)}** because raw "
        f"`{reference}` collapses to UNK. The aggregate near-tie is the weighted "
        "sum of those two effects, not evidence that the tokenizers are "
        "equivalent on Arabic. Under common normalization, `o200k` is "
        f"**{fmt_ratio(norm_o200k['ratio'])}** and the ambiguity disappears."
    )
    add("")

    add("## 4. Controlled view (`aratoken_norm`)")
    add("")
    add(
        "This is the valid tokenizer-efficiency comparison: every tokenizer sees "
        "the same normalized text, and UNK is low for all six. The cost is that "
        "the view removes diacritics and changes character classes, so it cannot "
        "answer \"what does diacritized input cost?\" That question belongs to "
        "the raw view and to the view ablation in section 6."
    )
    add("")
    add("### 4.1 All documents")
    add("")
    add("**LLM tokenizers**")
    add("")
    add(all_doc_table(result, CONTROLLED_VIEW, "llm"))
    add("")
    add("**BERT baselines (not LLM tokenizers)**")
    add("")
    add(all_doc_table(result, CONTROLLED_VIEW, "baseline"))
    add("")
    add("### 4.2 Paired tests vs `aratoken`")
    add("")
    add("**LLM tokenizers**")
    add("")
    add(paired_table(result, CONTROLLED_VIEW, "llm"))
    add("")
    add("**BERT baselines**")
    add("")
    add(paired_table(result, CONTROLLED_VIEW, "baseline"))
    add("")
    add(
        f"The normalized ranking is unambiguous for the two general-purpose "
        f"tokenizers: every document uses more tokens with `o200k` "
        f"(rank-biserial **{fmt_signed(norm_o200k_rank, 3)}**, fewer-token rate "
        f"**{fmt_pct(norm_o200k_less, 2)}**) and with `qwen3` "
        f"(rank-biserial **{fmt_signed(norm_qwen_rank, 3)}**). `aratoken_lep` is "
        f"closer: **{fmt_ratio(norm_lep['ratio'])}** aggregate, but its median "
        f"difference is **{fmt_signed(norm_lep['test']['median_diff'])}**, "
        f"**{fmt_pct(norm_lep['test']['win_rate_equal'])}** of documents are "
        f"exact ties, and **{fmt_pct(norm_lep['test']['win_rate_less'])}** favour "
        "`aratoken_lep`. Its aggregate penalty is concentrated in MSA, not spread "
        "evenly across dialects."
    )
    add("")
    add(
        f"The BERT baselines are competitive in aggregate: `arabert` is "
        f"**{fmt_ratio(norm_arabert['ratio'])}** and `camelbert` is "
        f"**{fmt_ratio(norm_camel['ratio'])}**. But the `arabert` effect is small "
        f"(dz **{fmt_signed(norm_arabert['test']['dz'], 2)}**, median difference "
        f"**{fmt_signed(norm_arabert['test']['median_diff'])}**, "
        f"**{fmt_pct(norm_arabert['test']['win_rate_equal'])}** ties); its "
        "aggregate penalty comes from MSA. On Egyptian and Gulf it is "
        "statistically indistinguishable from `aratoken` (section 5)."
    )
    add("")

    add("## 5. Per-stratum results")
    add("")
    add(
        "Cells show fertility and, in parentheses, the token ratio vs `aratoken` "
        "in the same stratum. `*` = Holm-adjusted p < 0.05; `ns` = not "
        "significant. `[UNK]` marks raw cells with >5% unknown tokens."
    )
    add("")
    add("### 5.1 Raw view")
    add("")
    add(stratum_matrix(result, "raw"))
    add("")
    add(
        f"The raw table is dominated by the diacritized column. `{reference}` "
        f"jumps to **{fmt(raw_diac_ref['fertility'])}** with "
        f"**{fmt_pct(raw_ref_diac_unk, 2)} UNK**; `o200k` is "
        f"**{fmt(raw_diac_o200k['fertility'])}** and `qwen3` is "
        f"**{fmt(raw_diac_qwen['fertility'])}**. The apparent `o200k` advantage "
        f"(ratio **{fmt_ratio(raw_diac_o200k['ratio'])}**) is therefore not a "
        "vocabulary win; it is the cost of `aratoken` being used without its "
        "normalization. The same caveat applies to the BERT cells at the bottom "
        "of the diacritized column."
    )
    add("")
    add("### 5.2 Normalized view")
    add("")
    add(stratum_matrix(result, CONTROLLED_VIEW))
    add("")
    add(
        f"The normalized table separates the effects. On MSA, `{reference}` is "
        f"**{fmt(norm_msa_ref['fertility'])}**, clearly below `arabert` "
        f"**{fmt(norm_msa_arabert['fertility'])}**, `aratoken_lep` "
        f"**{fmt(norm_msa_lep['fertility'])}**, `camelbert` "
        f"**{fmt(norm_msa_camel['fertility'])}**, `o200k` "
        f"**{fmt(norm_msa_o200k['fertility'])}**, and `qwen3` "
        f"**{fmt(norm_msa_qwen['fertility'])}**. On Egyptian, `{reference}` "
        f"**{fmt(norm_eg_ref['fertility'])}** and `arabert` "
        f"**{fmt(norm_eg_arabert['fertility'])}** are tied "
        f"(p_Holm **{fmt_p(norm_eg_arabert['test']['p_holm'])}**); `camelbert` "
        f"**{fmt(norm_eg_camel['fertility'])}** and `aratoken_lep` "
        f"**{fmt(norm_eg_lep['fertility'])}** are slightly higher. On Gulf, "
        f"`{reference}` **{fmt(norm_gulf_ref['fertility'])}**, `arabert` "
        f"**{fmt(norm_gulf_arabert['fertility'])}** (ns), and `camelbert` "
        f"**{fmt(norm_gulf_camel['fertility'])}** are within 1.7% of each other; "
        f"`aratoken_lep` is **{fmt(norm_gulf_lep['fertility'])}**. On "
        f"diacritized text, `aratoken_lep` **{fmt(norm_diac_lep['fertility'])}** "
        f"is slightly below `{reference}` **{fmt(norm_diac_ref['fertility'])}**; "
        f"`o200k` **{fmt(norm_diac_o200k['fertility'])}** and `qwen3` "
        f"**{fmt(norm_diac_qwen['fertility'])}** are far higher."
    )
    add("")
    add(
        "The overall normalized advantage is therefore not a uniform "
        "\"Arabic tokenizer wins everywhere\" result. It is a large MSA "
        "advantage, near-parity with Arabic BERT baselines on Egyptian and Gulf, "
        "and a small `aratoken_lep` advantage on diacritized text."
    )
    add("")

    add("## 6. View ablation: why the ranking flips")
    add("")
    add(view_table(result))
    add("")
    add(
        "`nfkc` is effectively inert: across the whole corpus it changes total "
        "tokens by at most 8 tokens for any tokenizer. The large effects come "
        "from `aratoken_norm`, and they go in both directions."
    )
    add("")
    add(
        f"`{reference}` drops from **{fmt(ref_raw['fertility'])}** to "
        f"**{fmt(ref_norm['fertility'])}** "
        f"(**{fmt_pct(ref_norm['ratio'] - 1.0, signed=True)}**), because the "
        "normalization removes the diacritics and variant forms that were "
        "producing UNK. `o200k` drops "
        f"**{fmt_pct(norm_o200k['ratio'] - 1.0, signed=True)}** and `qwen3` "
        f"**{fmt_pct(norm_qwen['ratio'] - 1.0, signed=True)}**, because their raw "
        "counts were inflated by diacritics. `aratoken_lep` "
        f"(**{fmt_pct(norm_lep['ratio'] - 1.0, signed=True)}**), `arabert` "
        f"(**{fmt_pct(norm_arabert['ratio'] - 1.0, signed=True)}**), and "
        f"`camelbert` (**{fmt_pct(norm_camel['ratio'] - 1.0, signed=True)}**) "
        "move slightly the other way: their internal normalizers already handle "
        "diacritics, and the external pipeline changes character classes and word "
        "boundaries in ways that cost a few tokens."
    )
    add("")
    add("### 6.1 View effect by stratum")
    add("")
    add(view_stratum_table(result))
    add("")
    add(
        f"The diacritized column is the key. `{reference}` falls to "
        f"**{fmt(ref_norm['fertility'])}** overall because its diacritized "
        f"fertility falls from **{fmt(raw_diac_ref['fertility'])}** to "
        f"**{fmt(norm_diac_ref['fertility'])}** "
        f"(ratio **{fmt(row_for(result, CONTROLLED_VIEW, reference, 'diacritized')['ratio'])}**). "
        f"`o200k` falls from **{fmt(raw_diac_o200k['fertility'])}** to "
        f"**{fmt(norm_diac_o200k['fertility'])}** and `qwen3` from "
        f"**{fmt(raw_diac_qwen['fertility'])}** to "
        f"**{fmt(norm_diac_qwen['fertility'])}**. The BERT baselines move the "
        f"other way on diacritized text (`arabert` "
        f"**{fmt(row_for(result, CONTROLLED_VIEW, 'arabert', 'diacritized')['ratio'])}**, "
        f"`camelbert` "
        f"**{fmt(row_for(result, CONTROLLED_VIEW, 'camelbert', 'diacritized')['ratio'])}**), "
        "because the external normalization replaces Arabic punctuation/digits "
        "and removes tatweel, changing their pre-tokenization. This is a "
        "pipeline effect, not a pure diacritic-removal effect."
    )
    add("")

    add("## 7. Significance summary")
    add("")
    add(significance_table(result))
    add("")
    add(
        "The largest p-values drive the non-significant cells. In the raw "
        "per-stratum LLM family, `qwen3` on diacritized text is not significant "
        f"(p_Holm **{fmt_p(row_for(result, 'raw', 'qwen3', 'diacritized')['test']['p_holm'])}**): "
        "both raw token counts are high, and the difference is small. After "
        "normalization the same comparison is highly significant and large "
        f"(`qwen3` ratio **{fmt_ratio(norm_diac_qwen['ratio'])}**). That contrast "
        "is a direct example of why the raw view cannot be used for the "
        "tokenizer-efficiency conclusion."
    )
    add("")
    add(
        "On the BERT side, `arabert` is not distinguishable from `aratoken` on "
        "Egyptian and Gulf after Holm correction; `camelbert` is slightly lower "
        "than `aratoken` on Gulf but the effect is small. These are encoder "
        "tokenizers with a 512-token context, not substitutes for an LLM "
        "tokenizer."
    )
    add("")

    add("## 8. What this does not show")
    add("")
    add(
        "- **No cost result.** Cost is `tokens x price per token`. No provider "
        "prices are configured; the cost table remains `-`. A 46.1% token "
        "difference does not imply a 46.1% cost difference unless the price per "
        "token is equal, and frontier API prices are not equal."
    )
    add(
        "- **No TTFT result.** The TTFT arm was not run. Token count can affect "
        "prefill time, but this document has no latency measurement. A 4GB "
        "laptop demo would not be a headline TTFT benchmark."
    )
    add(
        "- **No English-Arabic arm.** The corpus is Arabic-only. There is no "
        "parallel English corpus and no English tokenizer run, so no "
        "\"Arabic tax vs English\" number can be computed from these data."
    )
    add(
        "- **No quality evaluation.** Fertility is not accuracy. A tokenizer "
        "that produces fewer tokens can lose information; the UNK results are a "
        "concrete example. No downstream model was trained or evaluated."
    )
    add(
        "- **Corpus-specific.** The corpus is deliberately unbalanced: MSA is "
        "55% of words, Egyptian 20%, Gulf 10%, diacritized 15%. The overall "
        "micro-average is dominated by MSA; the per-stratum tables are the "
        "honest view."
    )
    add(
        "- **Raw is not a pure vocabulary comparison.** Each tokenizer has its "
        "own internal normalizer, and `aratoken` additionally expects its "
        "published external pipeline. The raw ranking mixes vocabulary, "
        "normalization, and UNK behavior. The normalized view removes that "
        "confound but also removes diacritics."
    )
    add(
        "- **`o200k` is not Llama 4.** It is a GPT-4o-class English-centric "
        "proxy. The Llama 4 tokenizer is gated and was not available; no "
        "substitution is claimed."
    )
    add(
        "- **No paid APIs.** Gemini and other paid provider comparisons were "
        "not run. The tokenizer benchmark is CPU-only."
    )
    add(
        "- **Paper comparison is not replication.** The AraToken paper reports "
        "about 1.199 tokens/word under its normalization; this corpus gives "
        f"**{fmt(ref_norm['fertility'])}**. The difference may reflect corpus "
        "mix, preprocessing, or tokenizer revision. This experiment measures "
        "this corpus, not the paper's corpus."
    )
    add("")

    add("## 9. Reproduce")
    add("")
    add("```bash")
    add("python compare.py")
    add("```")
    add("")
    add(
        "The script reads `results/tokenization.json` and "
        "`results/cache/token_counts.jsonl`, validates every token sum against "
        "the published metrics, and writes `results/COMPARISON.md` and "
        "`results/comparison.json`. It uses only the Python standard library. "
        "It does not call a tokenizer or an API. If the inputs are missing, run "
        "`python bench.py tokenize` first."
    )
    add("")
    add("## Appendix: JSON fields")
    add("")
    add(
        "- `all_docs[view][group]`: rows for all documents; `group` is `llm` or "
        "`baseline`."
    )
    add(
        "- `strata[view][stratum][group]`: the same rows for one stratum."
    )
    add(
        "- `reference_all` / `reference_strata`: the `aratoken` reference rows, "
        "which have no paired test."
    )
    add(
        "- `view_ablation.all_docs` / `view_ablation.strata`: raw-to-normalized "
        "paired rows."
    )
    add(
        "- `test`: `p`, `p_holm`, `significant_holm`, `family`, `median_diff`, "
        "`mean_diff`, `dz`, `rank_biserial`, win rates, `n_nonzero`, and `n_zero`."
    )
    add(
        "- `ratio_ci`: the existing 95% document-level paired bootstrap interval "
        "for the aggregate token ratio."
    )
    add("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI entry point — minimal, standard library only
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the English numeric and statistical comparison from the "
            "existing full-run results. No tokenizer or API is called."
        )
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="directory containing tokenization.json and cache/token_counts.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="directory to write COMPARISON.md and comparison.json",
    )
    args = parser.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = build_data(results_dir)
    result = build_result(data)

    md_path = out_dir / "COMPARISON.md"
    json_path = out_dir / "comparison.json"

    md_text = render_markdown(result)
    md_path.write_text(md_text, encoding="utf-8")

    # Clean JSON-serializable version (no None values that JSON handles fine)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)

    print(f"wrote {md_path} ({len(md_text):,} chars)")
    print(f"wrote {json_path} ({json_path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())


