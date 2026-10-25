# Methodology — scientific reference (English)

Every claim requires `n`, `CI`, `view`, `stratum`. No narrative; objective only.

> Before measurement: `AraToken paper reports 1.199`. After: `our corpus: 1.270 [1.255, 1.286]` (`n`=1,499, `view`=`aratoken_norm`, `stratum`=`__all__`). Difference = corpus/preprocessing/revision (`revision`=`6feb597`); not contradiction.

---

## 1. Design — three separate arms (`TOKENIZATION.md`, `TTFT.md` if run, `REPORT.md` if prices configured)

| Arm (`results/*.md`) | Measures (`tokenization.json` / `results/*.md`) | Does not measure | Mode |
|---|---|---|---|
| A. Tokenizer benchmark (`TOKENIZATION.md`) | `token inflation` (`fertility` = tokens/word), chars/token (`metrics[*].chars_per_token`), bytes/token (`bytes_per_token`), UNK rate (`unk_rate`), per-stratum (`scope`=`stratum`), per-view (`raw`/`nfkc`/`aratoken_norm`) | Model quality, cost, latency | CPU, deterministic (`core.py` + `bench.py`) |
| B. Cost audit (`REPORT.md`) | `cost ratio` = `token inflation` × `provider price` / `reference price` (`config.yaml` pricing) | GPU cost, energy, output cost, caching effects | Tokenizer + pricing snapshot (`provider-reported count` for `gemini`; `local tokenizer count` for others) |
| C. TTFT benchmark (`TTFT.md`) | `TTFT slope (ms/token)` (median / p95 TTFT vs input `tokens` from `tokenization.json`) | Isolated tokenizer effect (`ttft` measures model + provider); quality of response | Streaming provider / vLLM (`docs/RUNBOOK.md` section 4) |

Separation required: combining produces unverifiable claims (e.g., "tokenizer X faster" measured across different providers or without `provider`/`stratum` specified). Each arm has separate file (`docs/METHODOLOGY.md` sections 1, 4, 8). `results/COMPARISON.md` synthesizes A + statistical comparison; does not claim B or C results.

---

## 2. Corpus (`data/corpus.jsonl`, `data/corpus_manifest.json`)

`n_words` = 50,000 exactly (`len(text.split())` — operational, not linguistic; Arabic clitics `w-`, `al-`, `bi-`, `li-`, `ka-` remain attached). `n_docs` = 1,499. `strata` budget exact; last doc may be shorter (`min_words`) to hit exact budget.

| `stratum` | `n_docs` | `n_words` | `diacritic_ratio` (`corpus_manifest.json`) | Source (`corpus_manifest.json` `revision`) |
|---|---:|---:|---:|---|
| `msa` | 170 | 27,500 | 0.3% | `wikimedia/wikipedia` `b04c8d1ceb` |
| `egyptian` | 806 | 10,000 | 0.5% | `dataflare/arabic-dialect-corpus` `5e5bfb` |
| `gulf` | 394 | 5,000 | 0.0% | Same source (`5e5bfb`) |
| `diacritized` | 129 | 7,500 | 79.0% | `arbml/tashkeela` `96cc2daa` |

`corpus_sha256`: `fbc6e3972e95037f6e318eb402fb18d4de0ede428d888b16f71a372218c3c72a` (`manifest.json` `file_sha256`). Not redistributable (`corpus.jsonl` gitignored; Tashkeela GPL-2.0; Wikipedia CC-BY-SA-4.0).

`raw` word count = 50,000; `aratoken_norm` = 49,987 (13 `diacritized` docs contain standalone diacritic cluster → empty after `normalize_aratoken`). `tokenization.json` `metrics[*].words` uses each `view`'s word count; paired statistical tests (`compare.py`) use `raw` denominator (`docs/METHODOLOGY.md` section 6).

---

## 3. Tokenizer registry (`tokenization.json` `meta.tokenizers[*]`)

Every identifier exact; no substitution.

| `tokenizer` (`tokenizer`) | `class` | Source / adapter version (`tokenization.json` `meta.tokenizers[*]`) |
|---|---|---|
| `o200k` (`o200k`) | `english-centric-proxy` | `tiktoken:o200k_base:200019` (`encoding`=`o200k_base`, vocab=200,019). **Not `Llama 4`** (`llama4` = `meta-llama/Llama-4-Scout-17B-16E-Instruct`, `revision`=`92f3b159...`, gated, unavailable; `optional`=true). |
| `qwen3` (`qwen3`) | `massive-multilingual` | `hf:Qwen/Qwen3-8B@b968826d...:Qwen2TokenizerFast` (`repo`=`Qwen/Qwen3-8B`, vocab=151,669) |
| `aratoken` (`aratoken`) | `arabic-specialized` | `sentencepiece` (`repo`=`mariklolik/AraToken`, `revision`=`6feb597...`, vocab=150K, `published_normalization`=`aratoken`) |
| `aratoken_lep` (`aratoken_lep`) | `arabic-specialized-extended` | `hf:mariklolik/AraToken-Qwen3-0.6B-LEP` (`revision`=`6ad58e6...`, vocab=282,826). **Separate vocabulary** from `aratoken`; not same vocab expanded. |
| `arabert` (`arabert`) | `arabic-bert-baseline` | `hf:aubmindlab/bert-base-arabertv02` (`revision`=`016fb9d...`, vocab ≈30K, `context_window`=512) |
| `camelbert` (`camelbert`) | `arabic-dialect-bert-baseline` | `hf:CAMeL-Lab/bert-base-arabic-camelbert-mix` (`revision`=`9be35279...`, vocab ≈30K, `context_window`=512) |

`arabert`/`camelbert` excluded from LLM-tokenizer headline ranking (`results/COMPARISON.md` `tokenizers[*].is_baseline` = `true`); included for `token inflation` context (baseline comparison) and UNK audit only.

---

## 4. Views (`tokenization.json` `meta.views` = [`raw`, `nfkc`, `aratoken_norm`])

All three applied to same whitespace-normalized text (`normalize_whitespace` in `core.py` line 119).

- `raw`: no external normalization (`normalize_view` `raw` = `text`; `core.py` line 150). Each tokenizer uses its own internal normalizer (`unigram` for `aratoken`, `NFKC`+`lower` for `qwen3`, `WordPiece` for BERT). Deployment view; **not** a pure vocabulary-efficiency comparison (confounded by normalizer differences + UNK behavior on out-of-distribution input).
- `nfkc`: `unicodedata.normalize("NFKC")` (`core.py` line 153). Control: isolates whether Unicode normalization drives the effect. Result: inert (`tokenization.json` `metrics[*].tokens` `nfkc` vs `raw` ≤ 8 token difference for any tokenizer; `view_paired[*].ratio` ≈ 1.0). Not headline driver.
- `aratoken_norm`: `normalize_aratoken` (`core.py` line 136): `NFKC` → `strip_diacritics` (`DIACRITIC_RE` line 53) → `ARATOKEN_TRANSLATION` (line 56: alif variants → bare alif; Arabic/Indic digits → Western digits; Arabic punctuation → Latin; tatweel `ـ` removed) → lowercase ASCII Latin (`_lowercase_ascii_latin` line 132). Common pre-tokenization pipeline; applied to **all** tokenizers before tokenization (`core.py` `normalize_view` line 154; `count` method `core.py` line 649). **Does not measure diacritized-input cost** (removes diacritics); `raw` required for that question. Changes `n_words` in 13 `diacritized` docs (standalone diacritic cluster → empty); `metrics[*].words` uses each `view`'s own count; paired tests use `raw` denominator (`docs/METHODOLOGY.md` section 2; `compare.py`).

---

## 5. Metrics (`tokenization.json` `metrics[*]`; `results/COMPARISON.md` terminology rules)

Every number in `results/COMPARISON.md` includes `tokenizer`, `stratum`, `view`, `n_docs`, `n_words`, value, `CI` (`95% CI` from document-level paired bootstrap, `n_boot`=2000, `seed`=20261025, `tokenization.json` `meta.bootstrap`), and `unk_rate` (if `raw` > 5%). If any of `n`, `CI`, `view`, `stratum` is missing, the number is incomplete.

- **`token inflation`** (`metrics[*].fertility` = `sum(tokens) / sum(words)`; `results/COMPARISON.md` refers to it as `token inflation` in text, though `tokenization.json` field name remains `fertility` for consistency). Example (`tokenizer`=`aratoken`, `view`=`aratoken_norm`, `stratum`=`__all__`): 1.270 [1.255, 1.286], `n_docs`=1,499, `n_words`=49,987.
- **`cost ratio`** (`cost` = `token inflation` × `provider price` / `reference price`). **Blocked**: `results/REPORT.md` cost audit = `—` for all rows; `config.yaml` `pricing` = `illustrative_price_usd_per_1m: 0.5` (artificial equal-price anchor only; not a real provider price). No headline `cost ratio` without configured `provider price`; any claim of "2.5x more expensive" requires two real prices.
- **`provider-reported count`** (`gemini` `countTokens`, `optional`=true, `api_key_env`=`GEMINI_API_KEY`): `gemini` excluded from this run (`tokenization.json` `meta.tokenizers[*]` does not include `gemini` in benchmark results; `provider-reported count` label applies only if included). `local tokenizer count` applies to all reported metrics (`o200k`, `qwen3`, `aratoken`, `aratoken_lep`, `arabert`, `camelbert`).
- **Unknown rate (`metrics[*].unk_rate` = `sum(unk) / sum(tokens)`): `raw` `aratoken` = 26.19% overall (`unk`=30,356 of 115,908); `diacritized` `stratum` = 47.93% (`unk`=23,778 of 49,614). `raw` `arabert` `diacritized` = 98.48% (`unk`=7,399 of 7,513). `raw` `camelbert` `diacritized` = 98.59% (`unk`=7,399 of 7,505). Not evidence of efficient encoding; token-count artefact (`results/COMPARISON.md` section 3.1; `docs/METHODOLOGY.md` section 6.3; `docs/RESEARCH.md` point 6).
- **Token ratio (`metrics[*].ratio` from `tokenization.json` `paired[*].ratio` = `sum(candidate tokens) / sum(reference tokens)`; same denominator = `sum(words)`). Example (`tokenizer`=`qwen3`, `view`=`aratoken_norm`, `stratum`=`__all__`): 1.761 [1.740, 1.782].
- **Paired difference (`metrics[*].diff_tokens_per_word` from `tokenization.json` `paired[*].diff_tokens_per_word` = `(tokens_c - tokens_r) / words_raw`): positive = candidate uses more tokens per word than reference. Example (`tokenizer`=`o200k`, `view`=`raw`, `stratum`=`__all__`): +0.053 [0.001, 0.103]. Example (`tokenizer`=`aratoken_lep`, `view`=`raw`, `stratum`=`__all__`): -0.572 [-0.611, -0.535].
- **Bootstrap interval (`metrics[*].fertility_ci` = [lower, upper] from 2,000 paired replicates, `seed`=20261025, `ci`=0.95, `tokenization.json` `meta.bootstrap`). Descriptive only; not multiplicity-adjusted (`docs/METHODOLOGY.md` section 7.1).
- **Context capacity (`metrics[*].context_words_equivalent` = `context_window / fertility`; approximate words per context). `o200k` `raw` = 128,000 / 2.372 ≈ 53,971 words equivalent. Not quality measure.
- **`TTFT slope (ms/token)`**: `docs/METHODOLOGY.md` section 8 (`TTFT slope (ms/token)` definition); `docs/RUNBOOK.md` section 4 (protocol requires `nvidia-smi` verification + `vLLM`; `results/TTFT.md` = `not_executed` if not run). Not reported (`results/COMPARISON.md` section 2: "No TTFT result").
- **`token inflation` vs `cost ratio`** (`results/COMPARISON.md`): text uses `token inflation` (e.g., "token inflation = 2.372 [2.289, 2.458]"); `cost` mentioned only with `—` (`cost ratio` blocked by missing `provider price`).
- **`o200k proxy`** (`tokenizer`=`o200k`, `class`=`english-centric-proxy`): never labeled `Llama 4` in `results/COMPARISON.md` or any output; `llama4` (`optional`=true, `revision`=`92f3b159...`) unavailable; no substitution claim made (`docs/METHODOLOGY.md` section 5.2; `docs/RESEARCH.md` claim 5; `docs/RUNBOOK.md` checklist).
- **`provider-reported count`** (`gemini`): excluded (`optional`=true); `local tokenizer count` applies to all `metrics[*].tokens` (`results/COMPARISON.md` all tables labeled accordingly).
- **Every number completeness check**: `results/COMPARISON.md` line 1 header (`tokenizer`, `view`, `stratum`, `n_docs`, `n_words`, `fertility`, `CI`); table rows include `n_docs` (1,499 for `__all__`; per-stratum counts from `corpus_manifest.json`); paired test rows include `stratum`, `tokenizer`, `median_diff` [IQR], `mean_diff`, `dz`, `rank_biserial`, `win_rate_less`/`win_rate_equal`, `p`, `p_holm`, `significant` (`yes`/`no`). Any missing `n`, `CI`, `view`, or `stratum` = incomplete.

---

## 6. Statistical protocol (`results/COMPARISON.md` section 2; `docs/METHODOLOGY.md` section 7)

Every comparison paired by document (`compare.py`; `results/comparison.json`). Difference (`d_i`):

```
d_i = (tokens_candidate_i - tokens_reference_i) / words_raw_i
```

- **Reference (`reference_tokenizer`)**: `aratoken` (`results/tokenization.json` `meta.reference_tokenizer`).
- **Test**: two-sided paired Wilcoxon signed-rank (`wilcoxon_signed_rank`), normal approximation, average-rank tie correction (`tie_groups`), tie-corrected variance (`variance` = `n*(n+1)*(2n+1)/24 - sum(t^3 - t)/48`), continuity correction (`w_plus` > mean: `(w_plus - mean - 0.5) / sqrt(variance)`). Zero differences discarded (`nonzero`); `n_nonzero` and tie rate (`win_rate_equal` / `n`) reported.
- **Multiplicity** (`docs/METHODOLOGY.md` section 7.3; `results/COMPARISON.md` section 2): Holm step-down (`holm`) within declared families (`apply_holm`): all-doc `raw` LLM (3), all-doc `raw` BERT (2), all-doc `aratoken_norm` LLM (3), all-doc `aratoken_norm` BERT (2), per-stratum `raw` LLM (4 × 3 = 12), per-stratum `raw` BERT (4 × 2 = 8), per-stratum `aratoken_norm` LLM (4 × 3 = 12), per-stratum `aratoken_norm` BERT (4 × 2 = 8), all-doc view ablation (6). Per-stratum view ablation uses separate 6-tokenizer family per `stratum`. `results/COMPARISON.md` significance table (`section 7`) reports family, `n_tests`, `n_significant`, `not_significant` (tokenizer/stratum with `p_holm` ≥ 0.05).
- **Effect sizes (`results/COMPARISON.md` paired table)**: Cohen's `dz` (`mean_diff / sd_diff`, sample SD); rank-biserial (`(w_plus - w_minus) / (w_plus + w_minus)`). Positive = more tokens for candidate.
- **Weighting difference (`docs/METHODOLOGY.md` section 7.4)**: Wilcoxon weights documents equally; aggregate `ratio` weights words. They can disagree (`results/COMPARISON.md` section 3.4: `o200k` `raw` aggregate ratio 1.023 [1.001, 1.047] vs `median_diff` +0.357 vs `win_rate_less` 16.7% — aggregate pulled toward 1.0 by `diacritized` minority where `o200k` wins big due to `aratoken` `raw` UNK collapse). Both reported; interpretation requires both values (`COMPARISON.md` line 15: "positive `delta` means ..."; section 3.4 table with all statistics).
- **Implementation (`docs/METHODOLOGY.md` section 7; `compare.py`)**: Python standard library only (`import` list: `argparse`, `json`, `math`, `collections.defaultdict`, `datetime`, `pathlib`, `typing`). `scipy` unavailable; `numpy` 2.5.4 installed but not required for `compare.py`. `results/COMPARISON.md` line 55: `scipy` not installed and not required.

---

## 7. Threats / limitations (`docs/METHODOLOGY.md` section 9; `results/COMPARISON.md` section 8; `docs/RESEARCH.md` point 6; `docs/RUNBOOK.md` hardware verification)

- **No cost (`cost ratio`) result**: `token inflation` = 1.270 (`aratoken` `aratoken_norm`); `token inflation` = 1.856 (`o200k` `aratoken_norm`). No `provider price` configured (`config.yaml` `pricing` = `illustrative_price_usd_per_1m: 0.5` artificial anchor). `cost ratio` blocked; `results/REPORT.md` = `-`.
- **No TTFT (`TTFT slope`) result**: `docs/RUNBOOK.md` section 4 protocol defined (`nvidia-smi` verification required + `vLLM`); not executed (`results/TTFT.md` = `not_executed`; `hardware_unverified` if applicable). `results/COMPARISON.md` line 5: no `TTFT slope (ms/token)` claim made.
- **No English-Arabic arm**: `strata` = Arabic only; no English tokenizer (`stratum`=`en`) or English corpus; no English-vs-Arabic `token inflation` comparison possible (`results/COMPARISON.md` line 5: "no English arm").
- **No quality (`quality` / `accuracy`) evaluation**: `token inflation` (`fertility`) ≠ model quality. `arabert` `raw` `diacritized` `unk_rate` = 98.48% (`unk` = 7,399 of 7,513); `camelbert` `raw` `diacritized` = 98.59% (`unk` = 7,399 of 7,505). Not evidence of efficient encoding (`results/COMPARISON.md` section 3.1; `docs/METHODOLOGY.md` section 5.3; `docs/RESEARCH.md` point 6; `docs/METHODOLOGY.md` section 9.6).
- **Raw confounded (`docs/METHODOLOGY.md` section 4)**: `raw` mixes vocabulary, internal normalizer differences (`normalize_aratoken` vs tokenizer's own `NFKC`/`lower`/`WordPiece`), and UNK behavior (`aratoken` `raw` out-of-distribution). `aratoken_norm` removes confound but also removes diacritics; it answers vocabulary-efficiency questions (`token inflation` comparison), not diacritized-cost questions (`docs/METHODOLOGY.md` section 4.3; `docs/RESEARCH.md` H5; `results/COMPARISON.md` section 3.4; `docs/METHODOLOGY.md` section 9.4).
- **Corpus-specific (`docs/METHODOLOGY.md` section 3)**: micro-average (`metrics[*].scope`=`all`) word-weighted; MSA (`stratum`=`msa`) 55% dominates. Per-stratum (`scope`=`stratum`) is required for honest reading (`results/COMPARISON.md` section 5; `docs/METHODOLOGY.md` section 3.3). Macro-average (`stratum` mean) not headline.
- **Paper comparison (`docs/RESEARCH.md` point 2)**: `AraToken paper reports 1.199`; `our corpus: 1.270 [1.255, 1.286]` (`view`=`aratoken_norm`, `stratum`=`__all__`, `n_docs`=1,499, `revision`=`6feb597`). Not contradiction; reflects corpus (`msa` 55%, `egyptian` 20%, `gulf` 10%, `diacritized` 15%) vs paper's corpus, preprocessing (`normalize_aratoken`), tokenizer revision (`revision` pinned in `corpus_manifest.json` and `tokenization.json` `meta.tokenizers[*]`).
- **`provider-reported count` (`gemini`) excluded (`docs/METHODOLOGY.md` section 5; `docs/RUNBOOK.md` checklist)**: only `local tokenizer count` reported (`results/tokenization.json` `metrics[*].tokens`).
- **`o200k proxy` not `Llama 4` (`docs/METHODOLOGY.md` section 5; `docs/RESEARCH.md` claim 1)**: `o200k` (`class`=`english-centric-proxy`) never labeled `Llama 4`; `llama4` (`optional`=true, `revision` pinned) unavailable; no substitution claim made.

---

## 8. Reproduce (`docs/RUNBOOK.md` section 2-6; `docs/METHODOLOGY.md` section 11)

```bash
# Build (first run only; pinned revisions in config.yaml after manifest verification)
python bench.py corpus
# Verify: total_docs = 1,499; total_words = 50,000; revisions pinned (manifest file_sha256 recorded)
# Tokenizer benchmark (CPU only; local tokenizer count; no paid API)
python bench.py tokenize --tokenizers o200k,qwen3,aratoken,aratoken_lep,arabert,camelbert
# Verify: results/tokenization.json (90 metric rows); results/cache/token_counts.jsonl (26,982 rows = 6 tokenizers x 3 views x 1,499 docs)
# Statistical comparison (standard library; scipy not required)
python compare.py --results-dir results --out-dir results
# Verify: results/COMPARISON.md + results/comparison.json; every number includes n, CI, view, stratum; no Arabic text; terminology rules met (token inflation, cost ratio blocked, TTFT slope blocked, o200k proxy label, provider-reported count excluded, local tokenizer count present)
# Report synthesis (optional; REPORT.md not required for MVP; deleted to minimize files)
python bench.py report --out-dir results
```
