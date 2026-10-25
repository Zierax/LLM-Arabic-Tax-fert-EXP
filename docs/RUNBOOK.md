# Tokenization Results — English Executive Summary

Local tokenizer count (`local tokenizer count`) only. `provider-reported count` (`gemini`) excluded (`optional` = true, `api_key_env` not set). `o200k proxy` (`tiktoken:o200k_base`) is not `Llama 4` (`llama4` gated/unavailable, no substitution made).

---

## 1. Corpus (`results/tokenization.json` meta)

- `n_docs`: 1,499 (`stratum` = `__all__`)
- `n_words`: 50,000 (`raw` whitespace words); `aratoken_norm` = 49,987 (13 `diacritized` documents contain standalone diacritic cluster → empty after `normalize_aratoken`)
- `corpus_sha256`: `fbc6e3972e95037f6e318eb402fb18d4de0ede428d888b16f71a372218c3c72a` (`data/corpus_manifest.json`)
- `strata`: `msa` (170 docs / 27,500 words / 0.3% diacritics), `egyptian` (806 / 10,000 / 0.5%), `gulf` (394 / 5,000 / 0.0%), `diacritized` (129 / 7,500 / 79.0%)

---

## 2. Token inflation (`tokenization.json` `metrics[*].fertility` = token inflation)

Every value includes `tokenizer`, `view`, `scope`, `stratum`, `n_docs`, `n_words`, `tokens`, `fertility`, `fertility_ci` [lower, upper], `unk`, `unk_rate` (%), `context_words_equivalent`.

### 2.1 Headline view: `raw` (as delivered, no external normalization)

| Tokenizer (`tokenizer`) | `class` | `stratum` | `n_words` | Tokens | Fertility [95% CI] (`fertility`) | `unk_rate` (`unk_rate`) | Ratio vs `aratoken` (`paired[*].ratio`) |
|---|---|---|---:|---:|---|---|---|
| `aratoken_lep` | `arabic-specialized-extended` | `__all__` | 50,000 | 66,695 | 1.334 [1.314, 1.354] | 0.00% | 0.575x [0.541, 0.614] |
| `arabert` | `arabic-bert-baseline` | `__all__` | 50,000 | 63,758 | 1.275 [1.252, 1.298] | 12.82% | 0.550x [0.515, 0.586] |
| `camelbert` | `arabic-dialect-bert-baseline` | `__all__` | 50,000 | 68,075 | 1.361 [1.334, 1.388] | 11.92% | 0.587x [0.547, 0.629] |
| `aratoken` (ref) | `arabic-specialized` | `__all__` | 50,000 | 115,908 | 2.318 [2.195, 2.458] | 26.19% | 1.000 |
| `o200k` | `english-centric-proxy` | `__all__` | 50,000 | 118,582 | 2.372 [2.289, 2.458] | 0.00% | 1.023x [1.001, 1.047] |
| `qwen3` | `massive-multilingual` | `__all__` | 50,000 | 146,556 | 2.931 [2.824, 3.051] | 0.00% | 1.264x [1.239, 1.290] |

**Warning on `raw`:** `aratoken` `raw` (`fertility` = 2.318) has `unk_rate` = 26.19% overall (`unk` = 30,356 of 115,908 tokens) and 47.93% on `diacritized` (`stratum` = `diacritized`, `n_docs` = 129). This is out-of-distribution (`raw` applies no `normalize_aratoken` pipeline). The `arabert` (`unk_rate` = 98.48% on `diacritized`) and `camelbert` (`unk_rate` = 98.59% on `diacritized`) `raw` results are token-count artefacts: they emit `[UNK]` instead of encoding diacritics. Their low `raw` `fertility` is not evidence of efficient encoding.

### 2.2 Controlled view: `aratoken_norm` (valid tokenizer-efficiency comparison, common pre-tokenization pipeline)

| Tokenizer (`tokenizer`) | `stratum` | `view` | `n_words` (`words`) | Tokens (`tokens`) | `fertility` [CI] (`fertility_ci`) | `unk_rate` (`unk_rate`) | Ratio vs `aratoken` (`paired[*].ratio`) |
|---|---|---|---:|---:|---|---|---|
| `aratoken` (ref) | `__all__` | `aratoken_norm` | 49,987 | 63,487 | 1.270 [1.255, 1.286] | 0.31% | 1.000 [1.000, 1.000] |
| `aratoken_lep` | `__all__` | `aratoken_norm` | 49,987 | 67,853 | 1.357 [1.338, 1.377] | 0.00% | 1.069x [1.058, 1.079] |
| `arabert` | `__all__` | `aratoken_norm` | 49,987 | 66,108 | 1.323 [1.302, 1.345] | 0.26% | 1.041x [1.034, 1.048] |
| `camelbert` | `__all__` | `aratoken_norm` | 49,987 | 70,645 | 1.413 [1.389, 1.436] | 0.16% | 1.113x [1.103, 1.123] |
| `o200k` | `__all__` | `aratoken_norm` | 49,987 | 92,751 | 1.856 [1.835, 1.876] | 0.00% | 1.461x [1.447, 1.476] |
| `qwen3` | `__all__` | `aratoken_norm` | 49,987 | 111,791 | 2.236 [2.208, 2.265] | 0.00% | 1.761x [1.740, 1.782] |

**Note:** `aratoken_norm` changes word count in 13 `diacritized` docs (standalone diacritic cluster → empty). Fertility uses each `view`'s own word count (`results/tokenization.json` `metrics[*].words`); paired tests use `raw` denominator (`compare.py`).

---

## 3. Per-stratum (`stratum` breakdown, `raw`)

| Tokenizer (`tokenizer`) | `msa` (`stratum`) | `egyptian` | `gulf` | `diacritized` |
|---|---:|---:|---:|---:|
| `o200k` | 1.976 | 1.713 | 1.705 | 5.145 |
| `qwen3` | 2.413 | 2.028 | 2.028 | 6.636 |
| `aratoken` | 1.669 | 1.409 | 1.260 | 6.615 (`unk_rate`=47.93%) |
| `aratoken_lep` | 1.416 | 1.270 | 1.292 | 1.148 |
| `arabert` | 1.372 | 1.224 | 1.254 | 1.002 (`unk_rate`=98.48%) |
| `camelbert` | 1.523 | 1.250 | 1.239 | 1.001 (`unk_rate`=98.59%) |

The `diacritized` column shows why `raw` is confounded: `aratoken` jumps to 6.615 (47.93% UNK); `arabert`/`camelbert` collapse to ~1.0 (98.5% UNK). The `raw` `aratoken` `fertility` (2.318) is out-of-distribution; the valid reference measurement is `aratoken_norm` (1.270).

---

## 4. View ablation (`results/tokenization.json` `view_paired[*]`)

Paired comparison `raw` → `aratoken_norm` on identical docs (`view_paired[*]`):

| Tokenizer | `raw` / `norm` ratio | Change (`ratio` - 1) | `p` (`wilcoxon`) | `dz` (`Cohen's dz`) |
|---|---:|---:|---:|---:|
| `o200k` | 0.782 [0.751, 0.814] | -0.218 | 8.13e-22 | -0.308 |
| `qwen3` | 0.763 [0.728, 0.796] | -0.237 | 4.94e-19 | -0.309 |
| `aratoken` | 0.548 [0.518, 0.580] | -0.452 | 2.94e-101 | -0.402 |
| `arabert` | 1.037 [1.032, 1.042] | +0.037 | 2.9e-49 | +0.354 |
| `camelbert` | 1.038 [1.031, 1.044] | +0.038 | 1.5e-30 | +0.298 |
| `aratoken_lep` | 1.017 [1.015, 1.020] | +0.017 | 3.92e-53 | +0.363 |

`nfkc` is effectively inert (≤8 token change total for any tokenizer). The large effects come from `aratoken_norm`. `aratoken` drops most (`-45.2%` aggregate) because its `raw` is UNK-heavy; `o200k` and `qwen3` drop (~22%) because `raw` counts are inflated by diacritics; BERT baselines rise slightly (+3.7%, +3.8%) because the pipeline changes character classes/punctuation mapping.

---

## 5. Paired statistical tests (`compare.py` output → `results/COMPARISON.md`)

Reference: `aratoken`. Every row: `tokenizer`, `stratum`, `n_docs`, `mean_diff`, `median_diff` [IQR], `dz`, `rank_biserial`, `win_rate_less` / `win_rate_equal`, `p` (Wilcoxon), `p_holm` (Holm), significant (`p_holm` < 0.05).

**Normalized (`aratoken_norm`) LLM family (`llm` = `o200k`, `qwen3`, `aratoken_lep`):**

- `o200k`: `mean_diff` = +0.517, `median` = +0.483, `dz` = +2.498, `rank_biserial` = +1.000, `win_rate_less` = 0.1%, `p` = 4.19e-246, `p_holm` = 4.08e-246, significant: yes.
- `qwen3`: `mean_diff` = +0.828, `median` = +0.800, `dz` = +3.115, `rank_biserial` = +1.000, `win_rate_less` = 0.0%, `p` = 1.36e-246, `p_holm` = 4.08e-246, significant: yes.
- `aratoken_lep`: `mean_diff` = +0.049, `median` = 0.000, `dz` = +0.336, `rank_biserial` = +0.430, `win_rate_less` = 23.5%, `win_rate_equal` = 76.2%, `p` = 3.86e-32, `p_holm` = 4.08e-246, significant: yes.

**Normalized BERT family (`baseline` = `arabert`, `camelbert`):**

- `arabert`: `mean_diff` = +0.009, `median` = 0.000, `dz` = +0.090, `rank_biserial` = +0.118, `win_rate_less` = 28.1%, `win_rate_equal` = 35.4%, `p` = 0.00148, `p_holm` = 8.98e-21, significant: yes (small effect, MSA-driven).
- `camelbert`: `mean_diff` = +0.039, `median` = +0.047, `dz` = +0.254, `rank_biserial` = +0.314, `win_rate_less` = 28.8%, `win_rate_equal` = 19.9%, `p` = 4.49e-21, `p_holm` = 8.98e-21, significant: yes.

**Per-stratum (`stratum`) significant results (Holm-corrected `p_holm` < 0.05):**

- `msa`: `o200k` (yes), `qwen3` (yes), `aratoken_lep` (yes), `arabert` (yes), `camelbert` (yes).
- `egyptian`: `o200k` (yes), `qwen3` (yes), `aratoken_lep` (yes), `arabert` (no, `p_holm`=0.554), `camelbert` (yes, `p_holm`=0.000141).
- `gulf`: `o200k` (yes), `qwen3` (yes), `aratoken_lep` (yes, `p_holm`=0.0118), `arabert` (no, `p_holm`=0.112), `camelbert` (yes, `p_holm`=0.000853).
- `diacritized`: `o200k` (yes), `qwen3` (no, `p_holm`=0.0909 — near-tie because both `raw` counts are high), `aratoken_lep` (yes, `p_holm`=2.05e-06), `arabert` (yes, `p_holm`=6.11e-12), `camelbert` (yes, `p_holm`=1.32e-21).

---

## 6. View ablation (`results/COMPARISON.md` section 6; `tokenization.json` `view_paired[*]`)

Paired `raw` → `aratoken_norm` (`scope`=`all`):

| Tokenizer (`tokenizer`) | `norm`/`raw` ratio | `median_diff` | `dz` | `rank_biserial` | `win_rate_less` (`raw` > `norm`) | `win_rate_equal` | `p` (Wilcoxon) | `p_holm` |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `o200k` | 0.782 [0.751, 0.814] | 0.000 | -0.308 | -0.513 | 19.5% | 68.9% | 8.13e-22 | 0.000 |
| `qwen3` | 0.763 [0.728, 0.796] | 0.000 | -0.309 | -0.469 | 19.0% | 67.9% | 4.94e-19 | 0.000 |
| `aratoken` | 0.548 [0.518, 0.580] | 0.000 | -0.402 | -1.000 | 40.6% | 59.4% | 2.94e-101 | 0.000 |
| `arabert` | 1.037 [1.032, 1.042] | 0.000 | +0.354 | +0.929 | 1.6% | 77.6% | 2.9e-49 | 0.000 |
| `camelbert` | 1.038 [1.031, 1.044] | 0.000 | +0.299 | +0.709 | 5.0% | 76.7% | 1.5e-30 | 0.000 |
| `aratoken_lep` | 1.017 [1.015, 1.019] | 0.000 | +0.363 | +0.937 | 1.1% | 76.2% | 3.92e-53 | 0.000 |

`nfkc` is inert (≤8 token total change). The `diacritized` stratum drives the effect: `aratoken` drops from 6.615 (`raw`) to 1.203 (`norm`) (`ratio`=0.181); `o200k` from 5.145 to 1.667 (`ratio`=0.324); `qwen3` from 6.636 to 1.948 (`ratio`=0.293). `arabert` moves opposite on `diacritized` (`ratio`=1.237) because `normalize_aratoken` changes punctuation/digit mapping and removes tatweel. This is a pipeline effect (`docs/METHODOLOGY.md` section 4.3; `docs/RESEARCH.md` H5), not isolated diacritic removal.

---

## 7. Significance summary (`results/COMPARISON.md` section 7)

Family: `view_ablation_all` (6 tests, 1 significant failure per family not applicable here since all `p_holm` < 0.05 except none for this family — all 6 significant). Per-stratum families (`raw_stratum_llm`: 12 tests; `aratoken_norm_stratum_llm`: 12 tests; etc.). Non-significant cells: `qwen3` `raw` `diacritized` (`p_holm`=0.0909) — both raw counts high, difference small; after normalization same comparison is highly significant (`norm_diac_qwen` `ratio`=1.620, `dz`=+5.451, `p_holm`=6.58e-23). This is a direct demonstration that `raw` ranking is confounded and `aratoken_norm` is required for valid tokenizer-efficiency comparison.

---

## 8. What is blocked / not shown (`results/COMPARISON.md` section 8; `docs/METHODOLOGY.md` section 9; `docs/RUNBOOK.md`)

- **Cost (`cost ratio`)**: `results/REPORT.md` cost table = `-`. `token inflation` = 1.270 (`aratoken` `aratoken_norm`); `token inflation` = 1.856 (`o200k` `aratoken_norm`). No `provider price` configured; no `cost ratio` headline without price.
- **TTFT (`TTFT slope`)**: `docs/RUNBOOK.md` section 4 requires `nvidia-smi` verification + `vLLM`; not executed. `results/TTFT.md` = `not_executed`.
- **English-Arabic arm**: `strata` = 4 Arabic only (`msa`, `egyptian`, `gulf`, `diacritized`). No English tokenizer (`en`) or English corpus (`stratum`=`en`) exists; no `token inflation` comparison English/Arabic possible.
- **Quality**: `token inflation` ≠ accuracy. `arabert`/`camelbert` `diacritized` `raw` `unk_rate` ≈98% is a concrete counterexample.
- **Paper replication**: `AraToken paper reports 1.199`; `our corpus: 1.270 [1.255, 1.286]` (`view`=`aratoken_norm`, `stratum`=`__all__`, `n_docs`=1,499). Not a contradiction.
- **`provider-reported count`** (`gemini` `countTokens`) excluded (`optional`=true); `local tokenizer count` (`o200k`, `qwen3`, `aratoken`, `aratoken_lep`, `arabert`, `camelbert`) is the only count type reported.
- **`o200k proxy`** (`tiktoken:o200k_base`) never labeled `Llama 4`; `llama4` (`meta-llama/Llama-4-Scout-17B-16E-Instruct`) unavailable.
