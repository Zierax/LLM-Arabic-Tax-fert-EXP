# Arabic Tokenizer Fertility: Numeric and Statistical Comparison

**Generated:** 2026-10-25T23:39:17Z  
**Corpus:** 50,000 whitespace words / 1,499 documents; SHA256 `-`  
**Reference tokenizer:** `aratoken`  
**Views:** `raw` (as delivered), `nfkc` (Unicode only), `aratoken_norm` (controlled)  
**Bootstrap:** 2,000 document-level paired replicates, 95% CI, seed 20261025  
**Tests:** two-sided paired Wilcoxon signed-rank; Holm within declared families; alpha = 0.05  
**Sources:** `results/tokenization.json`, `results/cache/token_counts.jsonl`  

## Bottom line

1. **Controlled comparison (`aratoken_norm`):** `aratoken` has the lowest fertility among LLM tokenizers, **1.270 tokens/word** [1.255, 1.286]. `aratoken_lep` is **1.069x** [1.058, 1.079] vs `aratoken`; `o200k` is **1.461x** [1.447, 1.476]; `qwen3` is **1.761x** [1.740, 1.782]. All three paired differences remain significant after Holm correction.
2. **As-delivered (`raw`):** `aratoken_lep` is lowest at **1.334**, then `aratoken` **2.318**, `o200k` **2.372**, and `qwen3` **2.931**. This ordering is **not** a clean vocabulary ranking: `aratoken` raw emits **26.19% UNK overall** and **47.93% UNK on diacritized text**; the BERT baselines emit **98.48%** and **98.59% UNK** on diacritized text. Their raw fertility is a token-count artefact, not evidence of efficient encoding.
3. **The raw `o200k` vs `aratoken` near-tie is a composition effect.** The aggregate ratio is **1.023x [1.001, 1.047]**, but the median document has **+0.357 tokens/word** in `o200k`'s favour and **81.8%** of documents use more tokens with `o200k`. The aggregate is pulled toward 1.0 by the minority (**16.7%**) where `o200k` wins big on diacritized text because raw `aratoken` is UNK-heavy. Under common normalization, `o200k` is **1.461x** and **0.07%** of documents use fewer tokens with `o200k`; the rank-biserial is **+1.000**.
4. **The normalized win is MSA-driven.** On MSA, `aratoken` is **1.303** vs `arabert` **1.390**, `aratoken_lep` **1.443**, and `o200k` **1.989**. On Egyptian and Gulf, `aratoken` is statistically indistinguishable from `arabert` (`1.235` vs `1.231`; p_Holm = `0.554` on Egyptian). On diacritized text, `aratoken_lep` is slightly lower (**1.184** vs **1.203**).
5. **No cost, TTFT, English-Arabic, or quality claim is made here.** Token ratios are not prices; the cost table stays `-` until real provider prices are configured. There is no English arm, so no English-vs-Arabic tax can be computed from this run.

## 1. What is compared

The run covers **1,499 documents / 50,000 whitespace words** in four deliberately unbalanced strata. **Token inflation** (`fertility`) is `tokens / whitespace words`; it is a tokenizer metric, not model quality.

| Stratum | Docs | Words | Diacritics |
|---|---|---|---|
| `msa` | 170 | 27,500 | 0.3% |
| `egyptian` | 806 | 10,000 | 0.5% |
| `gulf` | 394 | 5,000 | 0.0% |
| `diacritized` | 129 | 7,500 | 79.0% |
| **Total** | **1,499** | **50,000** | **12.1%** |

The overall micro-average is word-weighted, so MSA (55% of words) dominates it. Diacritized text is 15% of words but a much larger share of raw tokens for tokenizers that do not handle diacritics.

**Definitions**

- **`raw`** applies no external normalization; each tokenizer uses only its own internal normalizer. This is the as-delivered deployment view.
- **`nfkc`** applies Unicode NFKC only. It is a control for the claim that Unicode normalization is the driver.
- **`aratoken_norm`** applies the published AraToken pipeline to every tokenizer: NFKC, diacritic removal, alif unification, Arabic digit and punctuation mapping, tatweel removal, and ASCII lowercasing. It is a common pre-tokenization view, not a model-specific normalizer.
- **UNK rate** is unknown tokens / total tokens. It is reported because a low token count produced by mapping text to `[UNK]` is not efficient tokenization. Raw `{reference}` is out-of-distribution without its published normalization; raw BERT WordPiece has no diacritized forms.
- **`arabert` and `camelbert`** are Arabic BERT WordPiece baselines (30k vocab, 512-token context). They are shown for context and excluded from the LLM-tokenizer ranking.
- **Word-count note:** raw has 50,000 words. `aratoken_norm` has 49,987 because 13 diacritized documents contain a standalone diacritic cluster that normalizes to an empty string. Fertility tables use each view's own word count, matching `tokenization.json`; paired tests use the original raw word count as a common denominator.

## 2. Statistical protocol

Every test is paired by document. For candidate `c` and reference `r`:

```text
d_i = (tokens_c_i - tokens_r_i) / words_raw_i
```

- **Test:** two-sided Wilcoxon signed-rank on `d_i`, normal approximation with average ranks for ties, tie-corrected variance, and continuity correction. Zero differences are discarded; `n_nonzero` and the tie rate are reported.
- **Multiplicity:** Holm step-down within declared families. The families are: all-doc raw LLM (3 tests), all-doc raw BERT (2), all-doc normalized LLM (3), all-doc normalized BERT (2), per-stratum raw LLM (12), per-stratum raw BERT (8), per-stratum normalized LLM (12), per-stratum normalized BERT (8), and all-doc view ablation (6). Per-stratum view-ablation tests use a separate six-tokenizer family per stratum.
- **Effect sizes:** Cohen's `dz = mean(d_i) / sd(d_i)` with sample SD; rank-biserial `(W+ - W-) / (W+ + W-)`. Positive values mean the candidate uses more tokens than the reference.
- **Intervals:** the 95% CIs are the existing document-level paired bootstrap intervals (2,000 replicates). They describe the aggregate token ratio and are not multiplicity-adjusted.
- **Weighting:** the Wilcoxon test weights documents equally; the aggregate ratio weights words. They answer different questions and can disagree. Both are reported. A significant Wilcoxon result does not imply a large aggregate ratio, and a near-1.0 aggregate ratio does not imply that most documents are tied.
- **Implementation:** `compare.py`, Python standard library only. `scipy` is not installed and is not required.

## 3. Raw view (as delivered)

This is what happens if the corpus text is passed to each tokenizer without the AraToken normalization. It is the deployment view, but it is **not** a valid vocabulary-efficiency ranking for tokenizers that emit UNK on this input.

### 3.1 UNK audit

| Tokenizer | Overall | MSA | Egyptian | Gulf | Diacritized |
|---|---|---|---|---|---|
| `aratoken` | 26.19% | 12.10% | 7.23% | 0.10% | 47.93% |
| `arabert` | 12.82% | 1.62% | 1.36% | 0.00% | 98.48% |
| `camelbert` | 11.92% | 1.32% | 1.33% | 0.00% | 98.59% |

`o200k`, `qwen3`, `aratoken_lep` have 0.00% UNK in the raw view. After `aratoken_norm`, overall UNK is `aratoken` 0.31%, `arabert` 0.26%, `camelbert` 0.16%; all other tokenizers are at 0.00%, and the diacritized stratum is 0.00% for every tokenizer.

The raw `aratoken` numbers are the clearest warning. Its overall fertility of **2.318** is built from **26.19% UNK tokens**; on diacritized text, **47.93%** of its tokens are UNK. That is not a tokenizer efficiently encoding Arabic; it is a tokenizer being fed an input distribution it was not trained for. The published pipeline is part of the tokenizer's contract.

### 3.2 All documents

**LLM tokenizers**

| Tokenizer | Class | Tokens | Fertility [95% CI] | Ratio vs aratoken [95% CI] | UNK rate |
|---|---|---|---|---|---|
| `aratoken` (ref) | arabic-specialized | 115,908 | 2.318 [2.195, 2.458] | 1.000 | 26.19% |
| `aratoken_lep` | arabic-specialized-extended | 66,695 | 1.334 [1.314, 1.354] | 0.575x [0.541, 0.614] | 0.00% |
| `o200k` | english-centric-proxy | 118,582 | 2.372 [2.289, 2.458] | 1.023x [1.001, 1.047] | 0.00% |
| `qwen3` | massive-multilingual | 146,556 | 2.931 [2.824, 3.051] | 1.264x [1.239, 1.290] | 0.00% |

**BERT baselines (not LLM tokenizers)**

| Tokenizer | Class | Tokens | Fertility [95% CI] | Ratio vs aratoken [95% CI] | UNK rate |
|---|---|---|---|---|---|
| `arabert` | arabic-bert-baseline | 63,758 | 1.275 [1.252, 1.298] | 0.550x [0.515, 0.586] | 12.82% |
| `camelbert` | arabic-dialect-bert-baseline | 68,075 | 1.361 [1.334, 1.388] | 0.587x [0.547, 0.629] | 11.92% |

### 3.3 Paired tests vs `aratoken`

**LLM tokenizers**

| Tokenizer | Median delta [IQR] | Mean delta | Cohen's dz | Rank-biserial | Docs cand<ref / equal | p (Wilcoxon) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| `aratoken_lep` | -0.071 -0.307 to +0.000 | -0.572 | -0.37 | -0.587 | 52.5% / 23.9% | 4.14e-66 | 4.14e-66 | yes |
| `o200k` | +0.357 +0.167 to +0.500 | +0.196 | +0.34 | +0.579 | 16.7% / 1.5% | 1.00e-82 | 2.01e-82 | yes |
| `qwen3` | +0.673 +0.452 to +0.858 | +0.631 | +1.76 | +0.976 | 5.4% / 1.9% | 1.16e-230 | 3.47e-230 | yes |

**BERT baselines**

| Tokenizer | Median delta [IQR] | Mean delta | Cohen's dz | Rank-biserial | Docs cand<ref / equal | p (Wilcoxon) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| `arabert` | -0.083 -0.333 to +0.000 | -0.625 | -0.40 | -0.794 | 60.4% / 24.3% | 8.64e-119 | 1.73e-118 | yes |
| `camelbert` | -0.085 -0.273 to +0.000 | -0.595 | -0.38 | -0.669 | 62.8% / 15.7% | 2.87e-94 | 2.87e-94 | yes |

Positive `delta` means the candidate uses more tokens per word than `aratoken`; negative means fewer. The raw tests describe token counts, not valid encodings: the `aratoken` reference and the BERT baselines are UNK-contaminated, so their signs should not be read as vocabulary quality.

### 3.4 Why the raw `o200k` result is a composition trap

`o200k` and `aratoken` look almost tied in the raw aggregate: **1.023x [1.001, 1.047]**. That single number hides two opposite populations:

| Statistic | Value |
|---|---|
| Aggregate token ratio | 1.023x [1.001, 1.047] |
| Median per-document delta | +0.357 |
| Documents where `o200k` uses more tokens | 81.8% |
| Documents where `o200k` uses fewer tokens | 16.7% |
| MSA stratum ratio | 1.184x |
| Diacritized stratum ratio | 0.778x |
| Diacritized `aratoken` UNK rate | 47.93% |

On MSA, `o200k` uses **+18.4%** more tokens than `aratoken`. On diacritized text it uses **-22.2%** because raw `aratoken` collapses to UNK. The aggregate near-tie is the weighted sum of those two effects, not evidence that the tokenizers are equivalent on Arabic. Under common normalization, `o200k` is **1.461x** and the ambiguity disappears.

## 4. Controlled view (`aratoken_norm`)

This is the valid tokenizer-efficiency comparison: every tokenizer sees the same normalized text, and UNK is low for all six. The cost is that the view removes diacritics and changes character classes, so it cannot answer "what does diacritized input cost?" That question belongs to the raw view and to the view ablation in section 6.

### 4.1 All documents

**LLM tokenizers**

| Tokenizer | Class | Tokens | Fertility [95% CI] | Ratio vs aratoken [95% CI] | UNK rate |
|---|---|---|---|---|---|
| `aratoken` (ref) | arabic-specialized | 63,487 | 1.270 [1.255, 1.286] | 1.000 | 0.31% |
| `aratoken_lep` | arabic-specialized-extended | 67,853 | 1.357 [1.338, 1.377] | 1.069x [1.058, 1.079] | 0.00% |
| `o200k` | english-centric-proxy | 92,751 | 1.856 [1.835, 1.876] | 1.461x [1.447, 1.476] | 0.00% |
| `qwen3` | massive-multilingual | 111,791 | 2.236 [2.208, 2.265] | 1.761x [1.740, 1.782] | 0.00% |

**BERT baselines (not LLM tokenizers)**

| Tokenizer | Class | Tokens | Fertility [95% CI] | Ratio vs aratoken [95% CI] | UNK rate |
|---|---|---|---|---|---|
| `arabert` | arabic-bert-baseline | 66,108 | 1.323 [1.302, 1.345] | 1.041x [1.034, 1.048] | 0.26% |
| `camelbert` | arabic-dialect-bert-baseline | 70,645 | 1.413 [1.390, 1.437] | 1.113x [1.103, 1.123] | 0.16% |

### 4.2 Paired tests vs `aratoken`

**LLM tokenizers**

| Tokenizer | Median delta [IQR] | Mean delta | Cohen's dz | Rank-biserial | Docs cand<ref / equal | p (Wilcoxon) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| `aratoken_lep` | +0.000 +0.000 to +0.091 | +0.049 | +0.34 | +0.430 | 23.5% / 33.1% | 3.86e-32 | 3.86e-32 | yes |
| `o200k` | +0.483 +0.357 to +0.636 | +0.498 | +2.50 | +1.000 | 0.1% / 0.2% | 4.19e-246 | 8.39e-246 | yes |
| `qwen3` | +0.800 +0.636 to +1.000 | +0.827 | +3.12 | +1.000 | 0.0% / 0.0% | 1.36e-246 | 4.08e-246 | yes |

**BERT baselines**

| Tokenizer | Median delta [IQR] | Mean delta | Cohen's dz | Rank-biserial | Docs cand<ref / equal | p (Wilcoxon) | p (Holm) | Significant |
|---|---|---|---|---|---|---|---|---|
| `arabert` | +0.000 -0.062 to +0.071 | +0.008 | +0.09 | +0.118 | 28.1% / 35.4% | 0.00148 | 0.00148 | yes |
| `camelbert` | +0.047 -0.077 to +0.143 | +0.039 | +0.25 | +0.314 | 28.8% / 19.9% | 4.49e-21 | 8.98e-21 | yes |

The normalized ranking is unambiguous for the two general-purpose tokenizers: every document uses more tokens with `o200k` (rank-biserial **+1.000**, fewer-token rate **0.07%**) and with `qwen3` (rank-biserial **+1.000**). `aratoken_lep` is closer: **1.069x** aggregate, but its median difference is **+0.000**, **33.1%** of documents are exact ties, and **23.5%** favour `aratoken_lep`. Its aggregate penalty is concentrated in MSA, not spread evenly across dialects.

The BERT baselines are competitive in aggregate: `arabert` is **1.041x** and `camelbert` is **1.113x**. But the `arabert` effect is small (dz **+0.09**, median difference **+0.000**, **35.4%** ties); its aggregate penalty comes from MSA. On Egyptian and Gulf it is statistically indistinguishable from `aratoken` (section 5).

## 5. Per-stratum results

Cells show fertility and, in parentheses, the token ratio vs `aratoken` in the same stratum. `*` = Holm-adjusted p < 0.05; `ns` = not significant. `[UNK]` marks raw cells with >5% unknown tokens.

### 5.1 Raw view

| Stratum | `aratoken` (ref) | `aratoken_lep` | `o200k` | `qwen3` | `arabert` | `camelbert` |
|---|---|---|---|---|---|---|
| msa (27,500 words) | 1.669 [UNK] | 1.416 (0.848*) | 1.976 (1.184*) | 2.413 (1.446*) | 1.372 (0.822*) | 1.523 (0.912*) |
| egyptian (10,000 words) | 1.409 [UNK] | 1.270 (0.902*) | 1.713 (1.216*) | 2.028 (1.440*) | 1.224 (0.869*) | 1.250 (0.887*) |
| gulf (5,000 words) | 1.260 | 1.292 (1.025*) | 1.705 (1.353*) | 2.028 (1.610*) | 1.254 (0.996 ns) | 1.239 (0.983*) |
| diacritized (7,500 words) | 6.615 [UNK] | 1.148 (0.173*) | 5.145 (0.778*) | 6.636 (1.003 ns) | 1.002 (0.151*) [UNK] | 1.001 (0.151*) [UNK] |

`*` = Holm-adjusted p < 0.05 within the relevant family; `ns` = not significant. `[UNK]` marks a cell with >5% unknown tokens; its fertility is a token-count artefact, not a valid efficiency measure. `arabert` and `camelbert` are BERT WordPiece baselines, not LLM tokenizers.

The raw table is dominated by the diacritized column. `aratoken` jumps to **6.615** with **47.93% UNK**; `o200k` is **5.145** and `qwen3` is **6.636**. The apparent `o200k` advantage (ratio **0.778x**) is therefore not a vocabulary win; it is the cost of `aratoken` being used without its normalization. The same caveat applies to the BERT cells at the bottom of the diacritized column.

### 5.2 Normalized view

| Stratum | `aratoken` (ref) | `aratoken_lep` | `o200k` | `qwen3` | `arabert` | `camelbert` |
|---|---|---|---|---|---|---|
| msa (27,500 words) | 1.303 | 1.443 (1.108*) | 1.989 (1.526*) | 2.433 (1.867*) | 1.390 (1.067*) | 1.535 (1.178*) |
| egyptian (10,000 words) | 1.235 | 1.284 (1.039*) | 1.705 (1.381*) | 2.016 (1.633*) | 1.231 (0.997 ns) | 1.251 (1.013*) |
| gulf (5,000 words) | 1.260 | 1.292 (1.025*) | 1.705 (1.353*) | 2.028 (1.610*) | 1.254 (0.996 ns) | 1.239 (0.983*) |
| diacritized (7,500 words) | 1.203 | 1.184 (0.985*) | 1.667 (1.386*) | 1.948 (1.620*) | 1.241 (1.032*) | 1.300 (1.081*) |

`*` = Holm-adjusted p < 0.05 within the relevant family; `ns` = not significant. `[UNK]` marks a cell with >5% unknown tokens; its fertility is a token-count artefact, not a valid efficiency measure. `arabert` and `camelbert` are BERT WordPiece baselines, not LLM tokenizers.

The normalized table separates the effects. On MSA, `aratoken` is **1.303**, clearly below `arabert` **1.390**, `aratoken_lep` **1.443**, `camelbert` **1.535**, `o200k` **1.989**, and `qwen3` **2.433**. On Egyptian, `aratoken` **1.235** and `arabert` **1.231** are tied (p_Holm **0.554**); `camelbert` **1.251** and `aratoken_lep` **1.284** are slightly higher. On Gulf, `aratoken` **1.260**, `arabert` **1.254** (ns), and `camelbert` **1.239** are within 1.7% of each other; `aratoken_lep` is **1.292**. On diacritized text, `aratoken_lep` **1.184** is slightly below `aratoken` **1.203**; `o200k` **1.667** and `qwen3` **1.948** are far higher.

The overall normalized advantage is therefore not a uniform "Arabic tokenizer wins everywhere" result. It is a large MSA advantage, near-parity with Arabic BERT baselines on Egyptian and Gulf, and a small `aratoken_lep` advantage on diacritized text.

## 6. View ablation: why the ranking flips

| Tokenizer | raw | nfkc | aratoken_norm | norm/raw [95% CI] | Change | p | p (Holm) | dz |
|---|---|---|---|---|---|---|---|---|
| `aratoken` | 2.318 | 2.318 | 1.270 | 0.548x [0.518, 0.580] | -45.2% | 2.94e-101 | 1.76e-100 | -0.40 |
| `o200k` | 2.372 | 2.372 | 1.856 | 0.782x [0.751, 0.814] | -21.8% | 8.13e-22 | 1.63e-21 | -0.31 |
| `qwen3` | 2.931 | 2.931 | 2.236 | 0.763x [0.728, 0.796] | -23.7% | 4.94e-19 | 4.94e-19 | -0.31 |
| `aratoken_lep` | 1.334 | 1.334 | 1.357 | 1.017x [1.015, 1.019] | +1.7% | 3.92e-53 | 1.96e-52 | +0.36 |
| `arabert` | 1.275 | 1.275 | 1.323 | 1.037x [1.032, 1.042] | +3.7% | 2.90e-49 | 1.16e-48 | +0.35 |
| `camelbert` | 1.361 | 1.362 | 1.413 | 1.038x [1.032, 1.045] | +3.8% | 1.50e-30 | 4.50e-30 | +0.30 |

`nfkc` is effectively inert: across the whole corpus it changes total tokens by at most 8 tokens for any tokenizer. The large effects come from `aratoken_norm`, and they go in both directions.

`aratoken` drops from **2.318** to **1.270** (**+0.0%**), because the normalization removes the diacritics and variant forms that were producing UNK. `o200k` drops **+46.1%** and `qwen3` **+76.1%**, because their raw counts were inflated by diacritics. `aratoken_lep` (**+6.9%**), `arabert` (**+4.1%**), and `camelbert` (**+11.3%**) move slightly the other way: their internal normalizers already handle diacritics, and the external pipeline changes character classes and word boundaries in ways that cost a few tokens.

### 6.1 View effect by stratum

| Tokenizer | msa (norm/raw) | egyptian (norm/raw) | gulf (norm/raw) | diacritized (norm/raw) |
|---|---|---|---|---|
| `aratoken` | 0.781* | 0.877* | 1.000 ns | 0.182* |
| `o200k` | 1.007* | 0.995* | 1.000 ns | 0.323* |
| `qwen3` | 1.008* | 0.994* | 1.000 ns | 0.293* |
| `aratoken_lep` | 1.020* | 1.011* | 1.000 ns | 1.030* |
| `arabert` | 1.013* | 1.006* | 1.000 ns | 1.237* |
| `camelbert` | 1.008* | 1.001 ns | 1.000 ns | 1.297* |

`*` = Holm-adjusted p < 0.05 within the six-tokenizer family for that stratum; `ns` = not significant. These are pipeline effects, not isolated diacritic effects: `aratoken_norm` also unifies alif forms, maps digits and punctuation, removes tatweel, and lowercases ASCII Latin.

The diacritized column is the key. `aratoken` falls to **1.270** overall because its diacritized fertility falls from **6.615** to **1.203** (ratio **1.000**). `o200k` falls from **5.145** to **1.667** and `qwen3` from **6.636** to **1.948**. The BERT baselines move the other way on diacritized text (`arabert` **1.032**, `camelbert` **1.081**), because the external normalization replaces Arabic punctuation/digits and removes tatweel, changing their pre-tokenization. This is a pipeline effect, not a pure diacritic-removal effect.

## 7. Significance summary

| Family | Tests | Significant after Holm | Not significant |
|---|---|---|---|
| raw / all docs / LLM | 3 | 3 | none |
| raw / all docs / BERT | 2 | 2 | none |
| normalized / all docs / LLM | 3 | 3 | none |
| normalized / all docs / BERT | 2 | 2 | none |
| raw / per stratum / LLM | 12 | 11 | `qwen3`/diacritized (p_Holm=0.0909) |
| raw / per stratum / BERT | 8 | 7 | `arabert`/gulf (p_Holm=0.112) |
| normalized / per stratum / LLM | 12 | 12 | none |
| normalized / per stratum / BERT | 8 | 6 | `arabert`/egyptian (p_Holm=0.554), `arabert`/gulf (p_Holm=0.224) |
| view ablation / all docs | 6 | 6 | none |

The largest p-values drive the non-significant cells. In the raw per-stratum LLM family, `qwen3` on diacritized text is not significant (p_Holm **0.0909**): both raw token counts are high, and the difference is small. After normalization the same comparison is highly significant and large (`qwen3` ratio **1.620x**). That contrast is a direct example of why the raw view cannot be used for the tokenizer-efficiency conclusion.

On the BERT side, `arabert` is not distinguishable from `aratoken` on Egyptian and Gulf after Holm correction; `camelbert` is slightly lower than `aratoken` on Gulf but the effect is small. These are encoder tokenizers with a 512-token context, not substitutes for an LLM tokenizer.

## 8. What this does not show

- **No cost result.** Cost is `tokens x price per token`. No provider prices are configured; the cost table remains `-`. A 46.1% token difference does not imply a 46.1% cost difference unless the price per token is equal, and frontier API prices are not equal.
- **No TTFT result.** The TTFT arm was not run. Token count can affect prefill time, but this document has no latency measurement. A 4GB laptop demo would not be a headline TTFT benchmark.
- **No English-Arabic arm.** The corpus is Arabic-only. There is no parallel English corpus and no English tokenizer run, so no "Arabic tax vs English" number can be computed from these data.
- **No quality evaluation.** Fertility is not accuracy. A tokenizer that produces fewer tokens can lose information; the UNK results are a concrete example. No downstream model was trained or evaluated.
- **Corpus-specific.** The corpus is deliberately unbalanced: MSA is 55% of words, Egyptian 20%, Gulf 10%, diacritized 15%. The overall micro-average is dominated by MSA; the per-stratum tables are the honest view.
- **Raw is not a pure vocabulary comparison.** Each tokenizer has its own internal normalizer, and `aratoken` additionally expects its published external pipeline. The raw ranking mixes vocabulary, normalization, and UNK behavior. The normalized view removes that confound but also removes diacritics.
- **`o200k` is not Llama 4.** It is a GPT-4o-class English-centric proxy. The Llama 4 tokenizer is gated and was not available; no substitution is claimed.
- **No paid APIs.** Gemini and other paid provider comparisons were not run. The tokenizer benchmark is CPU-only.
- **Paper comparison is not replication.** The AraToken paper reports about 1.199 tokens/word under its normalization; this corpus gives **1.270**. The difference may reflect corpus mix, preprocessing, or tokenizer revision. This experiment measures this corpus, not the paper's corpus.

## 9. Reproduce

```bash
python compare.py
```

The script reads `results/tokenization.json` and `results/cache/token_counts.jsonl`, validates every token sum against the published metrics, and writes `results/COMPARISON.md` and `results/comparison.json`. It uses only the Python standard library. It does not call a tokenizer or an API. If the inputs are missing, run `python bench.py tokenize` first.

## Appendix: JSON fields

- `all_docs[view][group]`: rows for all documents; `group` is `llm` or `baseline`.
- `strata[view][stratum][group]`: the same rows for one stratum.
- `reference_all` / `reference_strata`: the `aratoken` reference rows, which have no paired test.
- `view_ablation.all_docs` / `view_ablation.strata`: raw-to-normalized paired rows.
- `test`: `p`, `p_holm`, `significant_holm`, `family`, `median_diff`, `mean_diff`, `dz`, `rank_biserial`, win rates, `n_nonzero`, and `n_zero`.
- `ratio_ci`: the existing 95% document-level paired bootstrap interval for the aggregate token ratio.
