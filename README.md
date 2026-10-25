# Arabic Tokenizer Fertility Tax — Experiment 1

> **One-line idea:** tokenizing the same Arabic text with an English-centric, multilingual, or Arabic-specialized tokenizer can produce very different token counts — and therefore different cost and latency. This experiment measures that difference on one fixed Arabic corpus, not on isolated examples.

**Status:** the API-free MVP ran locally on the full 50,000-word corpus (1,499 documents, six open tokenizers, CPU only). Results are in `results/REPORT.md` and `results/TOKENIZATION.md`. The API and TTFT arms are optional and have not been run. Server steps are in `docs/RUNBOOK.md`.

---

## 1) What is measured?

Three separate things, each with a different interpretation:

| Arm | Question | Unit | Output |
|---|---|---|---|
| **1. Tokenizer benchmark** | How many tokens per word? How many characters per token? | tokenizer only (CPU) | `results/TOKENIZATION.md` |
| **2. Cost audit** | What does one million words cost at a provider price? | tokenizer × provider price | `results/REPORT.md` |
| **3. TTFT** | How long until the first token arrives? | model + provider + network | `results/TTFT.md` |

> **Important:** fertility is a tokenizer metric. API cost depends on the provider's price per token. TTFT depends on the model, network, and queueing. **Do not mix the three**, and do not say "AraToken is 2.5× faster" without specifying what is faster, at what price, and on which model.

---

## 2) Hypotheses — not results

These numbers come from the literature or the project brief. We test them; we do not assume them.

| Hypothesis | Source / claim | What we do |
|---|---|---|
| **H1:** an English/multilingual tokenizer has much higher fertility than an Arabic-specialized one | Petrov et al. 2023: gaps up to 15× across languages | Measure on 50k Arabic words and compute a paired bootstrap against AraToken |
| **H2:** AraToken reaches ~1.199 tokens/word | AraToken paper (arXiv 2512.18399) | Measure it on our corpus under `aratoken_norm`, and compare with `raw` |
| **H3:** Llama 4 fertility is ~3.0 | Project brief; **unverified** | Need the exact gated tokenizer; if unavailable, do not substitute `o200k` |
| **H4:** an English model costs ~2.5× more for Arabic | Project brief; **not a universal constant** | Compute `fertility × price` with two real prices, and separate token ratio from price difference |
| **H5:** normalization reduces fertility | CANDLE 2026: reduction up to 12.8% | Compare `raw` / `nfkc` / `aratoken_norm` on the same text, with paired Δ and 95% CI |

**Rule:** every number in `REPORT.md` is a result of this corpus and this run. Literature numbers are for comparison only, not a substitute for measurement.

---

## 3) Corpus

| Stratum | Words | Source | License | Why |
|---|---:|---|---|---|
| **MSA** | 27,500 | `wikimedia/wikipedia` — `20231101.ar` | CC-BY-SA-4.0 | Backbone of written Arabic |
| **Egyptian** | 10,000 | `dataflare/arabic-dialect-corpus` — label `Masri` | MIT | Egyptian dialect test |
| **Gulf** | 5,000 | Same source — label `Khaleeji` | MIT | Gulf dialect test |
| **Diacritized** | 7,500 | `arbml/tashkeela` — field `diacratized` | GPL-2.0 | Full diacritics, the hardest tokenizer case |
| **Total** | **50,000** | | | Exact whitespace words |

- The corpus is **deliberately unbalanced**: MSA is the majority, while dialects and diacritized text are larger than token examples. Results are shown **per stratum**, and the overall micro-average is affected by stratum size.
- The builder enforces the exact word budget, removes duplicates, and applies Arabic-letter and diacritic-ratio filters. The last document in a stratum may be shorter than `min_words` to hit the budget exactly.
- **Do not redistribute `data/corpus.jsonl`:** Tashkeela is GPL-2.0 and Wikipedia is CC-BY-SA. The file is gitignored. `data/smoke.jsonl` is original CC0 text and is not part of the 50k corpus.

Full details: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## 4) Tokenizers

| Name | Class | Source | Access | Note |
|---|---|---|---|---|
| `o200k` | English-centric proxy | `tiktoken` — `o200k_base` | Public | **Not** the Llama 4 tokenizer. A GPT-4o-class proxy |
| `llama4` | English-centric | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | **Gated** | Requires HF approval; if unavailable, leave it absent and do not substitute |
| `qwen3` | Massive multilingual | `Qwen/Qwen3-8B` | Public | Tokenizer only, not the model |
| `gemini` | Multilingual API | Gemini `countTokens` | API key | Tokenizer is not public; the count is provider-reported |
| `aratoken` | Arabic-specialized | `mariklolik/AraToken` — SentencePiece Unigram | Public | 150K vocab; 1.199 in the paper under normalization |
| `aratoken_lep` | Arabic-extended | `mariklolik/AraToken-Qwen3-0.6B-LEP` | Public | 282,826 vocab; for local paired TTFT only |
| `arabert` | Arabic BERT baseline | `aubmindlab/bert-base-arabertv02` | Public | Encoder baseline |
| `camelbert` | Arabic dialect BERT | `CAMeL-Lab/bert-base-arabic-camelbert-mix` | Public | Encoder baseline |

**`aratoken` ≠ `aratoken_lep`:** the first is the paper tokenizer (150K); the second is the modified model tokenizer (282,826) with layers 24–27 trained. A TTFT comparison between them is not a tokenizer-only ablation.

---

## 5) Quick start

### Server (Linux)

```bash
# 1) Check the hardware first — do not assume T100 is a known model name
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# 2) Python 3.10+ (tested on 3.12)
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 3) Keys (optional, depending on the arm)
export HF_TOKEN=...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...

# 4) Environment check
python bench.py doctor --deep

# 5) Build the corpus (first run only)
python bench.py corpus

# 6) Tokenization + cost
python bench.py tokenize

# 7) TTFT — either enable a provider in config.yaml or select one explicitly
python bench.py ttft --dry-run
python bench.py ttft --providers qwen3-0.6b-base-vllm,aratoken-qwen3-0.6b-lep-vllm

# 8) Combined report
python bench.py report
```

> `--config` is a global option and must come **before** the command:
> `python bench.py --config config.yaml tokenize`

### MVP without API (best for the laptop)

```bash
# 1) Corpus
python bench.py corpus

# 2) Tokenization — open tokenizers only
python bench.py tokenize --tokenizers o200k,qwen3,aratoken,aratoken_lep,arabert,camelbert

# 3) Report
python bench.py report
```

- No API calls and no GPU required.
- Gemini, Llama 4, and TTFT are outside the MVP; add them later.
- The metric here is a **token ratio**, not real dollars. The dollar table stays `—` until you fill in provider prices.
- The 4GB VRAM is irrelevant for the MVP; all tokenizers run on CPU. Local TTFT on 4GB is a demo, not a strong benchmark.

**Full-run result (2026-10-25, 50,000 words / 1,499 documents):**

| View | Lowest fertility among LLM tokenizers | Highest fertility among LLM tokenizers | Note |
|---|---|---|---|
| `raw` | `aratoken_lep` **1.334** [1.314, 1.354] | `qwen3` **2.931** [2.824, 3.051] | `arabert` / `camelbert` are WordPiece baselines: 1.275 / 1.361 — not LLM tokenizers |
| `aratoken_norm` | `aratoken` **1.270** [1.255, 1.286] | `qwen3` **2.236** [2.208, 2.265] | Unified comparison view after AraToken normalization; the order reverses vs `raw` |

- `aratoken_norm` removes diacritics, so it is not a substitute for `raw` when measuring diacritized input; use `raw` for that. It is a unified comparison view for the paper's number.
- `o200k` in `raw` = **2.372**, and `aratoken` = **2.318**; the difference is small and is not evidence of sufficiency.
- Paired CIs and per-stratum tables are in `results/REPORT.md`. Dollar cost is `—` because no real prices are configured; there is no price claim without documented prices.
- The smoke run (26 documents / 270 words) validates the pipeline only; it is not a research result.

### Local (smoke only)

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv\Scripts\python.exe bench.py doctor
.\.venv\Scripts\python.exe bench.py corpus --smoke
.\.venv\Scripts\python.exe bench.py tokenize --input data/smoke.jsonl --out-dir results/smoke
.\.venv\Scripts\python.exe bench.py report --out-dir results/smoke
```

The smoke run tests the pipeline on 26 documents / 270 words. **Its numbers are not research results** and must not be used as headlines.

---

## 6) Files

```
.
├── config.yaml          # Everything configurable: corpus, tokenizers, prices, TTFT
├── bench.py             # CLI: doctor / corpus / tokenize / ttft / report / all
├── core.py              # Normalization, corpus builder, adapters, metrics, reports
├── providers.py         # OpenAI-compatible streaming + Gemini REST
├── requirements.txt
├── data/
│   ├── smoke.jsonl      # Original CC0 fixture — not part of the 50k
│   └── corpus.jsonl     # Built locally, not tracked by git
├── docs/
│   ├── RESEARCH.md      # Fact-check + literature
│   ├── METHODOLOGY.md   # Design, equations, statistics, limitations
│   └── RUNBOOK.md       # Server steps, vLLM, troubleshooting
└── results/
    ├── TOKENIZATION.md
    ├── TTFT.md
    ├── REPORT.md
    ├── tokenization.json
    ├── ttft.json
    └── cache/
```

**Few files by design:** core logic in `core.py`, the CLI in `bench.py`, and providers in one file. No microservices or scattered packages.

---

## 7) Outputs — what to read

| File | Contents | When to read |
|---|---|---|
| `results/TOKENIZATION.md` | Fertility, chars/token, CIs, per-stratum, views, paired, cost | Start here |
| `results/TTFT.md` | Median/p95 TTFT, OLS slope, errors | After running TTFT |
| `results/REPORT.md` | Executive summary + all tables | For delivery/review |
| `results/tokenization.json` | The same data, machine-readable, plus metadata | Further analysis |
| `results/ttft_raw.jsonl` | Every raw request, including errors | Audit |
| `data/corpus_manifest.json` | Word counts, sources, revisions, SHA256 | Reproducibility |

---

## 8) Interpretation rules — read before any number

1. **Fertility is not quality.** A tokenizer that produces fewer tokens does not necessarily help the model understand better; the literature is mixed (Alyafeai 2021; Alrefaie 2024; Alakeel 2026).
2. **`raw` ≠ `aratoken_norm`.** In `raw` there is no external normalization; each tokenizer uses its own internal normalizer. In `aratoken_norm` we apply the AraToken pipeline to the text **for every** tokenizer before tokenization. That is why AraToken's raw number is above 1.199, while ~1.199 appears under `aratoken_norm`. But `aratoken_norm` removes diacritics, so `raw` is the view for measuring diacritized input cost.
3. **Cost = fertility × token price.** If the two prices differ, the cost ratio ≠ the fertility ratio. AraToken has no frontier API; any cost comparison with it is **counterfactual** (GPU-hour amortization, not an API price).
4. **TTFT is not an isolated tokenizer effect.** The slope against token count is the closest thing to a size effect; differences between providers are not necessarily caused by tokenization.
5. **CIs are descriptive.** Document-level bootstrap, not hypothesis tests with multiplicity correction. Overlapping CIs are not proof of "no difference".
6. **`o200k` is not Llama 4.** Do not label it Llama 4 in any report.
7. **T100 is not a known NVIDIA product name.** Verify with `nvidia-smi` and record the actual model (T10 16GB? T4? V100?). See `docs/RUNBOOK.md`.

---

## 9) Verification status

| Item | Status |
|---|---|
| Python/venv + dependencies | ✅ Works locally (Python 3.12.15) |
| `doctor`, `corpus --smoke`, `tokenize`, `report` | ✅ Works |
| View ablation paired + CI | ✅ Works on the full 50k corpus |
| TTFT dry-run | ✅ Works |
| TTFT streaming client | ✅ Tested against a fake SSE server |
| Full 50k corpus build | ✅ 50,000 words / 1,499 documents; SHA256 `fbc6e3972e95037f6e318eb402fb18d4de0ede428d888b16f71a372218c3c72a` |
| Six open tokenizers on the full corpus | ✅ All 1,499 documents; 90 metric rows |
| Llama 4 tokenizer | ⏳ Gated; unavailable locally |
| Gemini tokenizer | ⏳ Requires `GEMINI_API_KEY` |
| Falcon Arabic tokenizer | ❌ Not publicly downloadable — excluded |
| T100 GPU | ⏳ Unknown; verify on the server |
| API cost table | ⏳ All `—` until real prices are filled |
| TTFT benchmark | ⏳ Not run; local 4GB is demo only |

Source revisions are pinned in `config.yaml` and recorded in `data/corpus_manifest.json` (Wikipedia `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, dataflare `5e5bfb845bd8abefcee684bb97b05fea0f6231d9`, Tashkeela `96cc2daa40aa7357476ebcaf62392913d75e5a21`).

---

## 10) Documents

- [`docs/RESEARCH.md`](docs/RESEARCH.md) — fact-check, literature, references.
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — design, equations, TTFT protocol, limitations.
- [`docs/RUNBOOK.md`](docs/RUNBOOK.md) — server steps, vLLM, troubleshooting.

---

## License and use

The code is usable within this project; add a `LICENSE` file before public release. The data is not ours: review the licenses for Wikipedia (CC-BY-SA-4.0), dataflare (MIT), and Tashkeela (GPL-2.0). **Do not upload `data/corpus.jsonl` or any text derived from it to git or a public service.**
