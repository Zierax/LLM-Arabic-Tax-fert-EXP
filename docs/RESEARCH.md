# Research — brief fact-check

- **Petrov 2023**: up to 15x token difference across languages (`token inflation` varies). Not a specific Llama 4 number.
- **AraToken paper**: reports 1.199 (`aratoken_norm`). Our measurement: 1.270 [1.255, 1.286] (`n`=1499, `view`=`aratoken_norm`, `stratum`=`__all__`). Difference = corpus/preprocessing/revision, not contradiction.
- **No universal Arabic fertility number**: depends on `stratum`, `view`, tokenizer.
- **`o200k proxy`** is not Llama 4 (`llama4` gated/unavailable).
- **`provider-reported count`** (`gemini`) vs **`local tokenizer count`** (all others).
- **Fertility ≠ quality** (Alrefaie 2024; Alakeel 2026). Raw `aratoken` UNK-heavy (`unk_rate`=26.19% overall, 47.93% `diacritized`) is not efficiency.
- **`cost ratio`** requires real provider price; no prices configured (`results/REPORT.md` = `-`).
- **`TTFT slope (ms/token)`** requires verified GPU + vLLM; not measured.
- **No English-Arabic arm**: corpus Arabic-only; no parallel comparison possible.
