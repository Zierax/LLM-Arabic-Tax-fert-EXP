# تقرير التقطيع — Arabic Fertility Tax

- **تاريخ التشغيل:** 2026-10-25T11:44:48+00:00
- **المدخل:** `D:\Axioms\LLM Arabic Tax fert EXP\data\smoke.jsonl`
- **المستندات:** 26 — **الكلمات:** 270
- **الـ views:** `raw`, `nfkc`, `aratoken_norm`
- **المرجع:** `aratoken`

> **اقرأ قبل الأرقام:** fertility هنا مقياس tokenizer فقط. تكلفة الـ API تعتمد على سعر المزوّد، و TTFT يعتمد على الموديل والشبكة. لا تخلط الثلاثة. التفاصيل في `docs/METHODOLOGY.md`.

## 1) تكوين الكوربس


| Stratum | Docs | Words | Arabic letters | Diacritics | License |
|---|---|---|---|---|---|
| msa | 8 | 100 | 100.0% | 0.0% | CC0-1.0 |
| egyptian | 6 | 61 | 100.0% | 0.0% | CC0-1.0 |
| gulf | 6 | 59 | 100.0% | 0.0% | CC0-1.0 |
| diacritized | 6 | 50 | 100.0% | 81.2% | CC0-1.0 |
| **الإجمالي** | **26** | **270** | **100.0%** | **15.2%** |  |


> الكوربس المستخدم في هذا التقرير: **270 كلمة** whitespace. نسبة الحروف العربية ونسبة التشكيل محسوبتان على النص نفسه، وليستا تقديراً. هذا تشغيل جزئي (الهدف 50,000 كلمة)؛ لا تعرضه كرقم رئيسي.


## 2) النتيجة الأساسية — view = `raw`


| Tokenizer | Class | Docs | Fertility | 95% CI | Chars/token | Tokens / 270 words | × aratoken |
|---|---|---|---|---|---|---|---|
| o200k | english-centric-proxy | 26 | 2.507 | [1.958, 3.200] | 2.55 | 677 | 1.01× |
| qwen3 | massive-multilingual | 26 | 3.137 | [2.396, 4.024] | 2.04 | 847 | 1.27× |
| aratoken | arabic-specialized | 26 | 2.478 | [1.755, 3.434] | 2.58 | 669 | 1.00× |
| aratoken_lep | arabic-specialized-extended | 26 | 1.152 | [1.101, 1.202] | 5.54 | 311 | 0.46× |
| arabert | arabic-bert-baseline | 26 | 1.141 | [1.079, 1.205] | 5.60 | 308 | 0.46× |
| camelbert | arabic-dialect-bert-baseline | 26 | 1.189 | [1.122, 1.257] | 5.37 | 321 | 0.48× |


**Fertility** = tokens ÷ whitespace words. **CI** = 95% document-level bootstrap interval (not a model-confidence interval). **× aratoken** is token inflation relative to the reference tokenizer; it is not a price and does not include provider pricing.



> **Docs:** لو ظهر `X/Y`، فالـ tokenizer غطّى X مستنداً فقط من Y (مثلاً `--limit` أو فشل جزئي). في هذه الحالة `Tokens / target words` إسقاط خطي وليس قياساً كاملاً، والـ CI مبني على المستندات المتاحة فقط.


## 3) حسب الطبقة (per stratum)


| Tokenizer | msa | egyptian | gulf | diacritized |
|---|---|---|---|---|
| o200k | 1.460 | 2.000 | 1.797 | 6.060 |
| qwen3 | 1.860 | 2.344 | 2.136 | 7.840 |
| aratoken | 1.190 | 1.590 | 1.542 | 7.240 |
| aratoken_lep | 1.050 | 1.262 | 1.271 | 1.080 |
| arabert | 1.030 | 1.279 | 1.305 | 1.000 |
| camelbert | 1.100 | 1.295 | 1.390 | 1.000 |


> الطبقات ليست متساوية في الحجم: الفصحى 27,500 كلمة، المصري 10,000، الخليجي 5,000، المشكّل 7,500. المقارنة داخل كل طبقة، وليس بين الطبقات.



> **ملاحظة عن التشكيل:** بعض الـ tokenizers (مثل AraBERT/CAMeLBERT) تزيل التشكيل داخلياً، لذلك `raw` على الطبقة المشكّلة يظهر قريباً من 1.0. AraToken لا يزيل التشكيل داخلياً، لذلك `raw` عنده مرتفع، و`aratoken_norm` هو الـ view العادل للمقارنة. هذا سلوك الـ tokenizer، وليس تطبيعاً خارجياً.


## 4) تأثير التطبيع (view ablation)


| Tokenizer | `raw` | `nfkc` | `aratoken_norm` | Δ norm−raw | 95% CI للفرق | Ratio |
|---|---|---|---|---|---|---|
| o200k | 2.507 | 2.507 | 1.715 | -0.793 | [-1.474, -0.250] | 0.684× |
| qwen3 | 3.137 | 3.137 | 2.067 | -1.070 | [-1.942, -0.332] | 0.659× |
| aratoken | 2.478 | 2.478 | 1.178 | -1.300 | [-2.258, -0.541] | 0.475× |
| aratoken_lep | 1.152 | 1.152 | 1.181 | +0.030 | [0.007, 0.055] | 1.026× |
| arabert | 1.141 | 1.141 | 1.167 | +0.026 | [0.007, 0.050] | 1.023× |
| camelbert | 1.189 | 1.189 | 1.211 | +0.022 | [-0.007, 0.057] | 1.019× |


> `raw` = النص كما هو. `nfkc` = توحيد Unicode فقط. `aratoken_norm` = خط أنابيب AraToken (NFKC + حذف التشكيل + توحيد الألف + تطبيع الأرقام/الترقيم). الـ view يُطبَّق قبل الـ tokenizer؛ بعض الـ tokenizers تطبّع داخلياً أيضاً.

> **Δ** = `aratoken_norm` − `raw` بوحدة tokens/word، محسوب paired على نفس المستندات. القيمة السالبة تعني أن التطبيع قلّل الـ tokens. الـ CI هو document-level bootstrap للفرق، وليس CI حول نسبة. `Ratio` = tokens بعد التطبيع ÷ tokens قبلها.


## 5) الفرق paired مقابل `aratoken`


| Tokenizer | Ratio vs reference | 95% CI | Δ tokens/word | 95% CI | n docs |
|---|---|---|---|---|---|
| o200k | 1.012× | [0.929, 1.139] | +0.030 | [-0.233, 0.251] | 26 |
| qwen3 | 1.266× | [1.180, 1.396] | +0.659 | [0.545, 0.774] | 26 |
| aratoken_lep | 0.465× | [0.339, 0.659] | -1.326 | [-2.212, -0.606] | 26 |
| arabert | 0.460× | [0.338, 0.665] | -1.337 | [-2.190, -0.576] | 26 |
| camelbert | 0.480× | [0.338, 0.695] | -1.289 | [-2.246, -0.532] | 26 |



### view = `aratoken_norm`


| Tokenizer | Ratio vs reference | 95% CI | Δ tokens/word | 95% CI | n docs |
|---|---|---|---|---|---|
| o200k | 1.456× | [1.380, 1.540] | +0.537 | [0.450, 0.633] | 26 |
| qwen3 | 1.755× | [1.662, 1.854] | +0.889 | [0.802, 0.976] | 26 |
| aratoken_lep | 1.003× | [0.959, 1.058] | +0.004 | [-0.050, 0.066] | 26 |
| arabert | 0.991× | [0.961, 1.021] | -0.011 | [-0.048, 0.023] | 26 |
| camelbert | 1.028× | [0.985, 1.075] | +0.033 | [-0.019, 0.086] | 26 |




> الـ bootstrap paired يعيد أخذ نفس المستندات للطرفين، لذلك الـ CI أضيق من مقارنة عشوائية. النسبة = fertility للـ tokenizer ÷ fertility للمرجع.


## 6) التكلفة (cost audit) — view = `raw`


| Tokenizer | Input $/1M tokens | $/1M words | Cost ratio vs ref | Price as of |
|---|---|---|---|---|
| o200k | — | — | — | — |
| qwen3 | — | — | — | — |
| aratoken | — | — | — | — |
| aratoken_lep | — | — | — | — |
| arabert | — | — | — | — |
| camelbert | — | — | — | — |


> التكلفة = fertility × سعر المليون token. الحساب يستخدم الـ view الأساسي `raw` حتى تبقى الأرقام متسقة مع الجدول الأساسي. الأسعار الفعلية فارغة افتراضياً لأنها تتغير؛ املأ `pricing` في `config.yaml` من صفحة المزوّد يوم التشغيل. AraToken ليس له API frontier، لذلك مقارنة تكلفة الـ API معه هي **counterfactual** وليست سعراً حقيقياً.


## 7) حالة الـ tokenizers


| Tokenizer | Status | Class | Kind | Note / error |
|---|---|---|---|---|
| o200k | ok | english-centric-proxy | tiktoken | — |
| llama4 | unavailable | english-centric | hf | llama4: You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct. 401 Client Error. (Request ID: Root=1-6addeb92-7534159d6801bb751fa3fb8c;7 |
| qwen3 | ok | massive-multilingual | hf | — |
| gemini | unavailable | massive-multilingual-api | gemini_count | gemini: provider 'gemini-2.5-flash': missing environment variable 'GEMINI_API_KEY' |
| aratoken | ok | arabic-specialized | sentencepiece | — |
| aratoken_lep | ok | arabic-specialized-extended | hf | — |
| arabert | ok | arabic-bert-baseline | hf | — |
| camelbert | ok | arabic-dialect-bert-baseline | hf | — |
