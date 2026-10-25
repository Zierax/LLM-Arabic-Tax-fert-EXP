# تقرير Experiment 1 — ضريبة التقطيع العربية (Arabic Fertility Tax)


**تاريخ التقرير:** 2026-10-25T23:38:34+00:00  

**المشروع:** `arabic-fertility-tax`


> **اقرأ قبل الأرقام:** هذا التقرير يفصل ثلاثة أشياء: (1) fertility/compression للـ tokenizer، (2) تكلفة API حسب أسعار حقيقية، (3) TTFT على مستوى الموديل/المزوّد. لا تخلط بينها. التفاصيل الكاملة في `docs/METHODOLOGY.md`، والأدبيات في `docs/RESEARCH.md`.


## الملخص التنفيذي

- **أقل fertility بين tokenizers الـ LLM المتاحة في `raw`:** `aratoken_lep` عند **1.334** tokens/word (95% CI [1.314, 1.354]).
- **أعلى fertility بين tokenizers الـ LLM المتاحة في `raw`:** `qwen3` عند **2.931** tokens/word (95% CI [2.824, 3.051]).
- **بعد `aratoken_norm` (view موحّد للمقارنة):** أقل fertility `aratoken` عند **1.270**، وأعلى `qwen3` عند **2.236**. الترتيب يختلف عن `raw` لأن التطبيع وبعض الـ tokenizers يحذفون التشكيل؛ استخدم `raw` لقياس الإدخال المشكّل.
- **خط الأساس (BERT WordPiece) في `raw`:** `arabert` 1.275، `camelbert` 1.361. هذه tokenizers تشفيرية وليست توليدية؛ fertility المنخفضة عندها لا تعني تكلفة API أقل.
- **النسبة مقابل `aratoken` في `raw` بين tokenizers الـ LLM:** من **0.58×** إلى **1.26×** في نفس الكوربس (أقل من 1× يعني tokens أقل من المرجع؛ وليست سعراً).
- الكوربس: **50,000** كلمة، **1,499** مستند، بنسبة حروف عربية **94.4%**.
- **TTFT:** لا توجد نتائج كافية بعد. شغّل `python bench.py ttft` بعد تفعيل مزوّد واحد على الأقل.

> **الخلاصة لا تُكتب قبل القياس.** الأرقام المنشورة في الأدبيات (مثل 1.199 لـ AraToken أو 3.0 لـ Llama) ليست أرقام هذا الكوربس. هذا التقرير يقيسها على **50,000** كلمة، وهي كل الكوربس المستهدف، بنفس الـ views وبنفس البروتوكول.


## 1) تكوين الكوربس


| Stratum | Docs | Words | Arabic letters | Diacritics | License |
|---|---|---|---|---|---|
| msa | 170 | 27,500 | 91.1% | 0.3% | CC-BY-SA-4.0 |
| egyptian | 806 | 10,000 | 98.5% | 0.5% | MIT |
| gulf | 394 | 5,000 | 99.1% | 0.0% | MIT |
| diacritized | 129 | 7,500 | 99.9% | 79.0% | GPL-2.0 (data; do not redistribute the built corpus) |
| **الإجمالي** | **1,499** | **50,000** | **94.4%** | **11.3%** |  |


> الكوربس المستخدم في هذا التقرير: **50,000 كلمة** whitespace. نسبة الحروف العربية ونسبة التشكيل محسوبتان على النص نفسه، وليستا تقديراً. هذا هو حجم التشغيل الكامل المستهدف.


## 2) النتيجة الأساسية — view = `raw`


| Tokenizer | Class | Docs | Fertility | 95% CI | Chars/token | Tokens / 50,000 words | × aratoken |
|---|---|---|---|---|---|---|---|
| o200k | english-centric-proxy | 1,499 | 2.372 | [2.289, 2.458] | 2.57 | 118,582 | 1.02× |
| qwen3 | massive-multilingual | 1,499 | 2.931 | [2.824, 3.051] | 2.08 | 146,556 | 1.26× |
| aratoken | arabic-specialized | 1,499 | 2.318 | [2.195, 2.458] | 2.63 | 115,908 | 1.00× |
| aratoken_lep | arabic-specialized-extended | 1,499 | 1.334 | [1.314, 1.354] | 4.58 | 66,695 | 0.58× |
| arabert | arabic-bert-baseline | 1,499 | 1.275 | [1.252, 1.298] | 4.79 | 63,758 | 0.55× |
| camelbert | arabic-dialect-bert-baseline | 1,499 | 1.361 | [1.334, 1.388] | 4.48 | 68,075 | 0.59× |


**Fertility** = tokens ÷ whitespace words. **CI** = 95% document-level bootstrap interval (not a model-confidence interval). **× aratoken** is the token ratio relative to the reference tokenizer (below 1× means fewer tokens); it is not a price and does not include provider pricing.



> **Docs:** لو ظهر `X/Y`، فالـ tokenizer غطّى X مستنداً فقط من Y (مثلاً `--limit` أو فشل جزئي). في هذه الحالة `Tokens / target words` إسقاط خطي وليس قياساً كاملاً، والـ CI مبني على المستندات المتاحة فقط.


## 3) حسب الطبقة (per stratum)


| Tokenizer | msa | egyptian | gulf | diacritized |
|---|---|---|---|---|
| o200k | 1.976 | 1.713 | 1.705 | 5.145 |
| qwen3 | 2.413 | 2.028 | 2.028 | 6.636 |
| aratoken | 1.669 | 1.409 | 1.260 | 6.615 |
| aratoken_lep | 1.416 | 1.270 | 1.292 | 1.148 |
| arabert | 1.372 | 1.224 | 1.254 | 1.002 |
| camelbert | 1.523 | 1.250 | 1.239 | 1.001 |


> الطبقات ليست متساوية في الحجم: الفصحى 27,500 كلمة، المصري 10,000، الخليجي 5,000، المشكّل 7,500. المقارنة داخل كل طبقة، وليس بين الطبقات.



> **ملاحظة عن التشكيل:** بعض الـ tokenizers (مثل AraBERT/CAMeLBERT) تزيل التشكيل داخلياً، لذلك `raw` على الطبقة المشكّلة يظهر قريباً من 1.0. AraToken لا يزيل التشكيل داخلياً، لذلك `raw` عنده مرتفع، و`aratoken_norm` هو الـ view الموحّد للمقارنة مع رقم الورقة. لكن التطبيع يحذف التشكيل، فاستخدم `raw` لقياس تكلفة الإدخال المشكّل نفسه. هذا سلوك الـ tokenizer، وليس تطبيعاً خارجياً.


## 4) تأثير التطبيع (view ablation)


| Tokenizer | `raw` | `nfkc` | `aratoken_norm` | Δ norm−raw | 95% CI للفرق | Ratio |
|---|---|---|---|---|---|---|
| o200k | 2.372 | 2.372 | 1.856 | -0.517 | [-0.611, -0.425] | 0.782× |
| qwen3 | 2.931 | 2.931 | 2.236 | -0.695 | [-0.828, -0.574] | 0.763× |
| aratoken | 2.318 | 2.318 | 1.270 | -1.049 | [-1.181, -0.921] | 0.548× |
| aratoken_lep | 1.334 | 1.334 | 1.357 | +0.023 | [0.021, 0.026] | 1.017× |
| arabert | 1.275 | 1.275 | 1.323 | +0.047 | [0.041, 0.054] | 1.037× |
| camelbert | 1.361 | 1.362 | 1.413 | +0.051 | [0.044, 0.060] | 1.038× |


> `raw` = النص كما هو. `nfkc` = توحيد Unicode فقط. `aratoken_norm` = خط أنابيب AraToken (NFKC + حذف التشكيل + توحيد الألف + تطبيع الأرقام/الترقيم). الـ view يُطبَّق قبل الـ tokenizer؛ بعض الـ tokenizers تطبّع داخلياً أيضاً.

> **Δ** = `aratoken_norm` − `raw` بوحدة tokens/word، محسوب paired على نفس المستندات. القيمة السالبة تعني أن التطبيع قلّل الـ tokens. الـ CI هو document-level bootstrap للفرق، وليس CI حول نسبة. `Ratio` = tokens بعد التطبيع ÷ tokens قبلها.


## 5) الفرق paired مقابل `aratoken`


| Tokenizer | Ratio vs reference | 95% CI | Δ tokens/word | 95% CI | n docs |
|---|---|---|---|---|---|
| o200k | 1.023× | [1.001, 1.047] | +0.053 | [0.001, 0.103] | 1,499 |
| qwen3 | 1.264× | [1.239, 1.290] | +0.613 | [0.584, 0.642] | 1,499 |
| aratoken_lep | 0.575× | [0.541, 0.614] | -0.984 | [-1.124, -0.846] | 1,499 |
| arabert | 0.550× | [0.515, 0.586] | -1.043 | [-1.188, -0.907] | 1,499 |
| camelbert | 0.587× | [0.547, 0.629] | -0.957 | [-1.112, -0.813] | 1,499 |



### view = `aratoken_norm`


| Tokenizer | Ratio vs reference | 95% CI | Δ tokens/word | 95% CI | n docs |
|---|---|---|---|---|---|
| o200k | 1.461× | [1.447, 1.476] | +0.585 | [0.568, 0.602] | 1,499 |
| qwen3 | 1.761× | [1.740, 1.782] | +0.966 | [0.943, 0.990] | 1,499 |
| aratoken_lep | 1.069× | [1.058, 1.079] | +0.087 | [0.074, 0.101] | 1,499 |
| arabert | 1.041× | [1.034, 1.048] | +0.052 | [0.043, 0.062] | 1,499 |
| camelbert | 1.113× | [1.103, 1.123] | +0.143 | [0.130, 0.156] | 1,499 |




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
| qwen3 | ok | massive-multilingual | hf | — |
| aratoken | ok | arabic-specialized | sentencepiece | — |
| aratoken_lep | ok | arabic-specialized-extended | hf | — |
| arabert | ok | arabic-bert-baseline | hf | — |
| camelbert | ok | arabic-dialect-bert-baseline | hf | — |

## 8) TTFT

لم يتم تشغيل arm الـ TTFT بعد.


## 9) القيود وحدود التفسير

- **Fertility ليست جودة.** التقطيع الأقل ليس بالضرورة الأفضل للفهم أو التوليد؛ الأدبيات (مثل Morphemes Without Borders, LREC 2026) تحذر من الربط المباشر بين التوافق المورفولوجي والأداء.
- **AraBERT/CAMeLBERT خط أساس تشفيري.** هما WordPiece BERT tokenizers وليسا LLM tokenizers؛ fertility المنخفضة عندهما لا تعني تكلفة API أقل ولا جودة أعلى. استخدمهما كمرجع لغوي، لا كبديل عن tokenizer موديل توليدي.
- **تكلفة الـ API ليست ضريبة tokenizer فقط.** السعر لكل token يختلف بين المزوّدين، وهناك خصومات caching/batch، وفواتير output منفصلة. AraToken ليس له API frontier؛ مقارنة تكلفته بأي API هي **counterfactual**.
- **TTFT ليس تأثير tokenizer معزولاً.** يشمل الشبكة، queueing، معالجة الـ prompt، وأحياناً thinking. الـ slope مع عدد الـ tokens هو أقرب شيء لتأثير الحجم، وليس سبباً خالصاً للتقطيع.
- **الكوربس غير متوازن عمداً.** الفصحى 27,500 كلمة، المصري 10,000، الخليجي 5,000، المشكّل 7,500. لا تعمّم النتائج على «العربية» ككل من دون ذكر الطبقة.
- **مصادر البيانات مختلفة.** Wikipedia (CC-BY-SA)، dataflare (MIT)، Tashkeela (GPL-2.0). الكوربس المبني لا يُعاد توزيعه. الـ smoke fixture أصلي (CC0) وليس جزءاً من الـ 50k.
- **قيود الوصول:** Llama 4 gated؛ Gemini tokenizer غير عام (countTokens فقط)؛ Falcon Arabic tokenizer غير متاح للتحميل العام؛ T100 غير معروف بالضبط.
- **الـ CIs وصفية.** هي bootstrap على مستوى المستندات، وليست اختبارات فرض مع تصحيح multiplicity. لا تقرأ «تداخل الـ CI» كأنه «لا فرق».


## 10) الاستنساخ

```bash
python bench.py doctor --deep
python bench.py corpus
python bench.py tokenize --tokenizers o200k,qwen3,aratoken,aratoken_lep,arabert,camelbert
python bench.py report
# اختياري: python bench.py ttft
```

كل الأوامر تقبل `--config config.yaml` **قبل** اسم الأمر. النتائج الخام في `results/`، والكاش في `results/cache/`. بصمة الكوربس ومراجعات المصادر في `data/corpus_manifest.json`.

