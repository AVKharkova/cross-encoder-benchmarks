# Результаты бенчмаркинга моделей Cross-Encoder

| Модель | MRR | Hit Rate@5 | Latency (50 docs) | RAM/VRAM |
|---|---|---|---|---|
| ms-marco-TinyBERT-L-2-v2 | 0.878 | 100.0% | 0.129s | ~65 MB |
| ms-marco-MiniLM-L-6-v2 | 0.675 | 83.3% | 1.369s | ~90 MB |
| ms-marco-MiniLM-L-12-v2 | 0.869 | 91.7% | 2.637s | ~120 MB |
| mxbai-rerank-xsmall-v1 | 0.626 | 83.3% | 3.507s | ~120 MB |
| mmarco-mMiniLMv2-L12-H384-v1 | 0.760 | 91.7% | 1.104s | ~120 MB |
| cross-encoder-russian-msmarco | 0.785 | 91.7% | 3.479s | ~400 MB |
| bce-reranker-base_v1 | 0.806 | 83.3% | 4.316s | ~550 MB |
| jina-reranker-v2-base-multilingual | 0.711 | 100.0% | 25.327s | ~800 MB |
| bge-reranker-base | 0.769 | 91.7% | 3.801s | ~1.1 GB |
| bge-reranker-v2-m3 | 0.708 | 100.0% | 15.032s | ~2.2 GB |
| mxbai-rerank-base-v1 | 0.573 | 91.7% | 6.344s | ~550 MB |
| bge-reranker-large | 0.833 | 100.0% | 14.673s | ~1.5 GB |
| mxbai-rerank-large-v1 | 0.812 | 100.0% | 21.028s | ~1.2 GB |
| gte-Qwen2-1.5B-instruct | ERROR | ERROR | ERROR | ~3.5 GB |
| bge-reranker-v2-gemma | ERROR | ERROR | ERROR | ~4 GB |
| monot5-base-msmarco | 0.866 | 91.7% | 9.004s | ~1 GB |
| rankllama-v1-7b-lora-passage | ERROR | ERROR | ERROR | ~14 GB (FP16) |
| TILDEv2-TILDE200-exp | 0.351 | 66.7% | 16.531s | ~400 MB + ~1 GB (TILDE expansion) |
| Qwen3-Reranker-0.6B | 0.828 | 91.7% | 51.043s | ~1.2 GB |
| jina-reranker-v3 | 0.903 | 100.0% | 40.978s | ~1.2 GB |


*Примечание: Тестирование проводилось на синтетической выборке из 12 кейсов на русском и смешанном англо-русском языке (по 50 кандидатов каждый). Добавлены monoT5, RankLLaMA-7B, TILDEv2, Qwen3-Reranker-0.6B и jina-reranker-v3. Модели `BAAI/bge-reranker-v2-gemma`, `Alibaba-NLP/gte-Qwen2-1.5B-instruct` и `castorini/rankllama-v1-7b-lora-passage` на данном CPU-сервере не запущены: Gemma/RankLLaMA не помещаются в RAM, Qwen 1.5B считает >1.5 ч на 12 кейсов и был остановлен. Их веса удалены из кэша, но они оставлены в таблице и выводах как ERROR. Qwen3-Reranker-0.6B и jina-reranker-v3 удалось запустить после доработок (см. раздел «Ошибки и доработки»).

## Анализ результатов

### Ключевые наблюдения

1. **Лидер по качеству и скорости — `ms-marco-TinyBERT-L-2-v2`**  
   MRR 0.878, Hit Rate@5 100%, latency 0.129 с, размер ~65 МБ. Лучший выбор для production на CPU.

2. **monoT5-base показал высокое качество**  
   MRR 0.866, Hit Rate@5 91.7%, latency 9.0 с. Это сопоставимо с лучшими cross-encoder, хотя модель обучена на английском MS MARCO и не оптимизирована под русский.

3. **jina-reranker-v3 — лучшее качество среди протестированных**  
   MRR 0.903, Hit Rate@5 100.0%, latency 40.978 с. Listwise-архитектура на базе Qwen3 хорошо справляется с русскоязычными и смешанными кейсами, но требует chunked scoring (chunk_size=4) для запуска на CPU с 5.7 ГБ RAM.

4. **Qwen3-Reranker-0.6B — сопоставимое качество**  
   MRR 0.828, Hit Rate@5 91.7%, latency 51.043 с. Модель чувствительна к batch size: при batch_size > 1 процесс убивается OOM-killer на CPU. Для запуска использован batch_size=1.

5. **TILDEv2 уступает нейронным reranker**  
   MRR 0.351, Hit Rate@5 66.7%. На русскоязычных кейсах модель (BERT + лексическое exact matching) работает слабо, т.к. TILDE/TILDEv2 заточены под английскую лексику и требуют passage expansion. Результат получен с on-the-fly expansion.

6. **RankLLaMA-7B не запущен**  
   В окружении CPU + 5.7 ГБ RAM загрузка 7B модели невозможна. Требуется GPU или ≥16 ГБ RAM.

7. **Ошибки и доработки скрипта**
   - `jina-reranker-v2-base-multilingual` — ранее падал из-за отсутствия `einops`. После добавления `einops` в `requirements.txt` модель успешно запустилась (MRR 0.711, HR@5 100%).
   - `BAAI/bge-reranker-v2-gemma` и `castorini/rankllama-v1-7b-lora-passage` — не помещаются в RAM сервера (5.7 ГБ), процессы убиты OOM-killer. Их веса удалены из кэша, модели помечены `skip_on_server`.
   - `Alibaba-NLP/gte-Qwen2-1.5B-instruct` — технически загружается, но на CPU один кейс занимает ~270 с; полный бенчмарк занял бы >1.5 ч. Остановлен и помечен `skip_on_server`. В коде также исправлен отсутствующий `pad_token_id` у Qwen2-токенизатора.
   - `mxbai-rerank-base-v1` / `mxbai-rerank-large-v1` — первый полный прогон завершился OOM-kill из-за фрагментации памяти при последовательной загрузке. После перехода к отдельным процессам на модель и уменьшения batch size обе модели успешно отработали.
   - `Qwen/Qwen3-Reranker-0.6B` — при стандартном batch_size=8/4 падал с OOM. В `MODELS` задан `batch_size=1`, модель успешно отработала.
   - `jinaai/jina-reranker-v3` — listwise reranker пытался обработать все 50 документов разом и был убит OOM-killer. Добавлен `JinaV3Scorer` с chunked scoring (chunk_size=4) и `gc.collect()` между чанками; модель успешно отработала. В `run_benchmark.py` добавлена опция `--batch-size` для удобного управления batch size без редактирования кода.

### Рекомендации

| Сценарий | Рекомендуемая модель | Обоснование |
|---|---|---|
| Production / высокая нагрузка | `ms-marco-TinyBERT-L-2-v2` | Лучшее качество, минимальная задержка и память |
| Баланс качества и скорости | `ms-marco-MiniLM-L-12-v2` | MRR 0.869, latency 2.6 с |
| Нужна скорость, качество вторично | `ms-marco-MiniLM-L-6-v2` | 1.37 с, 90 МБ |
| Мультиязычность с ограниченными ресурсами | `mmarco-mMiniLMv2-L12-H384-v1` | 1.1 с, MRR 0.760 |
| Максимальный Hit Rate@5, latency не критична | `jina-reranker-v3` | 100% HR@5, MRR 0.903 (chunked x4 на CPU) |
| Максимальный Hit Rate@5, требуется стандартный CrossEncoder | `bge-reranker-large` | 100% HR@5, MRR 0.833 |
| Англоязычный pointwise reranker | `castorini/monot5-base-msmarco` | MRR 0.866, но медленнее TinyBERT |

### Дальнейшие шаги

1. Протестировать `RankLLaMA-7B`, `gte-Qwen2-1.5B-instruct`, `bge-reranker-v2-gemma`, `Qwen3-Reranker-0.6B` и `jina-reranker-v3` на машине с ≥16 ГБ RAM / GPU (с большим batch size и без chunked scoring).
2. Провести тестирование на репрезентативном датасете (сотни кейсов).
3. Замерить пропускную способность (throughput) и batch-инференс для выбранных кандидатов.
4. Оптимизировать TILDEv2 под русский язык (русский BERT + expansion) или использовать его только как лексический бейзлайн.
5. Добавить graceful degradation в `JinaV3Scorer`: автоматически уменьшать `chunk_size` при OOM, чтобы не подбирать его вручную.
