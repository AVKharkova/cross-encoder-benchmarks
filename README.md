# Cross-Encoder Benchmark

Скрипт для тестирования и сравнения различных моделей переранжирования (Cross-Encoders) для Retrieval-Augmented Generation (RAG) систем.

## Поддерживаемые модели

### Лёгкие / быстрые
- `cross-encoder/ms-marco-TinyBERT-L-2-v2` (~65 MB)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (~90 MB)
- `cross-encoder/ms-marco-MiniLM-L-12-v2` (~120 MB)
- `mixedbread-ai/mxbai-rerank-xsmall-v1` (~120 MB)

### Мультиязычные и русскоязычные
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (~120 MB) - Мультиязычный MiniLM
- `DiTy/cross-encoder-russian-msmarco` (~400 MB) - Оптимизация под русский язык
- `maidalun1020/bce-reranker-base_v1` (~550 MB)
- `jinaai/jina-reranker-v2-base-multilingual` (~800 MB) - Требует `trust_remote_code=True`

### Базовые и тяжёлые reranker
- `BAAI/bge-reranker-base` (~1.1 GB)
- `BAAI/bge-reranker-v2-m3` (~2.2 GB)
- `mixedbread-ai/mxbai-rerank-base-v1` (~550 MB)
- `BAAI/bge-reranker-large` (~1.5 GB)
- `mixedbread-ai/mxbai-rerank-large-v1` (~1.2 GB)

### LLM-based reranker
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (~3.5 GB) - Через `FlagEmbedding`. **На текущем CPU-сервере не запускается** (слишком медленно, >1.5 ч на бенчмарк).
- `BAAI/bge-reranker-v2-gemma` (~4 GB) - Через `FlagEmbedding`. **На текущем CPU-сервере не запускается** (OOM).

### Дополнительные reranker
- `castorini/monot5-base-msmarco` (~1 GB) - T5-based pointwise reranker
- `castorini/rankllama-v1-7b-lora-passage` (~14 GB FP16) - LLaMA-2 7B + LoRA sequence-classification reranker. **Требует GPU / ≥16 GB RAM**.
- `ielab/TILDEv2-TILDE200-exp` (~400 MB + TILDE expansion) - Лексический reranker с on-the-fly passage expansion

### Новые компактные мультиязычные reranker
- `Qwen/Qwen3-Reranker-0.6B` (~1.2 GB) - Компактный кросс-энкодер на базе Qwen3.
- `jinaai/jina-reranker-v3` (~1.2 GB) - Listwise reranker на базе Qwen3 (требует `trust_remote_code=True`).

## Описание метрик

В результате работы скрипта замеряются следующие ключевые показатели:

1. **MRR (Mean Reciprocal Rank)**: Оценивает позицию правильного ответа после переранжирования. 
   - Значение `1.0` означает, что нужный документ всегда на 1-м месте.
   - Значение `0.5` означает, что он в среднем на 2-м месте, и т.д. Чем ближе к единице, тем лучше.
2. **Hit Rate@5**: Показывает процент случаев, когда правильный ответ попадает в "Топ-5" кандидатов. Это важно для RAG систем, так как обычно генеративной модели (LLM) передаются именно 3-5 лучших кусков текста.
3. **Latency (50 docs)**: Время (в секундах), необходимое модели на инференс (сравнение вопроса с 50 текстовыми кандидатами). Меньше = быстрее.
4. **RAM/VRAM**: Примерное потребление оперативной или видеопамяти при загрузке модели.

## Установка

1. Склонируйте репозиторий.
2. Создайте виртуальное окружение (рекомендуется):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
*(Для тестирования моделей `Alibaba-NLP/gte-Qwen2-1.5B-instruct` и `BAAI/bge-reranker-v2-gemma` дополнительно потребуется выполнить `pip install FlagEmbedding`)*

## Предварительная загрузка моделей

Перед первым запуском бенчмарка можно скачать все модели в кэш Hugging Face Hub:

```bash
python download_models.py
```

Или через виртуальное окружение:

```bash
.venv/bin/python download_models.py
```

После этого `eval_rerankers.py` будет использовать локальные файлы вместо повторной загрузки.

Чтобы проверить, что все модели уже закэшированы, без попыток скачивания:

```bash
LOCAL_FILES_ONLY=1 python download_models.py
```

## Использование

Запустите скрипт:
```bash
python eval_rerankers.py
```

Или через виртуальное окружение:
```bash
.venv/bin/python eval_rerankers.py
```
Скрипт прогонит синтетическую базу данных (встроенную в код) через модели, вычислит метрики, а затем сохранит результаты в `reranker_benchmark_results.md`.

На машинах с небольшим объёмом RAM рекомендуется запускать каждую модель в отдельном процессе, чтобы память полностью освобождалась между моделями:
```bash
python run_benchmark.py
# продолжить с уже полученными результатами:
python run_benchmark.py --resume
# уменьшить batch size для тяжёлых моделей на CPU:
python run_benchmark.py --batch-size 1
```

Дополнительные опции `eval_rerankers.py`:
```bash
# список моделей с индексами
python eval_rerankers.py --list-models

# запуск одной модели
python eval_rerankers.py --model-index 0

# запуск по части имени
python eval_rerankers.py --model-name monot5

# уменьшить batch size на CPU с малым RAM
python eval_rerankers.py --batch-size 4

# пропустить модели, помеченные как неподдерживаемые на сервере
python eval_rerankers.py --skip-server
```

Датасет включает 12 кейсов на русском и смешанном англо-русском языке. Каждый кейс содержит 50 кандидатов: 49 жёстких негативов (в том числе query-specific hard negatives) и 1 правильный ответ. Типы кейсов: exact SKU match, brand substitution, technical spec, manual lookup, error code, multi-hop compatibility, numerical spec, multilingual query, obsolete replacement, symptom troubleshooting, compatibility matrix и operating range.
