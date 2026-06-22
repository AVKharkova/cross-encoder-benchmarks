# Cross-Encoder Benchmark

Скрипт для тестирования и сравнения различных моделей переранжирования (Cross-Encoders) для Retrieval-Augmented Generation (RAG) систем.

## Поддерживаемые модели
- `BAAI/bge-reranker-v2-m3` (2.2 GB) - Мультиязычный тяжеловес
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (90 MB) - Легкая модель
- `DiTy/cross-encoder-russian-msmarco` (400 MB) - Оптимизация под русский язык
- `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (Поддержка через `FlagEmbedding`)

## Установка

1. Склонируйте репозиторий.
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

Запустите скрипт:
```bash
python eval_rerankers.py
```
Скрипт прогонит синтетическую базу данных (встроенную в код) через модели, замерит MRR, Hit Rate@5 и Latency, а затем сохранит результаты в `reranker_benchmark_results.md`.
