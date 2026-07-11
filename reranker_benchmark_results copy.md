# Результаты бенчмаркинга моделей Cross-Encoder

| Модель | MRR | Hit Rate@5 | Latency (50 docs) | RAM/VRAM |
|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 0.667 | 60.0% | 1.102s | ~90 MB |
| cross-encoder-russian-msmarco | 0.667 | 60.0% | 2.517s | ~400 MB |
| bge-reranker-m3 | ERROR | ERROR | ERROR | ~2.2 GB |
| gte-Qwen2-1.5B-instruct | ERROR | ERROR | ERROR | ~3.5 GB |


*Примечание: Тестирование проводилось на синтетической выборке из 12 кейсов на русском и смешанном англо-русском языке (по 50 кандидатов каждый).*