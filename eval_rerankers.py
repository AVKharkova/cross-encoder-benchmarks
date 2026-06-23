import time
import gc
import logging
try:
    import torch
except ImportError:
    torch = None

from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. Фейковая база данных (Golden Dataset) ---
# Имитируем выдачу гибридного поиска (1 правильный ответ и 49 шумов)

HARD_NOISE = [
    "Насос Grundfos CR 10-3 (арт. 96517001) имеет производительность 10 м³/ч. Уплотнение HQQE.",
    "Масляный фильтр Fleetguard LF3349 используется на двигателях Cummins ISF 2.8.",
    "Масло Mobil SHC 624 (ISO VG 32) применяется в высокоскоростных подшипниках.",
    "Газоанализатор Dräger Pac 6000 калибруется смесью CO.",
    "Частотный преобразователь Danfoss VLT Micro Drive FC 51: ошибка E-04 означает перегрузку.",
    "Для насосов Sulzer серии AHLSTAR рекомендуется использовать торцевые уплотнения Burgmann.",
    "Самосвалы Caterpillar 777E требуют замены масла каждые 500 моточасов.",
    "Смазка SKF LGMT 2/0.4 подходит для подшипников электродвигателей.",
    "Датчик давления WIKA S-10 (арт. 8361001) имеет диапазон 0-100 бар.",
    "Реле температуры Danfoss KP 79 (арт. 060L112666) используется в системах охлаждения."
]

# Размножим шум до 49 элементов
noise_pool = (HARD_NOISE * 5)[:49]

DATASET = [
    {
        "question": "Какие характеристики у насоса Grundfos CR 32-4-2 A-F-A-E-HQQE?",
        "ground_truth": "Насос Grundfos CR 32-4-2 (арт. 96122005) имеет производительность 30 м³/ч и максимальный напор 60 м. Материал корпуса - чугун EN-GJL-200. Оснащен торцевым уплотнением HQQE.",
        "type": "exact_sku_match"
    },
    {
        "question": "Чем заменить фильтр Fleetguard LF9009 на самосвале Komatsu?",
        "ground_truth": "Для самосвалов Komatsu HD785-7 оригинальный масляный фильтр Fleetguard LF9009 может быть заменен на аналоги: Donaldson P553000 или Baldwin B7217. При замене обязательно проверьте уплотнительное кольцо.",
        "type": "brand_substitution"
    },
    {
        "question": "Какое масло лить в редуктор конвейера FLSmidth?",
        "ground_truth": "В приводные редукторы ленточных конвейеров FLSmidth серии T-Rex рекомендуется заливать синтетическое редукторное масло Mobil SHC 630 (ISO VG 220). Объем заливки указан в паспорте изделия.",
        "type": "technical_spec"
    },
    {
        "question": "Инструкция по калибровке газоанализатора Dräger X-am 5000",
        "ground_truth": "Для калибровки портативного газоанализатора Dräger X-am 5000 (арт. 8320000) используйте тестовую газовую смесь (кат. номер 6811130). Подключите адаптер, подайте газ с расходом 0.5 л/мин и дождитесь стабилизации показаний датчиков CatEx и O2.",
        "type": "manual_lookup"
    },
    {
        "question": "ошибка E-04 на частотнике Danfoss VLT AutomationDrive",
        "ground_truth": "Код ошибки E-04 (Mains Phase Loss) на преобразователях частоты Danfoss VLT AutomationDrive FC 302 указывает на обрыв одной из фаз питающей сети. Проверьте входные предохранители и напряжение на клеммах L1, L2, L3.",
        "type": "error_code"
    }
]

# Подготовим кандидатов для каждого кейса
for item in DATASET:
    # 49 шумов + 1 правильный = 50 кандидатов
    item["candidates"] = noise_pool.copy()
    item["candidates"].append(item["ground_truth"])

# --- 2. Модели для тестирования ---
MODELS = [
    {"name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "size": "~90 MB"},
    {"name": "DiTy/cross-encoder-russian-msmarco", "size": "~400 MB"},
    {"name": "BAAI/bge-reranker-v2-m3", "size": "~2.2 GB"},
    {"name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "size": "~3.5 GB", "use_flag_embedding": True}
]

# --- 3. Логика тестирования ---
def evaluate_models():
    results = []
    
    for model_info in MODELS:
        model_name = model_info["name"]
        logging.info(f"\n{'='*50}\nТестирование модели: {model_name}\n{'='*50}")
        
        try:
            kwargs = model_info.get("kwargs", {})
            # Загрузка модели
            start_load = time.time()
            if model_info.get("use_flag_embedding"):
                from FlagEmbedding import FlagReranker
                use_fp16 = torch.cuda.is_available() if torch else False
                model = FlagReranker(model_name, use_fp16=use_fp16)
            else:
                model = CrossEncoder(model_name, max_length=512, **kwargs)
            load_time = time.time() - start_load
            logging.info(f"Модель загружена за {load_time:.2f} сек.")
            
            latencies = []
            mrr_sum = 0.0
            hit_at_5_count = 0
            
            # Прогон по датасету
            for idx, item in enumerate(DATASET):
                question = item["question"]
                candidates = item["candidates"]
                ground_truth = item["ground_truth"]
                
                pairs = [[question, doc] for doc in candidates]
                
                # Замеряем время инференса
                start_infer = time.time()
                if model_info.get("use_flag_embedding"):
                    scores = model.compute_score(pairs)
                else:
                    scores = model.predict(pairs)
                infer_time = time.time() - start_infer
                latencies.append(infer_time)
                
                # Сортируем документы по убыванию оценки
                scored_docs = list(zip(scores, candidates))
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                
                # Ищем позицию правильного ответа (Rank)
                rank = -1
                for r, (score, doc) in enumerate(scored_docs):
                    if doc == ground_truth:
                        rank = r + 1 # 1-indexed
                        break
                
                if rank != -1:
                    mrr_sum += 1.0 / rank
                    if rank <= 5:
                        hit_at_5_count += 1
                        
                logging.info(f"--- Кейс {idx+1} ---")
                logging.info(f"Вопрос: {question}")
                logging.info(f"Ожидаемый ответ: {ground_truth}")
                logging.info(f"Топ-1 кандидат: {scored_docs[0][1]}")
                logging.info(f"Результат: Rank={rank}, Latency={infer_time:.3f}s\n")
                
            # Агрегация метрик
            avg_mrr = mrr_sum / len(DATASET)
            hit_rate_5 = (hit_at_5_count / len(DATASET)) * 100
            avg_latency = sum(latencies) / len(latencies)
            
            results.append({
                "Model": model_name.split('/')[-1],
                "MRR": f"{avg_mrr:.3f}",
                "Hit Rate@5": f"{hit_rate_5:.1f}%",
                "Latency (50 docs)": f"{avg_latency:.3f}s",
                "Size": model_info["size"]
            })
            
        except Exception as e:
            logging.error(f"Ошибка при тестировании {model_name}: {e}")
            results.append({
                "Model": model_name.split('/')[-1],
                "MRR": "ERROR",
                "Hit Rate@5": "ERROR",
                "Latency (50 docs)": "ERROR",
                "Size": model_info["size"]
            })
            
        finally:
            # Очистка памяти (предотвращение OOM)
            if 'model' in locals():
                del model
            gc.collect()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Очистка VRAM завершена.")
                
    return results

def print_and_save_results(results):
    md_table = "| Модель | MRR | Hit Rate@5 | Latency (50 docs) | RAM/VRAM |\n"
    md_table += "|---|---|---|---|---|\n"
    
    for r in results:
        md_table += f"| {r['Model']} | {r['MRR']} | {r['Hit Rate@5']} | {r['Latency (50 docs)']} | {r['Size']} |\n"
        
    print("\n\n=== РЕЗУЛЬТАТЫ БЕНЧМАРКИНГА ===")
    print(md_table)
    
    with open("reranker_benchmark_results.md", "w", encoding="utf-8") as f:
        f.write("# Результаты бенчмаркинга моделей Cross-Encoder\n\n")
        f.write(md_table)
        f.write("\n\n*Примечание: Тестирование проводилось на синтетической выборке из 5 русскоязычных кейсов (по 50 кандидатов каждый).*")
    print("Результаты сохранены в reranker_benchmark_results.md")

if __name__ == "__main__":
    results = evaluate_models()
    print_and_save_results(results)
