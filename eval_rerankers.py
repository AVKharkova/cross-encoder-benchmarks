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

GENERIC_NOISE = [
    "Процедура оформления отпуска описана во внутреннем регламенте.",
    "Система видеонаблюдения работает в штатном режиме.",
    "Вчера на сервере произошел сбой из-за перегрева дисков.",
    "Чтобы получить доступ к JIRA, напишите заявку в Service Desk.",
    "Отдел кадров находится на третьем этаже, кабинет 305.",
    "Настройка VPN для удаленной работы требует установки клиента Cisco AnyConnect.",
    "Корпоративная политика безопасности запрещает использование личных флешек.",
    "Заказ канцелярии производится через портал административно-хозяйственного отдела до 15 числа.",
    "Еженедельное совещание ИТ-отдела переносится на пятницу 10:00.",
    "Принтер на втором этаже нуждается в замене картриджа."
]

# Размножим шум до 49 элементов
noise_pool = (GENERIC_NOISE * 5)[:49]

DATASET = [
    {
        "question": "Как сбросить пароль от корпоративной почты?",
        "ground_truth": "Для сброса пароля от корпоративной почты перейдите на портал IDM (Identity Management), введите свой табельный номер и нажмите 'Забыли пароль'. Временный пароль придет по SMS.",
        "type": "exact_match"
    },
    {
        "question": "Где находится отдел кадров?",
        "ground_truth": "Служба управления персоналом (отдел кадров) расположена в главном офисе на третьем этаже, кабинет 305. Часы приема: с 10:00 до 16:00.",
        "type": "semantic_match"
    },
    {
        "question": "Как подать заявку на ремонт компьютера?",
        "ground_truth": "В случае поломки рабочей станции или ноутбука, необходимо создать тикет в системе Service Desk (раздел 'Ремонт и обслуживание ИТ-оборудования').",
        "type": "concept_match"
    },
    {
        "question": "какой VPN клиент нужен?",
        "ground_truth": "Подключение к внутренней сети компании осуществляется с помощью VPN-клиента Cisco AnyConnect. Инструкция по настройке доступна на интранет-портале.",
        "type": "typo_and_short"
    },
    {
        "question": "Ошибка 404 при входе в JIRA",
        "ground_truth": "Ошибка 404 при доступе к JIRA обычно означает, что ваш профиль не привязан к проекту. Обратитесь к администратору проекта для предоставления прав доступа.",
        "type": "technical_error"
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
                model = FlagReranker(model_name, use_fp16=True)
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
                        
                logging.info(f"Вопрос {idx+1}: Rank={rank}, Latency={infer_time:.3f}s")
                
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
