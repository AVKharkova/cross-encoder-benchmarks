import json
import time
import gc
import logging
import os
import re
import sys

try:
    import torch
except ImportError:
    torch = None

import numpy as np
from sentence_transformers import CrossEncoder

try:
    from FlagEmbedding import FlagReranker
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FlagReranker = None
    FLAG_EMBEDDING_AVAILABLE = False

try:
    from transformers import (
        T5Tokenizer,
        T5ForConditionalGeneration,
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
        BertConfig,
        BertModel,
        BertLMHeadModel,
        BertTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = PeftConfig = None
    PEFT_AVAILABLE = False

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_system_memory_bytes():
    """Возвращает объём физической памяти в байтах."""
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, AttributeError):
        return None


# --- 1. Фейковая база данных (Golden Dataset) ---
# Имитируем выдачу гибридного поиска (1 правильный ответ и 49 шумов).
# Шум специально подобран максимально близким к запросам, чтобы усложнить ранжирование.

HARD_NOISE = [
    # Похожие SKU и спецификации
    "Насос Grundfos CR 10-3 (арт. 96517001) имеет производительность 10 м³/ч. Уплотнение HQQE.",
    "Насос Grundfos CR 32-4-2 (арт. 96122004) имеет производительность 25 м³/ч и напор 45 м. Материал корпуса - нержавеющая сталь.",
    "Насос Grundfos CR 32-4-2 A-F-A-E-HQQE имеет производительность 32 м³/ч и максимальный напор 55 м.",
    "Насос Grundfos CR 32-4-2 (арт. 96122006) имеет производительность 30 м³/ч и максимальный напор 60 м. Материал корпуса - бронза.",
    # Фильтры и замены
    "Масляный фильтр Fleetguard LF3349 используется на двигателях Cummins ISF 2.8.",
    "Масляный фильтр Fleetguard LF9009 применяется на двигателях Caterpillar C15.",
    "Для самосвалов Komatsu HD785-7 оригинальный масляный фильтр Fleetguard LF9009 может быть заменен на аналоги: Donaldson P553000 или Baldwin B7218.",
    "Фильтр Fleetguard LF9009 может быть заменен на Donaldson P554004.",
    "Самосвалы Komatsu HD785-7 используют фильтр Fleetguard LF670.",
    # Масла и смазки
    "Масло Mobil SHC 624 (ISO VG 32) применяется в высокоскоростных подшипниках.",
    "В приводные редукторы ленточных конвейеров FLSmidth серии T-Rex рекомендуется заливать минеральное редукторное масло Mobil SHC 624 (ISO VG 32).",
    "Редуктор конвейера FLSmidth серии T-Rex требует масла ISO VG 460.",
    "Масло Mobil SHC 630 (ISO VG 220) применяется в низкоскоростных редукторах.",
    "Смазка SKF LGMT 2/0.4 подходит для подшипников электродвигателей.",
    "Смазка SKF LGMT 3/0.4 предназначена для подшипников с высокими нагрузками.",
    "Смазка SKF LGEP 2/0.4 применяется в высокотемпературных узлах.",
    # Газоанализаторы
    "Газоанализатор Dräger Pac 6000 калибруется смесью CO.",
    "Для калибровки портативного газоанализатора Dräger X-am 5000 используйте тестовую газовую смесь (кат. номер 6811131).",
    "Для калибровки газоанализатора Dräger X-am 5000 используйте расход газа 1.0 л/мин.",
    "Газоанализатор Dräger X-am 5000 калибруется смесью O2 и CH4.",
    # Частотники Danfoss
    "Частотный преобразователь Danfoss VLT Micro Drive FC 51: ошибка E-04 означает перегрузку.",
    "Код ошибки E-04 (Overcurrent) на преобразователях частоты Danfoss VLT AutomationDrive FC 302 указывает на перегрузку по току.",
    "Частотный преобразователь Danfoss VLT AutomationDrive FC 302: ошибка E-05 означает перенапряжение.",
    "Частотный преобразователь Danfoss VLT Micro Drive FC 51 и VLT AutomationDrive FC 302 используют одинаковые коды ошибок.",
    # Датчики и реле
    "Датчик давления WIKA S-10 (арт. 8361001) имеет диапазон 0-100 бар.",
    "Датчик давления WIKA A-10 (арт. 8361002) имеет диапазон 0-160 бар.",
    "Датчик давления WIKA S-11 (арт. 8361003) имеет диапазон 0-250 бар, выход 4-20 мА.",
    "Реле температуры Danfoss KP 79 (арт. 060L112666) используется в системах охлаждения.",
    "Реле температуры Danfoss KP 81 (арт. 060L112667) используется в системах отопления.",
    # Разное
    "Для насосов Sulzer серии AHLSTAR рекомендуется использовать торцевые уплотнения Burgmann.",
    "Самосвалы Caterpillar 777E требуют замены масла каждые 500 моточасов.",
    "Торцевое уплотнение HQQE не совместимо с насосами Grundfos серии CR.",
    "Насос Grundfos CR 32-4-2 A-F-A-E-HQQE оснащен торцевым уплотнением BUBE.",
]

# Размножим шум до 49 элементов
noise_pool = (HARD_NOISE * 2)[:49]

DATASET = [
    {
        "question": "Какие характеристики у насоса Grundfos CR 32-4-2 A-F-A-E-HQQE?",
        "ground_truth": "Насос Grundfos CR 32-4-2 (арт. 96122005) имеет производительность 30 м³/ч и максимальный напор 60 м. Материал корпуса - чугун EN-GJL-200. Оснащен торцевым уплотнением HQQE.",
        "type": "exact_sku_match",
        "hard_negatives": [
            "Насос Grundfos CR 32-4-2 A-F-A-E-HQQE имеет производительность 32 м³/ч и максимальный напор 55 м.",
            "Насос Grundfos CR 32-4-2 (арт. 96122004) имеет производительность 25 м³/ч и напор 45 м. Материал корпуса - нержавеющая сталь.",
            "Насос Grundfos CR 32-4-2 (арт. 96122006) имеет производительность 30 м³/ч и максимальный напор 60 м. Материал корпуса - бронза.",
            "Торцевое уплотнение HQQE не совместимо с насосами Grundfos серии CR.",
            "Насос Grundfos CR 32-4-2 A-F-A-E-HQQE оснащен торцевым уплотнением BUBE.",
        ]
    },
    {
        "question": "Чем заменить фильтр Fleetguard LF9009 на самосвале Komatsu?",
        "ground_truth": "Для самосвалов Komatsu HD785-7 оригинальный масляный фильтр Fleetguard LF9009 может быть заменен на аналоги: Donaldson P553000 или Baldwin B7217. При замене обязательно проверьте уплотнительное кольцо.",
        "type": "brand_substitution",
        "hard_negatives": [
            "Для самосвалов Komatsu HD785-7 оригинальный масляный фильтр Fleetguard LF9009 может быть заменен на аналоги: Donaldson P553000 или Baldwin B7218.",
            "Фильтр Fleetguard LF9009 может быть заменен на Donaldson P554004.",
            "Масляный фильтр Fleetguard LF9009 применяется на двигателях Caterpillar C15.",
            "Самосвалы Komatsu HD785-7 используют фильтр Fleetguard LF670.",
        ]
    },
    {
        "question": "Какое масло лить в редуктор конвейера FLSmidth?",
        "ground_truth": "В приводные редукторы ленточных конвейеров FLSmidth серии T-Rex рекомендуется заливать синтетическое редукторное масло Mobil SHC 630 (ISO VG 220). Объем заливки указан в паспорте изделия.",
        "type": "technical_spec",
        "hard_negatives": [
            "В приводные редукторы ленточных конвейеров FLSmidth серии T-Rex рекомендуется заливать минеральное редукторное масло Mobil SHC 624 (ISO VG 32).",
            "Редуктор конвейера FLSmidth серии T-Rex требует масла ISO VG 460.",
            "Масло Mobil SHC 630 (ISO VG 220) применяется в низкоскоростных редукторах.",
        ]
    },
    {
        "question": "Инструкция по калибровке газоанализатора Dräger X-am 5000",
        "ground_truth": "Для калибровки портативного газоанализатора Dräger X-am 5000 (арт. 8320000) используйте тестовую газовую смесь (кат. номер 6811130). Подключите адаптер, подайте газ с расходом 0.5 л/мин и дождитесь стабилизации показаний датчиков CatEx и O2.",
        "type": "manual_lookup",
        "hard_negatives": [
            "Для калибровки портативного газоанализатора Dräger X-am 5000 используйте тестовую газовую смесь (кат. номер 6811131).",
            "Для калибровки газоанализатора Dräger X-am 5000 используйте расход газа 1.0 л/мин.",
            "Газоанализатор Dräger X-am 5000 калибруется смесью O2 и CH4.",
            "Газоанализатор Dräger Pac 6000 калибруется смесью CO.",
        ]
    },
    {
        "question": "ошибка E-04 на частотнике Danfoss VLT AutomationDrive",
        "ground_truth": "Код ошибки E-04 (Mains Phase Loss) на преобразователях частоты Danfoss VLT AutomationDrive FC 302 указывает на обрыв одной из фаз питающей сети. Проверьте входные предохранители и напряжение на клеммах L1, L2, L3.",
        "type": "error_code",
        "hard_negatives": [
            "Код ошибки E-04 (Overcurrent) на преобразователях частоты Danfoss VLT AutomationDrive FC 302 указывает на перегрузку по току.",
            "Частотный преобразователь Danfoss VLT AutomationDrive FC 302: ошибка E-05 означает перенапряжение.",
            "Частотный преобразователь Danfoss VLT Micro Drive FC 51: ошибка E-04 означает перегрузку.",
        ]
    },
    {
        "question": "Какое торцевое уплотнение выбрать для насоса Grundfos CR 32-4-2 при агрессивной среде?",
        "ground_truth": "Для насосов Grundfos CR 32-4-2 в агрессивных средах рекомендуется торцевое уплотнение HQQE из карбида кремния/графита с фиксированной конструкцией вала. Артикул комплекта: 96455086.",
        "type": "multi_hop_compatibility",
        "hard_negatives": [
            "Для насосов Sulzer серии AHLSTAR рекомендуется использовать торцевые уплотнения Burgmann.",
            "Торцевое уплотнение HQQE не совместимо с насосами Grundfos серии CR.",
            "Насос Grundfos CR 32-4-2 A-F-A-E-HQQE оснащен торцевым уплотнением BUBE.",
        ]
    },
    {
        "question": "Датчик давления 0-250 бар, выход 4-20 мА, для гидросистемы",
        "ground_truth": "Датчик давления WIKA S-11 (арт. 8361003) имеет диапазон 0-250 бар, выход 4-20 мА, питание 10-30 В DC. Корпус из нержавеющей стали, защита IP65.",
        "type": "numerical_spec",
        "hard_negatives": [
            "Датчик давления WIKA S-10 (арт. 8361001) имеет диапазон 0-100 бар.",
            "Датчик давления WIKA A-10 (арт. 8361002) имеет диапазон 0-160 бар.",
        ]
    },
    {
        "question": "Pump Grundfos CR 32-4-2 performance and seal type",
        "ground_truth": "Насос Grundfos CR 32-4-2 (арт. 96122005) имеет производительность 30 м³/ч и максимальный напор 60 м. Материал корпуса - чугун EN-GJL-200. Оснащен торцевым уплотнением HQQE.",
        "type": "multilingual_query",
        "hard_negatives": [
            "Насос Grundfos CR 32-4-2 A-F-A-E-HQQE имеет производительность 32 м³/ч и максимальный напор 55 м.",
            "Насос Grundfos CR 32-4-2 (арт. 96122004) имеет производительность 25 м³/ч и напор 45 м. Материал корпуса - нержавеющая сталь.",
        ]
    },
    {
        "question": "Старый насос Grundfos CR 32-4-2 снят с производства, на что заменить?",
        "ground_truth": "Насос Grundfos CR 32-4-2 (арт. 96122005) имеет производительность 30 м³/ч и максимальный напор 60 м. Материал корпуса - чугун EN-GJL-200. Оснащен торцевым уплотнением HQQE.",
        "type": "obsolete_replacement",
        "hard_negatives": [
            "Насос Grundfos CR 10-3 (арт. 96517001) имеет производительность 10 м³/ч. Уплотнение HQQE.",
            "Насос Grundfos CR 32-4-2 (арт. 96122004) имеет производительность 25 м³/ч и напор 45 м. Материал корпуса - нержавеющая сталь.",
        ]
    },
    {
        "question": "Частотник Danfoss не запускается, горит ошибка, пропало питание",
        "ground_truth": "Код ошибки E-04 (Mains Phase Loss) на преобразователях частоты Danfoss VLT AutomationDrive FC 302 указывает на обрыв одной из фаз питающей сети. Проверьте входные предохранители и напряжение на клеммах L1, L2, L3.",
        "type": "symptom_troubleshooting",
        "hard_negatives": [
            "Код ошибки E-04 (Overcurrent) на преобразователях частоты Danfoss VLT AutomationDrive FC 302 указывает на перегрузку по току.",
            "Частотный преобразователь Danfoss VLT AutomationDrive FC 302: ошибка E-05 означает перенапряжение.",
        ]
    },
    {
        "question": "Смазка для подшипников электродвигателя, работающего при температуре до 120°C",
        "ground_truth": "Смазка SKF LGEP 2/0.4 предназначена для подшипников, работающих при повышенных температурах (до 130°C), высоких нагрузках и в условиях влажности.",
        "type": "compatibility_matrix",
        "hard_negatives": [
            "Смазка SKF LGMT 2/0.4 подходит для подшипников электродвигателей.",
            "Смазка SKF LGMT 3/0.4 предназначена для подшипников с высокими нагрузками.",
            "Масло Mobil SHC 624 (ISO VG 32) применяется в высокоскоростных подшипниках.",
        ]
    },
    {
        "question": "Реле температуры для холодильной установки с диапазоном -5...+30°C",
        "ground_truth": "Реле температуры Danfoss KP 79 (арт. 060L112666) используется в системах охлаждения, диапазон регулировки -5...+30°C, дифференциал 2-10 K.",
        "type": "operating_range",
        "hard_negatives": [
            "Реле температуры Danfoss KP 81 (арт. 060L112667) используется в системах отопления.",
            "Реле температуры Danfoss KP 79 (арт. 060L112666) используется в системах охлаждения.",
        ]
    },
]

# Подготовим кандидатов для каждого кейса:
# case-specific hard negatives в начале, затем общий шум, в конце правильный ответ.
# Итого: 49 негативов + 1 правильный = 50 кандидатов.
for item in DATASET:
    negatives = item.get("hard_negatives", []) + noise_pool.copy()
    negatives = negatives[:49]
    while len(negatives) < 49:
        negatives.extend(noise_pool)
        negatives = negatives[:49]
    item["candidates"] = negatives + [item["ground_truth"]]

# --- 2. Модели для тестирования ---
MODELS = [
    # Лёгкие / быстрые
    {"name": "cross-encoder/ms-marco-TinyBERT-L-2-v2", "size": "~65 MB"},
    {"name": "cross-encoder/ms-marco-MiniLM-L-6-v2", "size": "~90 MB"},
    {"name": "cross-encoder/ms-marco-MiniLM-L-12-v2", "size": "~120 MB"},
    {"name": "mixedbread-ai/mxbai-rerank-xsmall-v1", "size": "~120 MB"},

    # Мультиязычные и русскоязычные
    {"name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", "size": "~120 MB"},
    {"name": "DiTy/cross-encoder-russian-msmarco", "size": "~400 MB"},
    {"name": "maidalun1020/bce-reranker-base_v1", "size": "~550 MB"},
    {"name": "jinaai/jina-reranker-v2-base-multilingual", "size": "~800 MB",
     "kwargs": {"trust_remote_code": True, "max_length": 1024}},

    # Базовые и тяжёлые reranker
    {"name": "BAAI/bge-reranker-base", "size": "~1.1 GB"},
    {"name": "BAAI/bge-reranker-v2-m3", "size": "~2.2 GB"},
    {"name": "mixedbread-ai/mxbai-rerank-base-v1", "size": "~550 MB"},
    {"name": "BAAI/bge-reranker-large", "size": "~1.5 GB"},
    {"name": "mixedbread-ai/mxbai-rerank-large-v1", "size": "~1.2 GB"},

    # LLM-based reranker (через FlagEmbedding)
    {"name": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "size": "~3.5 GB", "use_flag_embedding": True, "skip_on_server": True},
    {"name": "BAAI/bge-reranker-v2-gemma", "size": "~4 GB", "use_flag_embedding": True, "skip_on_server": True},

    # Дополнительные модели по запросу
    {"name": "castorini/monot5-base-msmarco", "size": "~1 GB", "use_t5": True},
    {"name": "castorini/rankllama-v1-7b-lora-passage", "size": "~14 GB (FP16)", "use_rankllama": True, "skip_on_server": True},
    {"name": "ielab/TILDEv2-TILDE200-exp", "size": "~400 MB + ~1 GB (TILDE expansion)", "use_tildev2": True},

    # Новые компактные мультиязычные reranker (Qwen3)
    {"name": "Qwen/Qwen3-Reranker-0.6B", "size": "~1.2 GB", "batch_size": 1},
    {"name": "jinaai/jina-reranker-v3", "size": "~1.2 GB", "use_jina_v3": True, "batch_size": 1},
]


# --- Вспомогательные reranker ---

class MonoT5Scorer:
    """Pointwise T5 reranker (monoT5).

    Формат входа: Query: <q> Document: <d> Relevant:
    Скор — вероятность генерации токена true.
    """
    def __init__(self, model_name, device='cpu'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers не установлен")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device).eval()
        self.device = device
        true_id = self.tokenizer.encode("true", add_special_tokens=False)[0]
        false_id = self.tokenizer.encode("false", add_special_tokens=False)[0]
        self.label_ids = torch.tensor([false_id, true_id], device=device)

    def score(self, pairs, batch_size=8):
        scores = []
        decoder_input_ids_template = torch.full(
            (batch_size, 1),
            self.model.config.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            texts = [f"Query: {q} Document: {d} Relevant:" for q, d in batch]
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            cur_bs = inputs['input_ids'].size(0)
            decoder_input_ids = decoder_input_ids_template[:cur_bs]
            with torch.no_grad():
                outputs = self.model(**inputs, decoder_input_ids=decoder_input_ids)
                logits = outputs.logits[:, 0, :]
                selected = logits[:, self.label_ids]
                probs = torch.softmax(selected, dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
        return np.array(scores)


class RankLLaMAScorer:
    """Sequence-classification reranker на базе LLaMA-2 7B + LoRA (RankLLaMA)."""
    def __init__(self, model_name='castorini/rankllama-v1-7b-lora-passage', device='cpu'):
        if not TRANSFORMERS_AVAILABLE or not PEFT_AVAILABLE:
            raise ImportError("Для RankLLaMA нужны transformers и peft")
        config = PeftConfig.from_pretrained(model_name)
        base_name = config.base_model_name_or_path
        # meta-llama/Llama-2-7b-hf — gated; пробуем открытый эквивалент при необходимости
        fallback = 'NousResearch/Llama-2-7b-hf' if 'meta-llama' in base_name.lower() else base_name
        dtype = torch.float16 if device == 'cuda' else torch.float32
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_name,
                num_labels=1,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        except Exception as e:
            logging.warning(f"Не удалось загрузить базу {base_name}: {e}. Пробуем {fallback}")
            base_model = AutoModelForSequenceClassification.from_pretrained(
                fallback,
                num_labels=1,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(fallback)
        model = PeftModel.from_pretrained(base_model, model_name)
        self.model = model.merge_and_unload().to(device).eval()
        self.tokenizer = tokenizer
        self.device = device

    def score(self, pairs, batch_size=1):
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            texts_a = [f"query: {q}" for q, d in batch]
            texts_b = [f"document: {d}" for q, d in batch]
            inputs = self.tokenizer(
                texts_a,
                texts_b,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
            scores.extend(logits.cpu().tolist())
        return np.array(scores)


class JinaV3Scorer:
    """Listwise reranker jina-reranker-v3 (JinaForRanking на базе Qwen3).

    Модель принимает список документов целиком; на CPU с малым RAM дробим
    список на чанки и объединяем результаты, сохраняя исходные индексы.
    """
    def __init__(self, model_name='jinaai/jina-reranker-v3', device='cpu', max_chunk_size=4):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers не установлен")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        self.device = device
        self.max_chunk_size = max_chunk_size

    def score(self, pairs):
        # eval_rerankers передаёт пары для одного вопроса
        query = pairs[0][0]
        docs = [d for _, d in pairs]
        scores = np.zeros(len(docs), dtype=np.float64)
        for start in range(0, len(docs), self.max_chunk_size):
            chunk = docs[start:start + self.max_chunk_size]
            with torch.no_grad():
                results = self.model.rerank(query, chunk)
            for r in results:
                idx = start + r['index']
                if 0 <= idx < len(docs):
                    scores[idx] = r['relevance_score']
            # Снижаем пиковое потребление памяти между чанками на CPU
            if self.device == 'cpu':
                gc.collect()
        return scores


class TILDEv2(PreTrainedModel):
    """Кастомная архитектура TILDEv2 (скопирована из официального репозитория)."""
    config_class = BertConfig
    base_model_prefix = "tildev2"

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config)
        self.tok_proj = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def encode(self, **features):
        assert all(x in features for x in ['input_ids', 'attention_mask', 'token_type_ids'])
        model_out = self.bert(**features, return_dict=True)
        reps = self.tok_proj(model_out.last_hidden_state)
        tok_weights = torch.relu(reps)
        return tok_weights

    def mask_sep(self, qry_attention_mask):
        sep_pos = qry_attention_mask.sum(1).unsqueeze(1) - 1
        _zeros = torch.zeros_like(sep_pos)
        qry_attention_mask.scatter_(1, sep_pos.long(), _zeros)
        return qry_attention_mask

    def compute_tok_score_cart(self, doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask):
        qry_input_ids = qry_input_ids.unsqueeze(2).unsqueeze(3)
        doc_input_ids = doc_input_ids.unsqueeze(0).unsqueeze(1)
        exact_match = (doc_input_ids == qry_input_ids).float()
        scores_no_masking = torch.matmul(
            qry_reps.view(-1, 1),
            doc_reps.view(-1, 1).transpose(0, 1)
        )
        scores_no_masking = scores_no_masking.view(
            *qry_reps.shape[:2], *doc_reps.shape[:2])
        scores, _ = (scores_no_masking * exact_match).max(dim=3)
        tok_scores = (scores * qry_attention_mask.unsqueeze(2))[:, 1:].sum(1)
        return tok_scores

    def forward(self, qry_in, doc_in):
        doc_out = self.bert(**doc_in, return_dict=True)
        doc_reps = torch.relu(self.tok_proj(doc_out.last_hidden_state))
        doc_input_ids = doc_in['input_ids']
        qry_input_ids = qry_in['input_ids']
        qry_attention_mask = self.mask_sep(qry_in['attention_mask'].clone())
        qry_reps = torch.ones_like(qry_input_ids, dtype=torch.float32, device=doc_reps.device).unsqueeze(2)
        tok_scores = self.compute_tok_score_cart(
            doc_reps, doc_input_ids, qry_reps, qry_input_ids, qry_attention_mask
        )
        return tok_scores


class TILDEv2Scorer:
    """Прямое реранжирование через TILDEv2 с on-the-fly expansion кандидатов моделью TILDE."""

    # Минимальный набор английских стоп-слов для фильтрации expansion-токенов
    _STOP_WORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
        'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
        'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
        'just', 'don', 'should', 'now', 'definition',
    }

    def __init__(self, model_name='ielab/TILDEv2-TILDE200-exp', device='cpu', topk=200):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers не установлен")
        self.device = device
        self.topk = topk
        logging.info("Загрузка модели расширения TILDE (ielab/TILDE)...")
        self.tilde_model = BertLMHeadModel.from_pretrained("ielab/TILDE").to(device).eval()
        logging.info("Загрузка модели TILDEv2...")
        self.tildev2_model = TILDEv2.from_pretrained(model_name).to(device).eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._bad_ids = self._build_bad_ids()

    def _build_bad_ids(self):
        vocab = self.tokenizer.get_vocab()
        bad_ids = set()
        for sw in self._STOP_WORDS:
            ids = self.tokenizer(sw, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                bad_ids.add(ids[0])
        for token, tid in vocab.items():
            if tid in bad_ids:
                continue
            if token.startswith('#'):
                continue
            if not re.match(r"^[A-Za-z0-9_-]*$", token):
                bad_ids.add(tid)
        # ##s
        try:
            bad_ids.add(self.tokenizer.convert_tokens_to_ids('##s'))
        except Exception:
            pass
        return bad_ids

    def _expand(self, passages, batch_size=4):
        expanded = []
        for i in range(0, len(passages), batch_size):
            batch_texts = passages[i:i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors='pt',
            )
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']
            # Индикатор passage input — token id 1 на первой позиции
            input_ids[:, 0] = 1
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            with torch.no_grad():
                logits = self.tilde_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits[:, 0]
                selected = torch.topk(logits, self.topk, dim=-1).indices.cpu().numpy()
            orig_ids = input_ids.cpu().numpy()
            for j, orig in enumerate(orig_ids):
                # Убираем padding и первый специальный токен
                tokens = orig[orig != 0][1:].tolist()
                expand_ids = np.setdiff1d(selected[j], orig, assume_unique=True)
                expand_ids = np.setdiff1d(expand_ids, list(self._bad_ids), assume_unique=True)
                expanded.append(tokens + expand_ids.tolist())
        return expanded

    def score(self, pairs, batch_size=4):
        # eval_rerankers передаёт пары для одного вопроса
        docs = [d for _, d in pairs]
        query = pairs[0][0]
        expanded_docs = self._expand(docs, batch_size=batch_size)
        q_enc = self.tokenizer(
            query,
            max_length=16,
            truncation=True,
            return_tensors='pt',
        )
        q_input_ids = q_enc['input_ids'].to(self.device)
        q_attention_mask = q_enc['attention_mask'].to(self.device)
        # mask SEP
        sep_pos = q_attention_mask.sum(1).unsqueeze(1) - 1
        q_attention_mask.scatter_(1, sep_pos.long(), torch.zeros_like(sep_pos))
        q_reps = torch.ones_like(q_input_ids, dtype=torch.float32, device=self.device).unsqueeze(2)

        scores_all = []
        for i in range(0, len(expanded_docs), batch_size):
            batch_exp = expanded_docs[i:i + batch_size]
            max_len = max(len(x) for x in batch_exp)
            doc_input_ids = torch.zeros((len(batch_exp), max_len), dtype=torch.long, device=self.device)
            for j, x in enumerate(batch_exp):
                doc_input_ids[j, :len(x)] = torch.tensor(x, dtype=torch.long, device=self.device)
            doc_attention_mask = (doc_input_ids != 0).long()
            with torch.no_grad():
                doc_out = self.tildev2_model.bert(
                    input_ids=doc_input_ids,
                    attention_mask=doc_attention_mask,
                    return_dict=True,
                )
                doc_reps = torch.relu(self.tildev2_model.tok_proj(doc_out.last_hidden_state))
                tok_scores = self.tildev2_model.compute_tok_score_cart(
                    doc_reps, doc_input_ids, q_reps, q_input_ids, q_attention_mask
                )
            scores_all.extend(tok_scores[0].cpu().tolist())
        return np.array(scores_all)


# --- Загрузка моделей ---

def load_model(model_info):
    """Загружает модель в зависимости от типа и возвращает объект со scorer-ом."""
    model_name = model_info["name"]
    if os.environ.get("RERANKER_DEVICE"):
        device = os.environ["RERANKER_DEVICE"]
    else:
        device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'

    if model_info.get("use_t5"):
        return MonoT5Scorer(model_name, device=device)

    if model_info.get("use_rankllama"):
        # RankLLaMA 7B не запустится на CPU с <16 ГБ RAM; защищаемся от OOM-kill
        if device == 'cpu':
            mem = get_system_memory_bytes()
            if mem is not None and mem < 16 * 1024 ** 3:
                raise RuntimeError(
                    f"RankLLaMA 7B требует ~14-28 ГБ памяти, доступно {mem / 1024**3:.1f} ГБ. "
                    "Запуск на CPU в данном окружении невозможен."
                )
        return RankLLaMAScorer(model_name, device=device)

    if model_info.get("use_tildev2"):
        return TILDEv2Scorer(model_name, device=device)

    if model_info.get("use_jina_v3"):
        return JinaV3Scorer(model_name, device=device)

    if model_info.get("use_flag_embedding"):
        if not FLAG_EMBEDDING_AVAILABLE:
            raise ImportError("FlagEmbedding не установлен")
        # fp16 работает и на CPU, что существенно экономит память для больших LLM (Qwen, Gemma)
        use_fp16 = torch is not None
        reranker = FlagReranker(model_name, use_fp16=use_fp16)
        # Некоторые модели (Qwen2) не задают pad_token_id в config, что ломает батчинг >1
        if (
            hasattr(reranker, 'tokenizer')
            and hasattr(reranker, 'model')
            and reranker.tokenizer.pad_token is not None
            and reranker.model.config.pad_token_id is None
        ):
            reranker.model.config.pad_token_id = reranker.tokenizer.pad_token_id
        return reranker

    kwargs = model_info.get("kwargs", {})
    ce_kwargs = {"max_length": 512}
    ce_kwargs.update(kwargs)
    return CrossEncoder(model_name, **ce_kwargs)


def score_pairs(model, model_info, pairs):
    """Получает оценки для списка [query, doc] пар с ограничением batch size."""
    if model_info.get("use_flag_embedding"):
        # FlagReranker может пытаться прогнать все пары разом; на CPU с малым RAM
        # дробим на мелкие батчи.
        batch_size = model_info.get("batch_size", 4)
        scores = []
        for i in range(0, len(pairs), batch_size):
            scores.extend(model.compute_score(pairs[i:i + batch_size]))
        return np.array(scores)
    if any(model_info.get(k) for k in ("use_t5", "use_rankllama", "use_tildev2", "use_jina_v3")):
        return model.score(pairs)
    # CrossEncoder: уменьшаем batch size на CPU, чтобы снизить пиковое потребление памяти
    batch_size = model_info.get("batch_size", 8)
    return model.predict(pairs, batch_size=batch_size)


# --- 3. Логика тестирования ---
def evaluate_models():
    results = []

    for model_info in MODELS:
        model_name = model_info["name"]
        logging.info(f"\n{'='*50}\nТестирование модели: {model_name}\n{'='*50}")

        try:
            # Загрузка модели
            start_load = time.time()
            model = load_model(model_info)
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
                scores = score_pairs(model, model_info, pairs)
                infer_time = time.time() - start_infer
                latencies.append(infer_time)

                # Сортируем документы по убыванию оценки
                scored_docs = list(zip(scores, candidates))
                scored_docs.sort(key=lambda x: x[0], reverse=True)

                # Ищем позицию правильного ответа (Rank)
                rank = -1
                for r, (score, doc) in enumerate(scored_docs):
                    if doc == ground_truth:
                        rank = r + 1  # 1-indexed
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



def evaluate_single_model(model_info):
    """Запускает одну модель и возвращает результат (dict)."""
    model_name = model_info["name"]
    logging.info(f"\n{'='*50}\nТестирование модели: {model_name}\n{'='*50}")

    if model_info.get("skip_on_server"):
        reason = model_info.get("skip_reason", "Модель не запускается на данном сервере (недостаток памяти)")
        logging.warning(f"Пропуск {model_name}: {reason}")
        return {
            "Model": model_name.split('/')[-1],
            "MRR": "ERROR",
            "Hit Rate@5": "ERROR",
            "Latency (50 docs)": "ERROR",
            "Size": model_info["size"]
        }

    try:
        start_load = time.time()
        model = load_model(model_info)
        load_time = time.time() - start_load
        logging.info(f"Модель загружена за {load_time:.2f} сек.")

        latencies = []
        mrr_sum = 0.0
        hit_at_5_count = 0

        for idx, item in enumerate(DATASET):
            question = item["question"]
            candidates = item["candidates"]
            ground_truth = item["ground_truth"]

            pairs = [[question, doc] for doc in candidates]

            start_infer = time.time()
            scores = score_pairs(model, model_info, pairs)
            infer_time = time.time() - start_infer
            latencies.append(infer_time)

            scored_docs = list(zip(scores, candidates))
            scored_docs.sort(key=lambda x: x[0], reverse=True)

            rank = -1
            for r, (score, doc) in enumerate(scored_docs):
                if doc == ground_truth:
                    rank = r + 1
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

        avg_mrr = mrr_sum / len(DATASET)
        hit_rate_5 = (hit_at_5_count / len(DATASET)) * 100
        avg_latency = sum(latencies) / len(latencies)

        return {
            "Model": model_name.split('/')[-1],
            "MRR": f"{avg_mrr:.3f}",
            "Hit Rate@5": f"{hit_rate_5:.1f}%",
            "Latency (50 docs)": f"{avg_latency:.3f}s",
            "Size": model_info["size"]
        }

    except Exception as e:
        logging.error(f"Ошибка при тестировании {model_name}: {e}")
        return {
            "Model": model_name.split('/')[-1],
            "MRR": "ERROR",
            "Hit Rate@5": "ERROR",
            "Latency (50 docs)": "ERROR",
            "Size": model_info["size"]
        }

    finally:
        if 'model' in locals():
            del model
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("Очистка VRAM завершена.")


def evaluate_models():
    """Последовательный запуск всех моделей в одном процессе (для GPU/больших машин)."""
    results = []
    for model_info in MODELS:
        results.append(evaluate_single_model(model_info))
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
        f.write("\n\n*Примечание: Тестирование проводилось на синтетической выборке из 12 кейсов на русском и смешанном англо-русском языке (по 50 кандидатов каждый). Добавлены monoT5, RankLLaMA-7B и TILDEv2; RankLLaMA-7B на CPU пропущен при недостатке памяти.*")
    print("Результаты сохранены в reranker_benchmark_results.md")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark cross-encoder rerankers")
    parser.add_argument("--model-index", type=int, help="Запустить только модель с указанным индексом")
    parser.add_argument("--model-name", help="Запустить только модель по имени (частичное совпадение)")
    parser.add_argument("--device", help="Устройство (cpu/cuda). По умолчанию автоопределение")
    parser.add_argument("--batch-size", type=int, help="Batch size для инференса CrossEncoder/FlagEmbedding")
    parser.add_argument("--skip-server", action="store_true", help="Пропускать модели, помеченные skip_on_server")
    parser.add_argument("--list-models", action="store_true", help="Вывести список моделей с индексами и выйти")
    args = parser.parse_args()

    if args.list_models:
        for i, m in enumerate(MODELS):
            flags = []
            if m.get("skip_on_server"):
                flags.append("skip_on_server")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            print(f"{i}: {m['name']}{flag_str}")
        sys.exit(0)

    if args.device:
        # Запоминаем выбор устройства через переменную окружения, чтобы load_model его учитывал
        os.environ["RERANKER_DEVICE"] = args.device

    if args.batch_size:
        # Применяем batch size ко всем моделям, если не задан свой
        for m in MODELS:
            if "batch_size" not in m:
                m["batch_size"] = args.batch_size

    if args.model_index is not None:
        model_info = MODELS[args.model_index]
        result = evaluate_single_model(model_info)
        print(json.dumps(result, ensure_ascii=False))
    elif args.model_name:
        import json
        results = []
        for m in MODELS:
            if args.model_name.lower() in m["name"].lower():
                results.append(evaluate_single_model(m))
        print(json.dumps(results, ensure_ascii=False))
    else:
        results = evaluate_models()
        print_and_save_results(results)
