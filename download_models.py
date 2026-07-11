#!/usr/bin/env python3
"""Предварительная загрузка моделей reranker из Hugging Face Hub.

Запуск:
    python download_models.py

Для проверки локального кэша без попыток скачивания:
    LOCAL_FILES_ONLY=1 python download_models.py
"""

import os
import logging
from huggingface_hub import snapshot_download

from eval_rerankers import MODELS

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def download_models(local_files_only: bool = False) -> None:
    """Скачивает все модели из списка MODELS в кэш Hugging Face Hub."""
    for model_info in MODELS:
        model_name = model_info["name"]
        if model_info.get("skip_on_server"):
            logging.info(f"Пропуск {model_name}: не запускается на данном сервере")
            continue
        logging.info(f"Подготовка модели: {model_name}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_files_only=local_files_only,
            )
            logging.info(f"Модель готова: {model_name}")
        except Exception as e:
            logging.error(f"Ошибка при подготовке {model_name}: {e}")


if __name__ == "__main__":
    local_only = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
    download_models(local_files_only=local_only)
