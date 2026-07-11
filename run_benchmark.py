#!/usr/bin/env python3
"""Запускает eval_rerankers.py отдельным процессом для каждой модели.

Это позволяет полностью освобождать память между моделями и избегать
OOM-kill на машинах с небольшим объёмом RAM.

Usage:
    python run_benchmark.py
    python run_benchmark.py --resume  # пропустить модели, уже есть в reranker_benchmark_results.md
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

from eval_rerankers import MODELS, print_and_save_results

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

LOG_DIR = Path('logs')
RESULTS_FILE = Path('reranker_benchmark_results.md')


def parse_existing_results():
    """Читает уже сохранённые результаты из markdown-файла."""
    results = {}
    if not RESULTS_FILE.exists():
        return results
    inside_table = False
    for line in RESULTS_FILE.read_text(encoding='utf-8').splitlines():
        if line.startswith('| Модель ') or line.startswith('|---'):
            inside_table = True
            continue
        if inside_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')[1:-1]]
            if len(parts) >= 5:
                results[parts[0]] = {
                    'Model': parts[0],
                    'MRR': parts[1],
                    'Hit Rate@5': parts[2],
                    'Latency (50 docs)': parts[3],
                    'Size': parts[4],
                }
    return results


def run_model(index, model_info, resume=False, batch_size=None):
    model_name = model_info['name']
    short_name = model_name.split('/')[-1]
    logging.info(f"[{index + 1}/{len(MODELS)}] Запуск {model_name}")

    existing = parse_existing_results()
    if resume and short_name in existing:
        logging.info(f"  -> уже есть результат для {short_name}, пропускаем")
        return existing[short_name]

    if model_info.get('skip_on_server'):
        logging.warning(f"  -> {short_name} помечена как skip_on_server, не запускаем")
        return {
            'Model': short_name,
            'MRR': 'ERROR',
            'Hit Rate@5': 'ERROR',
            'Latency (50 docs)': 'ERROR',
            'Size': model_info['size'],
        }

    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"{short_name}.log"

    cmd = [sys.executable, 'eval_rerankers.py', str(index)]
    if batch_size is not None:
        cmd.extend(['--batch-size', str(batch_size)])

    try:
        with open(log_file, 'w', encoding='utf-8') as lf:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )

        text = log_file.read_text(encoding='utf-8')
        if proc.returncode != 0:
            logging.error(f"Ошибка при запуске {model_name} (код {proc.returncode}). Лог: {log_file}")
            return {
                'Model': short_name,
                'MRR': 'ERROR',
                'Hit Rate@5': 'ERROR',
                'Latency (50 docs)': 'ERROR',
                'Size': model_info['size'],
            }

        # Последняя непустая строка должна быть JSON с результатом
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        for line in reversed(lines):
            if line.startswith('{'):
                return json.loads(line)
        logging.error(f"Не удалось распарсить результат для {model_name}. Лог: {log_file}")
        return {
            'Model': short_name,
            'MRR': 'ERROR',
            'Hit Rate@5': 'ERROR',
            'Latency (50 docs)': 'ERROR',
            'Size': model_info['size'],
        }
    except subprocess.TimeoutExpired:
        logging.error(f"Таймаут при тестировании {model_name}")
        return {
            'Model': short_name,
            'MRR': 'TIMEOUT',
            'Hit Rate@5': 'TIMEOUT',
            'Latency (50 docs)': 'TIMEOUT',
            'Size': model_info['size'],
        }


def main():
    parser = argparse.ArgumentParser(description='Run reranker benchmark per model')
    parser.add_argument('--resume', action='store_true', help='Skip models already present in results file')
    parser.add_argument('--batch-size', type=int, help='Batch size для инференса CrossEncoder/FlagEmbedding')
    args = parser.parse_args()

    results = []
    for index, model_info in enumerate(MODELS):
        result = run_model(index, model_info, resume=args.resume, batch_size=args.batch_size)
        results.append(result)
        logging.info(
            f"Результат {model_info['name']}: MRR={result['MRR']}, "
            f"HR@5={result['Hit Rate@5']}, Latency={result['Latency (50 docs)']}"
        )

    print_and_save_results(results)


if __name__ == '__main__':
    main()
