"""
Точка запуска RAG-системы.

Режимы:
  ask          — задать один вопрос
  repl         — интерактивный режим
  benchmark    — прогнать список вопросов из JSON, сохранить результаты
  dump-prompts — выгрузить все промпты в папку для редактирования

Промпты задаются через --prompt-<name> <path/to/file.txt>.
Команда dump-prompts сохраняет текущие промпты в файлы — отредактируй
и передай обратно через --prompt-* для калибровки.

Примеры:
  python run.py ask "Сколько часов лекций?"
  python run.py repl --no-hyde --reranker-top-k 4
  python run.py benchmark questions.json --output results.json
  python run.py dump-prompts --prompts-dir my_prompts/
  python run.py ask "..." --prompt-router my_prompts/router.txt
  python run.py ask "..." --prompt-generate-single-simple my_prompts/generate_single_simple.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any
from models import QueryType, RAGResponse

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Реестр промптов
# Ключ — имя аргумента CLI (без --prompt-), значение — (модуль, атрибут)
# ---------------------------------------------------------------------------

PROMPT_REGISTRY: dict[str, tuple[str, str]] = {
    # rag/prompts.py
    "router-extract":   ("rag.router", "_PROMPT_EXTRACT_DISCIPLINES"),
    "router-single":    ("rag.router", "_PROMPT_CLASSIFY_SINGLE"),
    "router-zero":      ("rag.router", "_PROMPT_CLASSIFY_ZERO"),
    "paraphrase":              ("rag.prompts", "PARAPHRASE_PROMPT"),
    "decompose":               ("rag.prompts", "DECOMPOSE_PROMPT"),
    "hyde":                    ("rag.prompts", "HYDE_PROMPT"),
    "verify":                  ("rag.prompts", "VERIFY_PROMPT"),
    "fact-extract":            ("rag.prompts", "FACT_EXTRACT_PROMPT"),
    "generate-single-simple":  ("rag.prompts", "GENERATE_PROMPTS"),
    "generate-single-global":  ("rag.prompts", "GENERATE_PROMPTS"),
    "generate-multi-relation": ("rag.prompts", "GENERATE_PROMPTS"),
    "generate-multi-global":   ("rag.prompts", "GENERATE_PROMPTS"),
    # rag/time_filter.py
    "time-extract":            ("rag.time_filter", "EXTRACT_PROMPT"),
}

# Для GENERATE_PROMPTS — маппинг ключ аргумента → ключ словаря
GENERATE_KEYS: dict[str, str] = {
    "generate-single-simple":  "single.simple",
    "generate-single-global":  "single.global",
    "generate-multi-relation": "multi.relation",
    "generate-multi-global":   "multi.global",
}


def _load_prompt(path: str) -> str:
    """Загружает промпт из файла."""
    p = Path(path)
    if not p.exists():
        print(f"Ошибка: файл промпта не найден: {path}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def _apply_prompts(args: argparse.Namespace) -> None:
    """Патчит промпты в модулях согласно аргументам CLI."""
    import importlib

    for key in PROMPT_REGISTRY:
        arg_name = f"prompt_{key.replace('-', '_')}"
        path = getattr(args, arg_name, None)
        if not path:
            continue

        text = _load_prompt(path)
        module_name, attr = PROMPT_REGISTRY[key]
        module = importlib.import_module(module_name)

        if key in GENERATE_KEYS:
            # GENERATE_PROMPTS — словарь, патчим конкретный ключ
            getattr(module, attr)[GENERATE_KEYS[key]] = text
            log.info("Промпт %s загружен из %s", key, path)
        else:
            setattr(module, attr, text)
            log.info("Промпт %s загружен из %s", key, path)


# ---------------------------------------------------------------------------
# Патчинг калибровочных констант
# ---------------------------------------------------------------------------

def _apply_tuning(args: argparse.Namespace) -> None:
    import retrieval as ret
    import expander  as exp
    import config    as cfg

    ret.TOP_K_SINGLE          = args.top_k_single
    ret.TOP_K_PER_DISC        = args.top_k_per_disc
    ret.TOP_K_STAGE1          = args.top_k_stage1
    ret.MAX_DISCIPLINES_MULTI = args.max_disciplines
    ret.MATCH_THRESHOLD       = args.match_threshold
    cfg.RERANKER_TOP_K        = args.reranker_top_k

    if args.no_hyde:
        exp.HYDE_QUERY_TYPES = set()

    if args.no_paraphrase:
        original_expand = exp.QueryExpander.expand
        def _no_paraphrase(self, query, route, resolved):
            result = original_expand(self, query, route, resolved)
            result.paraphrases = []
            return result
        exp.QueryExpander.expand = _no_paraphrase


# ---------------------------------------------------------------------------
# Dump prompts
# ---------------------------------------------------------------------------

def cmd_dump_prompts(args: argparse.Namespace) -> None:
    """Выгружает все текущие промпты в файлы для редактирования."""
    import importlib

    out_dir = Path(args.prompts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for key, (module_name, attr) in PROMPT_REGISTRY.items():
        module = importlib.import_module(module_name)
        obj    = getattr(module, attr)

        if key in GENERATE_KEYS:
            text = obj[GENERATE_KEYS[key]]
        else:
            text = obj

        fname = f"{key.replace('-', '_')}.txt"
        (out_dir / fname).write_text(text, encoding="utf-8")
        print(f"  {fname}")

    print(f"\nПромпты сохранены в: {out_dir}/")
    print("\nЧтобы использовать отредактированный промпт:")
    print(f"  python run.py ask \"...\" --prompt-router {out_dir}/router.txt")


# ---------------------------------------------------------------------------
# Заглушки
# ---------------------------------------------------------------------------

class _NoopReranker:
    def rerank(self, query, chunks):
        return chunks

class _NoopFactExtractor:
    def try_extract(self, query, chunks, query_type):
        return None


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------

def _print_response(response, verbose: bool = False) -> None:
    print("\n" + "─" * 60)
    print(f"Тип запроса:       {response.query_type}")

    # Уточнение — отдельный вывод
    if response.query_type == QueryType.CLARIFY:
        print("─" * 60)
        print(response.answer)  # текст вопроса от LLM
        if response.clarification_candidates:
            print("\nВозможные варианты:")
            for i, c in enumerate(response.clarification_candidates, 1):
                print(f"  {i}. {c}")
        print()
        return

    print(f"Верифицирован:     {'✓' if response.is_verified else '✗'}")
    print(f"Прямое извлечение: {'да' if response.fact_extracted else 'нет'}")
    if response.verification_note:
        print(f"Заметка:           {response.verification_note}")
    print("─" * 60)
    print(response.answer)
    if verbose and response.chunks_used:
        print("\n── Блоки ──")
        for c in response.chunks_used:
            print(f"  [{c.score:+.3f}] {c.discipline} / {c.block_name}")
    print()


def _response_to_dict(query: str, response, elapsed: float) -> dict[str, Any]:
    return {
        "query":                    query,
        "answer":                   response.answer,
        "query_type":               response.query_type,
        "is_verified":              response.is_verified,
        "fact_extracted":           response.fact_extracted,
        "verification_note":        response.verification_note,
        "clarification_candidates": getattr(response, "clarification_candidates", []),  # ← добавить
        "elapsed_sec":              round(elapsed, 2),
        "chunks_used": [
            {"discipline": c.discipline,
             "block_name": c.block_name,
             "score":      round(c.score, 4)}
            for c in response.chunks_used
        ],
    }


# ---------------------------------------------------------------------------
# Режимы
# ---------------------------------------------------------------------------

def _make_pipeline(args):
    from pipeline import RAGPipeline
    pipeline = RAGPipeline(qdrant_url=args.qdrant, collection=args.collection)
    if args.no_reranker:
        pipeline._reranker = _NoopReranker()
    if args.no_fact_extractor:
        pipeline._fact_extractor = _NoopFactExtractor()
    return pipeline


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = _make_pipeline(args)
    t0       = time.perf_counter()
    response = pipeline.ask(args.query)
    elapsed  = time.perf_counter() - t0
    _print_response(response, verbose=args.verbose)
    print(f"Время: {elapsed:.2f} с")


def cmd_repl(args: argparse.Namespace) -> None:
    from models import QueryType
    pipeline = _make_pipeline(args)
    print("RAG готова. Введите вопрос или 'exit'.\n")

    while True:
        try:
            query = input("Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit", "выход"):
            break

        t0       = time.perf_counter()
        response = pipeline.ask(query)
        elapsed  = time.perf_counter() - t0
        _print_response(response, verbose=args.verbose)

        # Если нужно уточнение — даём выбрать и повторяем запрос
        if response.query_type == QueryType.CLARIFY:
            candidates = response.clarification_candidates
            try:
                raw = input("Ваш выбор (номер или название): ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            # Пользователь может ввести номер или часть названия
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < len(candidates):
                    clarified_query = f"{query} — {candidates[idx]}"
                else:
                    print("Неверный номер, попробуйте снова.")
                    continue
            else:
                clarified_query = f"{query} — {raw}"

            t0       = time.perf_counter()
            response = pipeline.ask(clarified_query)
            elapsed  = time.perf_counter() - t0
            _print_response(response, verbose=args.verbose)

        print(f"Время: {elapsed:.2f} с")


def cmd_benchmark(args: argparse.Namespace) -> None:
    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    queries = [
        (item if isinstance(item, str) else item.get("query", ""))
        for item in data
    ]
    queries = [q for q in queries if q]
    print(f"Загружено {len(queries)} вопросов из {args.input}\n")

    pipeline = _make_pipeline(args)
    results  = []

    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] {query[:70]}")
        t0       = time.perf_counter()
        response = pipeline.ask(query)
        elapsed  = time.perf_counter() - t0
        results.append(_response_to_dict(query, response, elapsed))

        status = "✓" if response.is_verified else "✗"
        flag   = " [direct]" if response.fact_extracted else ""
        print(f"         {status}{flag}  {elapsed:.2f}с  {response.answer[:80]}\n")

    out_path = args.output or args.input.replace(".json", "_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    verified = sum(1 for r in results if r["is_verified"])
    direct   = sum(1 for r in results if r["fact_extracted"])
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)
    print("── Итого ──")
    print(f"Вопросов:          {len(results)}")
    print(f"Верифицировано:    {verified}/{len(results)}")
    print(f"Прямое извлечение: {direct}/{len(results)}")
    print(f"Среднее время:     {avg_time:.2f} с")
    print(f"Результаты:        {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog            = "run.py",
        description     = "RAG-система для рабочих программ дисциплин",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    # Инфраструктура
    parser.add_argument("--qdrant",     default="http://localhost:6333")
    parser.add_argument("--collection", default="discipline_chunks")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Показывать использованные блоки")

    # Поиск
    g = parser.add_argument_group("поиск")
    g.add_argument("--top-k-single",    type=int,   default=6)
    g.add_argument("--top-k-per-disc",  type=int,   default=8)
    g.add_argument("--top-k-stage1",    type=int,   default=30)
    g.add_argument("--max-disciplines", type=int,   default=10)
    g.add_argument("--match-threshold", type=float, default=0.55,
                   help="Порог совпадения названий дисциплин (0–1)")

    # Реранкинг
    g = parser.add_argument_group("реранкинг")
    g.add_argument("--reranker-top-k", type=int, default=6)
    g.add_argument("--no-reranker",    action="store_true")

    # Расширение
    g = parser.add_argument_group("расширение запроса")
    g.add_argument("--no-hyde",       action="store_true")
    g.add_argument("--no-paraphrase", action="store_true")

    # Извлечение
    g = parser.add_argument_group("извлечение")
    g.add_argument("--no-fact-extractor", action="store_true")

    # Промпты — каждый принимает путь к .txt файлу
    g = parser.add_argument_group(
        "промпты",
        "Пути к файлам с промптами. Используйте dump-prompts для получения шаблонов.",
    )
    for key in PROMPT_REGISTRY:
        g.add_argument(
            f"--prompt-{key}",
            metavar="FILE",
            default=None,
            help=f"Файл с промптом для {key}",
        )

    # Субкоманды
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Задать один вопрос")
    p_ask.add_argument("query")

    sub.add_parser("repl", help="Интерактивный режим")

    p_bench = sub.add_parser("benchmark", help="Прогнать список вопросов")
    p_bench.add_argument("input")
    p_bench.add_argument("--output", default=None)

    p_dump = sub.add_parser("dump-prompts", help="Выгрузить промпты в файлы")
    p_dump.add_argument("--prompts-dir", default="prompts/",
                        help="Папка для сохранения промптов")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Промпты патчим ДО tuning и ДО создания пайплайна
    _apply_prompts(args)
    if args.cmd != "dump-prompts":
        _apply_tuning(args)

    if args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "repl":
        cmd_repl(args)
    elif args.cmd == "benchmark":
        cmd_benchmark(args)
    elif args.cmd == "dump-prompts":
        cmd_dump_prompts(args)


if __name__ == "__main__":
    main()
