"""
RAG-система для рабочих программ дисциплин.

Режимы:
  ask        — задать один вопрос
  repl       — интерактивный режим
  benchmark  — прогнать список вопросов из JSON, сохранить результаты

Примеры:
  python run.py ask "Сколько часов лекций?"
  python run.py repl --no-hyde --reranker-top-k 4
  python run.py benchmark questions.json --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from rag.domain.models import QueryType

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Заглушка реранкера
# ---------------------------------------------------------------------------

class _NoopReranker:
    def rerank(self, query, chunks):
        return chunks


# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

def _apply_tuning(args: argparse.Namespace) -> None:
    import rag.retrieval.retrieval as ret
    import rag.retrieval.expander as exp
    import rag.config.config as cfg

    ret.TOP_K_SINGLE   = args.top_k_single
    cfg.RERANKER_TOP_K = args.reranker_top_k

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
# Форматирование вывода
# ---------------------------------------------------------------------------

def _print_response(response, verbose: bool = False) -> None:
    print("\n" + "─" * 60)
    print(f"Тип запроса: {response.query_type}")

    if response.query_type == QueryType.CLARIFY:
        print("─" * 60)
        print(response.answer)
        if response.clarification_candidates:
            print("\nВозможные варианты:")
            for i, c in enumerate(response.clarification_candidates, 1):
                print(f"  {i}. {c}")
        print()
        return

    print(f"Верифицирован: {'✓' if response.is_verified else '✗'}")
    if response.verification_note:
        print(f"Заметка: {response.verification_note}")
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
        "verification_note":        response.verification_note,
        "clarification_candidates": getattr(response, "clarification_candidates", []),
        "elapsed_sec":              round(elapsed, 2),
        "chunks_used": [
            {"discipline": c.discipline,
             "block_name": c.block_name,
             "score":      round(c.score, 4)}
            for c in response.chunks_used
        ],
    }


# ---------------------------------------------------------------------------
# Пайплайн
# ---------------------------------------------------------------------------

def _make_pipeline(args):
    from rag.pipeline.pipeline import RAGPipeline
    pipeline = RAGPipeline(qdrant_url=args.qdrant, collection=args.collection)
    if args.no_reranker:
        pipeline._reranker = _NoopReranker()
    return pipeline


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------

def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = _make_pipeline(args)
    t0       = time.perf_counter()
    response = pipeline.ask(args.query)
    elapsed  = time.perf_counter() - t0
    _print_response(response, verbose=args.verbose)
    print(f"Время: {elapsed:.2f} с")


def cmd_repl(args: argparse.Namespace) -> None:
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

        if response.query_type == QueryType.CLARIFY:
            candidates = response.clarification_candidates
            try:
                raw = input("Ваш выбор (номер или название): ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if raw.isdigit():
                idx = int(raw) - 1
                clarified_query = f"{query} — {candidates[idx]}" if 0 <= idx < len(candidates) else query
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
        print(f"[{i}/{len(queries)}] {query}")
        t0       = time.perf_counter()
        response = pipeline.ask(query)
        elapsed  = time.perf_counter() - t0
        results.append(_response_to_dict(query, response, elapsed))

    out_path = args.output or args.input.replace(".json", "_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    verified = sum(1 for r in results if r["is_verified"])
    avg_time = sum(r["elapsed_sec"] for r in results) / len(results)
    print(f"\nВопросов:       {len(results)}")
    print(f"Верифицировано: {verified}/{len(results)}")
    print(f"Среднее время:  {avg_time:.2f} с")
    print(f"Результаты:     {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog            = "run.py",
        description     = "RAG-система для рабочих программ дисциплин",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--qdrant",     default="http://localhost:6333")
    parser.add_argument("--collection", default="discipline_chunks")
    parser.add_argument("--verbose", "-v", action="store_true")

    g = parser.add_argument_group("поиск")
    g.add_argument("--top-k-single",   type=int, default=6)

    g = parser.add_argument_group("реранкинг")
    g.add_argument("--reranker-top-k", type=int, default=6)
    g.add_argument("--no-reranker",    action="store_true")

    g = parser.add_argument_group("расширение запроса")
    g.add_argument("--no-hyde",       action="store_true")
    g.add_argument("--no-paraphrase", action="store_true")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Задать один вопрос")
    p_ask.add_argument("query")

    sub.add_parser("repl", help="Интерактивный режим")

    p_bench = sub.add_parser("benchmark", help="Прогнать список вопросов")
    p_bench.add_argument("input")
    p_bench.add_argument("--output", default=None)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _apply_tuning(args)

    if args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "repl":
        cmd_repl(args)
    elif args.cmd == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()