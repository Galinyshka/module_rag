"""
Модуль оценки качества RAG-системы.

Метрики:
  Faithfulness          — достоверность ответа относительно контекста (1–5)
  Answer Relevancy      — релевантность ответа вопросу (1–5)
  Answer Correctness    — корректность ответа относительно ground truth (1–5)
  Response Time         — время обработки запроса (сек)
  Router Accuracy       — точность классификации типа запроса (0/1 per sample)
  Discipline Precision  — доля верно предсказанных дисциплин из предсказанных
  Discipline Recall     — доля верно предсказанных дисциплин из эталонных
  Discipline F1         — гармоническое среднее precision и recall
  Discipline ExactMatch — полное совпадение множеств дисциплин (0/1)

Все LLM-метрики оцениваются моделью-судьёй (LLM_MODEL_EVAL),
которая должна отличаться от генерирующей модели.
Шкала 1–5 выбрана как более воспроизводимая и однозначная чем 1–10.

Запуск:
  python evaluate.py dataset.json
  python evaluate.py dataset.json --output results.json --workers 4
  python evaluate.py dataset.json --skip-llm-eval          # только роутер + дисциплины
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from openai import OpenAI

from pipeline import RAGPipeline
from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL_EVAL,
    LLM_MAX_TOKENS_EVAL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Промпты для оценки (модель-судья)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
Ты — беспристрастный эксперт по оценке качества ответов систем вопрос-ответ
по академическим документам (рабочим программам дисциплин).
Оценивай строго, объективно и воспроизводимо.
Отвечай ТОЛЬКО валидным JSON без markdown.
"""

FAITHFULNESS_PROMPT = """\
Оцени ДОСТОВЕРНОСТЬ ответа: насколько точно он отражает информацию из предоставленного контекста.
Высокий балл = все утверждения подкреплены контекстом.
Низкий балл = ответ содержит факты которых нет в контексте (галлюцинации).

Шкала:
  5 — все утверждения полностью соответствуют контексту, нет ничего лишнего
  4 — незначительные отклонения или обобщения, не искажающие смысл
  3 — часть утверждений не подкреплена контекстом, но нет прямых противоречий
  2 — заметные галлюцинации: факты которых нет в контексте
  1 — ответ преимущественно противоречит контексту или выдуман

Вопрос: {query}

Контекст (извлечённые документы):
{context}

Ответ системы:
{answer}

Отвечай ТОЛЬКО JSON:
{{"score": 1-5, "rationale": "1–2 предложения с конкретным обоснованием"}}
"""

RELEVANCY_PROMPT = """\
Оцени РЕЛЕВАНТНОСТЬ ответа: насколько точно он отвечает на заданный вопрос.
Высокий балл = ответ прямо и полно отвечает на вопрос.
Низкий балл = ответ уходит в сторону, слишком общий или отвечает на другой вопрос.

Шкала:
  5 — ответ прямо, конкретно и полно отвечает на вопрос
  4 — ответ релевантен, но содержит незначительные отступления или неполноту
  3 — ответ частично релевантен: затрагивает тему, но не отвечает на суть вопроса
  2 — ответ преимущественно нерелевантен или слишком общий
  1 — ответ не имеет отношения к вопросу

Вопрос: {query}

Ответ системы:
{answer}

Отвечай ТОЛЬКО JSON:
{{"score": 1-5, "rationale": "1–2 предложения с конкретным обоснованием"}}
"""

CORRECTNESS_PROMPT = """\
Оцени КОРРЕКТНОСТЬ ответа в сравнении с эталонным ответом.
Высокий балл = ответ передаёт все ключевые факты эталона без искажений.
Низкий балл = ответ упускает важные факты или искажает их.

Шкала:
  5 — все ключевые факты эталона присутствуют и переданы точно
  4 — большинство фактов верно, незначительные пропуски или неточности
  3 — часть ключевых фактов присутствует, часть пропущена или неточна
  2 — большинство ключевых фактов пропущено или искажено
  1 — ответ противоречит эталону или полностью его игнорирует

Важно: ответ не обязан быть дословным — оценивается фактическое совпадение, а не формулировка.

Вопрос: {query}

Эталонный ответ:
{ground_truth}

Ответ системы:
{answer}

Отвечай ТОЛЬКО JSON:
{{"score": 1-5, "rationale": "1–2 предложения с конкретным обоснованием"}}
"""


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _isnan(v: Any) -> bool:
    try:
        return math.isnan(v)
    except Exception:
        return False


def _safe_mean(vals: list[float]) -> float:
    return round(mean(vals), 3) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    return round(stdev(vals), 3) if len(vals) > 1 else 0.0


def _normalise(name: str) -> str:
    """Нормализация названия дисциплины для сравнения."""
    return name.strip().lower()


def _discipline_metrics(
    predicted: list[str],
    ground:    list[str],
) -> tuple[float, float, float, bool]:
    """
    Вычисляет метрики качества предсказания дисциплин.

    Args:
        predicted: дисциплины, которые вернул роутер
        ground:    эталонные дисциплины из датасета

    Returns:
        (precision, recall, f1, exact_match)

    Граничные случаи:
        - ground пуст → метрики не применимы, возвращает (nan, nan, nan, False).
          Такие семплы пропускаются при агрегации.
        - predicted пуст, ground непуст → precision=nan, recall=0, f1=0, exact=False
    """
    pred_set   = {_normalise(d) for d in predicted}
    ground_set = {_normalise(d) for d in ground}
    tp         = len(pred_set & ground_set)

    if pred_set:
        precision = tp / len(pred_set)
    else:
        precision = 1.0 if not ground_set else 0.0   # pred=[], ground=[] → 1.0; pred=[], ground=[X] → 0.0

    if ground_set:
        recall = tp / len(ground_set)
    else:
        recall = 1.0 if not pred_set else 0.0         # pred=[], ground=[] → 1.0; pred=[X], ground=[] → 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    exact_match = pred_set == ground_set
    return precision, recall, f1, exact_match


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    question:         str
    ground_truth:     str
    true_router:      str
    predicted_router: str
    answer:           str
    router_correct:   bool

    # Дисциплины
    ground_disciplines:    list[str] = field(default_factory=list)
    predicted_disciplines: list[str] = field(default_factory=list)
    discipline_precision:  float | None = None   # nan если ground пуст
    discipline_recall:     float | None = None
    discipline_f1:         float | None = None
    discipline_exact:      bool  | None = None   # None если ground пуст

    # LLM-метрики
    faithfulness:     float | None = None
    relevancy:        float | None = None
    correctness:      float | None = None
    response_time:    float = 0.0

    faithfulness_rationale:  str = ""
    relevancy_rationale:     str = ""
    correctness_rationale:   str = ""

    context_blocks:   list[str] = field(default_factory=list)
    error:            str = ""


@dataclass
class EvalSummary:
    total:               int
    router_accuracy:     float

    # Дисциплины (считается только по семплам с ground_discipline)
    discipline_samples:       int    = 0   # сколько семплов имеют ground_discipline
    discipline_exact_match:   float  = 0.0
    discipline_precision_mean: float = 0.0
    discipline_precision_std:  float = 0.0
    discipline_recall_mean:    float = 0.0
    discipline_recall_std:     float = 0.0
    discipline_f1_mean:        float = 0.0
    discipline_f1_std:         float = 0.0

    # LLM-метрики
    faithfulness_mean:   float = 0.0
    faithfulness_std:    float = 0.0
    relevancy_mean:      float = 0.0
    relevancy_std:       float = 0.0
    correctness_mean:    float = 0.0
    correctness_std:     float = 0.0
    response_time_mean:  float = 0.0
    response_time_std:   float = 0.0

    # По типам запросов
    by_query_type:       dict[str, dict[str, float]] = field(default_factory=dict)
    # Провалившиеся примеры (score ≤ 2 по любой LLM-метрике)
    low_quality_count:   int = 0


# ---------------------------------------------------------------------------
# Judge — модель-оценщик
# ---------------------------------------------------------------------------

class Judge:
    """Вызывает LLM_MODEL_EVAL для оценки по каждой метрике отдельно."""

    def __init__(self) -> None:
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    def _call(self, prompt: str) -> dict:
        resp = self._client.chat.completions.create(
            model      = LLM_MODEL_EVAL,
            max_tokens = LLM_MAX_TOKENS_EVAL,
            messages   = [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw.strip())

    def faithfulness(self, query: str, answer: str, context: str) -> tuple[float, str]:
        try:
            data = self._call(FAITHFULNESS_PROMPT.format(
                query=query, answer=answer, context=context
            ))
            return float(data["score"]), data.get("rationale", "")
        except Exception as exc:
            log.warning("Judge faithfulness error: %s", exc)
            return float("nan"), str(exc)

    def relevancy(self, query: str, answer: str) -> tuple[float, str]:
        try:
            data = self._call(RELEVANCY_PROMPT.format(query=query, answer=answer))
            return float(data["score"]), data.get("rationale", "")
        except Exception as exc:
            log.warning("Judge relevancy error: %s", exc)
            return float("nan"), str(exc)

    def correctness(self, query: str, answer: str, ground_truth: str) -> tuple[float, str]:
        try:
            data = self._call(CORRECTNESS_PROMPT.format(
                query=query, answer=answer, ground_truth=ground_truth
            ))
            return float(data["score"]), data.get("rationale", "")
        except Exception as exc:
            log.warning("Judge correctness error: %s", exc)
            return float("nan"), str(exc)


# ---------------------------------------------------------------------------
# Evaluator — оркестратор
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 4_000   # контекст для судьи — не весь, чтобы не перегружать


class Evaluator:
    def __init__(
        self,
        qdrant_url:  str = "http://localhost:6333",
        collection:  str = "discipline_chunks",
        workers:     int = 1,
    ) -> None:
        log.info("Инициализация RAG-пайплайна для оценки ...")
        self._pipeline = RAGPipeline(qdrant_url=qdrant_url, collection=collection)
        self._judge    = Judge()
        self._workers  = workers
        log.info("Модель-судья: %s", LLM_MODEL_EVAL)

    def evaluate_dataset(self, dataset: list[dict]) -> tuple[list[SampleResult], EvalSummary]:
        log.info("Оценка %d примеров (workers=%d) ...", len(dataset), self._workers)

        if self._workers > 1:
            results = self._run_parallel(dataset)
        else:
            results = [
                self._evaluate_one(item, i + 1, len(dataset))
                for i, item in enumerate(dataset)
            ]

        summary = self._compute_summary(results)
        return results, summary

    # ------------------------------------------------------------------
    # Один пример
    # ------------------------------------------------------------------

    def _evaluate_one(self, item: dict, idx: int, total: int) -> SampleResult:
        question          = item["question"]
        ground_truth      = item["ground_truth"]
        true_router       = item["router_type"]
        has_ground_discipline = "ground_discipline" in item
        ground_disciplines    = item.get("ground_discipline") or []
        log.info("[%d/%d] %s", idx, total, question[:70])

        result = SampleResult(
            question           = question,
            ground_truth       = ground_truth,
            true_router        = true_router,
            predicted_router   = "",
            answer             = "",
            router_correct     = False,
            ground_disciplines = ground_disciplines,
        )

        # 1. Запрос к RAG-пайплайну
        try:
            t0       = time.perf_counter()
            response = self._pipeline.ask(question)
            elapsed  = time.perf_counter() - t0

            result.answer           = response.answer
            result.predicted_router = response.query_type.value
            result.router_correct   = (response.query_type.value == true_router)
            result.response_time    = round(elapsed, 3)
            result.context_blocks   = [
                f"{c.discipline} / {c.block_name}"
                for c in response.chunks_used
            ]

            # Дисциплины из роутера
            # RAGResponse должен содержать поле disciplines (list[str]).
            # Если его нет — см. примечание в конце файла.
            result.predicted_disciplines = list(getattr(response, "disciplines", []) or [])

        except Exception as exc:
            result.error = str(exc)
            log.error("RAG error для '%s': %s", question[:50], exc)
            return result

        # 2. Метрики дисциплин (детерминированные, без LLM)
        prec, rec, f1, exact = _discipline_metrics(
            result.predicted_disciplines,
            ground_disciplines,
        )
        if has_ground_discipline:
            result.discipline_precision = prec
            result.discipline_recall    = rec
            result.discipline_f1        = f1
            result.discipline_exact     = exact

        # 3. Контекст для судьи
        context = "\n\n".join(
            f"[{c.block_name}]\n{c.text}"
            for c in response.chunks_used
        )[:MAX_CONTEXT_CHARS]

        # 4. LLM-оценки (последовательно — судья дорогой)
        score, rat = self._judge.faithfulness(question, result.answer, context)
        result.faithfulness, result.faithfulness_rationale = score, rat

        score, rat = self._judge.relevancy(question, result.answer)
        result.relevancy, result.relevancy_rationale = score, rat

        score, rat = self._judge.correctness(question, result.answer, ground_truth)
        result.correctness, result.correctness_rationale = score, rat

        # Лог одной строкой: все ключевые метрики
        disc_tag = ""
        if ground_disciplines:
            disc_tag = f" D={f1:.2f}({'✓' if exact else '✗'})"
        log.info(
            "  → router=%s%s  F=%.1f R=%.1f C=%.1f t=%.1fs",
            "✓" if result.router_correct else "✗",
            disc_tag,
            result.faithfulness or 0,
            result.relevancy    or 0,
            result.correctness  or 0,
            result.response_time,
        )
        return result

    # ------------------------------------------------------------------
    # Параллельный прогон
    # ------------------------------------------------------------------

    def _run_parallel(self, dataset: list[dict]) -> list[SampleResult]:
        total   = len(dataset)
        results = [None] * total

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {
                pool.submit(self._evaluate_one, item, i + 1, total): i
                for i, item in enumerate(dataset)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    log.error("Worker error [%d]: %s", idx, exc)
                    item = dataset[idx]
                    results[idx] = SampleResult(
                        question           = item["question"],
                        ground_truth       = item["ground_truth"],
                        true_router        = item["router_type"],
                        predicted_router   = "",
                        answer             = "",
                        router_correct     = False,
                        ground_disciplines = item.get("ground_discipline") or [],
                        error              = str(exc),
                    )
        return results

    # ------------------------------------------------------------------
    # Сводная статистика
    # ------------------------------------------------------------------

    def _compute_summary(self, results: list[SampleResult]) -> EvalSummary:

        def _vals(attr: str) -> list[float]:
            return [
                getattr(r, attr) for r in results
                if getattr(r, attr) is not None and not _isnan(getattr(r, attr))
            ]

        # --- LLM-метрики ---
        faith  = _vals("faithfulness")
        relev  = _vals("relevancy")
        corr   = _vals("correctness")
        times  = _vals("response_time")
        router = [1 if r.router_correct else 0 for r in results]

        # --- Дисциплины: только семплы с ground_discipline ---
        disc_results = [r for r in results if r.ground_disciplines]
        disc_prec    = [r.discipline_precision for r in disc_results
                        if r.discipline_precision is not None and not _isnan(r.discipline_precision)]
        disc_rec     = [r.discipline_recall    for r in disc_results
                        if r.discipline_recall    is not None and not _isnan(r.discipline_recall)]
        disc_f1      = [r.discipline_f1        for r in disc_results
                        if r.discipline_f1        is not None and not _isnan(r.discipline_f1)]
        disc_exact   = [1 if r.discipline_exact else 0 for r in disc_results
                        if r.discipline_exact is not None]

        # --- По типам запросов ---
        by_type: dict[str, dict] = {}
        for r in results:
            qt = r.true_router
            if qt not in by_type:
                by_type[qt] = {
                    "count":              0,
                    "faithfulness":       [],
                    "relevancy":          [],
                    "correctness":        [],
                    "router_correct":     [],
                    "discipline_f1":      [],
                    "discipline_exact":   [],
                }
            by_type[qt]["count"] += 1
            by_type[qt]["router_correct"].append(1 if r.router_correct else 0)

            for m in ("faithfulness", "relevancy", "correctness"):
                v = getattr(r, m)
                if v is not None and not _isnan(v):
                    by_type[qt][m].append(v)

            if r.ground_disciplines:
                if r.discipline_f1 is not None and not _isnan(r.discipline_f1):
                    by_type[qt]["discipline_f1"].append(r.discipline_f1)
                if r.discipline_exact is not None:
                    by_type[qt]["discipline_exact"].append(1 if r.discipline_exact else 0)

        by_type_summary = {
            qt: {
                "count":            d["count"],
                "faithfulness":     _safe_mean(d["faithfulness"]),
                "relevancy":        _safe_mean(d["relevancy"]),
                "correctness":      _safe_mean(d["correctness"]),
                "router_accuracy":  _safe_mean(d["router_correct"]),
                "discipline_f1":    _safe_mean(d["discipline_f1"]),
                "discipline_exact": _safe_mean(d["discipline_exact"]),
            }
            for qt, d in by_type.items()
        }

        low_quality = sum(
            1 for r in results
            if any(
                (getattr(r, m) or 5) <= 2
                for m in ("faithfulness", "relevancy", "correctness")
            )
        )

        return EvalSummary(
            total                  = len(results),
            router_accuracy        = round(_safe_mean(router), 3),

            discipline_samples     = len(disc_results),
            discipline_exact_match = round(_safe_mean(disc_exact), 3),
            discipline_precision_mean = _safe_mean(disc_prec),
            discipline_precision_std  = _safe_std(disc_prec),
            discipline_recall_mean    = _safe_mean(disc_rec),
            discipline_recall_std     = _safe_std(disc_rec),
            discipline_f1_mean        = _safe_mean(disc_f1),
            discipline_f1_std         = _safe_std(disc_f1),

            faithfulness_mean      = _safe_mean(faith),
            faithfulness_std       = _safe_std(faith),
            relevancy_mean         = _safe_mean(relev),
            relevancy_std          = _safe_std(relev),
            correctness_mean       = _safe_mean(corr),
            correctness_std        = _safe_std(corr),
            response_time_mean     = _safe_mean(times),
            response_time_std      = _safe_std(times),

            by_query_type          = by_type_summary,
            low_quality_count      = low_quality,
        )


# ---------------------------------------------------------------------------
# Вывод результатов
# ---------------------------------------------------------------------------

def print_summary(summary: EvalSummary) -> None:
    print("\n" + "═" * 64)
    print("  ИТОГИ ОЦЕНКИ RAG-СИСТЕМЫ")
    print("═" * 64)
    print(f"  Примеров:              {summary.total}")
    print(f"  Модель-судья:          {LLM_MODEL_EVAL}")
    print()
    print(f"  Router Accuracy:       {summary.router_accuracy:.3f}")
    print()

    if summary.discipline_samples > 0:
        print(f"  Метрики дисциплин ({summary.discipline_samples} семплов с ground_discipline):")
        print("  " + "─" * 46)
        print(f"  Exact Match          {summary.discipline_exact_match:.3f}")
        print(f"  Precision            {summary.discipline_precision_mean:.3f}  ±{summary.discipline_precision_std:.3f}")
        print(f"  Recall               {summary.discipline_recall_mean:.3f}  ±{summary.discipline_recall_std:.3f}")
        print(f"  F1                   {summary.discipline_f1_mean:.3f}  ±{summary.discipline_f1_std:.3f}")
        print()

    print("  Метрика             mean   std")
    print("  " + "─" * 40)
    print(f"  Faithfulness        {summary.faithfulness_mean:.2f}   ±{summary.faithfulness_std:.2f}")
    print(f"  Answer Relevancy    {summary.relevancy_mean:.2f}   ±{summary.relevancy_std:.2f}")
    print(f"  Answer Correctness  {summary.correctness_mean:.2f}   ±{summary.correctness_std:.2f}")
    print(f"  Response Time       {summary.response_time_mean:.2f}s  ±{summary.response_time_std:.2f}s")
    print()
    print(f"  Низкое качество (≤2 по любой LLM-метрике): {summary.low_quality_count}/{summary.total}")

    if summary.by_query_type:
        print()
        print("  По типам запросов:")
        print(f"  {'Тип':<20} {'N':>4}  {'F':>5}  {'R':>5}  {'C':>5}  {'Router':>7}  {'D-F1':>6}  {'D-EM':>5}")
        print("  " + "─" * 68)
        for qt, m in summary.by_query_type.items():
            print(
                f"  {qt:<20} {m['count']:>4}  "
                f"{m['faithfulness']:>5.2f}  "
                f"{m['relevancy']:>5.2f}  "
                f"{m['correctness']:>5.2f}  "
                f"{m['router_accuracy']:>7.3f}  "
                f"{m['discipline_f1']:>6.3f}  "
                f"{m['discipline_exact']:>5.3f}"
            )
    print("═" * 64 + "\n")


def save_results(
    results: list[SampleResult],
    summary: EvalSummary,
    path:    str,
) -> None:
    output = {
        "summary":     asdict(summary),
        "judge_model": LLM_MODEL_EVAL,
        "results":     [asdict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("Результаты сохранены: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description     = "Оценка качества RAG-системы",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", nargs="?", default=None, help="JSON-файл с датасетом")
    parser.add_argument("--ask", type=str, default=None, help="Оценить один вопрос без dataset.json")
    parser.add_argument("--output",         default=None,
                        help="Куда сохранить результаты (default: dataset_eval.json)")
    parser.add_argument("--qdrant",         default="http://localhost:6333")
    parser.add_argument("--collection",     default="discipline_chunks")
    parser.add_argument("--workers",        type=int, default=1,
                        help="Параллельность (осторожно: rate limits)")
    parser.add_argument("--limit",          type=int, default=None,
                        help="Оценить только первые N примеров")
    parser.add_argument("--skip-llm-eval",  action="store_true",
                        help="Только router accuracy + дисциплины, без LLM-судьи")
    args = parser.parse_args()

    if args.ask:
        dataset = [{
            "question": args.ask,
            "ground_truth": "",
            "router_type": "",
            "ground_discipline": [],
        }]
    else:
        with open(args.dataset, encoding="utf-8") as f:
            dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]
        log.info("Ограничение: первые %d примеров", args.limit)

    if args.skip_llm_eval:
        class _NoopJudge:
            def faithfulness(self, *a): return float("nan"), "skipped"
            def relevancy(self,    *a): return float("nan"), "skipped"
            def correctness(self,  *a): return float("nan"), "skipped"
        Judge.__new__ = lambda cls: _NoopJudge()

    evaluator = Evaluator(
        qdrant_url = args.qdrant,
        collection = args.collection,
        workers    = args.workers,
    )
    results, summary = evaluator.evaluate_dataset(dataset)

    print_summary(summary)

    out_path = args.output or args.dataset.replace(".json", "_eval.json")
    save_results(results, summary, out_path)


if __name__ == "__main__":
    main()
