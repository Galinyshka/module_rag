"""
Модуль оценки качества RAG-системы.

Метрики (без LLM по умолчанию):
  Router Accuracy       — точность классификации типа запроса
  Router Confusion      — матрица ошибок по всем типам запросов
  Discipline ExactMatch — полное совпадение множеств дисциплин (включая оба пустые)
  Discipline Count      — количество верно предсказанных дисциплин
  Verification Step     — с какой попытки система ответила (1/2/3)
  Is Verified           — доля верифицированных ответов
  Response Time         — время обработки запроса (сек)
  Context Tokens        — приблизительное число токенов контекста
  Answer Tokens         — приблизительное число токенов ответа

С флагом --with-llm-eval добавляются:
  Faithfulness          — достоверность ответа (1–5)
  Answer Relevancy      — релевантность ответа (1–5)
  Answer Correctness    — корректность ответа (1–5, требует ground_truth в датасете)

Результаты сохраняются в:
  results/{model_name}_{run}/summary.json
  results/{model_name}_{run}/samples.csv
  results/{model_name}_{run}/confusion_matrix.txt

Запуск:
  python evaluate.py --run 1
  python evaluate.py --run 1 --limit 20
  python evaluate.py --run 1 --with-llm-eval
  python evaluate.py --run 1 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

from pipeline import RAGPipeline
from config import LLM_MODEL_FAST

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

DATASET_PATH = "datasets_router/dataset.json"
MAX_JUDGE_CONTEXT_CHARS = 4_000


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Приблизительный подсчёт токенов: 1 токен ≈ 4 символа."""
    return max(0, len(text) // 4)

def _dict_to_sample(d: dict) -> SampleResult:
    return SampleResult(**{
        k: v for k, v in d.items()
        if k in SampleResult.__dataclass_fields__
    })

def _parse_verification_step(note: str) -> Optional[int]:
    """Извлекает номер шага из заметки верификатора: '[2/3] ...' → 2."""
    if not note:
        return None
    m = re.search(r'\[(\d)/3\]', note)
    return int(m.group(1)) if m else None


def _normalise(name: str) -> str:
    return name.strip().lower()


def _safe_mean(vals: list[float]) -> float:
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return round(mean(vals), 3) if vals else 0.0


def _safe_std(vals: list[float]) -> float:
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return round(stdev(vals), 3) if len(vals) > 1 else 0.0


def _make_output_dir(run: int) -> Path:
    model_safe = (
        LLM_MODEL_FAST
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )
    path = Path(f"results/{model_safe}_{run}")
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    question:               str
    true_router:            str
    predicted_router:       str
    router_correct:         bool

    ground_disciplines:     list[str]
    predicted_disciplines:  list[str]
    discipline_exact_match: bool   # set(pred) == set(ground), включая оба пустые
    discipline_correct_count: int  # |pred ∩ ground|

    answer:                 str
    is_verified:            bool
    verification_step:      Optional[int]  # 1, 2, 3 или None (нет шагов)
    verification_note:      str

    response_time:          float
    context_tokens:         int
    answer_tokens:          int

    # LLM-метрики — None если --with-llm-eval не указан
    faithfulness:              Optional[float] = None
    relevancy:                 Optional[float] = None
    correctness:               Optional[float] = None
    faithfulness_rationale:    str = ""
    relevancy_rationale:       str = ""
    correctness_rationale:     str = ""

    error: str = ""


@dataclass
class EvalSummary:
    run:           int
    model:         str
    dataset_path:  str
    total:         int
    with_llm_eval: bool

    # Router
    router_accuracy:   float
    per_type_accuracy: dict   # type → {count, correct, accuracy}
    confusion_matrix:  dict   # true → predicted → count

    # Дисциплины (все семплы, пустой список — валидный ground truth)
    discipline_samples:       int
    discipline_exact_match:   float   # доля точных совпадений
    discipline_correct_mean:  float   # среднее кол-во верно найденных дисциплин

    # Верификация
    is_verified_rate:        float
    verification_step_dist:  dict   # step_1/step_2/step_3/no_steps → доля

    # Производительность
    response_time_mean:   float
    response_time_std:    float
    context_tokens_mean:  float
    context_tokens_std:   float
    answer_tokens_mean:   float
    answer_tokens_std:    float

    # По типам запросов — полная разбивка всех метрик
    by_query_type: dict = field(default_factory=dict)

    # LLM-метрики (None если не считались)
    faithfulness_mean:  Optional[float] = None
    faithfulness_std:   Optional[float] = None
    relevancy_mean:     Optional[float] = None
    relevancy_std:      Optional[float] = None
    correctness_mean:   Optional[float] = None
    correctness_std:    Optional[float] = None
    low_quality_count:  int = 0


# ---------------------------------------------------------------------------
# Judge — модель-оценщик (только при --with-llm-eval)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """\
Ты — беспристрастный эксперт по оценке качества ответов систем вопрос-ответ
по академическим документам (рабочим программам дисциплин).
Оценивай строго, объективно и воспроизводимо.
Отвечай ТОЛЬКО валидным JSON без markdown.
"""

FAITHFULNESS_PROMPT = """\
Оцени ДОСТОВЕРНОСТЬ ответа: насколько точно он отражает информацию из контекста.
5 — все утверждения подкреплены контекстом
4 — незначительные отклонения, не искажающие смысл
3 — часть утверждений не подкреплена
2 — заметные галлюцинации
1 — преимущественно противоречит контексту

Вопрос: {query}
Контекст: {context}
Ответ: {answer}

JSON: {{"score": 1-5, "rationale": "1-2 предложения"}}
"""

RELEVANCY_PROMPT = """\
Оцени РЕЛЕВАНТНОСТЬ ответа: насколько точно он отвечает на вопрос.
5 — прямо и полно отвечает
4 — релевантен, незначительная неполнота
3 — частично релевантен
2 — преимущественно нерелевантен
1 — не имеет отношения к вопросу

Вопрос: {query}
Ответ: {answer}

JSON: {{"score": 1-5, "rationale": "1-2 предложения"}}
"""

CORRECTNESS_PROMPT = """\
Оцени КОРРЕКТНОСТЬ ответа относительно эталонного ответа.
5 — все ключевые факты верны
4 — большинство верно, незначительные неточности
3 — часть фактов верна
2 — большинство пропущено или искажено
1 — противоречит эталону

Вопрос: {query}
Эталон: {ground_truth}
Ответ: {answer}

JSON: {{"score": 1-5, "rationale": "1-2 предложения"}}
"""


class Judge:
    def __init__(self) -> None:
        from openai import OpenAI
        from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_EVAL, LLM_MAX_TOKENS_EVAL
        self._client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self._model = LLM_MODEL_EVAL
        self._max_tokens = LLM_MAX_TOKENS_EVAL

    def _call(self, prompt: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw.strip())

    def faithfulness(self, query: str, answer: str, context: str) -> tuple[float, str]:
        try:
            d = self._call(FAITHFULNESS_PROMPT.format(
                query=query, answer=answer, context=context
            ))
            return float(d["score"]), d.get("rationale", "")
        except Exception as e:
            log.warning("Judge faithfulness error: %s", e)
            return float("nan"), str(e)

    def relevancy(self, query: str, answer: str) -> tuple[float, str]:
        try:
            d = self._call(RELEVANCY_PROMPT.format(query=query, answer=answer))
            return float(d["score"]), d.get("rationale", "")
        except Exception as e:
            log.warning("Judge relevancy error: %s", e)
            return float("nan"), str(e)

    def correctness(self, query: str, answer: str, ground_truth: str) -> tuple[float, str]:
        try:
            d = self._call(CORRECTNESS_PROMPT.format(
                query=query, answer=answer, ground_truth=ground_truth
            ))
            return float(d["score"]), d.get("rationale", "")
        except Exception as e:
            log.warning("Judge correctness error: %s", e)
            return float("nan"), str(e)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    def __init__(
        self,
        qdrant_url:    str  = "http://localhost:6333",
        collection:    str  = "discipline_chunks",
        workers:       int  = 1,
        with_llm_eval: bool = False,
        only_route:    bool = False, 
    ) -> None:
        log.info("Инициализация RAG-пайплайна ...")
        self._pipeline = RAGPipeline(qdrant_url=qdrant_url, collection=collection)
        self._judge = Judge() if with_llm_eval else None
        self._workers = workers
        self._only_route   = only_route

    def evaluate_dataset(
        self, dataset: list[dict], out_dir: Path
    ) -> list[SampleResult]:
        results_path = out_dir / "results_live.jsonl"

        # Загружаем уже обработанные если файл существует
        done: dict[str, SampleResult] = {}
        if results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    done[d["question"]] = d
            log.info("Восстановлено %d уже обработанных вопросов", len(done))

        results = []
        with open(results_path, "a", encoding="utf-8") as f:
            for i, item in enumerate(dataset):
                q = item["question"]
                if q in done:
                    log.info("[%d/%d] пропускаем (уже есть): %s", i+1, len(dataset), q[:50])
                    # восстанавливаем объект из словаря
                    results.append(_dict_to_sample(done[q]))
                    continue

                r = self._evaluate_one(item, i + 1, len(dataset))
                results.append(r)
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                f.flush()  # сбрасываем буфер сразу

        return results
    # ------------------------------------------------------------------

    def _evaluate_one(self, item: dict, idx: int, total: int) -> SampleResult:
        question = item["question"]
        true_router = item.get("router_type", "")
        ground_disciplines = item.get("ground_discipline") or []

        log.info("[%d/%d] %s", idx, total, question[:70])

        result = SampleResult(
            question=question,
            true_router=true_router,
            predicted_router="",
            router_correct=False,
            ground_disciplines=ground_disciplines,
            predicted_disciplines=[],
            discipline_exact_match=False,
            discipline_correct_count=0,
            answer="",
            is_verified=False,
            verification_step=None,
            verification_note="",
            response_time=0.0,
            context_tokens=0,
            answer_tokens=0,
        )

        try:
            t0 = time.perf_counter()
            if self._only_route:
                # ── только роутер ──────────────────────────────────────────
                route = self._pipeline.route_only(question)
                result.response_time         = round(time.perf_counter() - t0, 3)
                result.predicted_router      = route.query_type.value
                result.router_correct        = (route.query_type.value == true_router)
                result.predicted_disciplines = list(route.disciplines or [])

                pred_set   = {_normalise(d) for d in result.predicted_disciplines}
                ground_set = {_normalise(d) for d in ground_disciplines}
                result.discipline_exact_match    = (pred_set == ground_set)
                result.discipline_correct_count  = len(pred_set & ground_set)

                log.info(
                    "  router=%s  disc=%s(%d)  t=%.2fs",
                    "✓" if result.router_correct else "✗",
                    "✓" if result.discipline_exact_match else "✗",
                    result.discipline_correct_count,
                    result.response_time,
                )
                return result
            
            response = self._pipeline.ask(question)
            result.response_time = round(time.perf_counter() - t0, 3)

            result.answer = response.answer
            result.predicted_router = response.query_type.value
            result.router_correct = (response.query_type.value == true_router)
            result.predicted_disciplines = list(response.disciplines or [])
            result.is_verified = response.is_verified
            result.verification_note = response.verification_note or ""
            result.verification_step = _parse_verification_step(result.verification_note)

            # Дисциплины
            pred_set = {_normalise(d) for d in result.predicted_disciplines}
            ground_set = {_normalise(d) for d in ground_disciplines}
            result.discipline_exact_match = (pred_set == ground_set)
            result.discipline_correct_count = len(pred_set & ground_set)

            # Токены (приблизительно)
            context_text = "".join(c.text for c in (response.chunks_used or []))
            result.context_tokens = _approx_tokens(context_text)
            result.answer_tokens = _approx_tokens(result.answer)

        except Exception as exc:
            result.error = str(exc)
            log.error("RAG error [%d]: %s", idx, exc)
            return result

        # LLM-метрики (опционально)
        if self._judge and not result.error:
            ground_truth = item.get("ground_truth", "")
            context_for_judge = "".join(
                f"[{c.block_name}]\n{c.text}\n"
                for c in (response.chunks_used or [])
            )[:MAX_JUDGE_CONTEXT_CHARS]

            s, r = self._judge.faithfulness(question, result.answer, context_for_judge)
            result.faithfulness, result.faithfulness_rationale = s, r

            s, r = self._judge.relevancy(question, result.answer)
            result.relevancy, result.relevancy_rationale = s, r

            if ground_truth:
                s, r = self._judge.correctness(question, result.answer, ground_truth)
                result.correctness, result.correctness_rationale = s, r

        # Лог одной строкой
        disc_tag = f" disc={'✓' if result.discipline_exact_match else '✗'}({result.discipline_correct_count})"
        step_tag = f" step={result.verification_step}" if result.verification_step else ""
        log.info(
            "  router=%s%s%s  t=%.1fs  ctx=%dtok  ans=%dtok",
            "✓" if result.router_correct else "✗",
            disc_tag, step_tag,
            result.response_time,
            result.context_tokens,
            result.answer_tokens,
        )
        return result

    # ------------------------------------------------------------------

    def _run_parallel(self, dataset: list[dict]) -> list[SampleResult]:
        total = len(dataset)
        results: list[Optional[SampleResult]] = [None] * total
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
                        question=item["question"],
                        true_router=item.get("router_type", ""),
                        predicted_router="",
                        router_correct=False,
                        ground_disciplines=item.get("ground_discipline") or [],
                        predicted_disciplines=[],
                        discipline_exact_match=False,
                        discipline_correct_count=0,
                        answer="",
                        is_verified=False,
                        verification_step=None,
                        verification_note="",
                        response_time=0.0,
                        context_tokens=0,
                        answer_tokens=0,
                        error=str(exc),
                    )
        return results


# ---------------------------------------------------------------------------
# Агрегация метрик
# ---------------------------------------------------------------------------

def compute_summary(
    results: list[SampleResult],
    run: int,
    with_llm_eval: bool,
) -> EvalSummary:

    total = len(results)
    valid = [r for r in results if not r.error]

    # --- Router ---
    router_correct_count = sum(1 for r in results if r.router_correct)

    all_types = sorted(
        {r.true_router for r in results}
        | {r.predicted_router for r in results if r.predicted_router}
    )

    confusion: dict[str, dict[str, int]] = {t: defaultdict(int) for t in all_types}
    for r in results:
        if r.predicted_router:
            confusion[r.true_router][r.predicted_router] += 1

    per_type: dict[str, dict] = {}
    for t in all_types:
        samples = [r for r in results if r.true_router == t]
        correct = sum(1 for r in samples if r.router_correct)
        per_type[t] = {
            "count":    len(samples),
            "correct":  correct,
            "accuracy": round(correct / len(samples), 3) if samples else 0.0,
        }

    # --- Дисциплины ---
    disc_exact = [r.discipline_exact_match for r in valid]
    disc_count = [float(r.discipline_correct_count) for r in valid]

    # --- Верификация ---
    n = len(valid) or 1
    is_verified_count = sum(1 for r in valid if r.is_verified)

    step_counts: dict[str, int] = defaultdict(int)
    for r in valid:
        key = f"step_{r.verification_step}" if r.verification_step else "no_steps"
        step_counts[key] += 1
    verification_step_dist = {
        k: round(v / n, 3) for k, v in sorted(step_counts.items())
    }

    # --- Производительность ---
    times  = [r.response_time  for r in valid]
    ctx_t  = [float(r.context_tokens) for r in valid]
    ans_t  = [float(r.answer_tokens)  for r in valid]

    summary = EvalSummary(
        run=run,
        model=LLM_MODEL_FAST,
        dataset_path=DATASET_PATH,
        total=total,
        with_llm_eval=with_llm_eval,

        router_accuracy=round(router_correct_count / total, 3) if total else 0.0,
        per_type_accuracy=per_type,
        confusion_matrix={k: dict(v) for k, v in confusion.items()},

        discipline_samples=len(disc_exact),
        discipline_exact_match=round(sum(disc_exact) / len(disc_exact), 3) if disc_exact else 0.0,
        discipline_correct_mean=_safe_mean(disc_count),

        is_verified_rate=round(is_verified_count / n, 3),
        verification_step_dist=verification_step_dist,

        response_time_mean=_safe_mean(times),
        response_time_std=_safe_std(times),
        context_tokens_mean=_safe_mean(ctx_t),
        context_tokens_std=_safe_std(ctx_t),
        answer_tokens_mean=_safe_mean(ans_t),
        answer_tokens_std=_safe_std(ans_t),
    )

    # --- LLM-метрики ---
    if with_llm_eval:
        def _llm_vals(attr: str) -> list[float]:
            return [
                v for r in valid
                if (v := getattr(r, attr)) is not None
                and not (isinstance(v, float) and math.isnan(v))
            ]

        faith = _llm_vals("faithfulness")
        relev = _llm_vals("relevancy")
        corr  = _llm_vals("correctness")

        summary.faithfulness_mean = _safe_mean(faith)
        summary.faithfulness_std  = _safe_std(faith)
        summary.relevancy_mean    = _safe_mean(relev)
        summary.relevancy_std     = _safe_std(relev)
        summary.correctness_mean  = _safe_mean(corr)
        summary.correctness_std   = _safe_std(corr)
        summary.low_quality_count = sum(
            1 for r in valid
            if any(
                (getattr(r, m) or 5) <= 2
                for m in ("faithfulness", "relevancy", "correctness")
            )
        )

    # --- По типам запросов ---
    by_type: dict[str, dict] = {}
    for true_t in all_types:
        samples = [r for r in valid if r.true_router == true_t]
        if not samples:
            continue
        n_t = len(samples)

        step_c: dict[str, int] = defaultdict(int)
        for r in samples:
            key = f"step_{r.verification_step}" if r.verification_step else "no_steps"
            step_c[key] += 1

        entry: dict = {
            "count":                   n_t,
            "router_accuracy":         round(sum(1 for r in samples if r.router_correct) / n_t, 3),
            "discipline_exact_match":  round(sum(r.discipline_exact_match for r in samples) / n_t, 3),
            "discipline_correct_mean": _safe_mean([float(r.discipline_correct_count) for r in samples]),
            "is_verified_rate":        round(sum(r.is_verified for r in samples) / n_t, 3),
            "verification_step_dist":  {k: round(v / n_t, 3) for k, v in sorted(step_c.items())},
            "response_time_mean":      _safe_mean([r.response_time for r in samples]),
            "response_time_std":       _safe_std([r.response_time for r in samples]),
            "context_tokens_mean":     _safe_mean([float(r.context_tokens) for r in samples]),
            "answer_tokens_mean":      _safe_mean([float(r.answer_tokens) for r in samples]),
        }

        if with_llm_eval:
            def _t_llm(attr: str) -> list[float]:
                return [
                    v for r in samples
                    if (v := getattr(r, attr)) is not None
                    and not (isinstance(v, float) and math.isnan(v))
                ]
            entry["faithfulness_mean"] = _safe_mean(_t_llm("faithfulness"))
            entry["relevancy_mean"]    = _safe_mean(_t_llm("relevancy"))
            entry["correctness_mean"]  = _safe_mean(_t_llm("correctness"))

        by_type[true_t] = entry

    summary.by_query_type = by_type

    return summary


# ---------------------------------------------------------------------------
# Сохранение результатов
# ---------------------------------------------------------------------------

def save_json(
    results: list[SampleResult],
    summary: EvalSummary,
    out_dir: Path,
) -> None:
    data = {
        "summary": asdict(summary),
        "results": [asdict(r) for r in results],
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("JSON → %s", path)


def save_csv(results: list[SampleResult], out_dir: Path) -> None:
    fields = [
        "question", "true_router", "predicted_router", "router_correct",
        "ground_disciplines", "predicted_disciplines",
        "discipline_exact_match", "discipline_correct_count",
        "is_verified", "verification_step", "verification_note",
        "response_time", "context_tokens", "answer_tokens",
        "faithfulness", "relevancy", "correctness",
        "faithfulness_rationale", "relevancy_rationale", "correctness_rationale",
        "answer", "error",
    ]
    path = out_dir / "samples.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = asdict(r)
            row["ground_disciplines"]    = "; ".join(row["ground_disciplines"])
            row["predicted_disciplines"] = "; ".join(row["predicted_disciplines"])
            writer.writerow({k: row.get(k, "") for k in fields})
    log.info("CSV  → %s", path)


def save_confusion_matrix(summary: EvalSummary, out_dir: Path) -> None:
    all_types = sorted(summary.confusion_matrix.keys())
    col_w = max((len(t) for t in all_types), default=10) + 2

    lines = [
        "CONFUSION MATRIX",
        "строки = истинный тип, столбцы = предсказанный тип\n",
    ]
    header = f"{'':>{col_w}}" + "".join(f"{t:>{col_w}}" for t in all_types)
    lines.append(header)
    lines.append("─" * len(header))

    for true_t in all_types:
        row = f"{true_t:>{col_w}}"
        for pred_t in all_types:
            count = summary.confusion_matrix.get(true_t, {}).get(pred_t, 0)
            marker = f"[{count}]" if true_t == pred_t and count > 0 else str(count)
            row += f"{marker:>{col_w}}"
        lines.append(row)

    # Точность по типам
    lines += ["", "ТОЧНОСТЬ ПО ТИПАМ", "─" * 44]
    for t, m in sorted(summary.per_type_accuracy.items()):
        bar = "█" * int(m["accuracy"] * 20)
        lines.append(
            f"{t:<26} {m['correct']:>3}/{m['count']:<3}  "
            f"{m['accuracy']:.3f}  {bar}"
        )

    path = out_dir / "confusion_matrix.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Матрица → %s", path)


# ---------------------------------------------------------------------------
# Вывод в консоль
# ---------------------------------------------------------------------------

def print_summary(summary: EvalSummary) -> None:
    w = 60
    print("\n" + "═" * w)
    print(f"  ОЦЕНКА RAG  |  run={summary.run}  |  {summary.model}")
    print("═" * w)
    print(f"  Примеров: {summary.total}")
    print()

    print(f"  Router Accuracy: {summary.router_accuracy:.3f}")
    print()
    print(f"  {'Тип':<26} {'N':>4}  {'✓':>4}  {'Acc':>6}")
    print("  " + "─" * 44)
    for t, m in sorted(summary.per_type_accuracy.items()):
        flag = "◀" if m["accuracy"] < 0.5 and m["count"] > 0 else ""
        print(f"  {t:<26} {m['count']:>4}  {m['correct']:>4}  {m['accuracy']:>6.3f}  {flag}")
    print()

    print(f"  Дисциплины ({summary.discipline_samples} семплов):")
    print(f"    ExactMatch:          {summary.discipline_exact_match:.3f}")
    print(f"    Верно найдено (μ):   {summary.discipline_correct_mean:.2f}")
    print()
    
    print(f"  Верифицировано:        {summary.is_verified_rate:.3f}")
    for step, share in summary.verification_step_dist.items():
        print(f"    {step:<12} {share:.3f}  ({round(share * summary.total)}/{summary.total})")
    print()

    print("  Производительность:")
    print(f"    Время (μ±σ):   {summary.response_time_mean:.2f}s ±{summary.response_time_std:.2f}")
    print(f"    Ctx tokens:    {summary.context_tokens_mean:.0f} ±{summary.context_tokens_std:.0f}")
    print(f"    Ans tokens:    {summary.answer_tokens_mean:.0f} ±{summary.answer_tokens_std:.0f}")

    if summary.with_llm_eval and summary.faithfulness_mean is not None:
        print()
        print("  LLM-метрики (μ±σ):")
        print(f"    Faithfulness:  {summary.faithfulness_mean:.2f} ±{summary.faithfulness_std:.2f}")
        print(f"    Relevancy:     {summary.relevancy_mean:.2f} ±{summary.relevancy_std:.2f}")
        print(f"    Correctness:   {summary.correctness_mean:.2f} ±{summary.correctness_std:.2f}")
        print(f"    Низкое кач-во: {summary.low_quality_count}/{summary.total}")

    if summary.by_query_type:
        print()
        print("  ПО ТИПАМ ЗАПРОСОВ:")
        print("  " + "─" * 78)

        has_llm = summary.with_llm_eval
        hdr = (
            f"  {'Тип':<26} {'N':>4}  {'Router':>6}  "
            f"{'Disc':>5}  {'Verif':>5}  {'Step1':>5}  "
            f"{'t(μ)':>6}  {'ctx':>6}  {'ans':>5}"
        )
        if has_llm:
            hdr += f"  {'Faith':>5}  {'Relev':>5}  {'Corr':>5}"
        print(hdr)
        print("  " + "─" * 78)

        for t, m in sorted(summary.by_query_type.items()):
            step1 = m["verification_step_dist"].get("step_1", 0.0)
            row = (
                f"  {t:<26} {m['count']:>4}  "
                f"{m['router_accuracy']:>6.3f}  "
                f"{m['discipline_exact_match']:>5.3f}  "
                f"{m['is_verified_rate']:>5.3f}  "
                f"{step1:>5.3f}  "
                f"{m['response_time_mean']:>6.2f}  "
                f"{m['context_tokens_mean']:>6.0f}  "
                f"{m['answer_tokens_mean']:>5.0f}"
            )
            if has_llm:
                row += (
                    f"  {m.get('faithfulness_mean', 0):>5.2f}"
                    f"  {m.get('relevancy_mean', 0):>5.2f}"
                    f"  {m.get('correctness_mean', 0):>5.2f}"
                )
            print(row)

    print("═" * w + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка RAG-системы",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run", type=int, required=True,
        help="Номер запуска — используется для имени папки результатов",
    )
    parser.add_argument("--dataset",    default=DATASET_PATH)
    parser.add_argument("--qdrant",     default="http://localhost:6333")
    parser.add_argument("--collection", default="discipline_chunks")
    parser.add_argument("--workers",    type=int, default=1,
                        help="Параллельность (осторожно: rate limits)")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Оценить только первые N примеров")
    parser.add_argument("--with-llm-eval", action="store_true",
                        help="Включить LLM-судью (faithfulness / relevancy / correctness)")
    parser.add_argument(
        "--only-route", action="store_true",
        help="Оценивать только роутер (без retrieval / generation / verification)",
    )
    
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    if args.limit:
        dataset = dataset[:args.limit]
        log.info("Ограничение: первые %d примеров", args.limit)

    evaluator = Evaluator(
        qdrant_url=args.qdrant,
        collection=args.collection,
        workers=args.workers,
        with_llm_eval=args.with_llm_eval,
        only_route=args.only_route
    )

    out_dir = _make_output_dir(args.run)
    results = evaluator.evaluate_dataset(dataset, out_dir=out_dir)
    summary = compute_summary(results, run=args.run, with_llm_eval=args.with_llm_eval)
    save_json(results, summary, out_dir)
    save_csv(results, out_dir)
    save_confusion_matrix(summary, out_dir)
    print_summary(summary)

    log.info("Все результаты сохранены в: %s", out_dir)


if __name__ == "__main__":
    main()