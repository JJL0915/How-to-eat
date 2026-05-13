"""
RAGAS 评测入口。

用法示例：
    python evaluation/run_ragas_eval.py
    python evaluation/run_ragas_eval.py --limit 5
    python evaluation/run_ragas_eval.py --skip-ragas
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CODE_DIR.parent
DEFAULT_DATASET_PATH = CODE_DIR / "evaluation" / "datasets" / "eval_dataset.jsonl"
DEFAULT_RUNS_DIR = CODE_DIR / "evaluation" / "runs"
DEFAULT_REPORTS_DIR = CODE_DIR / "evaluation" / "reports"
MAIN_CASE_TYPES = {"detail", "general", "list"}
BUSINESS_METRIC_NAMES = [
    "dish_hit_rate",
    "category_match_rate",
    "difficulty_match_rate",
    "ingredient_match_rate",
    "positive_term_match_rate",
    "negative_violation_rate",
    "valid_recommendation_rate",
]
PLAN_METRIC_NAMES = [
    "route_type_match",
    "answer_style_match",
    "target_dish_hit_rate",
    "focus_dish_match",
    "needs_clarification_match",
]


def _prepare_import_path() -> None:
    """确保评测脚本可以直接导入 code 目录下的 RAG 模块。"""
    code_dir = str(CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def _configure_console_encoding() -> None:
    """在 Windows 控制台中强制使用 UTF-8，避免中文和 emoji 日志输出失败。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 测试集。"""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, raw_line in enumerate(file, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON: {exc}") from exc
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """写入 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _validate_dataset(cases: List[Dict[str, Any]], dataset_path: Path) -> None:
    """校验测试集基础结构，默认主测试集只允许三类主任务。"""
    required_fields = ["id", "case_type", "user_input", "reference", "expected_dishes"]
    seen_ids = set()
    errors: List[str] = []
    is_default_dataset = dataset_path == DEFAULT_DATASET_PATH.resolve()

    for index, case in enumerate(cases, 1):
        for field in required_fields:
            if field not in case:
                errors.append(f"第 {index} 行缺少字段 {field}")

        case_id = case.get("id")
        if case_id in seen_ids:
            errors.append(f"样例 id 重复: {case_id}")
        seen_ids.add(case_id)

        case_type = case.get("case_type")
        if is_default_dataset and case_type not in MAIN_CASE_TYPES:
            errors.append(
                f"默认主测试集只允许 detail/general/list，{case_id} 使用了 {case_type}"
            )

    if errors:
        raise ValueError("测试集校验失败:\n" + "\n".join(errors))


def _safe_float(value: Any) -> Optional[float]:
    """把 RAGAS 返回值转换成可写入报告的浮点数。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _doc_summary(doc: Any, preview_length: int = 120) -> Dict[str, Any]:
    """提取文档摘要，避免 trace 文件过大。"""
    metadata = getattr(doc, "metadata", {}) or {}
    content = getattr(doc, "page_content", "") or ""
    return {
        "dish_name": metadata.get("dish_name", "未知菜品"),
        "category": metadata.get("category", "未知"),
        "difficulty": metadata.get("difficulty", "未知"),
        "source": metadata.get("source", ""),
        "parent_id": metadata.get("parent_id", ""),
        "chunk_id": metadata.get("chunk_id", ""),
        "rrf_score": metadata.get("rrf_score"),
        "filter_preference_score": metadata.get("filter_preference_score"),
        "query_preference_score": metadata.get("query_preference_score"),
        "final_retrieval_score": metadata.get("final_retrieval_score"),
        "content_preview": content[:preview_length].replace("\n", " "),
    }


def _build_parent_contexts(parent_docs: List[Any]) -> List[str]:
    """
    构造完整父文档上下文列表。

    这些内容用于 trace 排查，不直接作为第一版 RAGAS 输入。
    """
    contexts: List[str] = []
    for index, doc in enumerate(parent_docs, 1):
        metadata = getattr(doc, "metadata", {}) or {}
        content = getattr(doc, "page_content", "") or ""
        dish_name = metadata.get("dish_name", "未知菜品")
        category = metadata.get("category", "未知")
        difficulty = metadata.get("difficulty", "未知")
        header = f"【食谱 {index}】{dish_name} | 分类: {category} | 难度: {difficulty}"
        contexts.append(f"{header}\n{content}")

    return contexts or ["暂无相关食谱信息"]


def _build_llm_context(rag_system: Any, parent_docs: List[Any]) -> str:
    """记录现有生成模块实际构造出的上下文。"""
    if not parent_docs:
        return "暂无相关食谱信息"
    try:
        return rag_system.generation_module._build_context(parent_docs)
    except Exception as exc:
        return f"上下文构造失败: {exc}"


def _build_llm_contexts_for_ragas(rag_system: Any, parent_docs: List[Any]) -> List[str]:
    """
    构造传给 RAGAS 的上下文列表。

    第一版评测优先评估“实际交给生成模型的上下文”，避免完整父文档掩盖上下文截断问题。
    """
    return [_build_llm_context(rag_system, parent_docs)]


def _run_single_case(rag_system: Any, case: Dict[str, Any]) -> Dict[str, Any]:
    """运行单条测试样例并返回 trace。"""
    question = case["user_input"]
    started_at = time.perf_counter()
    error: Optional[str] = None

    raw_session_id = case.get("session_id") or case.get("conversation_id")
    session_id = str(raw_session_id).strip() if raw_session_id else None
    pipeline_result = None

    try:
        if session_id and case.get("reset_session"):
            rag_system.conversation_memory.clear(session_id)

        pipeline_result = rag_system.run_question_pipeline(
            question,
            stream=False,
            session_id=session_id,
            remember=True,
            verbose=False,
        )

    except Exception as exc:
        error = str(exc)
        pipeline_result = rag_system.empty_pipeline_result(
            question, answer=f"评测运行失败: {exc}"
        )

    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
    retrieved_contexts = _build_llm_contexts_for_ragas(
        rag_system, pipeline_result.relevant_docs
    )
    parent_contexts = _build_parent_contexts(pipeline_result.relevant_docs)
    parent_summaries = [_doc_summary(doc) for doc in pipeline_result.relevant_docs]

    return {
        "id": case.get("id", ""),
        "case_type": case.get("case_type", ""),
        "user_input": question,
        "reference": case.get("reference", ""),
        "expected_dishes": case.get("expected_dishes", []),
        "expected_constraints": case.get("expected_constraints", {}),
        "expected_plan": case.get("expected_plan", {}),
        "tags": case.get("tags", []),
        "session_id": session_id or "",
        "query_plan": pipeline_result.query_plan,
        "route_type": pipeline_result.route_type,
        "answer_style": pipeline_result.answer_style,
        "rewritten_query": pipeline_result.processing_question,
        "standalone_query": pipeline_result.processing_question,
        "target_dishes": pipeline_result.target_dishes,
        "focus_dish": pipeline_result.focus_dish,
        "needs_clarification": bool(
            pipeline_result.query_plan.get("needs_clarification", False)
        ),
        "clarification": pipeline_result.query_plan.get("clarification", ""),
        "conversation_state": pipeline_result.conversation_state,
        "conversation_context_preview": pipeline_result.conversation_context[:1000],
        "generation_context_preview": pipeline_result.generation_context[:1000],
        "filters": pipeline_result.filters,
        "preferences": pipeline_result.preferences,
        "retrieved_dishes": [doc["dish_name"] for doc in parent_summaries],
        "retrieved_chunks": [
            _doc_summary(doc) for doc in pipeline_result.relevant_chunks
        ],
        "retrieved_parent_docs": parent_summaries,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_parent_contexts": parent_contexts,
        "llm_context_preview": retrieved_contexts[0][:1000],
        "response": pipeline_result.answer,
        "refusal_reason": pipeline_result.refusal_reason,
        "latency_ms": latency_ms,
        "error": error,
    }


def _build_ragas_dataset_rows(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """转换为 RAGAS evaluate 需要的字段。"""
    return [
        {
            "user_input": trace["user_input"],
            "response": trace["response"],
            "retrieved_contexts": trace["retrieved_contexts"],
            "reference": trace["reference"],
        }
        for trace in traces
    ]


def _build_ragas_judge_llm(rag_system: Any) -> Any:
    """
    构造单独的 RAGAS 评测模型。

    评测阶段会请求结构化输出；关闭 thinking 并提高 max_tokens，可以减少评测器自身失败。
    """
    from langchain_openai import ChatOpenAI

    llm_kwargs = {
        "model": rag_system.config.llm_model,
        "temperature": 0,
        "max_tokens": max(rag_system.config.max_tokens, 8192),
        "api_key": rag_system.config.llm_api_key,
        "base_url": rag_system.config.llm_base_url,
    }
    return ChatOpenAI(**llm_kwargs)


def _run_ragas(
    traces: List[Dict[str, Any]],
    rag_system: Any,
    batch_size: Optional[int],
    ragas_timeout: int,
) -> List[Dict[str, Optional[float]]]:
    """运行 RAGAS 四个指标。"""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import (
            ContextPrecision,
            ContextRecall,
            Faithfulness,
            ResponseRelevancy,
        )

    dataset = Dataset.from_list(_build_ragas_dataset_rows(traces))
    metrics = [
        Faithfulness(),
        ResponseRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=_build_ragas_judge_llm(rag_system),
        embeddings=rag_system.index_module.embeddings,
        raise_exceptions=False,
        run_config=RunConfig(timeout=ragas_timeout),
        show_progress=True,
        batch_size=batch_size,
    )

    raw_rows = result.to_pandas().to_dict(orient="records")
    scores: List[Dict[str, Optional[float]]] = []
    for row in raw_rows:
        scores.append(
            {
                "faithfulness": _safe_float(row.get("faithfulness")),
                "response_relevancy": _safe_float(row.get("answer_relevancy")),
                "context_precision": _safe_float(row.get("context_precision")),
                "context_recall": _safe_float(row.get("context_recall")),
            }
        )
    return scores


def _empty_scores(count: int) -> List[Dict[str, Optional[float]]]:
    """在跳过 RAGAS 时填充空分数，保证输出格式稳定。"""
    return [
        {
            "faithfulness": None,
            "response_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }
        for _ in range(count)
    ]


def _merge_scores(
    traces: List[Dict[str, Any]],
    scores: List[Dict[str, Optional[float]]],
) -> List[Dict[str, Any]]:
    """把 trace 和 RAGAS 分数合并为最终结果行。"""
    rows: List[Dict[str, Any]] = []
    for trace, score in zip(traces, scores):
        business_metrics = _calculate_business_metrics(trace)
        plan_metrics = _calculate_plan_metrics(trace)
        rows.append(
            {
                "id": trace["id"],
                "case_type": trace["case_type"],
                "user_input": trace["user_input"],
                "expected_dishes": trace["expected_dishes"],
                "expected_constraints": trace["expected_constraints"],
                "expected_plan": trace["expected_plan"],
                "retrieved_dishes": trace["retrieved_dishes"],
                "session_id": trace["session_id"],
                "route_type": trace["route_type"],
                "answer_style": trace["answer_style"],
                "rewritten_query": trace["rewritten_query"],
                "standalone_query": trace["standalone_query"],
                "target_dishes": trace["target_dishes"],
                "focus_dish": trace["focus_dish"],
                "needs_clarification": trace["needs_clarification"],
                "query_plan": trace["query_plan"],
                "filters": trace["filters"],
                "preferences": trace["preferences"],
                "refusal_reason": trace["refusal_reason"],
                "latency_ms": trace["latency_ms"],
                "error": trace["error"],
                **plan_metrics,
                **business_metrics,
                **score,
            }
        )
    return rows


def _calculate_plan_metrics(trace: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """根据测试集声明的 expected_plan 计算 planner 命中情况。"""
    expected_plan = trace.get("expected_plan") or {}
    actual_plan = trace.get("query_plan") or {}
    metrics: Dict[str, Optional[float]] = {
        "route_type_match": None,
        "answer_style_match": None,
        "target_dish_hit_rate": None,
        "focus_dish_match": None,
        "needs_clarification_match": None,
    }
    if not expected_plan:
        return metrics

    if "route_type" in expected_plan:
        metrics["route_type_match"] = float(
            actual_plan.get("route_type") == expected_plan.get("route_type")
        )
    if "answer_style" in expected_plan:
        metrics["answer_style_match"] = float(
            actual_plan.get("answer_style") == expected_plan.get("answer_style")
        )
    if "target_dishes" in expected_plan:
        expected_targets = expected_plan.get("target_dishes") or []
        actual_targets = actual_plan.get("target_dishes") or []
        metrics["target_dish_hit_rate"] = _dish_hit_rate(
            expected_targets, actual_targets
        )
    if "focus_dish" in expected_plan:
        metrics["focus_dish_match"] = float(
            actual_plan.get("focus_dish", "") == expected_plan.get("focus_dish", "")
        )
    if "needs_clarification" in expected_plan:
        metrics["needs_clarification_match"] = float(
            bool(actual_plan.get("needs_clarification", False))
            == bool(expected_plan.get("needs_clarification", False))
        )
    return metrics


def _calculate_business_metrics(trace: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    计算推荐类业务约束指标。

    这些指标主要用于 general/list 的开放推荐问题，避免只用固定代表菜列表判断好坏。
    """
    constraints = trace.get("expected_constraints") or {}
    retrieved_docs = trace.get("retrieved_parent_docs") or []
    parent_contexts = trace.get("retrieved_parent_contexts") or []
    retrieved_dishes = trace.get("retrieved_dishes") or []
    expected_dishes = trace.get("expected_dishes") or []

    metrics: Dict[str, Optional[float]] = {
        "dish_hit_rate": _dish_hit_rate(expected_dishes, retrieved_dishes),
        "category_match_rate": None,
        "difficulty_match_rate": None,
        "ingredient_match_rate": None,
        "positive_term_match_rate": None,
        "negative_violation_rate": None,
        "valid_recommendation_rate": None,
    }

    if not constraints:
        return metrics

    total = len(retrieved_docs)
    if total == 0:
        for metric_name in BUSINESS_METRIC_NAMES:
            if metric_name != "dish_hit_rate":
                metrics[metric_name] = 0.0
        return metrics

    doc_records = []
    for index, doc in enumerate(retrieved_docs):
        context = parent_contexts[index] if index < len(parent_contexts) else ""
        doc_records.append((doc, context))

    if constraints.get("category"):
        metrics["category_match_rate"] = _rate(
            _matches_category(doc, constraints["category"]) for doc, _ in doc_records
        )

    if constraints.get("difficulty"):
        metrics["difficulty_match_rate"] = _rate(
            _matches_difficulty(doc, constraints["difficulty"])
            for doc, _ in doc_records
        )

    if constraints.get("ingredients_any") or constraints.get("ingredients_all"):
        metrics["ingredient_match_rate"] = _rate(
            _matches_ingredients(doc, context, constraints)
            for doc, context in doc_records
        )

    if constraints.get("positive_terms"):
        metrics["positive_term_match_rate"] = _rate(
            _contains_any(_doc_search_text(doc, context), constraints["positive_terms"])
            for doc, context in doc_records
        )

    if constraints.get("negative_terms"):
        metrics["negative_violation_rate"] = _rate(
            _contains_any(_doc_search_text(doc, context), constraints["negative_terms"])
            for doc, context in doc_records
        )

    metrics["valid_recommendation_rate"] = _rate(
        _matches_all_constraints(doc, context, constraints)
        for doc, context in doc_records
    )
    return metrics


def _dish_hit_rate(
    expected_dishes: List[str], retrieved_dishes: List[str]
) -> Optional[float]:
    """计算代表菜覆盖率；这是参考指标，不代表开放推荐题的完整正确率。"""
    if not expected_dishes:
        return None
    if not retrieved_dishes:
        return 0.0
    expected_set = set(expected_dishes)
    hits = sum(1 for dish in expected_set if dish in retrieved_dishes)
    return hits / len(expected_set)


def _rate(matches: Iterable[bool]) -> Optional[float]:
    """把布尔匹配结果转换成比例。"""
    values = list(matches)
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _doc_search_text(doc: Dict[str, Any], context: str) -> str:
    """构造用于业务约束匹配的文本。"""
    parts = [
        doc.get("dish_name", ""),
        doc.get("category", ""),
        doc.get("difficulty", ""),
        doc.get("content_preview", ""),
        context,
    ]
    return "\n".join(str(part) for part in parts if part)


def _matches_category(doc: Dict[str, Any], categories: List[str]) -> bool:
    """判断菜品分类是否命中约束。"""
    return doc.get("category") in set(categories)


def _matches_difficulty(doc: Dict[str, Any], difficulties: List[str]) -> bool:
    """判断菜品难度是否命中约束。"""
    return doc.get("difficulty") in set(difficulties)


def _matches_ingredients(
    doc: Dict[str, Any], context: str, constraints: Dict[str, Any]
) -> bool:
    """判断菜品文本是否满足食材约束。"""
    text = _doc_search_text(doc, context)
    ingredients_any = constraints.get("ingredients_any") or []
    ingredients_all = constraints.get("ingredients_all") or []

    if ingredients_any and not _contains_any(text, ingredients_any):
        return False
    if ingredients_all and not all(term in text for term in ingredients_all):
        return False
    return True


def _matches_all_constraints(
    doc: Dict[str, Any], context: str, constraints: Dict[str, Any]
) -> bool:
    """综合判断一条推荐是否满足所有已声明业务约束。"""
    if constraints.get("category") and not _matches_category(
        doc, constraints["category"]
    ):
        return False
    if constraints.get("difficulty") and not _matches_difficulty(
        doc, constraints["difficulty"]
    ):
        return False
    if (
        constraints.get("ingredients_any") or constraints.get("ingredients_all")
    ) and not _matches_ingredients(doc, context, constraints):
        return False
    text = _doc_search_text(doc, context)
    if constraints.get("positive_terms") and not _contains_any(
        text, constraints["positive_terms"]
    ):
        return False
    if constraints.get("negative_terms") and _contains_any(
        text, constraints["negative_terms"]
    ):
        return False
    return True


def _contains_any(text: str, terms: List[str]) -> bool:
    """判断文本是否包含任意一个关键词。"""
    return any(term in text for term in terms)


def _write_scores_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """写入简洁分数 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "case_type",
        "user_input",
        "expected_dishes",
        "expected_constraints",
        "expected_plan",
        "retrieved_dishes",
        "session_id",
        "route_type",
        "answer_style",
        "rewritten_query",
        "standalone_query",
        "target_dishes",
        "focus_dish",
        "needs_clarification",
        "query_plan",
        "filters",
        "preferences",
        "refusal_reason",
        "latency_ms",
        *PLAN_METRIC_NAMES,
        *BUSINESS_METRIC_NAMES,
        "faithfulness",
        "response_relevancy",
        "context_precision",
        "context_recall",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writable = dict(row)
            writable["expected_dishes"] = "、".join(row.get("expected_dishes") or [])
            writable["expected_constraints"] = json.dumps(
                row.get("expected_constraints") or {}, ensure_ascii=False
            )
            writable["expected_plan"] = json.dumps(
                row.get("expected_plan") or {}, ensure_ascii=False
            )
            writable["retrieved_dishes"] = "、".join(row.get("retrieved_dishes") or [])
            writable["target_dishes"] = "、".join(row.get("target_dishes") or [])
            writable["query_plan"] = json.dumps(
                row.get("query_plan") or {}, ensure_ascii=False
            )
            writable["filters"] = json.dumps(
                row.get("filters") or {}, ensure_ascii=False
            )
            writable["preferences"] = json.dumps(
                row.get("preferences") or {}, ensure_ascii=False
            )
            writer.writerow(writable)


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """计算均值，自动跳过空值。"""
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return statistics.mean(clean_values)


def _format_score(value: Optional[float]) -> str:
    """格式化分数。"""
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def _low_score_reasons(row: Dict[str, Any]) -> List[str]:
    """根据阈值给低分样例打诊断标签。"""
    reasons: List[str] = []
    thresholds = {
        "faithfulness": (0.7, "可能存在不忠实或幻觉"),
        "response_relevancy": (0.7, "回答相关性不足"),
        "context_precision": (0.6, "检索上下文噪声较大"),
        "context_recall": (0.6, "检索上下文召回不足"),
    }
    if row.get("case_type") in {"general", "list"}:
        thresholds.pop("context_recall")

    for field, (threshold, reason) in thresholds.items():
        value = row.get(field)
        if value is not None and value < threshold:
            reasons.append(reason)
    valid_rate = row.get("valid_recommendation_rate")
    if valid_rate is not None and valid_rate < 0.6:
        reasons.append("业务约束命中不足")
    violation_rate = row.get("negative_violation_rate")
    if violation_rate is not None and violation_rate > 0:
        reasons.append("违反排除条件")
    if row.get("error"):
        reasons.append("运行出错")
    return reasons


def _write_report(path: Path, rows: List[Dict[str, Any]], skipped_ragas: bool) -> None:
    """生成 Markdown 报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "faithfulness",
        "response_relevancy",
        "context_precision",
        "context_recall",
    ]

    lines: List[str] = []
    lines.append("# RAGAS 评测报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 样例数量：{len(rows)}")
    lines.append(f"- 是否跳过 RAGAS：{'是' if skipped_ragas else '否'}")
    lines.append("- 主评测集聚焦当前生产链路的 LLM planner 输出：detail、general、list")
    lines.append(
        "- list 是开放集合推荐题，context_recall 只表示测试集代表菜覆盖，不等价于推荐正确率"
    )
    lines.append(
        "- general/list 推荐类问题优先参考业务约束指标，expected_dishes 只是代表菜样例"
    )
    lines.append(
        "- RAGAS 上下文：使用现有生成模块实际传给 LLM 的上下文；完整父文档上下文保存在 trace 中"
    )
    lines.append(
        "- 分数为 N/A 表示该指标未返回有效结果，常见原因是评测模型超时或结构化输出失败"
    )
    lines.append("")

    lines.append("## 主任务参考平均分")
    lines.append("")
    lines.append("| 指标 | 平均分 |")
    lines.append("| --- | --- |")
    for metric in metric_names:
        lines.append(
            f"| {metric} | {_format_score(_mean(row.get(metric) for row in rows))} |"
        )
    lines.append("")

    plan_rows = [row for row in rows if row.get("expected_plan")]
    lines.append("## Planner 期望命中")
    lines.append("")
    if not plan_rows:
        lines.append("暂无带 expected_plan 的样例。")
    else:
        lines.append(
            "| 指标 | 平均命中率 |"
        )
        lines.append("| --- | ---: |")
        for metric in PLAN_METRIC_NAMES:
            lines.append(
                f"| {metric} | {_format_score(_mean(row.get(metric) for row in plan_rows))} |"
            )
    lines.append("")

    lines.append("## 按任务类型平均分")
    lines.append("")
    lines.append(
        "| case_type | 数量 | faithfulness | response_relevancy | context_precision | context_recall |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    case_types = sorted({row["case_type"] for row in rows})
    for case_type in case_types:
        group = [row for row in rows if row["case_type"] == case_type]
        scores = [
            _format_score(_mean(row.get(metric) for row in group))
            for metric in metric_names
        ]
        lines.append(f"| {case_type} | {len(group)} | {' | '.join(scores)} |")
    lines.append("")

    business_rows = [
        row
        for row in rows
        if row.get("case_type") in {"general", "list"}
        and row.get("expected_constraints")
    ]
    lines.append("## 推荐类业务约束指标")
    lines.append("")
    if not business_rows:
        lines.append("暂无带业务约束的 general/list 样例。")
    else:
        lines.append(
            "| case_type | 数量 | dish_hit_rate | category_match_rate | "
            "difficulty_match_rate | ingredient_match_rate | positive_term_match_rate | "
            "negative_violation_rate | valid_recommendation_rate |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        business_case_types = sorted({row["case_type"] for row in business_rows})
        for case_type in business_case_types:
            group = [row for row in business_rows if row["case_type"] == case_type]
            scores = [
                _format_score(_mean(row.get(metric) for row in group))
                for metric in BUSINESS_METRIC_NAMES
            ]
            lines.append(f"| {case_type} | {len(group)} | {' | '.join(scores)} |")
        scores = [
            _format_score(_mean(row.get(metric) for row in business_rows))
            for metric in BUSINESS_METRIC_NAMES
        ]
        lines.append(f"| overall | {len(business_rows)} | {' | '.join(scores)} |")
    lines.append("")

    low_rows = []
    for row in rows:
        reasons = _low_score_reasons(row)
        if reasons:
            low_rows.append((row, reasons))

    lines.append("## 低分或异常样例")
    lines.append("")
    if not low_rows:
        lines.append("暂无低分或异常样例。")
    else:
        lines.append("| id | 类型 | 问题 | 期望菜品 | 检索菜品 | 有效推荐率 | 诊断 |")
        lines.append("| --- | --- | --- | --- | --- | ---: | --- |")
        for row, reasons in low_rows:
            expected = "、".join(row.get("expected_dishes") or [])
            retrieved = "、".join(row.get("retrieved_dishes") or [])
            valid_rate = _format_score(row.get("valid_recommendation_rate"))
            lines.append(
                f"| {row['id']} | {row['case_type']} | {row['user_input']} | "
                f"{expected} | {retrieved} | {valid_rate} | {'；'.join(reasons)} |"
            )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行菜谱 RAG 的 RAGAS 评测")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="JSONL 测试集路径",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="trace 和分数输出目录",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=DEFAULT_REPORTS_DIR,
        help="Markdown 报告输出目录",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只运行前 N 条样例，便于调试",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="只生成 RAG trace，不调用 RAGAS 打分",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="RAGAS evaluate 的 batch_size",
    )
    parser.add_argument(
        "--ragas-timeout",
        type=int,
        default=180,
        help="RAGAS 单个指标任务的超时时间，单位秒",
    )
    return parser.parse_args()


def main() -> None:
    """主入口。"""
    _configure_console_encoding()
    args = _parse_args()
    dataset_path = args.dataset.resolve()
    runs_dir = args.runs_dir.resolve()
    reports_dir = args.reports_dir.resolve()

    _prepare_import_path()

    # 现有配置里的 data_path 和 index_save_path 是相对 code 目录的路径。
    os.chdir(CODE_DIR)

    from RecipeRAGSystem import DEFAULT_CONFIG, RecipeRAGSystem

    cases = _load_jsonl(dataset_path)
    _validate_dataset(cases, dataset_path)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise ValueError("测试集为空，无法运行评测")

    rag_system = RecipeRAGSystem(DEFAULT_CONFIG)
    rag_system.initialize_system()
    rag_system.build_knowledge_base()

    traces: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(
            f"[{index}/{len(cases)}] 运行样例 {case.get('id')}: {case.get('user_input')}"
        )
        traces.append(_run_single_case(rag_system, case))

    trace_path = runs_dir / "latest_trace.jsonl"
    _write_jsonl(trace_path, traces)
    print(f"trace 已写入: {trace_path}")

    if args.skip_ragas:
        scores = _empty_scores(len(traces))
    else:
        scores = _run_ragas(traces, rag_system, args.batch_size, args.ragas_timeout)

    score_rows = _merge_scores(traces, scores)
    scores_path = runs_dir / "latest_scores.csv"
    report_path = reports_dir / "latest_report.md"
    _write_scores_csv(scores_path, score_rows)
    _write_report(report_path, score_rows, skipped_ragas=args.skip_ragas)

    print(f"分数 CSV 已写入: {scores_path}")
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
