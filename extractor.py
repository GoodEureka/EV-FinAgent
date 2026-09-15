"""LLM-based metric extraction with retrieval and citation verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from llm_provider import (
    is_deepseek,
    parse_json_object,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    structured_response_format,
)
from pdf_parser import PageText, TextChunk, chunk_pages


PRIMARY_METRICS = (
    "revenue",
    "vehicle_delivery",
    "vehicle_margin",
    "gross_margin",
    "rd_expense",
    "cash_balance",
    "operating_cash_flow",
    "net_loss",
)

SUPPORTING_METRICS = (
    "gross_profit",
    "total_cost_of_sales",
    "vehicle_revenue",
    "vehicle_cost_of_sales",
)

METRICS = PRIMARY_METRICS + SUPPORTING_METRICS

KEYWORDS = {
    "revenue": ("revenue", "revenues", "收入", "营收"),
    "vehicle_delivery": ("deliveries", "delivered", "交付", "交付量"),
    "vehicle_margin": ("vehicle margin", "汽车毛利率", "车辆毛利率"),
    "gross_margin": ("gross margin", "毛利率"),
    "rd_expense": ("research and development", "r&d", "研发开支", "研发费用"),
    "cash_balance": (
        "cash and cash equivalents",
        "cash balance",
        "现金及现金等价物",
        "现金储备",
    ),
    "operating_cash_flow": (
        "operating cash flow",
        "cash used in operating",
        "经营活动现金流",
    ),
    "net_loss": ("net loss", "net income", "净亏损", "净利润"),
    "gross_profit": ("gross profit", "毛利润", "毛利"),
    "total_cost_of_sales": ("total cost of sales", "cost of sales", "营业成本", "销售成本"),
    "vehicle_revenue": ("vehicle sales", "revenues from vehicle sales", "汽车销售收入"),
    "vehicle_cost_of_sales": ("vehicle sales", "cost of sales", "汽车销售成本"),
}


@dataclass
class MetricEvidence:
    metric: str
    value: str | None
    source_page: int | None
    source_text: str | None
    verified: bool = False


@dataclass
class ExtractionResult:
    company: str
    year: str
    metrics: list[MetricEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "year": self.year,
            "metrics": [asdict(metric) for metric in self.metrics],
        }


def _schema() -> dict[str, Any]:
    item = {
        "type": "object",
        "properties": {
            "metric": {"type": "string", "enum": list(METRICS)},
            "value": {"type": ["string", "null"]},
            "source_page": {"type": ["integer", "null"]},
            "source_text": {"type": ["string", "null"]},
        },
        "required": ["metric", "value", "source_page", "source_text"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "year": {"type": "string"},
            "metrics": {
                "type": "array",
                "items": item,
                "minItems": len(METRICS),
                "maxItems": len(METRICS),
            },
        },
        "required": ["company", "year", "metrics"],
        "additionalProperties": False,
    }


def retrieve_chunks(chunks: Iterable[TextChunk], per_metric: int = 4) -> list[TextChunk]:
    """Small lexical retrieval layer: enough for an MVP without a vector DB."""
    pool = list(chunks)
    selected: dict[str, TextChunk] = {}
    for metric in METRICS:
        words = KEYWORDS[metric]
        ranked = sorted(
            pool,
            key=lambda chunk: sum(chunk.text.lower().count(word.lower()) for word in words),
            reverse=True,
        )
        for chunk in ranked[:per_metric]:
            if any(word.lower() in chunk.text.lower() for word in words):
                selected[chunk.chunk_id] = chunk
    return list(selected.values())


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _coerce_page_number(value: Any) -> int | None:
    """Accept an integer or a strict `PAGE 123` string from JSON-only providers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(?:page\s*)?(\d+)\s*", value, flags=re.IGNORECASE)
        if match:
            page = int(match.group(1))
            return page if page > 0 else None
    return None


def verify_and_normalize(
    raw: dict[str, Any], pages: list[PageText], company: str, year: str
) -> ExtractionResult:
    """Reject values whose quoted evidence cannot be found on the claimed page."""
    page_map = {page.page: _compact(page.text) for page in pages}
    received = {item.get("metric"): item for item in raw.get("metrics", [])}
    result: list[MetricEvidence] = []

    for name in METRICS:
        item = received.get(name, {})
        value = item.get("value")
        source_page = _coerce_page_number(item.get("source_page"))
        source_text = item.get("source_text")
        valid = (
            value is not None
            and source_page is not None
            and isinstance(source_text, str)
            and len(_compact(source_text)) >= 8
            and _compact(source_text) in page_map.get(source_page, "")
        )
        result.append(
            MetricEvidence(
                metric=name,
                value=str(value).strip() if valid else None,
                source_page=source_page if valid else None,
                source_text=source_text.strip() if valid else None,
                verified=valid,
            )
        )
    return ExtractionResult(company=company, year=str(year), metrics=result)


def extract_metrics(
    pages: list[PageText],
    company: str,
    year: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str = "openai",
    debug_raw: dict[str, Any] | None = None,
) -> ExtractionResult:
    from openai import OpenAI

    if company not in {"小鹏汽车", "理想汽车", "蔚来汽车"}:
        raise ValueError("MVP only supports 小鹏汽车、理想汽车、蔚来汽车")

    chunks = retrieve_chunks(chunk_pages(pages))
    if not chunks:
        return verify_and_normalize({}, pages, company, year)

    context = "\n\n".join(f"[PAGE {c.page}]\n{c.text}" for c in chunks)
    prompt_path = Path(__file__).parent / "prompts" / "extraction_prompt.txt"
    instructions = prompt_path.read_text(encoding="utf-8")
    resolved_base_url = resolve_base_url(provider, base_url)
    deepseek_mode = is_deepseek(provider, resolved_base_url)
    if deepseek_mode:
        example_metrics = [
            {"metric": name, "value": None, "source_page": None, "source_text": None}
            for name in METRICS
        ]
        instructions += (
            "\n你必须只输出一个合法 JSON 对象，不要输出 Markdown。JSON 格式示例：\n"
            + json.dumps(
                {"company": company, "year": str(year), "metrics": example_metrics},
                ensure_ascii=False,
            )
        )
    client = OpenAI(
        api_key=resolve_api_key(provider, api_key),
        base_url=resolved_base_url,
    )
    request: dict[str, Any] = {
        "model": resolve_model(provider, model),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": f"Company: {company}\nYear: {year}\n\nAnnual report excerpts:\n{context}",
            },
        ],
        "response_format": structured_response_format(
            provider,
            resolved_base_url,
            name="annual_report_metrics",
            schema=_schema(),
        ),
    }
    if deepseek_mode:
        # DeepSeek currently enables thinking by default. Structured extraction
        # is more reliable and cheaper in non-thinking mode, leaving the output
        # budget for the JSON object itself.
        request["max_tokens"] = 8192
        request["extra_body"] = {"thinking": {"type": "disabled"}}

    attempts = 2 if deepseek_mode else 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = client.chat.completions.create(**request)
            raw = parse_json_object(response.choices[0].message.content, "Metric extractor")
            if debug_raw is not None:
                debug_raw.clear()
                debug_raw.update(raw)
            return verify_and_normalize(raw, pages, company, year)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"The model did not return usable structured content: {last_error}")
