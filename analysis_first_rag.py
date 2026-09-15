"""Analysis-first RAG: open-ended document analysis followed by evidence grounding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from typing import Any, Iterable

from openai import OpenAI

from llm_provider import (
    is_deepseek,
    parse_json_object,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    structured_response_format,
)
from pdf_parser import PageText, TextChunk, chunk_pages


QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "search_terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["unit_id", "search_terms"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["queries"],
    "additionalProperties": False,
}

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "draft_report": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"},
            "claim_type": {"type": "string", "enum": ["disclosed_fact", "derived_metric", "analytical_inference", "qualitative_view"]},
            "importance": {"type": "string", "enum": ["high", "medium", "low"]},
            "section": {"type": "string"}, "calculation": {"type": "string"}},
            "required": ["text", "claim_type", "importance", "section", "calculation"], "additionalProperties": False}},
        "follow_up_questions": {"type": "array", "items": {"type": "object", "properties": {
            "question": {"type": "string"}, "required_evidence": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["question", "required_evidence", "reason"], "additionalProperties": False}},
    },
    "required": ["draft_report", "claims", "follow_up_questions"], "additionalProperties": False,
}

FINAL_REPORT_SCHEMA = {"type": "object", "properties": {"report": {"type": "string"}},
                       "required": ["report"], "additionalProperties": False}

CLAIMS_SCHEMA = {"type": "object", "properties": {"claims": ANALYSIS_SCHEMA["properties"]["claims"]},
                 "required": ["claims"], "additionalProperties": False}

INFERENCE_SCHEMA = {"type": "object", "properties": {"inferences": {"type": "array", "items": {
    "type": "object", "properties": {"inference_id": {"type": "string"}, "conclusion": {"type": "string"},
    "premise_ids": {"type": "array", "items": {"type": "string"}}, "reasoning": {"type": "string"},
    "caveat": {"type": "string"}, "dimension": {"type": "string"}},
    "required": ["inference_id", "conclusion", "premise_ids", "reasoning", "caveat", "dimension"],
    "additionalProperties": False}}}, "required": ["inferences"], "additionalProperties": False}

INFERENCE_AUDIT_SCHEMA = {"type": "object", "properties": {"reviews": {"type": "array", "items": {
    "type": "object", "properties": {"inference_id": {"type": "string"},
    "verdict": {"type": "string", "enum": ["supported", "weak", "unsupported"]},
    "revised_conclusion": {"type": "string"}, "explanation": {"type": "string"}},
    "required": ["inference_id", "verdict", "revised_conclusion", "explanation"],
    "additionalProperties": False}}}, "required": ["reviews"], "additionalProperties": False}

EVIDENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "audited_report": {"type": "string"},
        "overall_assessment": {"type": "string"},
        "verifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "unit_id": {"type": "string"},
                    "original_statement": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "supported",
                            "partially_supported",
                            "unsupported",
                            "contradicted",
                            "non_verifiable",
                        ],
                    },
                    "verified_statement": {"type": "string"},
                    "explanation": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page": {"type": "integer"},
                                "quote": {"type": "string"},
                                "relationship": {"type": "string"},
                            },
                            "required": ["page", "quote", "relationship"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "unit_id",
                    "original_statement",
                    "verdict",
                    "verified_statement",
                    "explanation",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["audited_report", "overall_assessment", "verifications"],
    "additionalProperties": False,
}

VERDICT_LABELS = {
    "supported": "证据支持",
    "partially_supported": "部分支持",
    "unsupported": "证据不足",
    "contradicted": "存在矛盾",
    "non_verifiable": "不可由年报核验",
}


@dataclass(frozen=True)
class AnalysisUnit:
    unit_id: str
    statement: str
    claim_type: str = "disclosed_fact"
    importance: str = "medium"
    section: str = "经营分析"
    calculation: str = ""
    priority_score: int = 0
    mandatory: bool = False


@dataclass(frozen=True)
class RetrievedEvidence:
    page: int
    chunk_id: str
    score: float
    text: str


def _client(provider: str, api_key: str | None, base_url: str | None) -> OpenAI:
    return OpenAI(
        api_key=resolve_api_key(provider, api_key),
        base_url=resolve_base_url(provider, base_url),
    )


def build_full_document_context(
    pages: Iterable[PageText], *, max_chars: int = 2_800_000
) -> tuple[str, dict[str, Any]]:
    """Build a page-tagged full-document prompt and refuse silent truncation."""
    usable = [page for page in pages if page.text]
    parts = [f"<<<PDF_PAGE_{page.page}>>>\n{page.text}" for page in usable]
    context = "\n\n".join(parts)
    if len(context) > max_chars:
        raise ValueError(
            f"全文解析后为 {len(context):,} 字符，超过当前安全上限 {max_chars:,}；"
            "为避免静默截断，请提高上限或启用分层全文摘要。"
        )
    return context, {
        "mode": "full_document",
        "included_pages": len(usable),
        "first_page": usable[0].page if usable else None,
        "last_page": usable[-1].page if usable else None,
        "input_characters": len(context),
        "truncated": False,
    }


def generate_open_analysis(
    pages: list[PageText],
    company: str,
    year: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str = "deepseek",
    max_chars: int = 2_800_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a readable draft, atomic claims and separate due-diligence questions."""
    context, coverage = build_full_document_context(pages, max_chars=max_chars)
    resolved_url = resolve_base_url(provider, base_url)
    deepseek_mode = is_deepseek(provider, resolved_url)
    prompt = (
        "你是银行新能源车产业金融分析师。请完整阅读以下带PDF页码标记的年度报告，"
        "用自然、连贯的中文撰写经营分析。"
        "你可以自主识别最重要的经营变化、业务模式、盈利与现金流质量、竞争位置、管理层叙述、"
        "潜在风险。draft_report只写基于本年报能够形成的分析，不得写问题句、待验证猜测或调查清单。"
        "无法由单份年报确认、需要跨年度数据、新闻、访谈或补充材料的问题，只放入follow_up_questions，"
        "绝不能混入draft_report。draft_report控制在2500个中文字符以内。claims固定返回空数组；"
        "后续阶段会单独提取主张。不得使用年报之外的知识。"
    )
    if deepseek_mode:
        prompt += "只输出合法JSON对象，键名必须是draft_report、claims、follow_up_questions，不要Markdown代码围栏。"
    request: dict[str, Any] = {
        "model": resolve_model(provider, model),
        "temperature": 0.25,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"公司：{company}\n报告年度：{year}\n\n{context}",
            },
        ],
        "response_format": structured_response_format(provider, resolved_url, name="analysis_package", schema=ANALYSIS_SCHEMA),
    }
    if deepseek_mode:
        request["max_tokens"] = 6000
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = _client(provider, api_key, resolved_url).chat.completions.create(**request)
    raw_content = response.choices[0].message.content
    if not raw_content or len(raw_content.strip()) < 100:
        raise ValueError("全文开放分析返回内容过短或为空")
    package = parse_json_object(raw_content, "Analysis package")
    if not isinstance(package.get("draft_report"), str) or not package["draft_report"].strip():
        raise ValueError("全文分析未返回draft_report")
    if not isinstance(package.get("claims"), list):
        package["claims"] = []
    questions = package.get("follow_up_questions", [])
    normalized_questions = []
    if isinstance(questions, list):
        for item in questions:
            if isinstance(item, str) and item.strip():
                normalized_questions.append({"question": item.strip(), "required_evidence": "需补充材料", "reason": "单份年报无法确认"})
            elif isinstance(item, dict) and str(item.get("question", "")).strip():
                normalized_questions.append({"question": str(item.get("question", "")).strip(),
                    "required_evidence": str(item.get("required_evidence", "需补充材料")).strip(),
                    "reason": str(item.get("reason", "单份年报无法确认")).strip()})
    package["follow_up_questions"] = normalized_questions
    package["draft_report"] = package["draft_report"].strip()
    return package, coverage


def generate_atomic_claims(analysis: str, *, model: str | None, api_key: str | None,
                           base_url: str | None, provider: str) -> list[dict[str, Any]]:
    resolved_url = resolve_base_url(provider, base_url)
    system = (
        "从经营分析正文中提取20至40条最小独立主张。数字、日期、趋势、事件、条款和强推论必须覆盖；"
        "纯修辞、标题、问题句和待调查事项不得成为claim。claim_type只能是disclosed_fact、derived_metric、"
        "analytical_inference或qualitative_view。不要补充正文没有的事实。只输出JSON对象{\"claims\":[...]}。"
    )
    request = {"model": resolve_model(provider, model), "temperature": 0,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": analysis}],
               "response_format": structured_response_format(provider, resolved_url, name="atomic_claims", schema=CLAIMS_SCHEMA)}
    if is_deepseek(provider, resolved_url):
        request["max_tokens"] = 8000
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = _client(provider, api_key, resolved_url).chat.completions.create(**request)
    raw = parse_json_object(response.choices[0].message.content, "Atomic claims")
    claims = raw.get("claims", [])
    if not isinstance(claims, list) or not claims:
        raise ValueError("主张提取阶段未返回非空claims数组")
    return claims


def split_analysis_units(analysis: str, max_units: int = 24) -> list[AnalysisUnit]:
    """Legacy prose splitter retained for compatibility; selection is priority based."""
    cleaned = re.sub(r"```.*?```", " ", analysis, flags=re.S)
    pieces = re.split(r"(?<=[。！？!?])\s*|\n+", cleaned)
    statements: list[str] = []
    for piece in pieces:
        raw_piece = piece.strip()
        if raw_piece.startswith("#"):
            continue
        statement = re.sub(r"^\s*[#>*-]+\s*", "", piece).strip()
        statement = re.sub(r"^\s*\d+[.、)）]\s*", "", statement)
        statement = re.sub(r"^\s*[一二三四五六七八九十]+[、.．]\s*", "", statement)
        statement = statement.replace("**", "").strip()
        statement = re.sub(r"\s+", " ", statement)
        if len(statement) < 8:
            continue
        if statement.endswith(("：", ":")):
            continue
        if len(statement) < 40 and not re.search(r"[。！？!?]$", raw_piece) and re.match(
            r"^[一二三四五六七八九十]+[、.．]", raw_piece
        ):
            continue
        if len(statement) > 600:
            subparts = re.split(r"(?<=[。！？!?；;])", statement)
        else:
            subparts = [statement]
        for subpart in subparts:
            subpart = subpart.strip()
            if len(subpart) >= 8:
                statements.append(subpart)
    return select_analysis_units(
        [AnalysisUnit(f"u{index}", statement) for index, statement in enumerate(statements, 1)], max_units
    )


def _claim_priority(unit: AnalysisUnit) -> tuple[int, bool]:
    text = unit.statement
    score = {"high": 40, "medium": 20, "low": 5}.get(unit.importance, 10)
    score += {"disclosed_fact": 30, "derived_metric": 35, "analytical_inference": 25,
              "qualitative_view": 0}.get(unit.claim_type, 10)
    has_number = bool(re.search(r"\d|%|％|亿元|万元|million|billion", text, re.I))
    has_trend = bool(re.search(r"同比|环比|增长|下降|增加|减少|改善|恶化|year.on.year|increase|decrease", text, re.I))
    has_inference = bool(re.search(r"说明|意味着|导致|证明|表明|因此|反映", text))
    has_risk = bool(re.search(r"风险|赎回|诉讼|违约|流动性|现金流|亏损|毛利|负债", text))
    score += 25 if has_number else 0
    score += 15 if has_trend else 0
    score += 15 if has_inference else 0
    score += 15 if has_risk else 0
    mandatory = unit.claim_type in {"disclosed_fact", "derived_metric"} and (has_number or has_trend or has_risk)
    mandatory = mandatory or (unit.claim_type == "analytical_inference" and has_inference)
    return score, mandatory


def select_analysis_units(units: list[AnalysisUnit], max_units: int) -> list[AnalysisUnit]:
    """Keep every mandatory claim; apply the budget only to lower-priority claims."""
    rescored = []
    for unit in units:
        score, mandatory = _claim_priority(unit)
        rescored.append(AnalysisUnit(**{**asdict(unit), "priority_score": max(score, unit.priority_score),
                                        "mandatory": unit.mandatory or mandatory}))
    required = [unit for unit in rescored if unit.mandatory]
    optional = sorted((unit for unit in rescored if not unit.mandatory), key=lambda item: item.priority_score, reverse=True)
    selected_ids = {unit.unit_id for unit in required + optional[:max(0, max_units - len(required))]}
    return [unit for unit in rescored if unit.unit_id in selected_ids]


def units_from_claims(claims: list[dict[str, Any]], max_units: int) -> list[AnalysisUnit]:
    candidates = []
    for index, claim in enumerate(claims, 1):
        if isinstance(claim, str):
            claim = {"text": claim, "claim_type": "disclosed_fact", "importance": "medium",
                     "section": "经营分析", "calculation": ""}
        if not isinstance(claim, dict):
            continue
        text_value = claim.get("text") or claim.get("statement") or claim.get("claim") or claim.get("content") or ""
        text = re.sub(r"\s+", " ", str(text_value)).strip()
        claim_type = str(claim.get("claim_type") or claim.get("type") or "disclosed_fact")
        type_aliases = {"fact": "disclosed_fact", "calculation": "derived_metric",
                        "inference": "analytical_inference", "view": "qualitative_view"}
        claim_type = type_aliases.get(claim_type, claim_type)
        if len(text) < 8 or claim_type == "qualitative_view":
            continue
        candidates.append(AnalysisUnit(f"c{index}", text, claim_type,
            str(claim.get("importance", "medium")), str(claim.get("section", "经营分析")),
            str(claim.get("calculation", ""))))
    return select_analysis_units(candidates, max_units)


def generate_retrieval_queries(
    units: list[AnalysisUnit],
    year: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str = "deepseek",
) -> dict[str, list[str]]:
    """Translate open-ended Chinese statements into annual-report search language."""
    resolved_url = resolve_base_url(provider, base_url)
    deepseek_mode = is_deepseek(provider, resolved_url)
    system = (
        "你是金融文档检索查询生成器。输入是从中文分析报告中自动拆出的自然语言句子。"
        "为每句话生成4至8个适合检索英文年报的短语，包含常见英文披露术语、关键数字（若有）、"
        f"年份{year}及必要同义词。不要判断句子真假，只生成检索词。"
    )
    if deepseek_mode:
        system += (
            "只输出JSON对象，不要Markdown。必须严格使用以下键名和形状："
            '{"queries":[{"unit_id":"u1","search_terms":'
            '["net loss 2025","operating cash flow 2025"]}]}。'
            "禁止改名为query、items、unit_queries或其他字段。"
        )
    request: dict[str, Any] = {
        "model": resolve_model(provider, model),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps([asdict(unit) for unit in units], ensure_ascii=False)},
        ],
        "response_format": structured_response_format(
            provider, resolved_url, name="retrieval_queries", schema=QUERY_SCHEMA
        ),
    }
    if deepseek_mode:
        request["max_tokens"] = 4096
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    valid_ids = {unit.unit_id for unit in units}
    result: dict[str, list[str]] = {}
    attempts = 2 if deepseek_mode else 1
    for attempt in range(attempts):
        if attempt:
            request["messages"].append(
                {
                    "role": "user",
                    "content": "上次字段格式不合格。仅返回包含queries数组的JSON，并为每个unit_id填写search_terms。",
                }
            )
        response = _client(provider, api_key, resolved_url).chat.completions.create(**request)
        raw = parse_json_object(response.choices[0].message.content, "Query generator")
        result = {}
        for item in raw.get("queries", []):
            unit_id = item.get("unit_id")
            terms = item.get("search_terms")
            if unit_id in valid_ids and isinstance(terms, list):
                result[unit_id] = [str(term).strip() for term in terms if str(term).strip()][:10]
        if len(result) >= max(1, math.ceil(len(units) * 0.8)):
            break
    for unit in units:
        result.setdefault(unit.unit_id, [unit.statement, year])
    return result


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    latin = re.findall(r"[a-z0-9%.-]+", lowered)
    chinese_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    chinese = [run[index : index + 2] for run in chinese_runs for index in range(max(0, len(run) - 1))]
    return latin + chinese


def retrieve_evidence(
    pages: list[PageText],
    units: list[AnalysisUnit],
    queries: dict[str, list[str]],
    *,
    top_k: int = 4,
) -> dict[str, list[RetrievedEvidence]]:
    """Small dependency-free BM25 retriever with page diversity."""
    chunks = chunk_pages(pages, chunk_size=2600, overlap=250)
    documents = [_tokens(chunk.text) for chunk in chunks]
    document_count = len(documents)
    if not document_count:
        return {unit.unit_id: [] for unit in units}
    document_frequency: dict[str, int] = {}
    for tokens in documents:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    average_length = sum(len(tokens) for tokens in documents) / document_count
    k1, b = 1.5, 0.75
    results: dict[str, list[RetrievedEvidence]] = {}
    for unit in units:
        query_text = " ".join([unit.statement, *queries.get(unit.unit_id, [])])
        query_tokens = list(dict.fromkeys(_tokens(query_text)))
        scored: list[tuple[float, TextChunk]] = []
        for chunk, tokens in zip(chunks, documents):
            if not tokens:
                continue
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                df = document_frequency.get(token, 0)
                inverse_frequency = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(tokens) / average_length)
                score += inverse_frequency * frequency * (k1 + 1) / denominator
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected: list[RetrievedEvidence] = []
        selected_pages: set[int] = set()
        for score, chunk in scored:
            if chunk.page in selected_pages:
                continue
            selected_pages.add(chunk.page)
            selected.append(RetrievedEvidence(chunk.page, chunk.chunk_id, round(score, 4), chunk.text))
            if len(selected) >= top_k:
                break
        results[unit.unit_id] = selected
    return results


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def validate_verification_output(
    raw: dict[str, Any],
    units: list[AnalysisUnit],
    retrieved: dict[str, list[RetrievedEvidence]],
) -> dict[str, Any]:
    unit_map = {unit.unit_id: unit for unit in units}
    verified_items = []
    for item in raw.get("verifications", []):
        unit_id = item.get("unit_id")
        if unit_id not in unit_map or item.get("verdict") not in VERDICT_LABELS:
            continue
        allowed = {evidence.page: evidence.text for evidence in retrieved.get(unit_id, [])}
        valid_evidence = []
        for evidence in item.get("evidence", []):
            page = evidence.get("page")
            quote = evidence.get("quote")
            if page in allowed and isinstance(quote, str) and _normalize_quote(quote) in _normalize_quote(allowed[page]):
                valid_evidence.append(
                    {
                        "page": page,
                        "quote": quote.strip(),
                        "relationship": str(evidence.get("relationship", "")).strip(),
                        "verified": True,
                    }
                )
        verdict = item["verdict"]
        explanation = str(item.get("explanation", "")).strip()
        if verdict in {"supported", "partially_supported"} and not valid_evidence:
            verdict = "unsupported"
            explanation = "模型未返回能在召回页面逐字匹配的证据；程序已降级为证据不足。"
        action = {
            "supported": "keep_or_clarify", "partially_supported": "split_and_qualify",
            "unsupported": "remove", "contradicted": "correct", "non_verifiable": "move_to_due_diligence",
        }[verdict]
        rewritten = str(item.get("verified_statement", "")).strip()
        if verdict in {"unsupported", "non_verifiable"}:
            rewritten = ""
        elif verdict in {"partially_supported", "contradicted"} and rewritten == unit_map[unit_id].statement:
            rewritten = ""
        verified_items.append(
            {
                "unit_id": unit_id,
                "original_statement": unit_map[unit_id].statement,
                "verdict": verdict,
                "verdict_label": VERDICT_LABELS[verdict],
                "verified_statement": rewritten,
                "revision_action": action,
                "explanation": explanation,
                "evidence": valid_evidence,
                "retrieved_pages": sorted(allowed),
            }
        )
    returned = {item["unit_id"] for item in verified_items}
    for unit in units:
        if unit.unit_id not in returned:
            verified_items.append(
                {
                    "unit_id": unit.unit_id,
                    "original_statement": unit.statement,
                    "verdict": "unsupported",
                    "verdict_label": VERDICT_LABELS["unsupported"],
                    "verified_statement": "",
                    "revision_action": "remove",
                    "explanation": "核验模型未返回该分析单元。",
                    "evidence": [],
                    "retrieved_pages": [item.page for item in retrieved.get(unit.unit_id, [])],
                }
            )
    model_audited_report = raw.get("audited_report")
    paragraphs = []
    for item in verified_items:
        if item["verdict"] not in {"supported", "partially_supported"} or not item["evidence"]:
            continue
        statement = item["verified_statement"] or item["original_statement"]
        pages = sorted({evidence["page"] for evidence in item["evidence"]})
        citation = " " + "".join(f"[PDF第{page}页]" for page in pages)
        paragraphs.append(statement.rstrip("。；;，,") + "。" + citation)
    audited_report = "\n\n".join(paragraphs) or "本次未形成具备逐字引文支持的可信分析结论。"
    overall_assessment = raw.get("overall_assessment")
    if not isinstance(overall_assessment, str) or not overall_assessment.strip():
        supported_count = sum(
            item["verdict"] in {"supported", "partially_supported"} for item in verified_items
        )
        overall_assessment = (
            f"共核验{len(verified_items)}条开放分析语句，其中{supported_count}条获得全部或部分证据支持；"
            "模型缺失的总体评价已由程序补充。"
        )
    return {
        "audited_report": audited_report.strip(),
        "model_audited_report_draft": (
            model_audited_report.strip() if isinstance(model_audited_report, str) else ""
        ),
        "overall_assessment": overall_assessment.strip(),
        "verifications": sorted(verified_items, key=lambda item: int(item["unit_id"][1:])),
    }


def build_inference_audit(
    verification: dict[str, Any], *, company: str, year: str, model: str | None,
    api_key: str | None, base_url: str | None, provider: str,
) -> dict[str, Any]:
    """Generate and independently review business inferences grounded in verified facts."""
    facts = []
    for item in verification["verifications"]:
        if item["verdict"] in {"supported", "partially_supported"} and item["evidence"]:
            facts.append({"fact_id": item["unit_id"],
                          "statement": item["verified_statement"] or item["original_statement"],
                          "pages": sorted({e["page"] for e in item["evidence"]})})
    if len(facts) < 2:
        return {"inferences": [], "counts": {"supported": 0, "weak": 0, "unsupported": 0}}
    resolved_url = resolve_base_url(provider, base_url)
    common = {"model": resolve_model(provider, model), "temperature": 0}
    generation_prompt = (
        "你是银行新能源汽车产业分析师。只根据verified_facts生成6至10条有业务价值的推论，覆盖增长质量、"
        "盈利改善、现金流质量、费用与研发、品牌战略、偿债或流动性风险。每条必须引用至少两个fact_id，"
        "清楚写出从前提到结论的推理过程，并说明边界条件。不得新增数字、事实、新闻或未来预测；不要重复客观事实。"
        "只输出JSON对象{\"inferences\":[...]}。"
    )
    request = {**common, "messages": [{"role": "system", "content": generation_prompt},
        {"role": "user", "content": json.dumps({"company": company, "year": year, "verified_facts": facts}, ensure_ascii=False)}],
        "response_format": structured_response_format(provider, resolved_url, name="business_inferences", schema=INFERENCE_SCHEMA)}
    if is_deepseek(provider, resolved_url):
        request["max_tokens"] = 6000; request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = _client(provider, api_key, resolved_url).chat.completions.create(**request)
    raw = parse_json_object(response.choices[0].message.content, "Business inferences")
    valid_fact_ids = {fact["fact_id"] for fact in facts}
    generated = []
    raw_inferences = raw.get("inferences") or raw.get("insights") or raw.get("analyses") or []
    for index, item in enumerate(raw_inferences, 1):
        if not isinstance(item, dict):
            continue
        raw_premises = (item.get("premise_ids") or item.get("fact_ids") or
                        item.get("supporting_fact_ids") or item.get("premises") or [])
        if isinstance(raw_premises, str):
            raw_premises = re.findall(r"[cu]\d+", raw_premises)
        normalized_ids = []
        for value in raw_premises:
            if isinstance(value, dict):
                value = value.get("fact_id") or value.get("id") or ""
            normalized_ids.extend(re.findall(r"[cu]\d+", str(value)) or [str(value)])
        premise_ids = [value for value in normalized_ids if value in valid_fact_ids]
        conclusion = str(item.get("conclusion") or item.get("inference") or item.get("statement") or "").strip()
        if conclusion and len(set(premise_ids)) >= 2:
            generated.append({"inference_id": str(item.get("inference_id") or f"i{index}"),
                "conclusion": conclusion, "premise_ids": list(dict.fromkeys(premise_ids)),
                "reasoning": str(item.get("reasoning") or item.get("rationale") or "").strip(),
                "caveat": str(item.get("caveat") or item.get("limitation") or
                              "该结论仅基于本年度报告中的已核验事实，不代表因果关系或未来预测。"
                              ).strip(),
                "dimension": str(item.get("dimension") or item.get("category") or "经营质量").strip()})
    if not generated:
        return {"inferences": [], "counts": {"supported": 0, "weak": 0, "unsupported": 0},
                "generation_diagnostic": raw}
    review_prompt = (
        "你是独立的信贷分析复核员。检查每条推论是否严格由所列前提支持，是否偷换因果、过度外推或忽略反例。"
        "verdict使用supported、weak、unsupported。weak必须给出更审慎的revised_conclusion；unsupported的"
        "revised_conclusion必须为空。不得增加新事实或数字。只输出JSON对象{\"reviews\":[...]}。"
    )
    review_request = {**common, "messages": [{"role": "system", "content": review_prompt},
        {"role": "user", "content": json.dumps({"verified_facts": facts, "candidate_inferences": generated}, ensure_ascii=False)}],
        "response_format": structured_response_format(provider, resolved_url, name="inference_audit", schema=INFERENCE_AUDIT_SCHEMA)}
    if is_deepseek(provider, resolved_url):
        review_request["max_tokens"] = 5000; review_request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = _client(provider, api_key, resolved_url).chat.completions.create(**review_request)
    review_raw = parse_json_object(response.choices[0].message.content, "Inference audit")
    reviews = review_raw.get("reviews") or review_raw.get("audits") or review_raw.get("results") or []
    review_map = {str(x.get("inference_id") or x.get("id")): x for x in reviews if isinstance(x, dict)}
    final = []
    counts = {"supported": 0, "weak": 0, "unsupported": 0}
    fact_map = {fact["fact_id"]: fact for fact in facts}
    for item in generated:
        review = review_map.get(item["inference_id"], {})
        verdict = str(review.get("verdict") or review.get("assessment") or "unsupported").lower()
        verdict = {"partially_supported": "weak", "partially supported": "weak",
                   "not_supported": "unsupported"}.get(verdict, verdict)
        if verdict not in counts:
            verdict = "unsupported"
        conclusion = item["conclusion"] if verdict == "supported" else str(
            review.get("revised_conclusion") or review.get("revised_statement") or "").strip()
        if verdict == "unsupported": conclusion = ""
        pages = sorted({page for pid in item["premise_ids"] for page in fact_map[pid]["pages"]})
        final.append({**item, "verdict": verdict, "verified_conclusion": conclusion,
                      "review_explanation": str(review.get("explanation") or review.get("reason") or
                                                review.get("rationale") or "复核结果已返回，但未附文字说明。"
                                                ).strip(),
                      "source_pages": pages})
        counts[verdict] += 1
    return {"inferences": final, "counts": counts}


def synthesize_grounded_report(
    verification: dict[str, Any], follow_up_questions: list[dict[str, Any]], inference_audit: dict[str, Any], *,
    company: str, year: str, model: str | None, api_key: str | None,
    base_url: str | None, provider: str,
) -> str:
    """Recompose a coherent report using only quote-verified analytical atoms."""
    admissible = []
    for item in verification["verifications"]:
        if item["verdict"] not in {"supported", "partially_supported", "contradicted"} or not item["evidence"]:
            continue
        admissible.append({
            "unit_id": item["unit_id"], "original": item["original_statement"],
            "verdict": item["verdict"], "revision_action": item["revision_action"],
            "verified_statement": item["verified_statement"],
            "explanation": item["explanation"], "evidence": item["evidence"],
        })
    if not admissible:
        return "本次未形成具备逐字引文支持的可信分析结论。"
    resolved_url = resolve_base_url(provider, base_url)
    approved_inferences = [item for item in inference_audit.get("inferences", [])
                           if item["verdict"] in {"supported", "weak"} and item["verified_conclusion"]]
    system = (
        "你是银行产业金融报告编辑。只能使用输入的admissible_claims重写一份连贯中文报告。"
        "supported可保留或澄清；partially_supported必须删除无证据部分并降低推论强度；"
        "contradicted必须依据证据明确纠正。禁止复制未修正的部分支持/矛盾原句，禁止新增数字、日期、事实或外部知识。"
        "允许使用approved_inferences形成分析理解，并明确使用‘这表明/据此推断/但不能据此认定’等审慎措辞。"
        "每个事实或推论后必须标注其证据页码[PDF第X页]。正文应包含经营表现、盈利现金、战略/行业位置和风险预警中"
        "有充分材料的部分；没有证据的章节可省略。待尽调问题不得写入主体结论。只输出JSON对象{\"report\":\"...\"}。"
    )
    payload = {"company": company, "year": year, "admissible_claims": admissible,
               "approved_inferences": approved_inferences}
    request = {
        "model": resolve_model(provider, model), "temperature": 0,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        "response_format": structured_response_format(provider, resolved_url, name="grounded_report", schema=FINAL_REPORT_SCHEMA),
    }
    if is_deepseek(provider, resolved_url):
        request["max_tokens"] = 6000
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    response = _client(provider, api_key, resolved_url).chat.completions.create(**request)
    raw = parse_json_object(response.choices[0].message.content, "Grounded report")
    report = str(raw.get("report", "")).strip()
    if not report:
        return verification["audited_report"]
    allowed_pages = {evidence["page"] for item in admissible for evidence in item["evidence"]}
    cited_pages = {int(value) for value in re.findall(r"\[PDF第(\d+)页\]", report)}
    evidence_text = " ".join(
        [company, year]
        + [item["original"] + " " + item["verified_statement"] + " " +
           " ".join(evidence["quote"] for evidence in item["evidence"]) for item in admissible]
        + [item["verified_conclusion"] for item in approved_inferences]
    )
    report_without_citations = re.sub(r"\[PDF第\d+页\]", "", report)
    number_pattern = r"(?<![A-Za-z])\d+(?:[,.]\d+)*(?:%|％)?"
    allowed_numbers = {value.replace(",", "") for value in re.findall(number_pattern, evidence_text)}
    report_numbers = {value.replace(",", "") for value in re.findall(number_pattern, report_without_citations)}
    if not cited_pages.issubset(allowed_pages) or not report_numbers.issubset(allowed_numbers):
        return verification["audited_report"]
    return report


def verify_analysis_with_evidence(
    open_analysis: str,
    units: list[AnalysisUnit],
    retrieved: dict[str, list[RetrievedEvidence]],
    *,
    company: str,
    year: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str = "deepseek",
) -> dict[str, Any]:
    """Give the first-stage prose plus retrieved pages to a second evidence judge."""
    evidence_bundles = []
    for unit in units:
        pages = [
            {"page": item.page, "score": item.score, "text": item.text[:2200]}
            for item in retrieved.get(unit.unit_id, [])
        ]
        evidence_bundles.append(
            {"unit_id": unit.unit_id, "statement": unit.statement, "retrieved_pages": pages}
        )
    resolved_url = resolve_base_url(provider, base_url)
    deepseek_mode = is_deepseek(provider, resolved_url)
    system = (
        "你是独立的金融证据审计员。输入包含第一阶段的开放式分析，以及程序针对每个分析单元"
        "召回的年报页面。逐项判断为supported、partially_supported、unsupported、contradicted或"
        "non_verifiable。引用必须逐字复制自给定页面并填写PDF页码，不得引用未召回页面。"
        "区分直接事实、合理推断和缺乏证据的因果解释。对partially_supported必须删去或弱化无证据部分；"
        "对contradicted必须按证据纠正；对unsupported和non_verifiable的verified_statement必须为空。"
        "不得把部分支持或矛盾的原句原样复制到verified_statement。audited_report必须是自然、连贯的中文经营分析，"
        "只保留证据支持或明确标注为推断的内容，并在关键句后标注[PDF第X页]；不得新增输入之外的事实。"
    )
    if deepseek_mode:
        system += (
            "只输出合法JSON对象，不要Markdown代码围栏。必须严格使用以下结构："
            '{"audited_report":"自然语言可信报告","overall_assessment":"总体评价",'
            '"verifications":[{"unit_id":"u1","original_statement":"原句",'
            '"verdict":"supported","verified_statement":"可信改写","explanation":"核验说明",'
            '"evidence":[{"page":123,"quote":"从输入页面逐字复制的短原文",'
            '"relationship":"直接支持"}]}]}。禁止使用unit_audits或把evidence写成字符串。'
        )
    payload = {
        "company": company,
        "year": year,
        "first_stage_open_analysis": open_analysis,
        "evidence_bundles": evidence_bundles,
    }
    request: dict[str, Any] = {
        "model": resolve_model(provider, model),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": structured_response_format(
            provider, resolved_url, name="evidence_audit", schema=EVIDENCE_SCHEMA
        ),
    }
    if deepseek_mode:
        request["max_tokens"] = 12000
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    attempts = 2 if deepseek_mode else 1
    raw: dict[str, Any] = {}
    for attempt in range(attempts):
        if attempt:
            request["messages"].append(
                {
                    "role": "user",
                    "content": (
                        "上次JSON字段格式不合格。请严格返回audited_report、overall_assessment和"
                        "verifications；每条evidence必须是包含page、quote、relationship的对象，"
                        "quote必须从对应retrieved_pages.text逐字复制。"
                    ),
                }
            )
        response = _client(provider, api_key, resolved_url).chat.completions.create(**request)
        raw = parse_json_object(response.choices[0].message.content, "Evidence verifier")
        if isinstance(raw.get("verifications"), list):
            break
    validated = validate_verification_output(raw, units, retrieved)
    validated["raw_model_output"] = raw
    return validated


def run_analysis_first_rag(
    pages: list[PageText],
    company: str,
    year: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str = "deepseek",
    max_units: int = 24,
    top_k: int = 4,
) -> dict[str, Any]:
    analysis_package, coverage = generate_open_analysis(
        pages,
        company,
        year,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
    )
    open_analysis = analysis_package["draft_report"]
    analysis_package["claims"] = generate_atomic_claims(
        open_analysis, model=model, api_key=api_key, base_url=base_url, provider=provider
    )
    units = units_from_claims(analysis_package["claims"], max_units=max_units)
    if not units:
        raise ValueError("开放分析无法拆分出可核验语句")
    queries = generate_retrieval_queries(
        units,
        year,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
    )
    retrieved = retrieve_evidence(pages, units, queries, top_k=top_k)
    verification = verify_analysis_with_evidence(
        open_analysis,
        units,
        retrieved,
        company=company,
        year=year,
        model=model,
        api_key=api_key,
        base_url=base_url,
        provider=provider,
    )
    inference_audit = build_inference_audit(
        verification, company=company, year=year, model=model, api_key=api_key,
        base_url=base_url, provider=provider,
    )
    verification["audited_report"] = synthesize_grounded_report(
        verification, analysis_package.get("follow_up_questions", []), inference_audit,
        company=company, year=year,
        model=model, api_key=api_key, base_url=base_url, provider=provider,
    )
    counts = {key: 0 for key in VERDICT_LABELS}
    for item in verification["verifications"]:
        counts[item["verdict"]] += 1
    return {
        "open_analysis": open_analysis,
        "follow_up_questions": analysis_package.get("follow_up_questions", []),
        "all_identified_claims": analysis_package.get("claims", []),
        "coverage": coverage,
        "analysis_units": [asdict(unit) for unit in units],
        "retrieval_queries": queries,
        "retrieved_evidence": {
            unit_id: [asdict(item) for item in items] for unit_id, items in retrieved.items()
        },
        "verification": verification,
        "inference_audit": inference_audit,
        "verdict_counts": counts,
        "audit": {
            "llm_stages": ["全文分析与尽调分离", "原子主张提取", "检索词扩展", "逐条事实复核",
                           "经营推论生成", "推论独立复核", "证据约束下的报告重写"],
            "analysis_unit_count": len(units),
            "identified_claim_count": len(analysis_package.get("claims", [])),
            "mandatory_claim_count": sum(unit.mandatory for unit in units),
            "selection_rule": "数字、趋势、重大金融风险与强推论强制审核；其余按金融重要性排序，不再等间隔抽样",
            "retrieved_page_links": sum(len(items) for items in retrieved.values()),
            "quote_verification": "所有保留引文均由Python在召回页面中逐字反校验",
        },
    }
