"""LLM explanations grounded only in deterministic results."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd
from openai import OpenAI

from llm_provider import (
    is_deepseek,
    parse_json_object,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    structured_response_format,
)
from risk_analyzer import RISK_RULEBOOK


REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "financial_health": {"type": "string"},
        "strength": {"type": "array", "items": {"type": "string"}},
        "risk": {"type": "array", "items": {"type": "string"}},
        "recommendation": {"type": "string"},
        "operating_assessment": {"type": "string"},
        "industry_position": {"type": "string"},
        "key_judgements": {"type": "array", "items": {"type": "string"}},
        "risk_outlook": {"type": "array", "items": {"type": "string"}},
        "credit_focus": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "company", "financial_health", "strength", "risk", "recommendation",
        "operating_assessment", "industry_position", "key_judgements",
        "risk_outlook", "credit_focus"
    ],
    "additionalProperties": False,
}

METRIC_LABELS = {
    "revenue": "营业收入",
    "vehicle_delivery": "汽车交付量",
    "vehicle_margin": "汽车毛利率",
    "gross_margin": "综合毛利率",
    "rd_expense": "研发投入",
    "cash_balance": "现金储备",
    "operating_cash_flow": "经营现金流",
    "net_loss": "净损益（正数为亏损、负数为盈利）",
    "gross_profit": "毛利润（计算支撑字段）",
    "total_cost_of_sales": "总销售成本（计算支撑字段）",
    "vehicle_revenue": "汽车销售收入（计算支撑字段）",
    "vehicle_cost_of_sales": "汽车销售成本（计算支撑字段）",
}


def _fmt_number(value: float | None, kind: str = "number") -> str:
    if value is None:
        return "无法计算"
    if kind == "percent":
        return f"{value:.1%}"
    if kind == "vehicles":
        return f"{value:,.0f} 辆"
    return f"{value:,.3f} 百万元"


def build_text_report(record: dict[str, Any]) -> str:
    """Build a human-readable, auditable final report without another LLM call."""
    extraction = record["extraction"]
    normalized = record["normalized"]
    derived = record["derived"]
    risk = record["risk"]
    narrative = record["report"]
    audit = record.get("audit", {})
    calculated_metrics = record.get("calculated_metrics", {})
    generated_at = audit.get("generated_at", "未记录")
    lines = [
        "EV-FinAgent 新能源车企经营研判与风险预警报告",
        "=" * 64,
        f"公司：{record['company']}",
        f"报告年度：{record['year']}",
        f"源文件：{audit.get('pdf_name', '未记录')}",
        f"生成时间：{generated_at}",
        f"模型服务：{audit.get('provider', record.get('provider', '未记录'))}",
        f"模型：{audit.get('model', '未记录')}",
        "",
        "一、执行摘要",
        "-" * 64,
        narrative["financial_health"],
        f"风险评分：{risk['risk_score']}/100（{risk['risk_level']}）",
        f"数据完整度：{risk['data_completeness']:.0%}",
        "",
        "二、经原文校验的经营指标",
        "-" * 64,
    ]

    normalized_kinds = {
        "vehicle_delivery": "vehicles",
        "vehicle_margin": "percent",
        "gross_margin": "percent",
    }
    for item in extraction["metrics"]:
        metric = item["metric"]
        label = METRIC_LABELS[metric]
        calculation = calculated_metrics.get(metric)
        verified = "是" if item.get("verified") else "否"
        disclosed_value = item.get("value") or "未取得"
        source_page = item.get("source_page") or "无"
        source_text = item.get("source_text") or "无可验证原文"
        if calculation:
            disclosed_value = "年报未直接披露；由已校验基础字段计算"
            verified = "计算输入均已校验"
            source_page = ", ".join(map(str, calculation["source_pages"]))
            input_text = ", ".join(
                f"{METRIC_LABELS.get(name, name)}={value:,.3f}百万元"
                for name, value in calculation["inputs"].items()
            )
            source_text = f"公式：{calculation['formula_cn']}；输入：{input_text}"
        lines.extend(
            [
                f"【{label}】",
                f"  年报披露值：{disclosed_value}",
                f"  计算层标准值：{_fmt_number(normalized.get(metric), normalized_kinds.get(metric, 'number'))}",
                f"  引用已校验：{verified}",
                f"  来源页码：{source_page}",
                f"  来源/计算说明：{source_text}",
                "",
            ]
        )

    net_result = normalized.get("net_loss")
    coverage_note = (
        "本期盈利，不适用亏损覆盖倍数"
        if net_result is not None and net_result <= 0
        else _fmt_number(derived.get("cash_loss_coverage"), "number").replace(" 百万元", " 倍")
    )
    lines.extend(
        [
            "三、确定性金融指标计算",
            "-" * 64,
            f"1. 现金亏损覆盖 = 现金储备 / 年度亏损：{coverage_note}",
            f"2. 研发费用率 = 研发投入 / 营收：{_fmt_number(derived.get('rd_intensity'), 'percent')}",
            f"3. 经营现金流率 = 经营现金流 / 营收：{_fmt_number(derived.get('operating_cash_flow_margin'), 'percent')}",
            f"4. 毛利率变化 = 本期毛利率 - 上期毛利率：{_fmt_number(derived.get('gross_margin_change'), 'percent')}",
            f"5. 综合毛利率：{_fmt_number(normalized.get('gross_margin'), 'percent')}"
            + ("（由已校验毛利润和营业收入计算）" if "gross_margin" in calculated_metrics else "（年报直接披露或未取得）"),
            f"6. 汽车毛利率：{_fmt_number(normalized.get('vehicle_margin'), 'percent')}"
            + ("（由已校验汽车收入和汽车成本计算）" if "vehicle_margin" in calculated_metrics else "（年报直接披露或未取得）"),
            "以上计算均由 Python 完成，LLM 不参与算术。",
            "",
            "四、风险评分与本次触发规则",
            "-" * 64,
            "评分起点：50分；分数越高表示经营风险越高；最终分数限制在0至100分。",
        ]
    )
    running_score = 50
    for signal in risk["signals"]:
        running_score += signal["impact"]
        lines.append(
            f"- {signal['name']}｜{signal['level']}｜{signal['impact']:+d}分｜"
            f"{signal['reason']}｜累计 {max(0, min(100, running_score))}分"
        )
    lines.extend([f"最终风险分数：{risk['risk_score']}分（{risk['risk_level']}）", ""])

    lines.extend(["五、完整评分规则", "-" * 64])
    for rule in RISK_RULEBOOK:
        impact = rule["分数影响"]
        impact_text = f"{impact:+d}分" if isinstance(impact, int) else str(impact)
        lines.append(
            f"- {rule['维度']}｜条件：{rule['条件']}｜影响：{impact_text}｜{rule['解释']}"
        )

    lines.extend(["", "六、AI综合研判", "-" * 64])
    if narrative.get("operating_assessment"):
        lines.extend(["经营状况判断：", narrative["operating_assessment"], ""])
    if narrative.get("industry_position"):
        lines.extend(["样本同业位置判断：", narrative["industry_position"], ""])
    if narrative.get("key_judgements"):
        lines.append("关键分析判断：")
        lines.extend(f"- {item}" for item in narrative["key_judgements"])
    lines.append("优势：")
    lines.extend([f"- {item}" for item in narrative["strength"]] or ["- 未识别到可靠优势"])
    lines.append("主要风险：")
    lines.extend([f"- {item}" for item in narrative["risk"]] or ["- 未识别到额外风险"])
    if narrative.get("risk_outlook"):
        lines.append("风险事件展望：")
        lines.extend(f"- {item}" for item in narrative["risk_outlook"])
    if narrative.get("credit_focus"):
        lines.append("银行关注重点：")
        lines.extend(f"- {item}" for item in narrative["credit_focus"])
    lines.extend(["贷前核验建议：", narrative["recommendation"], ""])

    business = record.get("business_analysis")
    if business:
        lines.extend(
            [
                "七、经营质量与样本同业位置",
                "-" * 64,
                f"经营质量得分：{business['quality_score']}/100（{business['quality_level']}；分数越高越好）",
                f"样本同业位置：{business['peer_position']['position']}",
                f"综合样本分位：{business['peer_position'].get('composite_percentile', '无法计算')}",
                business["peer_position"]["scope_note"],
                "维度诊断：",
            ]
        )
        for dimension in business["dimensions"].values():
            lines.append(
                f"- {dimension['label']}：{dimension['score']}/100｜权重 {dimension['weight']:.0%}"
            )
        lines.append("样本排名：")
        for ranking in business["peer_position"]["rankings"]:
            lines.append(
                f"- {ranking['label']}：第 {ranking['rank']}/{ranking['peer_count']} 名｜"
                f"样本分位 {ranking['percentile']:.0f}｜领先者：{ranking['leader']}"
            )

        lines.extend(["", "八、风险事件预警与传导路径", "-" * 64])
        warnings = business.get("event_warnings", [])
        if not warnings:
            lines.append("未触发预设事件规则；仍需持续监测季度经营变化。")
        for index, warning in enumerate(warnings, 1):
            lines.extend(
                [
                    f"{index}. {warning['event']}｜预警等级：{warning['severity']}",
                    f"   触发依据：{warning['trigger']}",
                    f"   传导路径：{warning['transmission_path']}",
                    f"   监控指标：{'、'.join(warning['monitoring_indicators'])}",
                    f"   银行动作：{warning['bank_action']}",
                ]
            )
        process_section = "九、处理过程与监督记录"
        limitation_section = "十、数据缺口、使用边界与免责声明"
    else:
        process_section = "七、处理过程与监督记录"
        limitation_section = "八、数据缺口、使用边界与免责声明"

    lines.extend(
        [
            process_section,
            "-" * 64,
            f"PDF页数：{audit.get('page_count', '未记录')}",
            f"有效文本页数：{audit.get('text_page_count', '未记录')}",
            f"提取文本字符数：{audit.get('text_char_count', '未记录')}",
            f"检索候选块数：{audit.get('retrieved_chunk_count', '未记录')}",
            f"候选来源页：{', '.join(map(str, audit.get('retrieved_pages', []))) or '未记录'}",
            f"基础字段引用校验通过：{audit.get('verified_metric_count', sum(bool(x.get('verified')) for x in extraction['metrics']))}/{len(extraction['metrics'])}",
            f"核心指标可用：{audit.get('available_primary_count', '未记录')}/8",
            "结构化抽取调用：仅从候选年报片段抽取8个核心指标及4个计算支撑字段、页码和原文。",
            "综合研判调用：仅综合Python生成的经营质量、同业位置、评分和事件预警证据。",
            "Python监督：字段白名单、引用反校验、单位换算、公式计算、同业排名、风险评分和事件规则。",
        ]
    )
    stage_seconds = audit.get("stage_seconds", {})
    for stage, seconds in stage_seconds.items():
        lines.append(f"- {stage}：{seconds:.3f}秒")

    primary_names = {
        "revenue", "vehicle_delivery", "vehicle_margin", "gross_margin",
        "rd_expense", "cash_balance", "operating_cash_flow", "net_loss",
    }
    missing = [
        METRIC_LABELS[x["metric"]]
        for x in extraction["metrics"]
        if x["metric"] in primary_names
        and not x.get("verified")
        and x["metric"] not in calculated_metrics
    ]
    lines.extend(
        [
            "",
            limitation_section,
            "-" * 64,
            f"未取得可验证数据：{', '.join(missing) if missing else '无'}。",
            "页码为PDF文件页序，可能与年报印刷页码不同。",
            "系统仅分析公开年报，不包含征信、担保、关联交易穿透、实时舆情和银行内部数据。",
            risk["disclaimer"],
            "所有结论均应由客户经理或风险人员复核后使用。",
        ]
    )
    dual = record.get("dual_path")
    if dual:
        verification = dual["verification"]
        counts = dual["verdict_counts"]
        lines.extend(
            [
                "",
                "附录A、Analysis-first RAG开放分析与证据审计",
                "-" * 64,
                "A1. 第一阶段分析草稿",
                "以下正文已排除待尽调问题，但其中事实与推论尚未经过证据回查：",
                dual["open_analysis"],
                "",
                "A2. 证据约束后的可信分析",
                verification["audited_report"],
                "",
                f"总体核验评价：{verification['overall_assessment']}",
                "核验统计："
                f"支持 {counts['supported']}；部分支持 {counts['partially_supported']}；"
                f"矛盾 {counts['contradicted']}；证据不足 {counts['unsupported']}；"
                f"不可由年报核验 {counts['non_verifiable']}。",
                "",
                "A3. 原子主张审计轨迹",
            ]
        )
        for item in verification["verifications"]:
            lines.extend(
                [
                    f"【{item['unit_id']}｜{item['verdict_label']}】",
                    f"  原始分析：{item['original_statement']}",
                    f"  可信改写：{item['verified_statement'] or '无'}",
                    f"  修订动作：{item.get('revision_action', '未记录')}",
                    f"  核验说明：{item['explanation']}",
                    f"  RAG召回页：{', '.join(map(str, item['retrieved_pages'])) or '无'}",
                ]
            )
            for evidence in item["evidence"]:
                lines.extend(
                    [
                        f"  已校验证据｜PDF第{evidence['page']}页｜{evidence['relationship']}",
                        f"  原文：{evidence['quote']}",
                    ]
                )
            lines.append("")
        follow_ups = dual.get("follow_up_questions", [])
        lines.extend(["A4. 补充尽调事项（不属于已成立结论）"])
        if follow_ups:
            for index, item in enumerate(follow_ups, 1):
                lines.extend([
                    f"{index}. {item.get('question', '')}",
                    f"   所需材料：{item.get('required_evidence', '未说明')}",
                    f"   原因：{item.get('reason', '未说明')}",
                ])
        else:
            lines.append("无。")
        lines.extend(["", "A5. 经营推论与推理链"])
        for item in dual.get("inference_audit", {}).get("inferences", []):
            lines.extend([f"【{item['inference_id']}｜{item['dimension']}｜{item['verdict']}】",
                          f"  可信结论：{item['verified_conclusion'] or '未通过'}",
                          f"  事实前提：{', '.join(item['premise_ids'])}",
                          f"  推理链：{item['reasoning']}", f"  适用边界：{item['caveat']}",
                          f"  独立复核：{item['review_explanation']}"])
        lines.extend(
            [
                "证据审计说明：第一阶段同时生成正文、原子主张和独立尽调事项；高价值主张按金融重要性选择；"
                "所有最终保留引文均由Python在RAG召回页面中逐字反校验。",
            ]
        )
    lines.extend(["", "报告结束"])
    return "\n".join(lines)


def _client(
    provider: str, api_key: str | None = None, base_url: str | None = None
) -> OpenAI:
    return OpenAI(
        api_key=resolve_api_key(provider, api_key),
        base_url=resolve_base_url(provider, base_url),
    )


def validate_report(
    raw: dict[str, Any],
    net_loss: float | None = None,
    rd_intensity: float | None = None,
) -> dict[str, Any]:
    required_strings = ("company", "financial_health", "recommendation")
    if any(not isinstance(raw.get(key), str) for key in required_strings):
        raise ValueError("Risk report has missing or invalid string fields")
    for key in ("strength", "risk"):
        if not isinstance(raw.get(key), list) or not all(
            isinstance(item, str) for item in raw[key]
        ):
            raise ValueError(f"Risk report field {key} must be a string array")
    validated = {key: raw[key] for key in (*required_strings, "strength", "risk")}
    advanced_strings = ("operating_assessment", "industry_position")
    advanced_arrays = ("key_judgements", "risk_outlook", "credit_focus")
    if any(key in raw for key in (*advanced_strings, *advanced_arrays)):
        if any(not isinstance(raw.get(key), str) for key in advanced_strings):
            raise ValueError("Strategic report has missing or invalid string fields")
        for key in advanced_arrays:
            if not isinstance(raw.get(key), list) or not all(isinstance(item, str) for item in raw[key]):
                raise ValueError(f"Strategic report field {key} must be a string array")
        validated.update({key: raw[key] for key in (*advanced_strings, *advanced_arrays)})
    combined = json.dumps(validated, ensure_ascii=False)
    if net_loss is not None and net_loss <= 0 and "净亏损" in combined:
        raise ValueError("Risk report contradicts deterministic net-income result")
    if net_loss is not None and net_loss > 0 and "本期实现净利润" in combined:
        raise ValueError("Risk report contradicts deterministic net-loss result")
    high_rd_phrases = ("研发投入占营收比例较高", "研发费用率较高", "研发压力较高")
    if rd_intensity is not None and rd_intensity <= 0.15 and any(
        phrase in combined for phrase in high_rd_phrases
    ):
        raise ValueError("Risk report contradicts deterministic low R&D pressure")
    return validated


def generate_report(
    company: str,
    normalized: dict[str, float | None],
    derived: dict[str, float | None],
    risk: dict[str, Any],
    *,
    calculated_metrics: dict[str, dict[str, Any]] | None = None,
    business_analysis: dict[str, Any] | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str = "openai",
) -> dict[str, Any]:
    llm_metrics = dict(normalized)
    net_result = llm_metrics.pop("net_loss", None)
    if net_result is not None:
        if net_result <= 0:
            llm_metrics["net_income"] = abs(net_result)
        else:
            llm_metrics["net_loss"] = net_result
    facts = {
        "base_metrics_rmb_million": llm_metrics,
        "python_calculated_metrics": calculated_metrics or {},
        "derived_metrics": derived,
        "risk_result": risk,
        "business_analysis": business_analysis or {},
    }
    resolved_base_url = resolve_base_url(provider, base_url)
    deepseek_mode = is_deepseek(provider, resolved_base_url)
    system_prompt = (
        "你是银行新能源车产业金融高级分析师。只能依据用户提供的已校验指标、Python经营质量分析、"
        "三家公司样本排名和事件预警规则进行综合研判，不得补充外部事实或预测具体事件必然发生。"
        "必须解释经营状况形成原因、利润与现金流质量、样本同业优势和短板。risk_outlook只能选择并"
        "综合business_analysis.event_warnings中已触发的情景，写清触发证据和可能传导，不得新增无依据事件。"
        "风险高低必须与risk_result和事件等级一致；建议应是贷前核验及贷后监控动作，不得直接做授信决定。"
    )
    if deepseek_mode:
        system_prompt += (
            "只输出合法 JSON 对象，不要输出 Markdown。格式示例："
            '{"company":"公司","financial_health":"概述","strength":["优势"],'
            '"risk":["风险"],"recommendation":"核验建议","operating_assessment":"经营判断",'
            '"industry_position":"样本同业位置","key_judgements":["判断"],'
            '"risk_outlook":["情景预警"],"credit_focus":["银行关注点"]}'
        )
    request: dict[str, Any] = dict(
        model=resolve_model(provider, model),
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"公司：{company}\n确定性分析结果：{json.dumps(facts, ensure_ascii=False)}"},
        ],
        response_format=structured_response_format(
            provider, resolved_base_url, name="risk_report", schema=REPORT_SCHEMA
        ),
    )
    if deepseek_mode:
        request["max_tokens"] = 4096
        request["extra_body"] = {"thinking": {"type": "disabled"}}
    attempts = 2 if deepseek_mode else 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = _client(provider, api_key, resolved_base_url).chat.completions.create(**request)
            return validate_report(
                parse_json_object(response.choices[0].message.content, "Risk reporter"),
                normalized.get("net_loss"),
                derived.get("rd_intensity"),
            )
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise RuntimeError(f"The model did not return a usable risk report: {last_error}")


def build_comparison(records: list[dict[str, Any]]) -> pd.DataFrame:
    labels = {
        "revenue": "营收（百万元）",
        "gross_margin": "毛利率",
        "rd_expense": "研发投入（百万元）",
        "cash_balance": "现金储备（百万元）",
        "vehicle_delivery": "交付量（辆）",
    }
    data: dict[str, list[Any]] = {"指标": list(labels.values())}
    for record in records:
        metrics = record["normalized"]
        values: list[Any] = []
        for key in labels:
            value = metrics.get(key)
            values.append(f"{value:.1%}" if key == "gross_margin" and value is not None else value)
        data[record["company"]] = values
    return pd.DataFrame(data)


def generate_comparison_commentary(
    records: list[dict[str, Any]],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str = "openai",
) -> str:
    facts = [
        {"company": r["company"], "metrics": r["normalized"], "risk": r["risk"]}
        for r in records
    ]
    response = _client(provider, api_key, base_url).chat.completions.create(
        model=resolve_model(provider, model),
        temperature=0.1,
        messages=[
            {
                "role": "system",
                "content": "仅依据给定结构化指标，用中文简要解释企业经营质量差异；指出数据缺口，不引入外部事实。",
            },
            {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
        ],
    )
    return response.choices[0].message.content or "暂无可用比较结论。"
