"""Streamlit UI for EV-FinAgent."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd
import streamlit as st

from analysis_first_rag import run_analysis_first_rag
from business_analyzer import analyze_business_quality
from extractor import PRIMARY_METRICS, SUPPORTING_METRICS, extract_metrics, retrieve_chunks
from financial_metrics import (
    calculate_financial_metrics,
    calculate_supported_margins,
    normalize_metrics,
)
from pdf_parser import chunk_pages, parse_pdf
from report_generator import (
    build_comparison,
    build_text_report,
    generate_comparison_commentary,
    generate_report,
)
from risk_analyzer import RISK_RULEBOOK, analyze_risk


SUPPORTED_COMPANIES = ["小鹏汽车", "理想汽车", "蔚来汽车"]
DISPLAY_NAMES = {
    "revenue": "营收",
    "vehicle_delivery": "汽车交付量",
    "vehicle_margin": "汽车毛利率",
    "gross_margin": "综合毛利率",
    "rd_expense": "研发投入",
    "cash_balance": "现金储备",
    "operating_cash_flow": "经营现金流",
    "net_loss": "净亏损/净利润",
    "gross_profit": "毛利润",
    "total_cost_of_sales": "总销售成本",
    "vehicle_revenue": "汽车销售收入",
    "vehicle_cost_of_sales": "汽车销售成本",
}


def format_metric_value(metric: str, value: float | None) -> str:
    if value is None:
        return "未取得"
    if metric in {"vehicle_margin", "gross_margin"}:
        return f"{value:.2%}"
    if metric == "vehicle_delivery":
        return f"{value:,.0f} 辆"
    return f"{value:,.3f} 百万元"

st.set_page_config(page_title="EV-FinAgent", page_icon="🚙", layout="wide")
st.title("EV-FinAgent")
st.caption("基于大语言模型的新能源车企经营风险分析助手 · 经营风险初筛，不替代银行授信审批")

with st.sidebar:
    st.header("分析设置")
    provider_label = st.selectbox("模型服务", ["DeepSeek", "OpenAI / 兼容接口"])
    provider = "deepseek" if provider_label == "DeepSeek" else "openai"
    company = st.selectbox("选择公司", SUPPORTED_COMPANIES)
    year = str(st.number_input("报告年度", min_value=2018, max_value=2030, value=2025, step=1))
    uploaded = st.file_uploader("上传年度报告 PDF", type=["pdf"])
    api_key = st.text_input(
        "API Key",
        value=(
            os.environ.get("DEEPSEEK_API_KEY", "")
            if provider == "deepseek"
            else os.environ.get("OPENAI_API_KEY", "")
        ),
        type="password",
        help="仅在当前进程使用，不会写入磁盘。也可通过环境变量配置。",
        key=f"api_key_{provider}",
    )
    base_url = st.text_input(
        "兼容接口 Base URL",
        value=(
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if provider == "deepseek"
            else os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ),
        key=f"base_url_{provider}",
    )
    model = st.text_input(
        "模型",
        value=(
            os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
            if provider == "deepseek"
            else os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        ),
        key=f"model_{provider}",
    )
    enable_dual_path = st.checkbox(
        "启用开放分析 + RAG证据回查",
        value=True,
        help="先生成分析、原子主张与独立尽调事项，再按优先级检索证据并重写报告。会增加4个LLM阶段。",
    )
    max_analysis_units = st.slider(
        "常规主张审核预算（高优先级主张不受此上限限制）", min_value=8, max_value=30, value=18, step=2,
        disabled=not enable_dual_path,
    )
    if provider == "deepseek":
        st.caption("DeepSeek 使用官方 JSON Output；字段与引用仍由 Python 严格校验。")
    analyze = st.button("开始分析", type="primary", use_container_width=True)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

if analyze:
    if uploaded is None:
        st.error("请先上传 PDF 年报。")
    elif not api_key:
        st.error("请配置 API Key。")
    else:
        try:
            with st.status("正在解析和分析年报……", expanded=True) as status:
                analysis_started = perf_counter()
                stage_seconds: dict[str, float] = {}

                total_steps = 7 if enable_dual_path else 6
                st.write(f"1/{total_steps} 解析 PDF 并保留页码")
                stage_started = perf_counter()
                pages = parse_pdf(uploaded.getvalue())
                stage_seconds["PDF解析"] = perf_counter() - stage_started
                if not any(page.text for page in pages):
                    raise ValueError("PDF 未提取到文本；扫描版年报需要先执行 OCR。")
                st.write(
                    f"已解析 {len(pages)} 页，其中 {sum(bool(page.text) for page in pages)} 页包含文本"
                )

                dual_path = None
                step_offset = 0
                if enable_dual_path:
                    st.write("2/7 全文分析与主张识别 → 动态RAG → 证据复核 → 报告重写")
                    stage_started = perf_counter()
                    dual_path = run_analysis_first_rag(
                        pages,
                        company,
                        year,
                        model=model,
                        api_key=api_key,
                        base_url=base_url or None,
                        provider=provider,
                        max_units=max_analysis_units,
                    )
                    stage_seconds["开放分析与证据回查"] = perf_counter() - stage_started
                    counts = dual_path["verdict_counts"]
                    st.write(
                        f"全文覆盖 {dual_path['coverage']['included_pages']} 页；"
                        f"识别 {dual_path['audit']['identified_claim_count']} 条主张，"
                        f"核验 {dual_path['audit']['analysis_unit_count']} 条高价值主张；"
                        f"支持 {counts['supported']}，部分支持 {counts['partially_supported']}，"
                        f"矛盾 {counts['contradicted']}，证据不足 {counts['unsupported']}"
                    )
                    step_offset = 1

                st.write(f"{2 + step_offset}/{total_steps} 检索指标相关上下文并进行结构化抽取")
                stage_started = perf_counter()
                retrieved = retrieve_chunks(chunk_pages(pages))
                stage_seconds["候选片段检索"] = perf_counter() - stage_started
                retrieved_pages = sorted({chunk.page for chunk in retrieved})
                st.write(f"召回 {len(retrieved)} 个候选片段，来自 {len(retrieved_pages)} 个页面")
                stage_started = perf_counter()
                extraction = extract_metrics(
                    pages,
                    company,
                    year,
                    model=model,
                    api_key=api_key,
                    base_url=base_url or None,
                    provider=provider,
                )
                stage_seconds["LLM结构化抽取"] = perf_counter() - stage_started
                extraction_dict = extraction.to_dict()
                verified_count = sum(item["verified"] for item in extraction_dict["metrics"])
                st.write(f"12 个基础字段中有 {verified_count} 个通过页码与原文反校验")

                st.write(f"{3 + step_offset}/{total_steps} 校验原文引用并统一数值单位")
                stage_started = perf_counter()
                normalized = normalize_metrics(extraction_dict)
                normalized, calculated_metrics = calculate_supported_margins(
                    normalized, extraction_dict
                )
                derived = calculate_financial_metrics(normalized)
                available_primary_count = sum(
                    normalized.get(name) is not None for name in PRIMARY_METRICS
                )
                stage_seconds["引用校验与金融计算"] = perf_counter() - stage_started
                if calculated_metrics:
                    calculated_labels = "、".join(
                        DISPLAY_NAMES[name] for name in calculated_metrics
                    )
                    st.write(f"Python使用已校验输入计算得到：{calculated_labels}")

                st.write(f"{4 + step_offset}/{total_steps} 计算经营质量与样本同业位置")
                stage_started = perf_counter()
                business_analysis = analyze_business_quality(company, year, normalized)
                stage_seconds["经营质量与同业分析"] = perf_counter() - stage_started
                st.write(
                    f"经营质量 {business_analysis['quality_score']}/100（{business_analysis['quality_level']}），"
                    f"{business_analysis['peer_position']['position']}"
                )

                st.write(f"{5 + step_offset}/{total_steps} 执行确定性风险评分与事件预警")
                stage_started = perf_counter()
                risk = analyze_risk(normalized, derived)
                stage_seconds["Python风险评分"] = perf_counter() - stage_started
                st.write(
                    f"风险分数 {risk['risk_score']}/100，等级为{risk['risk_level']}；"
                    f"触发 {len(business_analysis['event_warnings'])} 个情景预警"
                )

                st.write(f"{6 + step_offset}/{total_steps} 基于确定性证据生成综合研判")
                stage_started = perf_counter()
                report = generate_report(
                    company,
                    normalized,
                    derived,
                    risk,
                    calculated_metrics=calculated_metrics,
                    business_analysis=business_analysis,
                    model=model,
                    api_key=api_key,
                    base_url=base_url or None,
                    provider=provider,
                )
                stage_seconds["LLM风险解释"] = perf_counter() - stage_started
                audit = {
                    "generated_at": datetime.now().astimezone().isoformat(),
                    "pdf_name": uploaded.name,
                    "page_count": len(pages),
                    "text_page_count": sum(bool(page.text) for page in pages),
                    "text_char_count": sum(len(page.text) for page in pages),
                    "retrieved_chunk_count": len(retrieved),
                    "retrieved_pages": retrieved_pages,
                    "verified_metric_count": verified_count,
                    "available_primary_count": available_primary_count,
                    "calculated_metric_count": len(calculated_metrics),
                    "provider": provider,
                    "model": model,
                    "base_url": base_url,
                    "stage_seconds": stage_seconds,
                    "total_seconds": perf_counter() - analysis_started,
                }
                record = {
                    "company": company,
                    "year": year,
                    "extraction": extraction_dict,
                    "normalized": normalized,
                    "derived": derived,
                    "risk": risk,
                    "business_analysis": business_analysis,
                    "report": report,
                    "provider": provider,
                    "audit": audit,
                    "calculated_metrics": calculated_metrics,
                    "dual_path": dual_path,
                }
                text_report = build_text_report(record)
                record["text_report"] = text_report
                output_dir = Path(os.environ.get("EV_FINAGENT_OUTPUT_DIR", Path(__file__).resolve().parent.parent / "outputs"))
                output_dir.mkdir(parents=True, exist_ok=True)
                report_path = output_dir / f"{company}_{year}_经营研判与风险预警报告.txt"
                report_path.write_text(text_report, encoding="utf-8-sig")
                record["text_report_path"] = str(report_path)
                st.session_state.portfolio[company] = record
                st.write(f"完整报告已保存：{report_path}")
                status.update(label="分析完成", state="complete", expanded=False)
        except Exception as exc:
            st.exception(exc)

record = st.session_state.portfolio.get(company)
if record:
    tab_open, tab_audit, tab_metrics, tab_judgement, tab_warning, tab_risk, tab_report, tab_process, tab_sources, tab_full, tab_compare = st.tabs(
        ["全文开放分析", "证据审计", "经营指标", "经营研判", "事件预警", "风险评分", "AI 综合报告", "过程监督", "来源引用", "完整报告", "公司比较"]
    )

    with tab_open:
        dual = record.get("dual_path")
        if not dual:
            st.info("本次未启用“开放分析 + RAG证据回查”。")
        else:
            st.subheader("第一阶段：分析草稿")
            st.warning("正文已与待尽调问题分离，但事实和推论尚未经过证据回查；请以“证据审计”为准。")
            c1, c2, c3 = st.columns(3)
            c1.metric("全文覆盖页", dual["coverage"]["included_pages"])
            c2.metric("输入字符", f"{dual['coverage']['input_characters']:,}")
            c3.metric("是否截断", "否" if not dual["coverage"]["truncated"] else "是")
            st.markdown(dual["open_analysis"])
            follow_ups = dual.get("follow_up_questions", [])
            if follow_ups:
                with st.expander(f"补充尽调事项（{len(follow_ups)}项，不属于分析结论）"):
                    for index, item in enumerate(follow_ups, 1):
                        st.write(f"{index}. {item.get('question', '')}")
                        st.caption(f"所需材料：{item.get('required_evidence', '未说明')}｜原因：{item.get('reason', '未说明')}")

    with tab_audit:
        dual = record.get("dual_path")
        if not dual:
            st.info("本次未启用证据回查。")
        else:
            verification = dual["verification"]
            st.subheader("证据约束后的最终分析")
            st.markdown(verification["audited_report"])
            st.info(verification["overall_assessment"])
            inference_audit = dual.get("inference_audit", {})
            if inference_audit.get("inferences"):
                st.subheader("经营推论与推理链")
                for item in inference_audit["inferences"]:
                    with st.expander(f"{item['inference_id']}｜{item['dimension']}｜{item['verdict']}"):
                        st.write("**可信结论：**", item["verified_conclusion"] or "未通过")
                        st.write("**事实前提：**", item["premise_ids"])
                        st.write("**推理链：**", item["reasoning"])
                        st.write("**适用边界：**", item["caveat"])
                        st.write("**独立复核：**", item["review_explanation"])
            counts = dual["verdict_counts"]
            cols = st.columns(5)
            for col, key, label in zip(
                cols,
                ["supported", "partially_supported", "contradicted", "unsupported", "non_verifiable"],
                ["支持", "部分支持", "矛盾", "证据不足", "不可核验"],
            ):
                col.metric(label, counts[key])
            st.subheader("原子主张审计轨迹")
            for item in verification["verifications"]:
                with st.expander(
                    f"{item['unit_id']}｜{item['verdict_label']}｜{item['original_statement'][:45]}",
                    expanded=item["verdict"] in {"contradicted", "unsupported"},
                ):
                    st.write("**原始分析：**", item["original_statement"])
                    st.write("**可信改写：**", item["verified_statement"] or "无")
                    st.write("**修订动作：**", item.get("revision_action", "未记录"))
                    st.write("**核验说明：**", item["explanation"])
                    st.write("**RAG召回页：**", item["retrieved_pages"])
                    for evidence in item["evidence"]:
                        st.caption(f"PDF第 {evidence['page']} 页｜{evidence['relationship']}")
                        st.code(evidence["quote"], language=None)

    with tab_metrics:
        extracted = record["extraction"]["metrics"]
        extraction_map = {item["metric"]: item for item in extracted}
        rows = []
        for metric in PRIMARY_METRICS:
            item = extraction_map[metric]
            calculation = record.get("calculated_metrics", {}).get(metric)
            if item["verified"]:
                method = "年报直接披露"
                source = item["source_page"]
                verified = "是"
            elif calculation:
                method = "Python基于已校验字段计算"
                source = ", ".join(map(str, calculation["source_pages"]))
                verified = "计算输入均已校验"
            else:
                method = "未取得"
                source = None
                verified = "否"
            rows.append(
                {
                    "指标": DISPLAY_NAMES[metric],
                    "取得方式": method,
                    "年报披露值": item["value"] or "未直接披露",
                    "标准值/计算值": format_metric_value(metric, record["normalized"].get(metric)),
                    "来源页": source,
                    "验证状态": verified,
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("金额统一为人民币百万元。系统计算值只在全部输入通过原文校验后产生，并与直接披露值明确区分。")
        with st.expander("查看毛利率计算支撑字段", expanded=False):
            support_rows = [
                {
                    "字段": DISPLAY_NAMES[name],
                    "原始值": extraction_map[name]["value"] or "未取得",
                    "标准值": format_metric_value(name, record["normalized"].get(name)),
                    "来源页": extraction_map[name]["source_page"],
                    "引用已校验": "是" if extraction_map[name]["verified"] else "否",
                }
                for name in SUPPORTING_METRICS
            ]
            st.dataframe(pd.DataFrame(support_rows), hide_index=True, use_container_width=True)
            for metric, calculation in record.get("calculated_metrics", {}).items():
                st.code(
                    f"{DISPLAY_NAMES[metric]} = {calculation['formula_cn']} = {calculation['value']:.2%}",
                    language=None,
                )

    with tab_judgement:
        business = record["business_analysis"]
        c1, c2, c3 = st.columns(3)
        c1.metric("经营质量", f"{business['quality_score']} / 100")
        c2.metric("质量判断", business["quality_level"])
        c3.metric("同业位置", business["peer_position"]["position"])
        st.caption(business["method_note"])
        st.subheader("五维经营质量诊断")
        dimension_rows = [
            {
                "维度": item["label"],
                "得分（越高越好）": item["score"],
                "权重": f"{item['weight']:.0%}",
                "加权贡献": round(item["score"] * item["weight"], 1),
            }
            for item in business["dimensions"].values()
        ]
        st.dataframe(pd.DataFrame(dimension_rows), hide_index=True, use_container_width=True)
        st.subheader("三家公司样本排名")
        peer_rows = [
            {
                "指标": row["label"],
                "本公司排名": f"{row['rank']} / {row['peer_count']}",
                "样本分位": f"{row['percentile']:.0f}",
                "同业中位数": row["peer_median"],
                "领先公司": row["leader"],
            }
            for row in business["peer_position"]["rankings"]
        ]
        st.dataframe(pd.DataFrame(peer_rows), hide_index=True, use_container_width=True)
        st.info(business["peer_position"]["scope_note"])

    with tab_warning:
        warnings = record["business_analysis"]["event_warnings"]
        st.subheader("基于财务信号的情景预警")
        st.caption("预警表示在当前数据组合下需要监控的风险情景，不代表事件一定发生。")
        if not warnings:
            st.success("未触发预设事件规则；仍需持续监测季度数据和外部事件。")
        for warning in warnings:
            title = f"{warning['severity']}预警｜{warning['event']}"
            with st.expander(title, expanded=warning["severity"] == "高"):
                st.write("**触发依据：**", warning["trigger"])
                st.write("**风险传导：**", warning["transmission_path"])
                st.write("**持续监控：**", "、".join(warning["monitoring_indicators"]))
                st.write("**银行动作：**", warning["bank_action"])

    with tab_process:
        audit = record.get("audit", {})
        st.subheader("本次分析流水线")
        process_rows = [
            {"步骤": "PDF按页解析", "执行者": "Python / PyMuPDF", "监督点": "页数、文本页数、字符数"},
            {"步骤": "全文开放分析", "执行者": "LLM", "监督点": "完整页覆盖、无固定字段约束、不引入外部资料"},
            {"步骤": "开放分析分句", "执行者": "Python", "监督点": "只拆分、不改写第一阶段原文"},
            {"步骤": "检索词扩展", "执行者": "LLM", "监督点": "只生成中英文年报检索词、不判断真假"},
            {"步骤": "动态证据召回", "执行者": "Python / BM25", "监督点": "每个分析单元保留召回页和分数"},
            {"步骤": "逐句证据复核", "执行者": "LLM + Python", "监督点": "五类判定、引文逐字反校验、无证据自动降级"},
            {"步骤": "候选片段检索", "执行者": "Python", "监督点": "关键词召回、候选页可查看"},
            {"步骤": "经营指标抽取", "执行者": "LLM", "监督点": "8个核心指标+4个支撑字段、缺失返回null"},
            {"步骤": "引用反校验", "执行者": "Python", "监督点": "页码存在且原文逐字匹配"},
            {"步骤": "金融指标计算", "执行者": "Python", "监督点": "单位、公式、正负语义"},
            {"步骤": "经营与同业研判", "执行者": "Python", "监督点": "五维质量分、三家公司同年排名"},
            {"步骤": "风险评分与事件预警", "执行者": "Python", "监督点": "透明规则、触发依据和传导链"},
            {"步骤": "综合研判", "执行者": "LLM", "监督点": "只能综合已触发证据、禁止外部臆测"},
        ]
        st.dataframe(pd.DataFrame(process_rows), hide_index=True, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PDF页数", audit.get("page_count", "—"))
        c2.metric("候选片段", audit.get("retrieved_chunk_count", "—"))
        c3.metric("核心指标可用", f"{audit.get('available_primary_count', 0)}/8")
        c4.metric("总耗时", f"{audit.get('total_seconds', 0):.2f}秒")
        st.write("候选来源页：", audit.get("retrieved_pages", []))
        if audit.get("stage_seconds"):
            st.dataframe(
                pd.DataFrame(
                    [{"阶段": key, "耗时（秒）": round(value, 3)} for key, value in audit["stage_seconds"].items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.info("启用双通道时包含五个逻辑LLM阶段：全文开放分析、检索词扩展、证据复核、结构化指标抽取和综合研判。Python负责页码、引文、计算、排名与规则校验。")

    with tab_risk:
        left, right, third = st.columns(3)
        left.metric("风险分数", f"{record['risk']['risk_score']} / 100")
        right.metric("风险等级", record["risk"]["risk_level"])
        third.metric("数据完整度", f"{record['risk']['data_completeness']:.0%}")
        st.progress(record["risk"]["risk_score"] / 100)
        st.dataframe(pd.DataFrame(record["risk"]["signals"]), hide_index=True, use_container_width=True)
        st.subheader("本次评分如何得到")
        running_score = 50
        score_rows = [{"项目": "评分起点", "影响": 50, "累计分数": 50, "解释": "统一初始分"}]
        for signal in record["risk"]["signals"]:
            running_score = max(0, min(100, running_score + signal["impact"]))
            score_rows.append(
                {
                    "项目": signal["name"],
                    "影响": f"{signal['impact']:+d}",
                    "累计分数": running_score,
                    "解释": signal["reason"],
                }
            )
        st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
        with st.expander("查看完整评分规则", expanded=False):
            st.write("分数越高表示经营风险越高；最终分数限制在 0–100。")
            st.dataframe(pd.DataFrame(RISK_RULEBOOK), hide_index=True, use_container_width=True)
        st.warning(record["risk"]["disclaimer"])

    with tab_report:
        report = record["report"]
        st.subheader("执行判断")
        st.write(report["financial_health"])
        if report.get("operating_assessment"):
            st.subheader("经营状况")
            st.write(report["operating_assessment"])
        if report.get("industry_position"):
            st.subheader("样本同业位置")
            st.write(report["industry_position"])
        if report.get("key_judgements"):
            st.subheader("关键判断")
            for item in report["key_judgements"]:
                st.write(f"- {item}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("优势")
            for item in report["strength"]:
                st.write(f"- {item}")
        with col2:
            st.subheader("主要风险")
            for item in report["risk"]:
                st.write(f"- {item}")
        st.subheader("贷前核验建议")
        st.write(report["recommendation"])
        if report.get("credit_focus"):
            st.subheader("贷后监控重点")
            for item in report["credit_focus"]:
                st.write(f"- {item}")

    with tab_sources:
        if record.get("calculated_metrics"):
            st.subheader("系统计算指标的可追溯依据")
            for metric, calculation in record["calculated_metrics"].items():
                with st.expander(
                    f"{DISPLAY_NAMES[metric]} · 系统计算 {calculation['value']:.2%}",
                    expanded=True,
                ):
                    st.write("公式：", calculation["formula_cn"])
                    st.write("来源页：", calculation["source_pages"])
                    st.json(calculation["inputs"])
                    if calculation.get("cross_check"):
                        check = calculation["cross_check"]
                        st.write(
                            "交叉校验：",
                            "通过" if check.get("passed") is not False else "未通过",
                            check.get("formula"),
                        )
        st.subheader("全部基础字段原文")
        for item in record["extraction"]["metrics"]:
            with st.expander(
                f"{DISPLAY_NAMES[item['metric']]} · "
                + (f"第 {item['source_page']} 页" if item["source_page"] else "无可靠来源")
            ):
                if item["verified"]:
                    st.code(item["source_text"], language=None)
                else:
                    st.info("未在候选上下文中找到并通过原文反校验，按 null 处理。")

    with tab_full:
        st.subheader("可直接阅读和归档的完整报告")
        st.caption(f"服务器保存位置：{record.get('text_report_path', '未保存')}")
        st.download_button(
            "下载完整 TXT 报告",
            data=record["text_report"].encode("utf-8-sig"),
            file_name=f"{company}_{record['year']}_经营研判与风险预警报告.txt",
            mime="text/plain",
            type="primary",
        )
        st.text_area("报告预览", record["text_report"], height=700)

    with tab_compare:
        records = list(st.session_state.portfolio.values())
        if len(records) < 2:
            st.info("依次上传并分析至少两家公司年报后，可在此进行横向比较。")
        else:
            st.dataframe(build_comparison(records), hide_index=True, use_container_width=True)
            if st.button("生成三家公司经营质量比较"):
                try:
                    st.write(
                        generate_comparison_commentary(
                            records,
                            model=model,
                            api_key=api_key,
                            base_url=base_url or None,
                            provider=provider,
                        )
                    )
                except Exception as exc:
                    st.exception(exc)
else:
    st.info("从左侧上传年报并开始分析。当前 MVP 仅支持小鹏汽车、理想汽车和蔚来汽车。")
