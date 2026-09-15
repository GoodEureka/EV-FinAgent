from analysis_first_rag import (
    AnalysisUnit,
    build_full_document_context,
    retrieve_evidence,
    split_analysis_units,
    select_analysis_units,
    units_from_claims,
    validate_verification_output,
)
from pdf_parser import PageText


def test_full_document_context_keeps_page_tags_without_truncation():
    pages = [PageText(1, "alpha"), PageText(2, "beta")]
    context, audit = build_full_document_context(pages)
    assert "<<<PDF_PAGE_1>>>" in context
    assert "<<<PDF_PAGE_2>>>" in context
    assert audit["included_pages"] == 2
    assert audit["truncated"] is False


def test_full_document_context_refuses_silent_truncation():
    try:
        build_full_document_context([PageText(1, "a" * 100)], max_chars=20)
    except ValueError as exc:
        assert "静默截断" in str(exc)
    else:
        raise AssertionError("Oversized context must not be silently truncated")


def test_free_prose_is_split_only_after_generation():
    analysis = "公司收入增长明显。经营现金流已经转正，但仍然处于亏损状态。\n现金流改善的持续性需要进一步验证。"
    units = split_analysis_units(analysis)
    assert len(units) == 3
    assert units[0].unit_id == "u1"


def test_splitter_preserves_year_and_ignores_heading():
    analysis = "## 经营分析\n2025年对公司而言是重要年份。\n**值得关注的风险：**\n1. 经营现金流转负需要关注。"
    units = split_analysis_units(analysis)
    assert units[0].statement.startswith("2025年")
    assert all(not item.statement.endswith("：") for item in units)
    assert any(item.statement.startswith("经营现金流") for item in units)


def test_dynamic_retrieval_returns_relevant_page():
    pages = [
        PageText(1, "General corporate information."),
        PageText(8, "Net cash provided by operating activities was RMB 2,992 million in 2025."),
    ]
    units = split_analysis_units("公司2025年经营现金流已经转正。")
    retrieved = retrieve_evidence(
        pages,
        units,
        {units[0].unit_id: ["net cash provided by operating activities", "2025"]},
        top_k=1,
    )
    assert retrieved[units[0].unit_id][0].page == 8


def test_unverifiable_model_quote_downgrades_supported_verdict():
    pages = [PageText(5, "Revenue increased to RMB 10 billion in 2025.")]
    units = split_analysis_units("公司2025年收入增长到人民币100亿元。")
    retrieved = retrieve_evidence(
        pages,
        units,
        {units[0].unit_id: ["revenue increased", "2025"]},
        top_k=1,
    )
    raw = {
        "audited_report": "公司收入增长。",
        "overall_assessment": "基本可信。",
        "verifications": [
            {
                "unit_id": units[0].unit_id,
                "original_statement": units[0].statement,
                "verdict": "supported",
                "verified_statement": "公司收入增长。",
                "explanation": "有直接证据。",
                "evidence": [
                    {"page": 5, "quote": "This quote does not exist.", "relationship": "支持"}
                ],
            }
        ],
    }
    validated = validate_verification_output(raw, units, retrieved)
    assert validated["verifications"][0]["verdict"] == "unsupported"
    assert validated["verifications"][0]["evidence"] == []


def test_priority_selection_keeps_numeric_claims_beyond_budget():
    units = [AnalysisUnit("u1", "这是一个转折性年份，但只是中性定性表达。", "qualitative_view", "low")]
    units += [AnalysisUnit(f"u{i}", f"2025年交付量为{i}万辆。", "disclosed_fact", "high") for i in range(2, 6)]
    selected = select_analysis_units(units, max_units=2)
    assert len(selected) == 4
    assert all(unit.mandatory for unit in selected)
    assert all("交付量" in unit.statement for unit in selected)


def test_claim_conversion_excludes_qualitative_views_and_keeps_inferences():
    claims = [
        {"text": "2025年是一个转折性年份。", "claim_type": "qualitative_view", "importance": "low"},
        {"text": "全年交付326028辆，同比增长46.9%。", "claim_type": "disclosed_fact", "importance": "high"},
        {"text": "销量结构说明新品牌扩大了价格覆盖。", "claim_type": "analytical_inference", "importance": "medium"},
    ]
    selected = units_from_claims(claims, max_units=1)
    assert len(selected) == 2
    assert all("转折性" not in unit.statement for unit in selected)


def test_partial_support_requires_real_rewrite():
    pages = [PageText(7, "R&D expenses decreased because products were at different development stages.")]
    units = [AnalysisUnit("c1", "研发费用下降意味着公司主动压缩长期投入。", "analytical_inference")]
    retrieved = retrieve_evidence(pages, units, {"c1": ["R&D expenses decreased"]}, top_k=1)
    quote = retrieved["c1"][0].text
    raw = {"audited_report": "", "overall_assessment": "", "verifications": [{
        "unit_id": "c1", "original_statement": units[0].statement, "verdict": "partially_supported",
        "verified_statement": units[0].statement, "explanation": "只能支持费用下降。",
        "evidence": [{"page": 7, "quote": quote, "relationship": "部分支持"}],
    }]}
    validated = validate_verification_output(raw, units, retrieved)
    assert validated["verifications"][0]["verified_statement"] == ""
    assert validated["verifications"][0]["revision_action"] == "split_and_qualify"
