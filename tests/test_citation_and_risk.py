from extractor import METRICS, verify_and_normalize
from pdf_parser import PageText
from risk_analyzer import analyze_risk


def test_unverifiable_citation_is_rejected():
    raw = {
        "metrics": [
            {
                "metric": "revenue",
                "value": "RMB 10 billion",
                "source_page": 2,
                "source_text": "Revenue was RMB 10 billion.",
            }
        ]
    }
    result = verify_and_normalize(raw, [PageText(1, "Revenue was RMB 10 billion.")], "小鹏汽车", "2024")
    revenue = result.metrics[0]
    assert revenue.metric == "revenue"
    assert revenue.value is None
    assert len(result.metrics) == len(METRICS)


def test_valid_citation_survives_whitespace_normalization():
    text = "Revenue was RMB 10 billion for the year."
    raw = {
        "metrics": [
            {
                "metric": "revenue",
                "value": "RMB 10 billion",
                "source_page": 1,
                "source_text": "Revenue was RMB 10 billion for the year.",
            }
        ]
    }
    result = verify_and_normalize(raw, [PageText(1, text)], "理想汽车", "2024")
    assert result.metrics[0].verified is True


def test_deepseek_page_string_is_safely_normalized():
    text = "Revenue was RMB 10 billion for the year."
    raw = {
        "metrics": [
            {
                "metric": "revenue",
                "value": "RMB 10 billion",
                "source_page": "PAGE 1",
                "source_text": text,
            }
        ]
    }
    result = verify_and_normalize(raw, [PageText(1, text)], "理想汽车", "2025")
    assert result.metrics[0].verified is True
    assert result.metrics[0].source_page == 1


def test_risk_score_is_bounded_and_explainable():
    base = {"gross_margin": -0.02, "x": 1.0}
    derived = {"cash_loss_coverage": 0.5, "operating_cash_flow_margin": -0.2, "rd_intensity": 0.3}
    result = analyze_risk(base, derived)
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] == "高风险"
    assert result["signals"]


def test_profitable_company_does_not_get_missing_cash_coverage_penalty():
    base = {
        "revenue": 100_000.0,
        "net_loss": -1_000.0,
        "gross_margin": 0.2,
        "cash_balance": 40_000.0,
    }
    derived = {
        "cash_loss_coverage": None,
        "operating_cash_flow_margin": 0.1,
        "rd_intensity": 0.1,
        "gross_margin_change": None,
    }
    result = analyze_risk(base, derived)
    profit_signal = next(item for item in result["signals"] if item["name"] == "净利润率")
    liquidity_signal = next(item for item in result["signals"] if item["name"] == "流动性")
    assert profit_signal["impact"] == -8
    assert liquidity_signal["impact"] == -5


def test_deep_loss_is_not_fully_offset_by_cash_coverage():
    base = {
        "revenue": 100_000.0,
        "net_loss": 20_000.0,
        "gross_margin": 0.14,
        "cash_balance": 60_000.0,
    }
    derived = {
        "cash_loss_coverage": 3.0,
        "operating_cash_flow_margin": 0.03,
        "rd_intensity": 0.12,
        "gross_margin_change": None,
    }
    result = analyze_risk(base, derived)
    profit_signal = next(item for item in result["signals"] if item["name"] == "净利润率")
    liquidity_signal = next(item for item in result["signals"] if item["name"] == "流动性")
    assert profit_signal["impact"] == 22
    assert liquidity_signal["impact"] == 0
    assert result["risk_level"] == "中风险"
