from llm_provider import (
    DEEPSEEK_BASE_URL,
    resolve_base_url,
    resolve_model,
    structured_response_format,
)
from report_generator import validate_report


def test_deepseek_provider_uses_json_object():
    response_format = structured_response_format(
        "deepseek", None, name="example", schema={"type": "object"}
    )
    assert response_format == {"type": "json_object"}
    assert resolve_base_url("deepseek", None) == DEEPSEEK_BASE_URL
    assert resolve_model("deepseek", None).startswith("deepseek-")


def test_openai_provider_keeps_strict_schema():
    response_format = structured_response_format(
        "openai", None, name="example", schema={"type": "object"}
    )
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_report_validation_rejects_wrong_shape():
    valid = {
        "company": "理想汽车",
        "financial_health": "稳健",
        "strength": ["现金流为正"],
        "risk": ["竞争激烈"],
        "recommendation": "核验订单",
    }
    assert validate_report(valid) == valid

    invalid = dict(valid, strength="not-an-array")
    try:
        validate_report(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid report shape must be rejected")


def test_report_validation_rejects_profit_loss_contradiction():
    contradictory = {
        "company": "理想汽车",
        "financial_health": "公司本期出现净亏损。",
        "strength": [],
        "risk": ["净亏损扩大"],
        "recommendation": "核验财务数据",
    }
    try:
        validate_report(contradictory, net_loss=-1_000.0)
    except ValueError:
        pass
    else:
        raise AssertionError("profit/loss contradiction must be rejected")


def test_report_validation_rejects_rd_level_contradiction():
    contradictory = {
        "company": "理想汽车",
        "financial_health": "研发投入占营收比例较高。",
        "strength": [],
        "risk": ["研发压力较高"],
        "recommendation": "核验研发项目",
    }
    try:
        validate_report(contradictory, rd_intensity=0.10)
    except ValueError:
        pass
    else:
        raise AssertionError("R&D risk-level contradiction must be rejected")
