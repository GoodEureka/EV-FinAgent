from financial_metrics import (
    calculate_financial_metrics,
    calculate_supported_margins,
    parse_value,
)


def test_unit_normalization():
    assert parse_value("revenue", "RMB 13.0 billion") == 13_000
    assert parse_value("revenue", "RMB112,312,511 thousand") == 112_312.511
    assert parse_value("net_loss", "RMB1,139,428 thousand (net income)") == -1_139.428
    assert parse_value("net_loss", "RMB2.5 billion net loss") == 2_500
    assert parse_value("cash_balance", "人民币 120 亿元") == 12_000
    assert abs(parse_value("gross_margin", "18.1%") - 0.181) < 1e-9
    assert parse_value("vehicle_delivery", "20.2 万辆") == 202_000
    assert parse_value(
        "operating_cash_flow",
        "RMB 1.5 billion",
        "Net cash used in operating activities was RMB 1.5 billion.",
    ) == -1_500


def test_deterministic_calculation():
    current = {
        "cash_balance": 20_000,
        "net_loss": 5_000,
        "rd_expense": 3_000,
        "revenue": 30_000,
        "operating_cash_flow": -1_500,
        "gross_margin": 0.12,
    }
    result = calculate_financial_metrics(current, {"gross_margin": 0.08})
    assert result["cash_loss_coverage"] == 4
    assert result["rd_intensity"] == 0.1
    assert result["operating_cash_flow_margin"] == -0.05
    assert abs(result["gross_margin_change"] - 0.04) < 1e-9


def test_margins_are_computed_only_from_verified_inputs():
    normalized = {
        "revenue": 112_312.511,
        "gross_profit": 20_985.058,
        "total_cost_of_sales": -91_327.453,
        "vehicle_revenue": 106_683.100,
        "vehicle_cost_of_sales": -87_591.473,
        "gross_margin": None,
        "vehicle_margin": None,
    }
    extraction = {
        "metrics": [
            {"metric": name, "verified": True, "source_page": 123}
            for name in (
                "revenue",
                "gross_profit",
                "total_cost_of_sales",
                "vehicle_revenue",
                "vehicle_cost_of_sales",
            )
        ]
    }
    enriched, calculations = calculate_supported_margins(normalized, extraction)
    assert abs(enriched["gross_margin"] - 0.18684523935182965) < 1e-12
    assert abs(enriched["vehicle_margin"] - 0.17895643264959493) < 1e-12
    assert calculations["gross_margin"]["cross_check"]["passed"] is True
    assert calculations["gross_margin"]["source_pages"] == [123]


def test_margin_calculation_rejects_unverified_or_inconsistent_inputs():
    normalized = {
        "revenue": 100.0,
        "gross_profit": 90.0,
        "total_cost_of_sales": -80.0,
        "gross_margin": None,
        "vehicle_margin": None,
    }
    extraction = {
        "metrics": [
            {"metric": name, "verified": True, "source_page": 1}
            for name in ("revenue", "gross_profit", "total_cost_of_sales")
        ]
    }
    enriched, calculations = calculate_supported_margins(normalized, extraction)
    assert enriched["gross_margin"] is None
    assert "gross_margin" not in calculations
