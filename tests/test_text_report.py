from report_generator import build_text_report


def test_text_report_contains_audit_rules_and_sources():
    record = {
        "company": "理想汽车",
        "year": "2025",
        "provider": "deepseek",
        "extraction": {
            "metrics": [
                {
                    "metric": "revenue",
                    "value": "RMB 100 billion",
                    "source_page": 12,
                    "source_text": "Total revenue was RMB 100 billion.",
                    "verified": True,
                }
            ]
        },
        "normalized": {
            "revenue": 100_000.0,
            "net_loss": -1_000.0,
        },
        "derived": {
            "cash_loss_coverage": None,
            "rd_intensity": 0.10,
            "operating_cash_flow_margin": -0.05,
            "gross_margin_change": None,
        },
        "risk": {
            "risk_score": 48,
            "risk_level": "中风险",
            "data_completeness": 0.75,
            "signals": [
                {"name": "经营现金流", "impact": 8, "level": "中", "reason": "经营现金流率为 -5.0%"}
            ],
            "disclaimer": "仅用于经营风险初筛。",
        },
        "report": {
            "financial_health": "整体风险中等。",
            "strength": ["本期盈利"],
            "risk": ["现金流为负"],
            "recommendation": "核验现金流。",
        },
        "audit": {
            "pdf_name": "annual-report.pdf",
            "page_count": 200,
            "text_page_count": 200,
            "text_char_count": 500_000,
            "retrieved_chunk_count": 20,
            "retrieved_pages": [12, 18],
            "verified_metric_count": 6,
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "stage_seconds": {"PDF解析": 1.0},
        },
    }
    text = build_text_report(record)
    assert "完整评分规则" in text
    assert "结构化抽取调用" in text
    assert "Total revenue was RMB 100 billion." in text
    assert "经营现金流率 = 经营现金流 / 营收" in text
    assert "仅用于经营风险初筛" in text
