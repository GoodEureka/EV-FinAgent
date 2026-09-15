from business_analyzer import analyze_business_quality, build_event_warnings, load_peer_benchmarks


def _metrics(company: str):
    return next(item for item in load_peer_benchmarks("2025") if item["company"] == company)


def test_peer_position_uses_three_verified_companies():
    analysis = analyze_business_quality("理想汽车", "2025", _metrics("理想汽车"))
    assert analysis["peer_position"]["peer_count"] == 3
    revenue_rank = next(
        row for row in analysis["peer_position"]["rankings"] if row["metric"] == "revenue"
    )
    assert revenue_rank["rank"] == 1
    assert revenue_rank["leader"] == "理想汽车"


def test_profit_cash_divergence_triggers_warning():
    analysis = analyze_business_quality("理想汽车", "2025", _metrics("理想汽车"))
    events = {item["event"] for item in analysis["event_warnings"]}
    assert "利润未转化为经营现金流" in events


def test_loss_and_positive_cash_flow_is_not_treated_as_no_risk():
    analysis = analyze_business_quality("蔚来汽车", "2025", _metrics("蔚来汽车"))
    events = {item["event"] for item in analysis["event_warnings"]}
    assert "亏损延续与外部融资依赖" in events
    assert "正经营现金流可持续性" in events
    assert analysis["quality_score"] < 60
