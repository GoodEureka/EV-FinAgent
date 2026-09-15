"""Transparent, deterministic MVP risk rules (not a bank credit model)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


RISK_RULEBOOK = [
    {"维度": "净利润率", "条件": "净利润率 ≥ 5%", "分数影响": -15, "解释": "盈利基础较强"},
    {"维度": "净利润率", "条件": "0 ≤ 净利润率 < 5%", "分数影响": -8, "解释": "已盈利但安全垫有限"},
    {"维度": "净利润率", "条件": "-5% < 净利润率 < 0", "分数影响": 6, "解释": "小幅亏损"},
    {"维度": "净利润率", "条件": "-15% < 净利润率 ≤ -5%", "分数影响": 14, "解释": "亏损压力较高"},
    {"维度": "净利润率", "条件": "净利润率 ≤ -15%", "分数影响": 22, "解释": "深度亏损"},
    {"维度": "流动性", "条件": "盈利且现金/营收 ≥ 30%", "分数影响": -5, "解释": "现金缓冲较强"},
    {"维度": "流动性", "条件": "亏损且现金覆盖期 ≥ 5年", "分数影响": -4, "解释": "短期现金缓冲较充足，但不抵消亏损风险"},
    {"维度": "流动性", "条件": "亏损且2年 ≤ 现金覆盖期 < 5年", "分数影响": 0, "解释": "现金缓冲尚可"},
    {"维度": "流动性", "条件": "亏损且1年 ≤ 现金覆盖期 < 2年", "分数影响": 8, "解释": "现金缓冲一般"},
    {"维度": "流动性", "条件": "亏损且现金覆盖期 < 1年", "分数影响": 16, "解释": "流动性缓冲不足"},
    {"维度": "经营现金流", "条件": "经营现金流率 < -10%", "分数影响": 16, "解释": "经营现金流明显承压"},
    {"维度": "经营现金流", "条件": "-10% ≤ 经营现金流率 < 0", "分数影响": 8, "解释": "经营现金流为负"},
    {"维度": "经营现金流", "条件": "0 ≤ 经营现金流率 < 10%", "分数影响": -5, "解释": "经营现金流为正"},
    {"维度": "经营现金流", "条件": "经营现金流率 ≥ 10%", "分数影响": -10, "解释": "经营现金流表现较强"},
    {"维度": "毛利率", "条件": "毛利率 < 0", "分数影响": 18, "解释": "主营业务尚未形成正毛利"},
    {"维度": "毛利率", "条件": "0 ≤ 毛利率 < 8%", "分数影响": 10, "解释": "盈利缓冲较薄"},
    {"维度": "毛利率", "条件": "8% ≤ 毛利率 < 12%", "分数影响": 5, "解释": "毛利安全垫有限"},
    {"维度": "毛利率", "条件": "12% ≤ 毛利率 < 18%", "分数影响": -3, "解释": "毛利水平尚可"},
    {"维度": "毛利率", "条件": "毛利率 ≥ 18%", "分数影响": -8, "解释": "毛利水平较好"},
    {"维度": "盈利改善", "条件": "毛利率同比下降 ≥ 3个百分点", "分数影响": 10, "解释": "盈利能力显著恶化"},
    {"维度": "盈利改善", "条件": "毛利率同比提升 ≥ 3个百分点", "分数影响": -8, "解释": "盈利能力显著改善"},
    {"维度": "研发压力", "条件": "研发费用率 > 25%", "分数影响": 10, "解释": "研发投入对利润和现金形成较高压力"},
    {"维度": "研发压力", "条件": "15% < 研发费用率 ≤ 25%", "分数影响": 5, "解释": "研发投入压力中等"},
    {"维度": "研发压力", "条件": "研发费用率 ≤ 15%", "分数影响": 0, "解释": "研发投入强度相对可控"},
    {"维度": "数据缺失", "条件": "关键指标无法计算", "分数影响": "每项+3或+4", "解释": "不确定性本身计入初筛风险"},
]


@dataclass(frozen=True)
class RiskSignal:
    name: str
    impact: int
    level: str
    reason: str


def analyze_risk(
    base: dict[str, float | None], derived: dict[str, float | None]
) -> dict:
    """Return a 0-100 risk score; a higher score means higher operating risk."""
    score = 50
    signals: list[RiskSignal] = []

    def add(name: str, impact: int, level: str, reason: str) -> None:
        nonlocal score
        score += impact
        signals.append(RiskSignal(name, impact, level, reason))

    revenue = base.get("revenue")
    net_loss = base.get("net_loss")
    net_margin = None if net_loss is None or not revenue else -net_loss / revenue
    if net_margin is None:
        add("净利润率", 5, "数据不足", "无法计算净利润率")
    elif net_margin >= 0.05:
        add("净利润率", -15, "低", f"净利润率为 {net_margin:.1%}")
    elif net_margin >= 0:
        add("净利润率", -8, "低", f"净利润率为 {net_margin:.1%}，已盈利但安全垫有限")
    elif net_margin > -0.05:
        add("净利润率", 6, "中", f"净利润率为 {net_margin:.1%}，仍处于小幅亏损")
    elif net_margin > -0.15:
        add("净利润率", 14, "中", f"净利润率为 {net_margin:.1%}")
    else:
        add("净利润率", 22, "高", f"净利润率为 {net_margin:.1%}，亏损程度较深")

    coverage = derived.get("cash_loss_coverage")
    cash_to_revenue = None if not revenue else (base.get("cash_balance") or 0) / revenue
    if net_loss is not None and net_loss <= 0:
        if base.get("cash_balance") is None or revenue is None:
            add("流动性", 3, "数据不足", "无法计算现金/营收")
        elif cash_to_revenue >= 0.30:
            add("流动性", -5, "低", f"现金占营收 {cash_to_revenue:.1%}")
        elif cash_to_revenue >= 0.15:
            add("流动性", 0, "低", f"现金占营收 {cash_to_revenue:.1%}")
        else:
            add("流动性", 6, "中", f"现金占营收仅 {cash_to_revenue:.1%}")
    elif coverage is None:
        add("流动性", 4, "数据不足", "无法计算现金储备对年度亏损的覆盖倍数")
    elif coverage < 1:
        add("流动性", 16, "高", f"现金仅覆盖约 {coverage:.2f} 年当前亏损")
    elif coverage < 2:
        add("流动性", 8, "中", f"现金覆盖约 {coverage:.2f} 年当前亏损")
    elif coverage < 5:
        add("流动性", 0, "低", f"现金覆盖约 {coverage:.2f} 年当前亏损，但不抵消亏损风险")
    else:
        add("流动性", -4, "低", f"现金覆盖约 {coverage:.2f} 年当前亏损")

    ocf_margin = derived.get("operating_cash_flow_margin")
    if ocf_margin is None:
        add("经营现金流", 4, "数据不足", "无法计算经营现金流率")
    elif ocf_margin < -0.10:
        add("经营现金流", 16, "高", f"经营现金流率为 {ocf_margin:.1%}")
    elif ocf_margin < 0:
        add("经营现金流", 8, "中", f"经营现金流率为 {ocf_margin:.1%}")
    elif ocf_margin < 0.10:
        add("经营现金流", -5, "低", f"经营现金流率为 {ocf_margin:.1%}")
    else:
        add("经营现金流", -10, "低", f"经营现金流率为 {ocf_margin:.1%}")

    margin = base.get("gross_margin")
    if margin is None:
        add("毛利率", 4, "数据不足", "年报中未取得可验证毛利率")
    elif margin < 0:
        add("毛利率", 18, "高", f"毛利率为 {margin:.1%}，主营业务尚未形成正毛利")
    elif margin < 0.08:
        add("毛利率", 10, "中", f"毛利率为 {margin:.1%}，盈利缓冲较薄")
    elif margin < 0.12:
        add("毛利率", 5, "中", f"毛利率为 {margin:.1%}，毛利安全垫有限")
    elif margin < 0.18:
        add("毛利率", -3, "低", f"毛利率为 {margin:.1%}")
    else:
        add("毛利率", -8, "低", f"毛利率为 {margin:.1%}")

    trend = derived.get("gross_margin_change")
    if trend is not None:
        if trend <= -0.03:
            add("盈利改善", 10, "高", f"毛利率同比下降 {abs(trend):.1%}")
        elif trend >= 0.03:
            add("盈利改善", -8, "低", f"毛利率同比提升 {trend:.1%}")

    rd = derived.get("rd_intensity")
    if rd is None:
        add("研发压力", 3, "数据不足", "无法计算研发费用率")
    elif rd > 0.25:
        add("研发压力", 10, "高", f"研发费用占营收 {rd:.1%}")
    elif rd > 0.15:
        add("研发压力", 5, "中", f"研发费用占营收 {rd:.1%}")
    else:
        add("研发压力", 0, "低", f"研发费用占营收 {rd:.1%}")

    score = max(0, min(100, score))
    band = "高风险" if score >= 70 else "中风险" if score >= 40 else "低风险"
    primary_metrics = (
        "revenue",
        "vehicle_delivery",
        "vehicle_margin",
        "gross_margin",
        "rd_expense",
        "cash_balance",
        "operating_cash_flow",
        "net_loss",
    )
    completeness = sum(base.get(name) is not None for name in primary_metrics) / len(primary_metrics)
    return {
        "risk_score": score,
        "risk_level": band,
        "data_completeness": round(completeness, 3),
        "signals": [asdict(signal) for signal in signals],
        "disclaimer": "该评分仅用于经营风险初筛，不构成授信审批结论。",
    }
