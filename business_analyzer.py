"""Deterministic business-quality, peer-position and event-warning analysis."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from financial_metrics import safe_divide


PEER_METRICS = {
    "revenue": "营收规模",
    "vehicle_delivery": "交付规模",
    "gross_margin": "综合毛利率",
    "vehicle_margin": "汽车毛利率",
    "net_margin": "净利润率",
    "operating_cash_flow_margin": "经营现金流率",
    "cash_balance": "现金储备",
}


def load_peer_benchmarks(year: str, path: str | Path | None = None) -> list[dict[str, Any]]:
    benchmark_path = Path(path) if path else Path(__file__).parent / "data" / f"peer_benchmarks_{year}.json"
    if not benchmark_path.exists():
        return []
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if str(payload.get("year")) != str(year):
        return []
    return payload.get("companies", [])


def _analytical_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    revenue = metrics.get("revenue")
    net_loss = metrics.get("net_loss")
    return {
        **metrics,
        "net_margin": safe_divide(-net_loss if net_loss is not None else None, revenue),
        "operating_cash_flow_margin": safe_divide(metrics.get("operating_cash_flow"), revenue),
        "rd_intensity": safe_divide(metrics.get("rd_expense"), revenue),
        "cash_to_revenue": safe_divide(metrics.get("cash_balance"), revenue),
        "cash_loss_coverage": safe_divide(
            metrics.get("cash_balance"), net_loss if net_loss is not None and net_loss > 0 else None
        ),
    }


def build_peer_position(
    company: str,
    metrics: dict[str, float | None],
    peers: list[dict[str, Any]],
) -> dict[str, Any]:
    current = _analytical_metrics(metrics)
    peer_rows = []
    seen = set()
    for peer in peers:
        name = peer.get("company")
        if not name or name in seen:
            continue
        seen.add(name)
        peer_rows.append({"company": name, **_analytical_metrics(peer)})
    if company not in seen:
        peer_rows.append({"company": company, **current})

    rankings: list[dict[str, Any]] = []
    percentiles: list[float] = []
    for key, label in PEER_METRICS.items():
        values = [(row["company"], row.get(key)) for row in peer_rows if row.get(key) is not None]
        current_value = current.get(key)
        if current_value is None or len(values) < 2:
            continue
        ordered = sorted(values, key=lambda item: item[1], reverse=True)
        rank = next(index for index, item in enumerate(ordered, 1) if item[0] == company)
        percentile = 100.0 if len(ordered) == 1 else 100 * (len(ordered) - rank) / (len(ordered) - 1)
        percentiles.append(percentile)
        rankings.append(
            {
                "metric": key,
                "label": label,
                "value": current_value,
                "rank": rank,
                "peer_count": len(ordered),
                "percentile": round(percentile, 1),
                "peer_median": statistics.median(value for _, value in values),
                "leader": ordered[0][0],
            }
        )

    composite = sum(percentiles) / len(percentiles) if percentiles else None
    if composite is None:
        position = "缺少同年同业样本"
    elif composite >= 67:
        position = "样本同业领先"
    elif composite >= 45:
        position = "样本同业中游偏强"
    elif composite >= 30:
        position = "样本同业中游偏弱"
    else:
        position = "样本同业相对落后"
    return {
        "position": position,
        "composite_percentile": round(composite, 1) if composite is not None else None,
        "peer_count": len(peer_rows),
        "rankings": rankings,
        "scope_note": "仅比较小鹏、理想、蔚来三家同年度公开年报，不代表整个新能源汽车行业排名。",
    }


def _score_dimensions(
    metrics: dict[str, float | None], peer_position: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    values = _analytical_metrics(metrics)
    net_margin = values["net_margin"]
    gross_margin = values.get("gross_margin")
    ocf_margin = values["operating_cash_flow_margin"]
    coverage = values["cash_loss_coverage"]
    cash_to_revenue = values["cash_to_revenue"]
    rd = values["rd_intensity"]

    if net_margin is None:
        net_score = 40
    elif net_margin >= 0.05:
        net_score = 90
    elif net_margin >= 0:
        net_score = 75
    elif net_margin > -0.05:
        net_score = 55
    elif net_margin > -0.15:
        net_score = 35
    else:
        net_score = 15
    if gross_margin is None:
        margin_score = 40
    elif gross_margin >= 0.18:
        margin_score = 85
    elif gross_margin >= 0.12:
        margin_score = 65
    elif gross_margin >= 0.08:
        margin_score = 45
    else:
        margin_score = 20
    profitability = round(net_score * 0.65 + margin_score * 0.35)

    if ocf_margin is None:
        cash_score = 40
    elif ocf_margin >= 0.10:
        cash_score = 90
    elif ocf_margin >= 0:
        cash_score = 70
    elif ocf_margin >= -0.10:
        cash_score = 40
    else:
        cash_score = 15
    if net_margin is not None and net_margin >= 0 and ocf_margin is not None and ocf_margin < 0:
        cash_score = min(cash_score, 30)
    if net_margin is not None and net_margin < 0 and ocf_margin is not None and ocf_margin > 0:
        cash_score = min(cash_score, 65)

    if net_margin is not None and net_margin >= 0:
        if cash_to_revenue is None:
            liquidity = 45
        elif cash_to_revenue >= 0.30:
            liquidity = 90
        elif cash_to_revenue >= 0.15:
            liquidity = 70
        else:
            liquidity = 45
    elif coverage is None:
        liquidity = 35
    elif coverage >= 5:
        liquidity = 80
    elif coverage >= 2:
        liquidity = 65
    elif coverage >= 1:
        liquidity = 45
    else:
        liquidity = 20

    if rd is None:
        innovation = 40
    elif 0.08 <= rd <= 0.15:
        innovation = 75
    elif 0.05 <= rd <= 0.20:
        innovation = 65
    elif rd > 0.25:
        innovation = 35
    else:
        innovation = 50
    market = peer_position.get("composite_percentile")
    market_score = round(market) if market is not None else 50

    return {
        "profitability": {"label": "盈利质量", "score": profitability, "weight": 0.30},
        "cash_flow": {"label": "现金流质量", "score": cash_score, "weight": 0.25},
        "liquidity": {"label": "流动性缓冲", "score": liquidity, "weight": 0.20},
        "market_position": {"label": "样本同业位置", "score": market_score, "weight": 0.15},
        "innovation": {"label": "研发投入可持续性", "score": innovation, "weight": 0.10},
    }


def build_event_warnings(
    metrics: dict[str, float | None],
    peer_position: dict[str, Any],
) -> list[dict[str, Any]]:
    values = _analytical_metrics(metrics)
    net_margin = values["net_margin"]
    ocf_margin = values["operating_cash_flow_margin"]
    gross_margin = values.get("gross_margin")
    vehicle_margin = values.get("vehicle_margin")
    coverage = values["cash_loss_coverage"]
    rd = values["rd_intensity"]
    warnings: list[dict[str, Any]] = []

    def add(event: str, severity: str, trigger: str, path: str, monitors: list[str], action: str) -> None:
        warnings.append(
            {
                "event": event,
                "severity": severity,
                "trigger": trigger,
                "transmission_path": path,
                "monitoring_indicators": monitors,
                "bank_action": action,
            }
        )

    if net_margin is not None and net_margin >= 0 and ocf_margin is not None and ocf_margin < 0:
        add(
            "利润未转化为经营现金流",
            "高",
            f"净利润率为 {net_margin:.1%}，但经营现金流率为 {ocf_margin:.1%}",
            "回款放缓或营运资金占用上升 → 经营现金流持续流出 → 流动性缓冲下降 → 短期融资需求增加",
            ["应收账款周转天数", "存货周转天数", "合同负债", "应付账款变化", "季度经营现金流"],
            "核验利润与经营现金流差异的构成，并对营运资金占用开展压力测试。",
        )
    if net_margin is not None and net_margin < 0:
        severity = "高" if net_margin <= -0.10 else "中"
        add(
            "亏损延续与外部融资依赖",
            severity,
            f"净利润率为 {net_margin:.1%}",
            "持续亏损 → 净资产与现金缓冲消耗 → 再融资依赖提高 → 授信续作和偿债能力承压",
            ["季度净亏损率", "自由现金流", "现金及受限现金", "有息负债到期分布", "融资公告"],
            "核验亏损收窄路径、未来十二个月资金缺口和备用融资安排。",
        )
    if net_margin is not None and net_margin < 0 and ocf_margin is not None and ocf_margin > 0:
        add(
            "正经营现金流可持续性",
            "中",
            f"企业仍亏损，但经营现金流率为正（{ocf_margin:.1%}）",
            "营运资金释放或非现金费用支撑当期现金流 → 若应付增加或预收放缓发生逆转 → 经营现金流可能再次转负",
            ["经营现金流与EBITDA差异", "应付账款增速", "预收及合同负债", "供应商账期", "连续季度经营现金流"],
            "穿透经营现金流来源，区分主营造血、营运资金释放和一次性因素。",
        )
    peer_margin = next(
        (row for row in peer_position.get("rankings", []) if row["metric"] == "gross_margin"), None
    )
    if gross_margin is not None and (
        gross_margin < 0.15 or (peer_margin and gross_margin + 0.02 < peer_margin["peer_median"])
    ):
        add(
            "价格竞争导致盈利缓冲收窄",
            "中",
            f"综合毛利率为 {gross_margin:.1%}，同业样本中位数为 {peer_margin['peer_median']:.1%}" if peer_margin else f"综合毛利率为 {gross_margin:.1%}",
            "终端降价或产品结构下沉 → 单车毛利下降 → 亏损扩大或现金回收能力减弱",
            ["季度综合毛利率", "汽车毛利率", "单车收入", "促销折扣", "原材料成本"],
            "设置季度毛利率预警线，并核验降价情景下的盈亏平衡销量。",
        )
    if vehicle_margin is not None and vehicle_margin < 0.15:
        add(
            "汽车业务毛利安全垫偏薄",
            "中",
            f"汽车毛利率为 {vehicle_margin:.1%}",
            "单车盈利缓冲有限 → 销量不及预期或成本反弹 → 毛利润快速下滑 → 现金流与偿债缓冲承压",
            ["分车型毛利率", "单车物料成本", "产能利用率", "质保费用率", "产品结构"],
            "按车型和价格带核验毛利，进行销量下降与成本上升的联合压力测试。",
        )
    if coverage is not None and coverage < 2:
        add(
            "现金缓冲不足",
            "高" if coverage < 1 else "中",
            f"现金仅覆盖约 {coverage:.2f} 年当前亏损",
            "亏损延续 → 现金储备快速下降 → 外部融资窗口收紧时出现流动性缺口",
            ["现金亏损覆盖期", "受限现金比例", "短债余额", "未来资本开支", "未使用授信额度"],
            "建立十二个月滚动现金流预测并核验可动用现金及备用授信。",
        )
    if rd is not None and rd > 0.10 and net_margin is not None and net_margin < 0:
        add(
            "研发投入转化不及预期",
            "中",
            f"研发费用率为 {rd:.1%}，且企业仍处于亏损状态",
            "研发持续高投入 → 新车型或技术商业化不及预期 → 费用刚性延长亏损周期 → 融资需求增加",
            ["研发费用率", "新车型订单与交付", "研发资本化比例", "车型迭代周期", "研发人员变化"],
            "核验主要研发项目里程碑、量产计划和投入回收假设。",
        )
    bottom_count = sum(row["rank"] == row["peer_count"] for row in peer_position.get("rankings", []))
    if bottom_count >= 3:
        add(
            "样本同业竞争位置承压",
            "中",
            f"在 {bottom_count} 个可比维度位于三家样本末位",
            "规模、盈利或现金指标相对落后 → 价格与研发竞争承压 → 融资议价能力下降",
            ["样本同业排名", "市场份额", "订单增速", "毛利率差距", "现金储备差距"],
            "将同业差距纳入季度跟踪，并核验管理层缩小差距的可执行措施。",
        )
    return warnings


def analyze_business_quality(
    company: str,
    year: str,
    normalized: dict[str, float | None],
    peers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    peers = peers if peers is not None else load_peer_benchmarks(year)
    peer_position = build_peer_position(company, normalized, peers)
    dimensions = _score_dimensions(normalized, peer_position)
    quality_score = round(sum(item["score"] * item["weight"] for item in dimensions.values()))
    quality_level = "经营质量较强" if quality_score >= 75 else "经营质量中等" if quality_score >= 55 else "经营质量偏弱"
    values = _analytical_metrics(normalized)
    positive_signals = []
    if values["net_margin"] is not None and values["net_margin"] >= 0:
        positive_signals.append(f"实现净利润，净利润率为 {values['net_margin']:.1%}")
    if values["operating_cash_flow_margin"] is not None and values["operating_cash_flow_margin"] > 0:
        positive_signals.append(f"经营现金流为正，经营现金流率为 {values['operating_cash_flow_margin']:.1%}")
    if normalized.get("gross_margin") is not None and normalized["gross_margin"] >= 0.15:
        positive_signals.append(f"综合毛利率为 {normalized['gross_margin']:.1%}")
    return {
        "quality_score": quality_score,
        "quality_level": quality_level,
        "analytical_metrics": values,
        "dimensions": dimensions,
        "peer_position": peer_position,
        "positive_signals": positive_signals,
        "event_warnings": build_event_warnings(normalized, peer_position),
        "method_note": "经营质量分越高越好；风险分越高风险越高。两者均为可解释的演示规则，不是银行生产评级模型。",
    }
