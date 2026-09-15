"""Deterministic parsing, unit normalization and financial calculations."""

from __future__ import annotations

import math
import re
from typing import Any


AMOUNT_METRICS = {
    "revenue",
    "rd_expense",
    "cash_balance",
    "operating_cash_flow",
    "net_loss",
    "gross_profit",
    "total_cost_of_sales",
    "vehicle_revenue",
    "vehicle_cost_of_sales",
}
PERCENT_METRICS = {"vehicle_margin", "gross_margin"}
ALL_METRICS = (
    "revenue",
    "vehicle_delivery",
    "vehicle_margin",
    "gross_margin",
    "rd_expense",
    "cash_balance",
    "operating_cash_flow",
    "net_loss",
    "gross_profit",
    "total_cost_of_sales",
    "vehicle_revenue",
    "vehicle_cost_of_sales",
)

PRIMARY_METRICS = (
    "revenue",
    "vehicle_delivery",
    "vehicle_margin",
    "gross_margin",
    "rd_expense",
    "cash_balance",
    "operating_cash_flow",
    "net_loss",
)


def _number(text: str) -> float | None:
    normalized = text.replace(",", "").replace("，", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", normalized)
    if not match:
        return None
    value = float(match.group())
    parentheses = re.search(r"\(\s*[-+]?\d", normalized)
    return -abs(value) if parentheses else value


def parse_value(metric: str, value: str | None, source_text: str | None = None) -> float | None:
    """Normalize money to RMB million, ratios to decimals, deliveries to vehicles."""
    if value is None:
        return None
    number = _number(value)
    if number is None:
        return None
    combined = f"{value} {source_text or ''}".lower()

    if metric in PERCENT_METRICS:
        return number / 100 if "%" in combined or abs(number) > 1 else number
    if metric == "vehicle_delivery":
        if any(token in combined for token in ("million", "百万")):
            return number * 1_000_000
        if any(token in combined for token in ("thousand", "千辆", "千台")):
            return number * 1_000
        if "万" in combined:
            return number * 10_000
        return number
    if metric in AMOUNT_METRICS:
        # Annual reports often use RMB million. Normalize every amount to RMB million.
        multiplier = 1.0  # documented fallback: assume RMB million
        if any(token in combined for token in ("billion", "十亿")):
            multiplier = 1_000
        elif any(token in combined for token in ("million", "百万元")):
            multiplier = 1
        elif any(token in combined for token in ("rmb'000", "rmb 000", "thousand", "千元")):
            multiplier = 1 / 1_000
        elif "亿元" in combined or re.search(r"\d\s*亿", combined):
            multiplier = 100
        elif "万元" in combined or re.search(r"\d\s*万", combined):
            multiplier = 1 / 100
        amount = number * multiplier
        if metric == "operating_cash_flow" and amount > 0 and any(
            token in combined
            for token in ("used in operating", "used for operating", "operating cash outflow", "经营活动所用", "经营活动现金流出")
        ):
            amount = -amount
        if metric == "net_loss":
            if any(token in combined for token in ("net income", "net profit", "净利润")):
                amount = -abs(amount)  # Convention: negative net_loss means profitable.
            elif any(token in combined for token in ("net loss", "净亏损")):
                amount = abs(amount)
        return amount
    return number


def normalize_metrics(extraction: dict[str, Any]) -> dict[str, float | None]:
    normalized: dict[str, float | None] = {}
    for item in extraction.get("metrics", []):
        normalized[item["metric"]] = parse_value(
            item["metric"], item.get("value"), item.get("source_text")
        )
    return {name: normalized.get(name) for name in ALL_METRICS}


def calculate_supported_margins(
    normalized: dict[str, float | None], extraction: dict[str, Any]
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    """Compute margins only from verified, normalized annual-report inputs."""
    enriched = dict(normalized)
    calculations: dict[str, dict[str, Any]] = {}
    evidence = {
        item["metric"]: item
        for item in extraction.get("metrics", [])
        if item.get("verified")
    }

    def pages_for(*metrics: str) -> list[int]:
        return sorted(
            {
                evidence[name]["source_page"]
                for name in metrics
                if name in evidence and evidence[name].get("source_page") is not None
            }
        )

    revenue = enriched.get("revenue")
    gross_profit = enriched.get("gross_profit")
    total_cost = enriched.get("total_cost_of_sales")
    gross_inputs_verified = all(name in evidence for name in ("revenue", "gross_profit"))
    if enriched.get("gross_margin") is None and gross_inputs_verified and revenue and revenue > 0 and gross_profit is not None:
        cross_check_passed: bool | None = None
        cross_check_difference: float | None = None
        if total_cost is not None and "total_cost_of_sales" in evidence:
            expected_gross_profit = revenue - abs(total_cost)
            cross_check_difference = expected_gross_profit - gross_profit
            cross_check_passed = abs(cross_check_difference) / revenue <= 0.005
        if cross_check_passed is not False:
            value = gross_profit / revenue
            enriched["gross_margin"] = value
            calculations["gross_margin"] = {
                "value": value,
                "method": "computed_from_verified_inputs",
                "formula": "gross_profit / revenue",
                "formula_cn": "毛利润 / 营业收入",
                "inputs": {"gross_profit": gross_profit, "revenue": revenue},
                "source_pages": pages_for("gross_profit", "revenue", "total_cost_of_sales"),
                "cross_check": {
                    "formula": "revenue - abs(total_cost_of_sales) = gross_profit",
                    "passed": cross_check_passed,
                    "difference_rmb_million": cross_check_difference,
                },
            }

    vehicle_revenue = enriched.get("vehicle_revenue")
    vehicle_cost = enriched.get("vehicle_cost_of_sales")
    vehicle_inputs_verified = all(
        name in evidence for name in ("vehicle_revenue", "vehicle_cost_of_sales")
    )
    if (
        enriched.get("vehicle_margin") is None
        and vehicle_inputs_verified
        and vehicle_revenue
        and vehicle_revenue > 0
        and vehicle_cost is not None
        and abs(vehicle_cost) <= vehicle_revenue
    ):
        value = (vehicle_revenue - abs(vehicle_cost)) / vehicle_revenue
        enriched["vehicle_margin"] = value
        calculations["vehicle_margin"] = {
            "value": value,
            "method": "computed_from_verified_inputs",
            "formula": "(vehicle_revenue - abs(vehicle_cost_of_sales)) / vehicle_revenue",
            "formula_cn": "（汽车销售收入 - 汽车销售成本）/ 汽车销售收入",
            "inputs": {
                "vehicle_revenue": vehicle_revenue,
                "vehicle_cost_of_sales": vehicle_cost,
            },
            "source_pages": pages_for("vehicle_revenue", "vehicle_cost_of_sales"),
            "cross_check": None,
        }

    return enriched, calculations


def safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or math.isclose(denominator, 0):
        return None
    return numerator / denominator


def calculate_financial_metrics(
    current: dict[str, float | None], previous: dict[str, float | None] | None = None
) -> dict[str, float | None]:
    """All derived values are deterministic and never delegated to the LLM."""
    net_loss = current.get("net_loss")
    annual_loss = net_loss if net_loss is not None and net_loss > 0 else None
    derived = {
        "cash_loss_coverage": safe_divide(current.get("cash_balance"), annual_loss),
        "rd_intensity": safe_divide(current.get("rd_expense"), current.get("revenue")),
        "operating_cash_flow_margin": safe_divide(
            current.get("operating_cash_flow"), current.get("revenue")
        ),
        "gross_margin_change": None,
    }
    if previous:
        current_margin = current.get("gross_margin")
        previous_margin = previous.get("gross_margin")
        if current_margin is not None and previous_margin is not None:
            derived["gross_margin_change"] = current_margin - previous_margin
    return derived
