"""Command-line runner for the Analysis-first RAG claim-audit pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from analysis_first_rag import run_analysis_first_rag
from pdf_parser import parse_pdf


def render(result: dict, company: str, year: str, source: Path) -> str:
    audit = result["audit"]
    verification = result["verification"]
    counts = result["verdict_counts"]
    lines = [
        f"{company} {year} Analysis-first RAG可信分析报告（Claim审计版）",
        "=" * 72,
        f"源文件：{source}",
        f"全文覆盖：{result['coverage']['included_pages']}页，{result['coverage']['input_characters']:,}字符",
        f"识别原子主张：{audit['identified_claim_count']}条",
        f"进入审核：{audit['analysis_unit_count']}条（强制审核{audit['mandatory_claim_count']}条）",
        f"结果：支持{counts['supported']}，部分支持{counts['partially_supported']}，"
        f"矛盾{counts['contradicted']}，证据不足{counts['unsupported']}，不可核验{counts['non_verifiable']}",
        f"选择规则：{audit['selection_rule']}", "", "一、证据约束后的最终分析", "-" * 72,
        verification["audited_report"], "", "二、第一阶段分析草稿（未经逐条核验）", "-" * 72,
        result["open_analysis"], "", "三、补充尽调事项（不属于已成立结论）", "-" * 72,
    ]
    questions = result.get("follow_up_questions", [])
    if questions:
        for index, item in enumerate(questions, 1):
            lines.extend([f"{index}. {item.get('question', '')}",
                          f"   所需材料：{item.get('required_evidence', '未说明')}",
                          f"   原因：{item.get('reason', '未说明')}"])
    else:
        lines.append("无。")
    lines.extend(["", "四、经营推论与推理链", "-" * 72])
    for item in result.get("inference_audit", {}).get("inferences", []):
        lines.extend([f"【{item['inference_id']}｜{item['dimension']}｜{item['verdict']}】",
                      f"结论：{item['verified_conclusion'] or '未通过'}",
                      f"事实前提：{', '.join(item['premise_ids'])}", f"推理：{item['reasoning']}",
                      f"边界：{item['caveat']}", f"复核：{item['review_explanation']}",
                      f"来源页：{', '.join(map(str, item['source_pages']))}", ""])
    lines.extend(["", "五、原子主张审计轨迹", "-" * 72])
    for item in verification["verifications"]:
        lines.extend([f"【{item['unit_id']}｜{item['verdict_label']}｜{item.get('revision_action', '')}】",
                      f"原始主张：{item['original_statement']}",
                      f"可信改写：{item['verified_statement'] or '无'}",
                      f"核验说明：{item['explanation']}",
                      f"召回页：{', '.join(map(str, item['retrieved_pages'])) or '无'}"])
        for evidence in item["evidence"]:
            lines.extend([f"证据：PDF第{evidence['page']}页｜{evidence['relationship']}", evidence["quote"]])
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--company", default="蔚来汽车")
    parser.add_argument("--year", default="2025")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--budget", type=int, default=18)
    args = parser.parse_args()
    result = run_analysis_first_rag(parse_pdf(args.pdf), args.company, args.year,
                                    provider="deepseek", model=args.model, max_units=args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(result, args.company, args.year, args.pdf), encoding="utf-8")
    if args.json_output:
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output), "claims": result["audit"]["identified_claim_count"],
                      "audited": result["audit"]["analysis_unit_count"],
                      "mandatory": result["audit"]["mandatory_claim_count"],
                      "follow_ups": len(result.get("follow_up_questions", [])),
                      "inference_counts": result.get("inference_audit", {}).get("counts", {}),
                      "verdict_counts": result["verdict_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
