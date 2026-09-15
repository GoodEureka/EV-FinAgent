"""Re-run only the inference stages from an existing claim-audit JSON."""
import argparse
import json
from pathlib import Path
from analysis_first_rag import build_inference_audit

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    record = json.loads(a.input.read_text(encoding="utf-8"))
    result = build_inference_audit(record["verification"], company="蔚来汽车", year="2025",
                                   model=None, api_key=None, base_url=None, provider="deepseek")
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.get("counts", {}), ensure_ascii=False))


if __name__ == "__main__":
    main()
