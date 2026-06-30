"""Export vul4j repair outcomes to JSON and Markdown matrix."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

ALL_VUL4J_IDS = [f"VUL4J-{index}" for index in range(1, 80)]


@dataclass(frozen=True)
class ExperimentPaths:
    baseline: Path
    evo_repair: Path


def build_parser(workspace: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export baseline/evoRepair success-failure lists and a 1/0 Markdown matrix for vul4j.",
    )
    parser.add_argument(
        "--qwen3-max-baseline",
        type=Path,
        default=workspace / "minisweagent" / "output" / "vul4j" / "qwen3-max",
        help="Directory of qwen3-max baseline experiment.",
    )
    parser.add_argument(
        "--qwen3-max-evorepair",
        type=Path,
        default=workspace / "minisweagent" / "output" / "vul4j" / "qwen3-max-230-k-10-n-5-v1",
        help="Directory of qwen3-max evoRepair experiment.",
    )
    parser.add_argument(
        "--qwen35-plus-baseline",
        type=Path,
        default=workspace / "minisweagent" / "output" / "vul4j" / "qwen3.5-plus-baseline-v1",
        help="Directory of qwen3.5-plus baseline experiment.",
    )
    parser.add_argument(
        "--qwen35-plus-evorepair",
        type=Path,
        default=workspace / "minisweagent" / "output" / "vul4j" / "qwen3.5-plus-230-k-10-n-5-v2",
        help="Directory of qwen3.5-plus evoRepair experiment.",
    )
    parser.add_argument(
        "--qwen3-coder-plus-baseline",
        type=Path,
        default=workspace / "minisweagent" / "output" / "vul4j" / "qwen3-coder-plus",
        help="Directory of qwen3-coder-plus baseline experiment.",
    )
    parser.add_argument(
        "--qwen3-coder-plus-evorepair",
        type=Path,
        default=workspace / "minisweagent" / "output" / "vul4j" / "qwen3-coder-plus-230-k-10-n-1-v2",
        help="Directory of qwen3-coder-plus evoRepair experiment.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=workspace / "analysis_output" / "vul4j_success_failure_by_model.json",
        help="Output path of JSON summary.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=workspace / "analysis_output" / "vul4j_repair_matrix.md",
        help="Output path of Markdown matrix table.",
    )
    return parser


def load_success_map(experiment_dir: Path) -> dict[str, bool]:
    results_path = experiment_dir / "evaluation" / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"results.json not found: {results_path}")

    raw_data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(raw_data, dict):
        raise ValueError(f"invalid results format in {results_path}: expected JSON object")

    success_map: dict[str, bool] = {}
    for cve_id, payload in raw_data.items():
        cve_key = str(cve_id)
        if isinstance(payload, dict):
            success_map[cve_key] = payload.get("status") == "success"
        elif isinstance(payload, bool):
            success_map[cve_key] = payload
        else:
            success_map[cve_key] = False

    missing_ids = [cve_id for cve_id in ALL_VUL4J_IDS if cve_id not in success_map]
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"incomplete results in {results_path}: missing {len(missing_ids)} ids, for example: {preview}"
        )

    return success_map


def build_cve_lists(success_map: dict[str, bool]) -> dict[str, list[str]]:
    success_cves = [cve_id for cve_id in ALL_VUL4J_IDS if success_map.get(cve_id, False)]
    failed_cves = [cve_id for cve_id in ALL_VUL4J_IDS if not success_map.get(cve_id, False)]
    return {"success_cves": success_cves, "failed_cves": failed_cves}


def build_markdown_table(model_success_maps: dict[str, dict[str, dict[str, bool]]]) -> str:
    column_order = [
        ("qwen3-max", "baseline"),
        ("qwen3-max", "evoRepair"),
        ("qwen3.5-plus", "baseline"),
        ("qwen3.5-plus", "evoRepair"),
        ("qwen3-coder-plus", "baseline"),
        ("qwen3-coder-plus", "evoRepair"),
    ]

    headers = ["cve"] + [f"{model} {variant}" for model, variant in column_order]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for cve_id in ALL_VUL4J_IDS:
        row_values = [cve_id]
        for model, variant in column_order:
            is_success = model_success_maps[model][variant].get(cve_id, False)
            row_values.append("1" if is_success else "0")
        lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    workspace = Path(__file__).resolve().parents[1]
    parser = build_parser(workspace)
    args = parser.parse_args()

    model_experiments = {
        "qwen3-max": ExperimentPaths(
            baseline=args.qwen3_max_baseline,
            evo_repair=args.qwen3_max_evorepair,
        ),
        "qwen3.5-plus": ExperimentPaths(
            baseline=args.qwen35_plus_baseline,
            evo_repair=args.qwen35_plus_evorepair,
        ),
        "qwen3-coder-plus": ExperimentPaths(
            baseline=args.qwen3_coder_plus_baseline,
            evo_repair=args.qwen3_coder_plus_evorepair,
        ),
    }

    model_success_maps: dict[str, dict[str, dict[str, bool]]] = {}
    json_output: dict[str, dict[str, dict[str, list[str]]]] = {}

    for model_name, experiment_paths in model_experiments.items():
        baseline_success_map = load_success_map(experiment_paths.baseline)
        evorepair_success_map = load_success_map(experiment_paths.evo_repair)
        model_success_maps[model_name] = {
            "baseline": baseline_success_map,
            "evoRepair": evorepair_success_map,
        }
        json_output[model_name] = {
            "baseline": build_cve_lists(baseline_success_map),
            "evoRepair": build_cve_lists(evorepair_success_map),
        }

    json_text = json.dumps(json_output, indent=2, ensure_ascii=False) + "\n"
    markdown_text = build_markdown_table(model_success_maps)

    write_text(args.json_output, json_text)
    write_text(args.md_output, markdown_text)

    print(f"JSON written to: {args.json_output}")
    print(f"Markdown written to: {args.md_output}")


if __name__ == "__main__":
    main()
