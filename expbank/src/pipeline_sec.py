#!/usr/bin/env python3
"""
ExpBank SEC-bench pipeline.

This module reuses the base ExpBank Pipeline and adapts SEC-bench specifics:
1) Success-set loading from SEC evaluator `report_*.jsonl`,
2) Vulnerability metadata loading from SEC JSONL dataset,
3) Trajectory filename parsing for non-CVE instance ids (e.g. project.cve-xxxx-xxxx),
4) Flat output layout for direct `-o` output roots.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from src.pipeline import Pipeline, logger

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPBANK_ROOT = Path(__file__).resolve().parents[1]
_CVE_RE = re.compile(r"cve-\d{4}-\d+", re.IGNORECASE)


class SECBPipeline(Pipeline):
    """Pipeline adapter for SEC-bench data/evaluation formats."""

    def __init__(
        self,
        config_path: str = "src/config/run/pipeline_config_sec.yaml",
        input_dir: str | None = None,
        output_dir: str | None = None,
        summary_path: str | None = None,
        vul_info_path: str | None = None,
        db_dir: str | None = None,
        report_mode: str | None = None,
    ):
        super().__init__(
            config_path=config_path,
            input_dir=input_dir,
            output_dir=output_dir,
            summary_path=summary_path,
            vul_info_path=vul_info_path,
            db_dir=db_dir,
        )
        if report_mode:
            self.config.setdefault("input", {})["evaluation_report_mode"] = report_mode
        self._instance_to_cve: dict[str, str] = {}

        # SEC raw_traj subdirs are typically instance IDs, not "CVE-*".
        self.config.setdefault("input", {}).setdefault("prefix", "")

    def _setup_turn_directory(self, base_dir: Path) -> Path:
        """
        Use a flat output layout for SEC-bench.

        SEC single-turn runners already pass an explicit output root, so this
        adapter avoids creating nested turn_{n} directories.
        """
        run_dir = Path(base_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "extracted").mkdir(parents=True, exist_ok=True)
        (run_dir / "exp_json").mkdir(parents=True, exist_ok=True)
        (run_dir / "vec_DB").mkdir(parents=True, exist_ok=True)
        logger.info(f"Created run directory (SEC flat): {run_dir}")
        return run_dir

    @staticmethod
    def _unique_paths(paths: list[Path]) -> list[Path]:
        seen: set[str] = set()
        uniq: list[Path] = []
        for path in paths:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(path)
        return uniq

    @staticmethod
    def _resolve_existing_path(path_value: str) -> Path | None:
        raw_path = Path(path_value).expanduser()
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend([Path.cwd() / raw_path, EXPBANK_ROOT / raw_path, REPO_ROOT / raw_path, raw_path])

        for candidate in SECBPipeline._unique_paths(candidates):
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _extract_cve_id(text: str) -> str:
        match = _CVE_RE.search(text)
        if match:
            return match.group(0).upper()
        return ""

    @staticmethod
    def _normalize_cwe_info(raw: Any) -> dict[str, dict[str, str]]:
        if not isinstance(raw, dict):
            return {}
        cwe_info: dict[str, dict[str, str]] = {}
        for key, value in raw.items():
            cwe_id = str(key).strip()
            if not cwe_id:
                continue
            if isinstance(value, dict):
                cwe_info[cwe_id] = {
                    "name": str(value.get("name", "") or ""),
                    "description": str(value.get("description", "") or ""),
                }
            else:
                cwe_info[cwe_id] = {"name": "", "description": str(value or "")}
        return cwe_info

    @staticmethod
    def _extract_instance_id_from_traj(traj_file: Path) -> str:
        name = traj_file.name
        if name.endswith(".traj.json"):
            return name[: -len(".traj.json")]
        return traj_file.parent.name or traj_file.stem

    def _get_vulnerability_info_path(self) -> Path | None:
        """Resolve SEC vulnerability_info path in a portable way."""
        input_cfg = self.config.get("input", {})
        raw_traj_dir = Path(input_cfg.get("raw_traj_dir", "."))
        relative_path = Path(
            input_cfg.get(
                "vulnerability_info_relative_path",
                "minisweagent/data/secb_official_cve201_dataset.jsonl",
            )
        )

        candidates: list[Path] = []
        configured_path = input_cfg.get("vulnerability_info")
        if configured_path:
            resolved = self._resolve_existing_path(str(configured_path))
            if resolved:
                return resolved
            raw_configured = Path(str(configured_path)).expanduser()
            if raw_configured.is_absolute():
                candidates.append(raw_configured)
            else:
                candidates.extend(
                    [Path.cwd() / raw_configured, EXPBANK_ROOT / raw_configured, REPO_ROOT / raw_configured]
                )

        candidates.extend(
            [
                raw_traj_dir / relative_path,
                raw_traj_dir.parent / relative_path,
                Path.cwd() / relative_path,
                EXPBANK_ROOT / relative_path,
                REPO_ROOT / relative_path,
            ]
        )

        for candidate in self._unique_paths(candidates):
            if candidate.exists():
                return candidate

        logger.warning(f"Vulnerability info file not found, attempted paths: {len(self._unique_paths(candidates))}")
        return None

    def _load_vulnerability_rows(self, vul_path: Path) -> list[dict[str, Any]]:
        """Load SEC vulnerability rows from JSONL or JSON."""
        if vul_path.suffix.lower() in {".jsonl", ".jl"}:
            rows: list[dict[str, Any]] = []
            with vul_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
            return rows

        with vul_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            rows = []
            for key, value in data.items():
                if not isinstance(value, dict):
                    continue
                item = dict(value)
                item.setdefault("instance_id", str(key))
                rows.append(item)
            return rows
        return []

    def _normalize_vul_row(self, row: dict[str, Any]) -> dict[str, Any]:
        instance_id = str(row.get("instance_id", "")).strip()
        cve_id = str(row.get("cve_id", "")).strip().upper()
        if not cve_id and instance_id:
            cve_id = self._extract_cve_id(instance_id)

        description = str(row.get("cve_description", "")).strip() or str(row.get("bug_description", "")).strip()
        programming_language = (
            str(row.get("programing_language", "")).strip()
            or str(row.get("programming_language", "")).strip()
            or "Unknown"
        )

        return {
            "instance_id": instance_id,
            "cve_id": cve_id,
            "cve_description": description,
            "cwe_info": self._normalize_cwe_info(row.get("cwe_info", {})),
            "vul_func": row.get("vul_func", []) if isinstance(row.get("vul_func", []), list) else [],
            "fix_func": row.get("fix_func", []) if isinstance(row.get("fix_func", []), list) else [],
            "programming_language": programming_language,
        }

    def _get_vulnerability_info(self) -> dict[str, dict]:
        """Load SEC vulnerability metadata map keyed by instance_id and cve_id."""
        vul_path = self._get_vulnerability_info_path()
        if not vul_path or not vul_path.exists():
            logger.warning("Vulnerability info file does not exist")
            self._instance_to_cve = {}
            return {}

        rows = self._load_vulnerability_rows(vul_path)
        if not rows:
            logger.warning(f"No rows loaded from vulnerability info file: {vul_path}")
            self._instance_to_cve = {}
            return {}

        vul_map: dict[str, dict[str, Any]] = {}
        instance_to_cve: dict[str, str] = {}

        for row in rows:
            item = self._normalize_vul_row(row)
            instance_id = item["instance_id"]
            cve_id = item["cve_id"]
            if not instance_id and not cve_id:
                continue
            if instance_id:
                vul_map[instance_id] = item
            if cve_id:
                vul_map[cve_id] = item
            if instance_id and cve_id:
                instance_to_cve[instance_id] = cve_id

        self._instance_to_cve = instance_to_cve
        logger.info(f"Loaded SEC vulnerability info: {len(instance_to_cve)} instance mappings from {vul_path}")
        return vul_map

    def _get_report_paths(self) -> list[Path]:
        """Get candidate SEC evaluation report files (report_*.jsonl)."""
        input_cfg = self.config.get("input", {})
        raw_traj_dir = Path(input_cfg.get("raw_traj_dir", "."))
        report_mode = str(input_cfg.get("evaluation_report_mode", "medium")).strip().lower() or "medium"

        configured_summary = input_cfg.get("summary_path")
        candidates: list[Path] = []
        if configured_summary:
            p = Path(str(configured_summary)).expanduser()
            if p.is_file():
                candidates.append(p)
            elif p.is_dir():
                candidates.append(p / f"report_{report_mode}.jsonl")
                candidates.extend(sorted(p.glob("report_*.jsonl")))

        candidates.extend(
            [
                raw_traj_dir / "evaluation" / f"report_{report_mode}.jsonl",
                raw_traj_dir.parent / "evaluation" / f"report_{report_mode}.jsonl",
                raw_traj_dir / "evaluation" / "report_medium.jsonl",
                raw_traj_dir.parent / "evaluation" / "report_medium.jsonl",
                raw_traj_dir / f"report_{report_mode}.jsonl",
                raw_traj_dir.parent / f"report_{report_mode}.jsonl",
            ]
        )

        report_paths: list[Path] = []
        for path in self._unique_paths(candidates):
            if path.exists() and path.is_file() and path.suffix.lower() == ".jsonl":
                report_paths.append(path)
        return report_paths

    def _get_success_set(self) -> set:
        """
        Load successful identifiers for SEC-bench.

        Success entries include both instance IDs and CVE IDs to keep compatibility
        with downstream filtering logic.
        """
        success_set: set[str] = set()
        report_paths = self._get_report_paths()

        if not report_paths:
            logger.warning("No SEC evaluation report_*.jsonl found, all trajectories will be marked fail")
            return set()

        for report_path in report_paths:
            loaded = 0
            with report_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict) or not bool(row.get("success", False)):
                        continue
                    instance_id = str(row.get("instance_id", "")).strip()
                    if not instance_id:
                        continue
                    success_set.add(instance_id)
                    cve_id = self._instance_to_cve.get(instance_id) or self._extract_cve_id(instance_id)
                    if cve_id:
                        success_set.add(cve_id)
                    loaded += 1
            logger.info(f"Loaded {loaded} successful rows from {report_path}")

        logger.info(f"SEC success set size: {len(success_set)}")
        return success_set

    def _get_poc_only_success_cves(self, base_input_dir: Path) -> set:
        """SEC patch evaluation currently has no poc_only split; no skip set is needed."""
        _ = base_input_dir
        return set()

    def _extract_trajectory(
        self, traj_file: Path, success_set: set, vul_info_map: dict, output_dir: Path
    ) -> Path | None:
        """
        Extract trajectory + vulnerability metadata for SEC format.

        SEC trajectory filenames are often `<instance_id>.traj.json`, where instance_id
        is not CVE-prefixed. We normalize them to CVE-based extracted filenames so that
        the downstream experience extractor can parse CVE IDs consistently.
        """
        try:
            with traj_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error(f"Failed to load trajectory file {traj_file}: {exc}")
            return None

        messages = data.get("messages")
        if not isinstance(messages, list):
            logger.warning(f"Invalid trajectory format {traj_file}: missing messages")
            return None

        traj_list: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            msg = {
                "role": message.get("role"),
                "content": message.get("content"),
            }
            if message.get("role") == "assistant":
                msg["tool_calls"] = message.get("extra", {}).get("tool_calls", [])
            traj_list.append(msg)

        if not traj_list:
            logger.warning(f"Empty trajectory file {traj_file}")
            return None

        instance_id = self._extract_instance_id_from_traj(traj_file)
        cve_id = self._instance_to_cve.get(instance_id) or self._extract_cve_id(instance_id) or instance_id

        success = instance_id in success_set or cve_id in success_set
        output_name = f"{cve_id}_{'success' if success else 'fail'}.json"

        vul_info = vul_info_map.get(instance_id) or vul_info_map.get(cve_id)
        if vul_info is None:
            vul_info = {
                "cve_id": cve_id,
                "cve_description": "",
                "cwe_info": {},
                "vul_func": [],
                "fix_func": [],
                "programming_language": "Unknown",
            }

        final_data = {"traj": traj_list, "vulnerability_info": vul_info}
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Trajectory extracted: {output_path}")
        return output_path

    def _init_experience_extractor(self, extracted_dir: Path, exp_json_dir: Path):
        """Initialize extractor with portable llm config path resolution."""
        from src.agent.experience_extractor_agent import ExperienceExtractor

        llm_cfg = str(self.config["llm"]["config_path"])
        resolved_llm_cfg = self._resolve_existing_path(llm_cfg)
        if resolved_llm_cfg is None:
            raise FileNotFoundError(f"LLM config file does not exist: {llm_cfg}")

        self._extractor = ExperienceExtractor(
            config_path=str(resolved_llm_cfg),
            extracted_dir=str(extracted_dir),
            output_dir=str(exp_json_dir),
            template_name=self.config["llm"]["template_name"],
            client_type=self.config["llm"].get("client_type", "openai"),
        )
        return self._extractor


def main():
    """CLI entrypoint for SEC-bench pipeline."""
    parser = argparse.ArgumentParser(
        description="ExpBank SEC-bench Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        default="src/config/run/pipeline_config_sec.yaml",
        help="Config file path (default: src/config/run/pipeline_config_sec.yaml)",
    )
    parser.add_argument("--input-dir", "-i", default=None, help="Input directory (overrides raw_traj_dir)")
    parser.add_argument("--output-dir", "-o", default=None, help="Output directory (overrides base_dir)")
    parser.add_argument("--db-dir", default=None, help="Specify existing vector database directory")
    parser.add_argument("--summary-path", default=None, help="Evaluation report path or directory")
    parser.add_argument("--vul-info-path", default=None, help="Vulnerability info path (JSON or JSONL)")
    parser.add_argument(
        "--report-mode",
        default=None,
        choices=["strict", "medium", "generous"],
        help="Preferred SEC report mode when summary-path is a directory",
    )
    parser.add_argument(
        "--stages", "-s", default=None, help="Execution stages, comma-separated (extract,experience,index)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate config, do not execute")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum number of files to process")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers (default: 4)")
    parser.add_argument("--clear-db", action="store_true", default=False, help="Clear vector database")

    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None

    pipeline = SECBPipeline(
        config_path=args.config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        vul_info_path=args.vul_info_path,
        db_dir=args.db_dir,
        report_mode=args.report_mode,
    )
    stats = pipeline.run_all(
        stages=stages,
        dry_run=args.dry_run,
        max_files=args.max_files,
        workers=args.workers,
        clear_db=args.clear_db,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
