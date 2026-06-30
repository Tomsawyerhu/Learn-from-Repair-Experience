#!/usr/bin/env python3

"""Build enriched SEC-bench official CVE dataset.

This script builds `data/secb_official_cve201_dataset.jsonl` from:
1. Official instance list in SEC-bench output
2. SEC-bench Hugging Face metadata (`data/eval-cve.jsonl`)
3. CVEProject cvelistV5 JSON (for CVE descriptions and CWE IDs)
4. MITRE CWE catalog (for CWE name/description)

The output schema is aligned with current mini-swe-agent PatchEval-style fields.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTANCE_LIST = REPO_ROOT / "SEC-bench/output/official_supported_cve_instances.txt"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data/secb_official_cve201_dataset.jsonl"
DEFAULT_REPO_LANG_PATH = REPO_ROOT / "SEC-bench/secb/data/repository-language-mappings.augmented.jsonl"

HF_DATASET_REPO = "SEC-bench/SEC-bench"
HF_DATASET_FILENAME = "data/eval-cve.jsonl"

CWE_CATALOG_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
NVD_CVE_API_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"

CVE_PATTERN = re.compile(r"cve-\d{4}-\d+", re.IGNORECASE)
CWE_PATTERN = re.compile(r"CWE-\d+", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build enriched SEC-bench official CVE dataset.")
    parser.add_argument(
        "--instance-list",
        type=Path,
        default=DEFAULT_INSTANCE_LIST,
        help="Path to official instance list file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--repo-lang-file",
        type=Path,
        default=DEFAULT_REPO_LANG_PATH,
        help="Repository language mapping JSONL path.",
    )
    parser.add_argument(
        "--hf-local-dir",
        type=Path,
        default=Path("/tmp/secb_hf_data"),
        help="Local cache dir for Hugging Face files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def normalize_language(language: str) -> str:
    value = (language or "").strip().lower()
    mapping = {
        "c": "C",
        "c++": "C++",
        "cpp": "C++",
        "python": "Python",
        "java": "Java",
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "go": "Go",
        "rust": "Rust",
        "php": "PHP",
        "ruby": "Ruby",
        "c#": "C#",
    }
    if not value:
        return "Unknown"
    return mapping.get(value, language.strip() if language else "Unknown")


def extract_cve_id(instance_id: str) -> str:
    match = CVE_PATTERN.search(instance_id)
    if not match:
        raise ValueError(f"Unable to parse CVE ID from instance_id: {instance_id}")
    return match.group(0).upper()


def read_instance_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    deduped = list(dict.fromkeys(ids))
    return deduped


def read_repo_language_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            repo = str(row.get("repo", "")).strip().lower()
            repo_url = str(row.get("repo_url", "")).strip().lower()
            lang = normalize_language(str(row.get("language", "")).strip())
            if not lang:
                continue
            if repo:
                mapping[repo] = lang
            if repo_url:
                mapping[repo_url] = lang
    return mapping


def load_hf_eval_cve(local_dir: Path) -> dict[str, dict[str, Any]]:
    local_dir.mkdir(parents=True, exist_ok=True)
    file_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        filename=HF_DATASET_FILENAME,
        local_dir=str(local_dir),
    )

    mapping: dict[str, dict[str, Any]] = {}
    with Path(file_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instance_id = str(row.get("instance_id", "")).strip()
            if instance_id:
                mapping[instance_id] = row
    return mapping


def fetch_json(url: str, session: requests.Session, timeout: int) -> dict[str, Any] | None:
    try:
        response = session.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def fetch_nvd_record(
    cve_id: str,
    session: requests.Session,
    timeout: int,
    max_retries: int = 5,
) -> dict[str, Any] | None:
    url = NVD_CVE_API_URL.format(cve_id=cve_id)
    backoff = 1.0
    for _ in range(max_retries):
        try:
            response = session.get(url, timeout=timeout)
        except requests.RequestException:
            response = None

        if response is None:
            time.sleep(backoff)
            backoff *= 2
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError:
                return None

        # NVD can throttle; retry on rate-limit/server-side failures.
        if response.status_code in {403, 429, 500, 502, 503, 504}:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                sleep_seconds = max(float(retry_after), backoff)
            else:
                sleep_seconds = backoff
            time.sleep(sleep_seconds)
            backoff = min(backoff * 2, 60)
            continue

        return None
    return None


def build_cvelist_url(cve_id: str) -> str:
    _, year, number = cve_id.split("-")
    bucket = f"{int(number) // 1000}xxx"
    return (
        f"https://raw.githubusercontent.com/CVEProject/cvelistV5/main/cves/"
        f"{year}/{bucket}/{cve_id}.json"
    )


def get_english_description(cvelist_record: dict[str, Any]) -> str:
    containers = cvelist_record.get("containers", {})
    cna = containers.get("cna", {}) if isinstance(containers, dict) else {}
    descriptions = cna.get("descriptions", []) if isinstance(cna, dict) else []

    for entry in descriptions:
        if not isinstance(entry, dict):
            continue
        if entry.get("lang") == "en":
            value = str(entry.get("value", "")).strip()
            if value:
                return value

    for entry in descriptions:
        if not isinstance(entry, dict):
            continue
        value = str(entry.get("value", "")).strip()
        if value:
            return value

    return ""


def collect_cwe_ids(cvelist_record: dict[str, Any], text_fallback: str) -> list[str]:
    ids: list[str] = []

    containers = cvelist_record.get("containers", {})
    if isinstance(containers, dict):
        container_values: list[dict[str, Any]] = []
        cna = containers.get("cna")
        if isinstance(cna, dict):
            container_values.append(cna)
        adp = containers.get("adp")
        if isinstance(adp, list):
            for item in adp:
                if isinstance(item, dict):
                    container_values.append(item)

        for container in container_values:
            problem_types = container.get("problemTypes", [])
            if not isinstance(problem_types, list):
                continue
            for problem in problem_types:
                if not isinstance(problem, dict):
                    continue
                descriptions = problem.get("descriptions", [])
                if not isinstance(descriptions, list):
                    continue
                for desc in descriptions:
                    if not isinstance(desc, dict):
                        continue
                    candidates = []
                    cwe_id = desc.get("cweId")
                    if isinstance(cwe_id, str):
                        candidates.append(cwe_id)
                    description = desc.get("description")
                    if isinstance(description, str):
                        candidates.append(description)

                    for candidate in candidates:
                        for found in CWE_PATTERN.findall(candidate):
                            ids.append(found.upper())

    if not ids and text_fallback:
        for found in CWE_PATTERN.findall(text_fallback):
            ids.append(found.upper())

    # Keep deterministic order
    unique_ids = sorted(set(ids), key=lambda value: int(value.split("-")[1]))
    return unique_ids


def collect_nvd_cwe_ids(nvd_record: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    vulnerabilities = nvd_record.get("vulnerabilities", [])
    if not isinstance(vulnerabilities, list) or not vulnerabilities:
        return ids

    cve_data = vulnerabilities[0].get("cve", {})
    if not isinstance(cve_data, dict):
        return ids

    weaknesses = cve_data.get("weaknesses", [])
    if not isinstance(weaknesses, list):
        return ids

    for weakness in weaknesses:
        if not isinstance(weakness, dict):
            continue
        descriptions = weakness.get("description", [])
        if not isinstance(descriptions, list):
            continue
        for desc in descriptions:
            if not isinstance(desc, dict):
                continue
            value = str(desc.get("value", "")).strip()
            if not value:
                continue
            for found in CWE_PATTERN.findall(value):
                ids.append(found.upper())

    unique_ids = sorted(set(ids), key=lambda value: int(value.split("-")[1]))
    return unique_ids


def collect_nvd_description(nvd_record: dict[str, Any]) -> str:
    vulnerabilities = nvd_record.get("vulnerabilities", [])
    if not isinstance(vulnerabilities, list) or not vulnerabilities:
        return ""

    cve_data = vulnerabilities[0].get("cve", {})
    if not isinstance(cve_data, dict):
        return ""

    descriptions = cve_data.get("descriptions", [])
    if not isinstance(descriptions, list):
        return ""

    for desc in descriptions:
        if not isinstance(desc, dict):
            continue
        if desc.get("lang") == "en":
            value = str(desc.get("value", "")).strip()
            if value:
                return value

    for desc in descriptions:
        if not isinstance(desc, dict):
            continue
        value = str(desc.get("value", "")).strip()
        if value:
            return value

    return ""


def parse_repo_from_cvelist(cvelist_record: dict[str, Any]) -> str:
    containers = cvelist_record.get("containers", {})
    cna = containers.get("cna", {}) if isinstance(containers, dict) else {}

    affected = cna.get("affected", []) if isinstance(cna, dict) else []
    for item in affected:
        if not isinstance(item, dict):
            continue
        vendor = str(item.get("vendor", "")).strip()
        product = str(item.get("product", "")).strip()
        if vendor and product and vendor.lower() != "n/a" and product.lower() != "n/a":
            candidate = f"{vendor}/{product}".lower()
            if "/" in candidate:
                return candidate

    references = cna.get("references", []) if isinstance(cna, dict) else []
    for ref in references:
        if not isinstance(ref, dict):
            continue
        url = str(ref.get("url", "")).strip()
        if not url:
            continue
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if "github.com" not in host and "gitlab.com" not in host:
            continue
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}".lower()

    return ""


def parse_repo_url(repo_value: str) -> str:
    value = repo_value.strip()
    if not value:
        return ""
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"https://github.com/{value}"


def extract_repo_key(repo_url: str) -> str:
    if not repo_url:
        return ""
    parsed = urlparse(repo_url)
    if not parsed.netloc:
        return repo_url.lower().strip("/")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1]}".lower()


def download_cwe_catalog(session: requests.Session, timeout: int) -> dict[str, dict[str, str]]:
    response = session.get(CWE_CATALOG_URL, timeout=timeout)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        xml_name = next((name for name in archive.namelist() if name.endswith(".xml")), None)
        if xml_name is None:
            raise ValueError("Unable to locate CWE XML in archive.")
        xml_bytes = archive.read(xml_name)

    root = ElementTree.fromstring(xml_bytes)
    cwe_map: dict[str, dict[str, str]] = {}

    for tag in ["Weakness", "Category"]:
        for node in root.findall(f".//{{*}}{tag}"):
            cwe_id = node.attrib.get("ID")
            cwe_name = node.attrib.get("Name", "").strip()
            if not cwe_id:
                continue
            key = f"CWE-{cwe_id}"
            desc_node = node.find("{*}Description")
            description = ""
            if desc_node is not None:
                description = "".join(desc_node.itertext()).strip()
                description = re.sub(r"\s+", " ", description)
            cwe_map[key] = {
                "name": cwe_name,
                "description": description,
            }

    return cwe_map


def build_cwe_info(cwe_ids: list[str], cwe_catalog: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for cwe_id in cwe_ids:
        catalog_entry = cwe_catalog.get(cwe_id, {})
        name = str(catalog_entry.get("name", "")).strip()
        description = str(catalog_entry.get("description", "")).strip()

        if not name:
            name = cwe_id
        if not description:
            description = "No CWE description available from MITRE catalog."

        result[cwe_id] = {
            "name": name,
            "description": description,
        }

    if not result:
        result["CWE-UNKNOWN"] = {
            "name": "Unknown CWE",
            "description": "CWE ID is not provided in the CVE record.",
        }
    return result


def guess_work_dir(instance_id: str) -> str:
    project = instance_id.split(".", 1)[0]
    return f"/src/{project}"


def build_record(
    instance_id: str,
    hf_row: dict[str, Any] | None,
    cvelist_record: dict[str, Any] | None,
    nvd_record: dict[str, Any] | None,
    repo_lang_map: dict[str, str],
    cwe_catalog: dict[str, dict[str, str]],
) -> dict[str, Any]:
    cve_id = extract_cve_id(instance_id)

    cve_description = ""
    cwe_ids: list[str] = []
    repo_key = ""

    if cvelist_record:
        cve_description = get_english_description(cvelist_record)
        cwe_ids = collect_cwe_ids(cvelist_record, cve_description)
        repo_key = parse_repo_from_cvelist(cvelist_record)

    if nvd_record:
        if not cwe_ids:
            cwe_ids = collect_nvd_cwe_ids(nvd_record)
        if not cve_description:
            cve_description = collect_nvd_description(nvd_record)

    if hf_row:
        bug_description = str(hf_row.get("bug_description", "")).strip()
        if not cve_description:
            cve_description = bug_description
    else:
        bug_description = ""

    if not cve_description:
        cve_description = f"No CVE description extracted for {cve_id}."

    if hf_row and str(hf_row.get("repo", "")).strip():
        repo_value = str(hf_row.get("repo", "")).strip()
    elif repo_key:
        repo_value = repo_key
    else:
        repo_value = instance_id.split(".", 1)[0]
    repo_url = parse_repo_url(repo_value)
    repo_key_normalized = extract_repo_key(repo_url)

    if hf_row:
        language = normalize_language(str(hf_row.get("lang", "")))
    else:
        language = "Unknown"
    if language == "Unknown":
        language = repo_lang_map.get(repo_key_normalized, language)
        if language == "Unknown":
            language = repo_lang_map.get(repo_url.lower(), language)

    work_dir = ""
    if hf_row:
        work_dir = str(hf_row.get("work_dir", "")).strip()
    if not work_dir:
        work_dir = guess_work_dir(instance_id)

    cwe_info = build_cwe_info(cwe_ids, cwe_catalog)

    record: dict[str, Any] = {
        "cve_id": cve_id,
        "instance_id": instance_id,
        "cve_description": cve_description,
        "cwe_info": cwe_info,
        "repo": repo_url,
        "programing_language": language,
        "vul_func": [],
        "work_dir": work_dir,
    }

    if bug_description:
        record["bug_description"] = bug_description
    if hf_row and "project_name" in hf_row:
        record["project_name"] = hf_row.get("project_name")
    if hf_row and "sanitizer" in hf_row:
        record["sanitizer"] = hf_row.get("sanitizer")

    return record


def main() -> None:
    args = parse_args()
    instance_ids = read_instance_ids(args.instance_list)
    if not instance_ids:
        raise ValueError(f"No instance IDs found: {args.instance_list}")

    repo_lang_map = read_repo_language_map(args.repo_lang_file)
    hf_rows = load_hf_eval_cve(args.hf_local_dir)

    session = requests.Session()
    cwe_catalog = download_cwe_catalog(session, args.timeout)

    records: list[dict[str, Any]] = []
    missing_hf_instances: list[str] = []
    missing_cvelist_instances: list[str] = []
    missing_nvd_instances: list[str] = []
    unknown_lang_instances: list[str] = []
    unknown_cwe_instances: list[str] = []

    for instance_id in instance_ids:
        hf_row = hf_rows.get(instance_id)
        if hf_row is None:
            missing_hf_instances.append(instance_id)

        cve_id = extract_cve_id(instance_id)
        cvelist_url = build_cvelist_url(cve_id)
        cvelist_record = fetch_json(cvelist_url, session, args.timeout)
        if cvelist_record is None:
            missing_cvelist_instances.append(instance_id)

        need_nvd = True
        if cvelist_record is not None:
            preliminary_desc = get_english_description(cvelist_record)
            preliminary_cwes = collect_cwe_ids(cvelist_record, preliminary_desc)
            need_nvd = not preliminary_cwes

        nvd_record = None
        if need_nvd:
            nvd_record = fetch_nvd_record(cve_id, session, args.timeout)
            if nvd_record is None:
                missing_nvd_instances.append(instance_id)

        record = build_record(
            instance_id=instance_id,
            hf_row=hf_row,
            cvelist_record=cvelist_record,
            nvd_record=nvd_record,
            repo_lang_map=repo_lang_map,
            cwe_catalog=cwe_catalog,
        )
        if record["programing_language"] == "Unknown":
            unknown_lang_instances.append(instance_id)
        if "CWE-UNKNOWN" in record["cwe_info"]:
            unknown_cwe_instances.append(instance_id)

        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Built dataset: {args.output}")
    print(f"Records: {len(records)}")
    print(f"Missing HF metadata: {len(missing_hf_instances)}")
    if missing_hf_instances:
        print("  " + ", ".join(missing_hf_instances))
    print(f"Missing cvelist records: {len(missing_cvelist_instances)}")
    if missing_cvelist_instances:
        print("  " + ", ".join(missing_cvelist_instances))
    print(f"Missing NVD records (queried on demand): {len(missing_nvd_instances)}")
    if missing_nvd_instances:
        print("  " + ", ".join(missing_nvd_instances))
    print(f"Unknown language records: {len(unknown_lang_instances)}")
    if unknown_lang_instances:
        print("  " + ", ".join(unknown_lang_instances))
    print(f"Unknown CWE records: {len(unknown_cwe_instances)}")
    if unknown_cwe_instances:
        print("  " + ", ".join(unknown_cwe_instances))


if __name__ == "__main__":
    main()
