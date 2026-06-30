from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def iter_vector_db_dirs(artifact_root: Path) -> list[Path]:
    db_dirs = []
    patterns = [
        "*/turn_*/vec_DB_before_update",
        "*/turn_*/vec_DB_before_updated",
    ]
    for pattern in patterns:
        db_dirs.extend(artifact_root.glob(pattern))
    return sorted({path.resolve() for path in db_dirs if path.is_dir()})


def count_success_records(db_dir: Path) -> int:
    sqlite_path = db_dir / "chroma.sqlite3"
    if not sqlite_path.exists():
        return 0

    conn = sqlite3.connect(sqlite_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(DISTINCT em.id)
            FROM embedding_metadata em
            WHERE em.key = 'repair_result' AND em.string_value = 'success'
            """
        )
        row = cursor.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def write_count_file(db_dir: Path, success_count: int) -> Path:
    output_path = db_dir.parent / f"{db_dir.name}_success_count.txt"
    output_path.write_text(f"success_experience_count: {success_count}\n", encoding="utf-8")
    return output_path


def get_count_file_path(db_dir: Path) -> Path:
    return db_dir.parent / f"{db_dir.name}_success_count.txt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count the number of successful experiences in vec_DB_before_update(d) for each turn of each experiment directory under artifact, and write to a txt file at the same level as the experience database"
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifact",
        help="Artifact root directory path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each vector database",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing statistics txt files",
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    if not artifact_root.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_root}")

    db_dirs = iter_vector_db_dirs(artifact_root)
    if not db_dirs:
        print(f"No vector database directories found: {artifact_root}")
        return

    print(f"Found {len(db_dirs)} vector database directories, starting statistics...")
    success = 0
    failed = 0
    skipped = 0
    for idx, db_dir in enumerate(db_dirs, start=1):
        output_path = get_count_file_path(db_dir)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            success_count = count_success_records(db_dir)
            output_path = write_count_file(db_dir, success_count)
            success += 1
            if args.verbose:
                print(f"[OK] {db_dir} -> {success_count} ({output_path})")
            elif idx % 25 == 0:
                print(f"Progress: {idx}/{len(db_dirs)}")
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {db_dir}: {exc}")

    print(f"Completed: {success} succeeded, {failed} failed, {skipped} skipped")


if __name__ == "__main__":
    main()
