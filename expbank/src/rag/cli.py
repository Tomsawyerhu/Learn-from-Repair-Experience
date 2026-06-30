"""
Experience RAG Command Line Interface

Provides command line tools for experience indexing and retrieval.

Usage:
    # Index all experiences
    python -m src.rag.cli index --dir src/exp_json

    # Index single file
    python -m src.rag.cli index --file src/exp_json/CVE-2015-3295.json

    # Retrieve experiences
    python -m src.rag.cli search --cwe CWE-89 --description "SQL injection vulnerability..."

    # View statistics
    python -m src.rag.cli stats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.rag.cold_start import run_cold_start
from src.rag.embedder import BGEEmbedder
from src.rag.experience_retriever import ExperienceRetriever
from src.rag.experience_store import ExperienceStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cmd_index(args):
    """
    Index command handler
    """
    logger.info("Initializing vector encoder...")
    embedder = BGEEmbedder(model_path=args.model_path)

    logger.info("Initializing experience store...")
    store = ExperienceStore(db_path=args.db_path, embedder=embedder)

    if args.file:
        # Single file indexing
        logger.info(f"Indexing file: {args.file}")
        count = store.add_experience(args.file, overwrite=not args.no_overwrite)
        print(f"Successfully added/updated {count} experience records")

    elif args.dir:
        # Batch directory indexing
        logger.info(f"Batch indexing directory: {args.dir}")
        stats = store.add_experiences_from_dir(args.dir, overwrite=not args.no_overwrite)
        print(json.dumps(stats, ensure_ascii=False, indent=2))

    else:
        print("Please specify --file or --dir parameter")
        sys.exit(1)

    # Show final statistics
    final_stats = store.get_stats()
    print(f"\nExperience store statistics: {final_stats['total_records']} records")


def cmd_search(args):
    """
    Search command handler
    """
    logger.info("Initializing vector encoder...")
    embedder = BGEEmbedder(model_path=args.model_path)

    logger.info("Initializing experience store...")
    store = ExperienceStore(db_path=args.db_path, embedder=embedder)

    retriever = ExperienceRetriever(store, embedder)

    # Parse CWE list
    cwe_ids = args.cwe.split(",") if args.cwe else None

    # Execute search
    results = retriever.retrieve(
        cve_description=args.description,
        cwe_ids=cwe_ids,
        exclude_cve_id=args.exclude_cve,
        k=args.k,
        min_quality_score=args.min_quality_score,
    )

    if results:
        print(f"\nFound {len(results)} related experiences:\n")
        for i, exp in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"ID: {exp['id']}")
            print(f"CVE: {exp['cve_id']}")
            print(f"CWE: {exp['cwe_ids']}")
            print(f"Tags: {exp['tags']}")
            print(f"Similarity: {exp['similarity_score']:.4f}")
            print(f"Quality Score: {exp['quality_score']}/5")
            print(f"Repair Result: {exp['repair_result']}")
            print("Experience Content:")
            # Experience is now a Markdown string, truncate for display
            exp_content = exp["experience"]
            if len(exp_content) > 500:
                print(exp_content[:500] + "\n...[truncated]")
            else:
                print(exp_content)
            print()
    else:
        print("No related experiences found")


def cmd_stats(args):
    """
    Stats command handler
    """
    logger.info("Initializing experience store...")

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(path=args.db_path, settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection("cwe_experiences")

        # Basic statistics
        stats = {"db_path": args.db_path, "total_records": collection.count()}

        # Get quality score distribution and repair result statistics
        all_records = collection.get(include=["metadatas"])
        score_counts = {}
        repair_counts = {"success": 0, "failure": 0, "unknown": 0}

        if all_records and all_records["metadatas"]:
            for metadata in all_records["metadatas"]:
                # Count scores
                score = metadata.get("quality_score", 0)
                score_key = str(score)
                score_counts[score_key] = score_counts.get(score_key, 0) + 1

                # Count repair results
                result = metadata.get("repair_result", "unknown")
                if result in repair_counts:
                    repair_counts[result] += 1
                else:
                    repair_counts["unknown"] += 1

        # Sort by score
        sorted_counts = dict(sorted(score_counts.items()))
        stats["quality_score_distribution"] = sorted_counts

        # Add repair result statistics
        stats["repair_result_distribution"] = repair_counts

        print(json.dumps(stats, ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"Failed to get statistics: {e}")
        sys.exit(1)


def cmd_cold_start(args):
    """
    Cold start command handler
    """
    import json

    logger.info("Starting cold start of experience store...")

    stats = run_cold_start(
        input_path=args.input,
        output_dir=args.output,
        db_path=args.db_path,
        model_path=args.model_path,
        no_index=args.no_index,
    )

    print("\nCold start completed!")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def cmd_clear(args):
    """
    Clear database command handler
    """
    if not args.confirm:
        print("Warning: This operation will clear all experience data!")
        print("Please use --confirm parameter to confirm operation")
        sys.exit(1)

    logger.info("Initializing vector encoder...")
    embedder = BGEEmbedder(model_path=args.model_path)

    logger.info("Initializing experience store...")
    store = ExperienceStore(db_path=args.db_path, embedder=embedder)

    store.clear()
    print("Experience store has been cleared")


def main():
    parser = argparse.ArgumentParser(
        description="Experience RAG Command Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch index
  python -m src.rag.cli index --dir src/exp_json
  
  # Retrieve experiences
  python -m src.rag.cli search --cwe CWE-89 --description "SQL injection related description..."
  
  # View statistics
  python -m src.rag.cli stats
        """,
    )

    # Common parameters
    parser.add_argument("--db-path", default="vec_DB", help="ChromaDB database path (default: vec_DB)")
    parser.add_argument(
        "--model-path", default="models/bge-large-en-v1.5", help="BGE model path (default: models/bge-large-en-v1.5)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index subcommand
    index_parser = subparsers.add_parser("index", help="Index experiences to database")
    index_parser.add_argument("--file", "-f", help="Single experience JSON file path")
    index_parser.add_argument("--dir", "-d", help="Experience JSON file directory")
    index_parser.add_argument(
        "--no-overwrite", action="store_true", help="Do not overwrite existing records (default overwrites)"
    )
    index_parser.set_defaults(func=cmd_index)

    # search subcommand
    search_parser = subparsers.add_parser("search", help="Search related experiences")
    search_parser.add_argument("--description", "-D", required=True, help="CVE description (for semantic matching)")
    search_parser.add_argument("--cwe", "-c", help="CWE ID filter (multiple separated by comma, e.g., CWE-89,CWE-79)")
    search_parser.add_argument("--exclude-cve", "-e", help="CVE ID to exclude")
    search_parser.add_argument("-k", type=int, default=5, help="Number of results to return (default: 5)")
    search_parser.add_argument(
        "--min-quality-score",
        "-q",
        type=int,
        default=0,
        help="Minimum experience quality score (1-5, default: 0 no filter)",
    )
    search_parser.set_defaults(func=cmd_search)

    # stats subcommand
    stats_parser = subparsers.add_parser("stats", help="View experience store statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # cold-start subcommand
    cold_start_parser = subparsers.add_parser("cold-start", help="Cold start experience store")
    cold_start_parser.add_argument(
        "--input", "-i", default="datasets/input.json", help="Input data file path (default: datasets/input.json)"
    )
    cold_start_parser.add_argument(
        "--output",
        "-o",
        default="datasets/cold_start",
        help="Output experience JSON directory (default: datasets/cold_start)",
    )
    cold_start_parser.add_argument(
        "--no-index", action="store_true", help="Only generate JSON, do not index to database"
    )
    cold_start_parser.set_defaults(func=cmd_cold_start)

    # clear subcommand
    clear_parser = subparsers.add_parser("clear", help="Clear experience store")
    clear_parser.add_argument("--confirm", action="store_true", help="Confirm clear operation")
    clear_parser.set_defaults(func=cmd_clear)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
