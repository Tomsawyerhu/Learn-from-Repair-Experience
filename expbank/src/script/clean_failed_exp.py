import json
import os
from pathlib import Path

def clean_failed_extractions(exp_dir: str = "exp_json"):
    """
    Delete experience JSON files where cwe_ids contains "UNKNOWN".
    These typically indicate extraction errors (e.g. LLM failure).
    """
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"Directory not found: {exp_path}")
        return

    deleted_count = 0
    total_count = 0
    
    print(f"Scanning {exp_path} for failed extractions...")
    
    for json_file in exp_path.glob("*.json"):
        total_count += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if cwe_ids contains "UNKNOWN"
            cwe_ids = data.get("cwe_ids", [])
            if "UNKNOWN" in cwe_ids:
                print(f"Deleting {json_file.name} (CWE: UNKNOWN, Error: {data.get('extraction_error', 'N/A')})")
                os.remove(json_file)
                deleted_count += 1
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")

    print(f"\nCleanup complete.")
    print(f"Total files scanned: {total_count}")
    print(f"Deleted files: {deleted_count}")

if __name__ == "__main__":
    # Adjust path if running from project root or script dir
    # Adjust path if running from project root or script dir
    # script is in src/script/clean_failed_exp.py
    # parents[0] = script dir
    # parents[1] = src dir
    # parents[2] = project root (expbank)
    src_dir = Path(__file__).resolve().parents[1]
    target_dir = project_root / "exp_json"
    
    clean_failed_extractions(str(target_dir))
