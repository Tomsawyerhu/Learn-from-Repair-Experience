import os
import json
import re

def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "minisweagent", "output")

    # Ensure base directory exists
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Iterate over each model directory in output
    for model_name in os.listdir(base_dir):
        model_dir = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"Processing model: {model_name}")
        results = []

        # Set to track unique fixed CVEs across turns for this model
        cumulative_fixed_cves = set()

        # Find all turn directories
        turns = []
        for entry in os.listdir(model_dir):
            if entry.startswith("turn_") and os.path.isdir(os.path.join(model_dir, entry)):
                try:
                    turn_num = int(entry.split("_")[1])
                    turns.append((turn_num, entry))
                except ValueError:
                    continue

        # Sort turns by number (1, 2, 3...)
        turns.sort()

        for turn_num, turn_dirname in turns:
            summary_path = os.path.join(model_dir, turn_dirname, "evaluation", "summary.json")

            # Extract success info for this turn
            current_turn_fixed = set()

            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        data = json.load(f)

                        # Navigate to successful_cves in strict_evaluation
                        # Structure: strict_evaluation -> successful_cves -> { "Python": [...], "Go": [...] }
                        strict_eval = data.get("strict_evaluation", {})
                        successful_cves_dict = strict_eval.get("successful_cves", {})

                        # Collect all CVE IDs from all languages
                        if isinstance(successful_cves_dict, dict):
                            for lang, cve_list in successful_cves_dict.items():
                                if isinstance(cve_list, list):
                                    current_turn_fixed.update(cve_list)

                except Exception as e:
                    print(f"  Error reading {summary_path}: {e}")
            else:
                print(f"  No evaluation summary found for Turn {turn_num}")

            # Update cumulative set
            original_count = len(cumulative_fixed_cves)
            cumulative_fixed_cves.update(current_turn_fixed)
            new_count = len(cumulative_fixed_cves)
            added = new_count - original_count

            print(f"  Turn {turn_num}: Found {len(current_turn_fixed)} fixed (New: {added}). Cumulative: {new_count}")
            results.append(f"Turn {turn_num}: {new_count}")

        # If we found any results, write them to a txt file in the model directory
        if results:
            output_file = os.path.join(model_dir, "cumulative_repair_counts.txt")
            try:
                with open(output_file, 'w') as f:
                    f.write("\n".join(results) + "\n")
                print(f"  Saved results to: {output_file}")
            except Exception as e:
                print(f"  Error writing output file: {e}")
        else:
            print("  No evaluation data found.")

if __name__ == "__main__":
    main()
