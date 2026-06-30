#!/usr/bin/env python3
"""
Statistics visualization script.
Reads `statistics.json` files under model output directories and generates
comparison line charts for cumulative cost and fix success count.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use a non-interactive backend to avoid display issues.
matplotlib.use('Agg')

# Keep a broad font fallback list.
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # Avoid minus-sign rendering issues.


def load_statistics(stats_path: Path) -> Dict:
    """Load a `statistics.json` file."""
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to read {stats_path}: {e}")
        return None


def extract_turn_data(stats: Dict) -> Tuple[List[int], List[float], List[int]]:
    """
    Extract turn-level data from statistics.

    Returns:
        - turn_numbers: turn index list [1, 2, 3, ...]
        - cumulative_costs: cumulative cost list
        - cumulative_success: cumulative success count list
    """
    if not stats or 'turns' not in stats:
        return [], [], []

    turn_numbers = []
    cumulative_costs = []
    cumulative_success = []

    total_cost = 0.0

    # Extract data in turn order.
    for turn_name in sorted(stats['turns'].keys(), key=lambda x: int(x.split('_')[1])):
        turn_data = stats['turns'][turn_name]
        turn_num = int(turn_name.split('_')[1])

        # Accumulate costs.
        turn_cost = turn_data.get('costs', {}).get('total', 0.0)
        total_cost += turn_cost

        # Cumulative success count.
        cumulative_count = turn_data.get('evaluation', {}).get('cumulative', {}).get('count', 0)

        turn_numbers.append(turn_num)
        cumulative_costs.append(total_cost)
        cumulative_success.append(cumulative_count)

    return turn_numbers, cumulative_costs, cumulative_success


def find_model_statistics(output_dir: Path, prefix: str = None) -> Dict[str, Dict]:
    """
    Find all model `statistics.json` files under the output directory.

    Args:
        output_dir: Output directory path.
        prefix: Optional model-directory name prefix filter.

    Returns: {model_name: statistics_data}
    """
    model_stats = {}

    # Iterate through all subdirectories in output.
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # If a prefix is specified, only process matching directories.
        if prefix and not model_dir.name.startswith(prefix):
            continue

        # Look for statistics/statistics.json.
        stats_path = model_dir / 'statistics' / 'statistics.json'
        if stats_path.exists():
            stats = load_statistics(stats_path)
            if stats:
                model_name = model_dir.name
                model_stats[model_name] = stats
                print(f"✓ Loaded model: {model_name}")

    return model_stats


def plot_cumulative_costs(model_data: Dict[str, Tuple], output_path: Path):
    """Plot cumulative cost-by-turn curves for all models."""
    plt.figure(figsize=(12, 6))

    # Define color and marker styles.
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, (model_name, (turns, costs, _)) in enumerate(model_data.items()):
        if not turns:
            continue

        plt.plot(turns, costs,
                marker=markers[idx % len(markers)],
                label=model_name,
                color=colors[idx],
                linewidth=2,
                markersize=8,
                alpha=0.8)

    plt.xlabel('Turn', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Cost ($)', fontsize=12, fontweight='bold')
    plt.title('Model Cost Accumulation by Turn', fontsize=14, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Cost accumulation chart saved: {output_path}")
    plt.close()


def plot_cumulative_success(model_data: Dict[str, Tuple], output_path: Path):
    """Plot cumulative fix-success curves for all models."""
    plt.figure(figsize=(12, 6))

    # Define color and marker styles.
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for idx, (model_name, (turns, _, success)) in enumerate(model_data.items()):
        if not turns:
            continue

        plt.plot(turns, success,
                marker=markers[idx % len(markers)],
                label=model_name,
                color=colors[idx],
                linewidth=2,
                markersize=8,
                alpha=0.8)

    plt.xlabel('Turn', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Success Count', fontsize=12, fontweight='bold')
    plt.title('Model Fix Success Accumulation by Turn', fontsize=14, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Success accumulation chart saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison visualizations for model statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                                          # Use default dir minisweagent/output
  %(prog)s --input-dir /path/to/output              # Specify input directory
  %(prog)s --prefix deepseek                        # Compare only models starting with deepseek
  %(prog)s --prefix gpt-5-mini --output-dir ./figs  # Filter gpt-5-mini models and set output dir
        '''
    )

    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('minisweagent/output'),
        help='Model output directory path (default: minisweagent/output)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('.'),
        help='Chart output directory (default: current directory)'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default=None,
        help='Model directory name prefix filter; process only matching model directories (e.g., deepseek, gpt-5-mini)'
    )

    args = parser.parse_args()

    # Validate input directory.
    if not args.input_dir.exists():
        print(f"❌ Error: input directory does not exist: {args.input_dir}")
        return 1

    # Create output directory.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("📊 Model Statistics Visualization")
    print(f"{'='*60}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.prefix:
        print(f"Filter prefix: {args.prefix}")
    print(f"{'='*60}\n")

    # Find and load statistics for all models.
    model_stats = find_model_statistics(args.input_dir, args.prefix)

    if not model_stats:
        if args.prefix:
            print(f"❌ Error: no model directory or statistics.json found with prefix '{args.prefix}'")
        else:
            print("❌ Error: no statistics.json files found")
        return 1

    print(f"\nFound statistics for {len(model_stats)} models\n")

    # Extract turn-level data for each model.
    model_data = {}
    for model_name, stats in model_stats.items():
        turns, costs, success = extract_turn_data(stats)
        if turns:
            model_data[model_name] = (turns, costs, success)
            print(f"  {model_name}: {len(turns)} turns, final cost=${costs[-1]:.2f}, success count={success[-1]}")

    if not model_data:
        print("❌ Error: failed to extract valid turn data")
        return 1

    print(f"\n{'='*60}")
    print("📈 Generating charts...")
    print(f"{'='*60}\n")

    # Generate charts.
    cost_plot_path = args.output_dir / 'model_cost_accumulation.png'
    success_plot_path = args.output_dir / 'model_success_accumulation.png'

    plot_cumulative_costs(model_data, cost_plot_path)
    plot_cumulative_success(model_data, success_plot_path)

    print(f"\n{'='*60}")
    print("✅ Done!")
    print(f"{'='*60}\n")

    return 0


if __name__ == '__main__':
    exit(main())
