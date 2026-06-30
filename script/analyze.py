#!/usr/bin/env python3
"""
Experience database effectiveness analysis script - basic statistical comparison.

Features:
1. Scan experiment outputs under a given directory, with model-prefix filtering.
2. Grouped statistics for success rate, cost, step count, and more.
3. Turn accumulation support: aggregate data up to a target turn and auto-detect max turn.
4. Sort experiments by number of repaired cases.
5. Paired-sample analysis to identify cases with significant gains from experience.
6. Additional metrics:
   - Average steps for successful cases
   - First-attempt success rate (success on first verification call)
   - Average cost per step (overall and successful cases separately)
7. Rich table output + Markdown report export.

Usage examples:
    # Analyze all experiments with prefix gpt-5-mini, auto-detect max turn
    python experiment_stats_comparison.py --base-dir ./output --model-prefix gpt-5-mini

    # Analyze multiple models up to turn 3
    python experiment_stats_comparison.py --base-dir ./output --model-prefix gpt-5-mini deepseek-v3.2 --max-turn 3

    # Sort by repaired case count
    python experiment_stats_comparison.py --base-dir ./output --model-prefix gpt-5-mini --sort-by-success

    # Specify output directory and max turn
    python experiment_stats_comparison.py --base-dir ./output --model-prefix gpt-5-mini --max-turn 2 --output-dir ./analysis_results
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
except ImportError:
    print("Please install the rich package: pip install rich")
    sys.exit(1)

console = Console()


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TrajectoryStats:
    """Statistics for a single trajectory."""
    cve_id: str
    experiment_name: str
    turn: int
    exit_status: str
    submitted: bool  # Whether the agent submitted a patch
    repair_success: bool  # True repair success (validated by evaluation)
    num_steps: int
    num_api_calls: int
    total_cost: float
    commands_count: int = 0
    errors_count: int = 0
    verification_attempts: int = 0
    first_submit_success: bool = False  # Success on first submission
    traj_file_path: str = ""

    @property
    def cost_per_step(self) -> float:
        """Average cost per step."""
        return self.total_cost / self.num_steps if self.num_steps > 0 else 0.0


@dataclass
class ExperimentGroup:
    """Experiment group (different experience configs for the same model)."""
    model_prefix: str
    experiments: Dict[str, 'ExperimentStats'] = field(default_factory=dict)

    def get_baseline(self) -> Optional['ExperimentStats']:
        """Get the baseline experiment (no-experience config, i.e. '0')."""
        for exp in self.experiments.values():
            if exp.experience_config == "0":
                return exp
        return None


@dataclass
class ExperimentStats:
    """Statistics for a single experiment."""
    experiment_name: str
    model_prefix: str
    experience_config: str  # "0", "230", "230+770", "230-k-3", etc.
    has_experience: bool
    turn: int

    # Experience retrieval parameters
    exp_retrieve_k: int = 0  # Number of retrieved experiences
    exp_return_n: int = 0    # Number of returned experiences

    # Aggregated metrics
    total_cves: int = 0
    successful_cves: int = 0
    failed_cves: int = 0
    error_cves: int = 0  # Failed due to errors, not unsuccessful submission

    # Averages
    avg_steps: float = 0.0
    avg_cost: float = 0.0
    avg_api_calls: float = 0.0
    avg_verification_attempts: float = 0.0

    # Totals
    total_cost: float = 0.0
    total_steps: int = 0

    # Statistics for successful cases
    success_avg_steps: float = 0.0
    success_avg_cost: float = 0.0

    # Additional metrics
    first_submit_success_count: int = 0  # Count of first-submission successes
    avg_cost_per_step: float = 0.0  # Average cost per step
    success_avg_cost_per_step: float = 0.0  # Average cost per step for successful cases

    # Detailed entries
    trajectories: Dict[str, TrajectoryStats] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successful_cves / self.total_cves if self.total_cves > 0 else 0.0

    @property
    def success_rate_str(self) -> str:
        return f"{self.success_rate:.1%}"

    @property
    def first_submit_success_rate(self) -> float:
        """First-submission success rate (among successful cases)."""
        return self.first_submit_success_count / self.successful_cves if self.successful_cves > 0 else 0.0

    @property
    def first_submit_success_rate_str(self) -> str:
        return f"{self.first_submit_success_rate:.1%}"


@dataclass
class PairedSample:
    """Paired sample: same CVE under with/without experience conditions."""
    cve_id: str
    model_prefix: str
    experience_config: str

    # Baseline (without experience)
    baseline_success: bool
    baseline_steps: int
    baseline_cost: float
    baseline_errors: int
    baseline_status: str

    # Experiment (with experience)
    exp_success: bool
    exp_steps: int
    exp_cost: float
    exp_errors: int
    exp_status: str

    # Computed metrics
    @property
    def success_change(self) -> int:
        """Success change: 1=improved, 0=unchanged, -1=degraded."""
        if self.exp_success and not self.baseline_success:
            return 1
        elif not self.exp_success and self.baseline_success:
            return -1
        return 0

    @property
    def step_reduction(self) -> int:
        return self.baseline_steps - self.exp_steps

    @property
    def step_reduction_pct(self) -> float:
        if self.baseline_steps == 0:
            return 0.0
        return self.step_reduction / self.baseline_steps

    @property
    def cost_reduction(self) -> float:
        return self.baseline_cost - self.exp_cost

    @property
    def effectiveness_score(self) -> float:
        """Composite effectiveness score (0-100)."""
        score = 0.0

        # Success improvement (max 50 points)
        if self.success_change > 0:
            score += 50
        elif self.baseline_success and self.exp_success:
            score += 15

        # Step reduction (max 30 points)
        if self.baseline_steps > 0 and self.step_reduction > 0:
            pct = min(1.0, self.step_reduction_pct)
            score += pct * 30

        # Cost reduction (max 15 points)
        if self.baseline_cost > 0 and self.cost_reduction > 0:
            cost_pct = min(1.0, self.cost_reduction / self.baseline_cost)
            score += cost_pct * 15

        # Error reduction (max 5 points)
        error_reduction = self.baseline_errors - self.exp_errors
        if error_reduction > 0:
            score += min(5, error_reduction)

        return score


# ============================================================================
# Parsing functions
# ============================================================================

def parse_experiment_name(exp_name: str) -> Tuple[str, str, bool]:
    """
    Parse an experiment directory name.

    Returns: (model_prefix, experience_config, has_experience)

    Examples:
    - gpt-5-mini-0 -> ("gpt-5-mini", "0", False)
    - gpt-5-mini-230 -> ("gpt-5-mini", "230", True)
    - gpt-5-mini-230+770 -> ("gpt-5-mini", "230+770", True)
    - gpt-5-mini-230-k-3 -> ("gpt-5-mini", "230-k-3", True)
    - gpt-5-mini-230-k-10-n-7 -> ("gpt-5-mini", "230-k-10-n-7", True)
    - deepseek-v3.2-100steps-0 -> ("deepseek-v3.2-100steps", "0", False)
    """
    # Common experience config patterns (sorted by complexity, complex first)
    config_patterns = [
        r'-(\d+\+\d+(?:-k-\d+)?(?:-n-\d+)?)$',  # 230+770, 230+770-k-3
        r'-(\d+-k-\d+-n-\d+)$',                  # 230-k-10-n-7
        r'-(\d+-k-\d+)$',                        # 230-k-3
        r'-(\d+-n-\d+)$',                        # 230-n-1
        r'-(\d+)$',                              # 0, 230, 1000
    ]

    for pattern in config_patterns:
        match = re.search(pattern, exp_name)
        if match:
            config = match.group(1)
            model_prefix = exp_name[:match.start()]
            has_exp = config != "0"
            return model_prefix, config, has_exp

    return exp_name, "unknown", False


def detect_max_turn(base_dir: str) -> int:
    """
    Detect the maximum turn index present in experiment directories.

    Args:
        base_dir: Base directory for experiment outputs

    Returns:
        Largest turn number
    """
    max_turn = 0

    if not os.path.exists(base_dir):
        return max_turn

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        if entry.startswith('.') or entry in ['backup', 'statistics', 'analysis_output']:
            continue

        # Check turns under this experiment directory.
        for turn_entry in os.listdir(entry_path):
            if turn_entry.startswith('turn_'):
                try:
                    turn_num = int(turn_entry.split('_')[1])
                    max_turn = max(max_turn, turn_num)
                except (ValueError, IndexError):
                    pass

    return max_turn if max_turn > 0 else 1


def discover_experiments(base_dir: str, model_prefixes: List[str] = None) -> Dict[str, List[str]]:
    """
    Discover experiment directories and group by model prefix.

    Args:
        base_dir: Base directory for experiment outputs
        model_prefixes: List of model prefixes to filter. None means all.

    Returns:
        {model_prefix: [experiment_dirs]}
    """
    discovered = defaultdict(list)

    if not os.path.exists(base_dir):
        console.print(f"[red]Error: directory does not exist: {base_dir}[/red]")
        return discovered

    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        # Skip special directories.
        if entry.startswith('.') or entry in ['backup', 'statistics', 'analysis_output']:
            continue

        model_prefix, config, _ = parse_experiment_name(entry)

        # Filter model prefixes.
        if model_prefixes:
            if not any(entry.startswith(p) for p in model_prefixes):
                continue

        discovered[model_prefix].append(entry_path)

    return discovered


def load_evaluation_results(exp_dir: str, turn: int) -> set:
    """
    Load truly repaired CVE IDs from evaluation/summary.json.

    Returns:
        Set of successfully repaired CVE IDs
    """
    eval_path = os.path.join(exp_dir, f'turn_{turn}', 'evaluation', 'summary.json')
    successful_cves = set()

    if not os.path.exists(eval_path):
        return successful_cves

    try:
        with open(eval_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract all successful CVEs from strict_evaluation.successful_cves.
        strict_eval = data.get('strict_evaluation', {})
        successful_by_lang = strict_eval.get('successful_cves', {})

        for lang, cves in successful_by_lang.items():
            for cve in cves:
                successful_cves.add(cve)

    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        pass

    return successful_cves


def parse_trajectory(traj_path: str, successful_cves: set = None) -> Optional[TrajectoryStats]:
    """
    Parse a single trajectory file.

    Args:
        traj_path: Trajectory file path
        successful_cves: Set of truly repaired CVEs (from evaluation)
    """
    try:
        with open(traj_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError) as e:
        return None

    info = data.get('info', {})
    messages = data.get('messages', [])

    # Extract CVE ID.
    cve_id = Path(traj_path).stem.replace('.traj', '')

    # Exit status.
    exit_status = info.get('exit_status', 'Unknown')
    submitted = exit_status == 'Submitted'

    # True repair success requires evaluation verification.
    if successful_cves is not None:
        repair_success = cve_id in successful_cves
    else:
        # If evaluation data is missing, fall back to submitted status (but marked).
        repair_success = False

    # Model stats.
    model_stats = info.get('model_stats', {})
    total_cost = model_stats.get('instance_cost', 0.0)
    api_calls = model_stats.get('api_calls', 0)

    # Count steps.
    num_steps = sum(1 for msg in messages if msg.get('role') == 'assistant')

    # Count commands and errors.
    commands_count = 0
    errors_count = 0
    verification_attempts = 0

    for msg in messages:
        content = msg.get('content', '')

        # Count bash commands.
        bash_matches = re.findall(r'```bash', content)
        commands_count += len(bash_matches)

        # Detect verification attempts.
        if 'fix-run.sh' in content:
            verification_attempts += 1

        # Detect errors (non-zero return code).
        if '<returncode>' in content:
            non_zero = re.findall(r'<returncode>([1-9]\d*)</returncode>', content)
            errors_count += len(non_zero)

    return TrajectoryStats(
        cve_id=cve_id,
        experiment_name="",
        turn=0,
        exit_status=exit_status,
        submitted=submitted,
        repair_success=repair_success,
        num_steps=num_steps,
        num_api_calls=api_calls,
        total_cost=total_cost,
        commands_count=commands_count,
        errors_count=errors_count,
        verification_attempts=verification_attempts,
        first_submit_success=repair_success and verification_attempts == 1,
        traj_file_path=traj_path
    )


def analyze_experiment(exp_dir: str, max_turn: int = 1) -> Optional[ExperimentStats]:
    """
    Analyze a single experiment directory and accumulate across turns.

    Args:
        exp_dir: Experiment directory path
        max_turn: Maximum turn; accumulates data from turn_1 to turn_max_turn
    """
    exp_name = os.path.basename(exp_dir)
    model_prefix, exp_config, has_exp = parse_experiment_name(exp_name)

    stats = ExperimentStats(
        experiment_name=exp_name,
        model_prefix=model_prefix,
        experience_config=exp_config,
        has_experience=has_exp,
        turn=max_turn
    )

    # Read multi_turn_config.json to get exp_retrieve_k and exp_return_n.
    config_path = os.path.join(exp_dir, 'multi_turn_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                stats.exp_retrieve_k = config.get('exp_retrieve_k', 0)
                stats.exp_return_n = config.get('exp_return_n', 0)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Accumulated counters across turns.
    total_steps = 0
    total_cost = 0.0
    total_api_calls = 0
    total_verifications = 0
    success_steps = 0
    success_cost = 0.0
    success_count = 0
    first_submit_success_count = 0

    # Process each turn.
    for turn in range(1, max_turn + 1):
        # Locate raw_traj directory.
        turn_dir = os.path.join(exp_dir, f'turn_{turn}', 'raw_traj')
        if not os.path.exists(turn_dir):
            continue

        # Load evaluation results to get truly repaired CVE list.
        # If evaluation is missing, the experiment is incomplete; skip it.
        successful_cves = load_evaluation_results(exp_dir, turn)
        eval_path = os.path.join(exp_dir, f'turn_{turn}', 'evaluation', 'summary.json')
        if not os.path.exists(eval_path):
            continue  # Skip incomplete experiment runs.

        # Iterate CVEs.
        for cve_dir in os.listdir(turn_dir):
            cve_path = os.path.join(turn_dir, cve_dir)
            if not os.path.isdir(cve_path):
                continue

            traj_files = [f for f in os.listdir(cve_path) if f.endswith('.traj.json')]
            if not traj_files:
                continue

            traj_path = os.path.join(cve_path, traj_files[0])
            traj = parse_trajectory(traj_path, successful_cves)

            if traj is None:
                continue

            traj.experiment_name = exp_name
            traj.turn = turn

            # In accumulated mode, track whether this CVE already succeeded in prior turns.
            # If already successful earlier, do not count it again in later turns.
            if traj.cve_id in stats.trajectories:
                # Update only if it previously failed but now succeeds.
                prev_traj = stats.trajectories[traj.cve_id]
                if not prev_traj.repair_success and traj.repair_success:
                    # Promote from failure to success and update aggregate stats.
                    stats.successful_cves += 1
                    stats.failed_cves = max(0, stats.failed_cves - 1)
                    success_steps += traj.num_steps
                    success_cost += traj.total_cost
                    success_count += 1
                    if traj.first_submit_success:
                        first_submit_success_count += 1
                    stats.trajectories[traj.cve_id] = traj
                continue  # Already handled; skip.

            stats.trajectories[traj.cve_id] = traj
            stats.total_cves += 1

            # Use true repair-success status.
            if traj.repair_success:
                stats.successful_cves += 1
                success_steps += traj.num_steps
                success_cost += traj.total_cost
                success_count += 1
                if traj.first_submit_success:
                    first_submit_success_count += 1
            elif traj.submitted:
                # Submitted but failed verification.
                stats.failed_cves += 1
            else:
                # No submission (error or timeout).
                stats.error_cves += 1

            total_steps += traj.num_steps
            total_cost += traj.total_cost
            total_api_calls += traj.num_api_calls
            total_verifications += traj.verification_attempts

    # Return None when there is no data.
    if stats.total_cves == 0:
        return None

    # Compute aggregate metrics.
    stats.total_steps = total_steps
    stats.total_cost = total_cost
    stats.first_submit_success_count = first_submit_success_count

    if stats.total_cves > 0:
        stats.avg_steps = total_steps / stats.total_cves
        stats.avg_cost = total_cost / stats.total_cves
        stats.avg_api_calls = total_api_calls / stats.total_cves
        stats.avg_verification_attempts = total_verifications / stats.total_cves

    # Average cost per step.
    if total_steps > 0:
        stats.avg_cost_per_step = total_cost / total_steps

    if success_count > 0:
        stats.success_avg_steps = success_steps / success_count
        stats.success_avg_cost = success_cost / success_count
        # Average cost per step for successful cases.
        if success_steps > 0:
            stats.success_avg_cost_per_step = success_cost / success_steps

    return stats


# ============================================================================
# Paired analysis
# ============================================================================

def create_paired_samples(group: ExperimentGroup) -> List[PairedSample]:
    """Create paired samples for one experiment group."""
    baseline = group.get_baseline()
    if baseline is None:
        return []

    paired = []

    for exp_name, exp_stats in group.experiments.items():
        if not exp_stats.has_experience:
            continue

        # Find shared CVEs.
        common_cves = set(baseline.trajectories.keys()) & set(exp_stats.trajectories.keys())

        for cve_id in common_cves:
            base_traj = baseline.trajectories[cve_id]
            exp_traj = exp_stats.trajectories[cve_id]

            sample = PairedSample(
                cve_id=cve_id,
                model_prefix=group.model_prefix,
                experience_config=exp_stats.experience_config,

                baseline_success=base_traj.repair_success,
                baseline_steps=base_traj.num_steps,
                baseline_cost=base_traj.total_cost,
                baseline_errors=base_traj.errors_count,
                baseline_status=base_traj.exit_status,

                exp_success=exp_traj.repair_success,
                exp_steps=exp_traj.num_steps,
                exp_cost=exp_traj.total_cost,
                exp_errors=exp_traj.errors_count,
                exp_status=exp_traj.exit_status,
            )

            paired.append(sample)

    # Sort by effectiveness score.
    paired.sort(key=lambda x: x.effectiveness_score, reverse=True)

    return paired


# ============================================================================
# Output functions
# ============================================================================

def print_experiment_summary(groups: Dict[str, ExperimentGroup], verbose: bool = False, sort_by_success: bool = False):
    """
    Print experiment summary.

    Args:
        groups: Experiment groups
        verbose: Verbose mode
        sort_by_success: Whether to sort by repaired count (default: k asc, n asc)
    """
    console.print("\n")
    console.print(Panel.fit("[bold cyan]Experiment Group Overview[/bold cyan]", border_style="cyan"))

    for model_prefix, group in groups.items():
        console.print(f"\n[bold magenta]Model: {model_prefix}[/bold magenta]")

        # Build main table.
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        table.add_column("k", style="dim", justify="center")
        table.add_column("n", style="dim", justify="center")
        table.add_column("Config", style="cyan", justify="center")
        table.add_column("CVE Count", justify="right")
        table.add_column("Successes", justify="right")
        table.add_column("Success Rate", justify="right")
        table.add_column("Avg Steps", justify="right")
        table.add_column("Avg Cost", justify="right")
        table.add_column("Total Cost", justify="right")

        # Sort: no-experience first, then by k/n (or by repaired count).
        if sort_by_success:
            sorted_exps = sorted(
                group.experiments.values(),
                key=lambda e: (e.has_experience, -e.successful_cves, e.exp_retrieve_k, e.exp_return_n)
            )
        else:
            sorted_exps = sorted(
                group.experiments.values(),
                key=lambda e: (e.has_experience, e.exp_retrieve_k, e.exp_return_n, e.experience_config)
            )

        for exp in sorted_exps:
            # Color success rate.
            rate = exp.success_rate
            if rate >= 0.7:
                rate_style = "green"
            elif rate >= 0.5:
                rate_style = "yellow"
            else:
                rate_style = "red"

            config_display = exp.experience_config
            if not exp.has_experience:
                config_display = f"[dim]{config_display} (baseline)[/dim]"

            # Display k and n.
            k_display = str(exp.exp_retrieve_k) if exp.exp_retrieve_k > 0 else "-"
            n_display = str(exp.exp_return_n) if exp.exp_return_n > 0 else "-"

            table.add_row(
                k_display,
                n_display,
                config_display,
                str(exp.total_cves),
                str(exp.successful_cves),
                f"[{rate_style}]{exp.success_rate_str}[/{rate_style}]",
                f"{exp.avg_steps:.1f}",
                f"${exp.avg_cost:.4f}",
                f"${exp.total_cost:.2f}"
            )

        console.print(table)

        # Additional: detailed efficiency metrics table.
        console.print("\n[bold]Detailed Efficiency Metrics[/bold]")

        detail_table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
        detail_table.add_column("Config", style="cyan", justify="center")
        detail_table.add_column("Successful Cases\nAvg Steps", justify="right")
        detail_table.add_column("First Success\nCount/Rate", justify="right")
        detail_table.add_column("Cost Per Step\n(All)", justify="right")
        detail_table.add_column("Cost Per Step\n(Successful)", justify="right")

        # Get baseline for comparison.
        baseline = group.get_baseline()

        for exp in sorted_exps:
            config_display = exp.experience_config
            if not exp.has_experience:
                config_display = f"[dim]{config_display} (baseline)[/dim]"

            # Average steps for successful cases.
            success_steps_str = f"{exp.success_avg_steps:.1f}" if exp.successful_cves > 0 else "-"

            # First-submission success rate.
            first_submit_str = f"{exp.first_submit_success_count}/{exp.first_submit_success_rate_str}" if exp.successful_cves > 0 else "-"

            # Cost per step (compared with baseline).
            cost_per_step_str = f"${exp.avg_cost_per_step:.5f}"
            if baseline and exp.has_experience and baseline.avg_cost_per_step > 0:
                diff_pct = (exp.avg_cost_per_step - baseline.avg_cost_per_step) / baseline.avg_cost_per_step * 100
                if diff_pct < -5:
                    cost_per_step_str = f"[green]{cost_per_step_str} ({diff_pct:+.1f}%)[/green]"
                elif diff_pct > 5:
                    cost_per_step_str = f"[red]{cost_per_step_str} ({diff_pct:+.1f}%)[/red]"

            # Cost per step for successful cases.
            success_cost_per_step_str = f"${exp.success_avg_cost_per_step:.5f}" if exp.successful_cves > 0 else "-"

            detail_table.add_row(
                config_display,
                success_steps_str,
                first_submit_str,
                cost_per_step_str,
                success_cost_per_step_str
            )

        console.print(detail_table)


def print_paired_analysis(all_paired: Dict[str, List[PairedSample]]):
    """Print paired-analysis results."""
    console.print("\n")
    console.print(Panel.fit("[bold cyan]Paired Sample Analysis[/bold cyan]", border_style="cyan"))

    for model_prefix, samples in all_paired.items():
        if not samples:
            continue

        console.print(f"\n[bold magenta]Model: {model_prefix}[/bold magenta]")

        # Aggregate counts.
        total = len(samples)
        improved = sum(1 for s in samples if s.success_change > 0)
        degraded = sum(1 for s in samples if s.success_change < 0)
        both_success = sum(1 for s in samples if s.baseline_success and s.exp_success)
        both_fail = sum(1 for s in samples if not s.baseline_success and not s.exp_success)

        # Efficiency improvements.
        efficiency_improved = sum(1 for s in samples if s.step_reduction > 0 and s.baseline_success and s.exp_success)

        console.print(f"  Total paired samples: {total}")
        if total > 0:
            console.print(f"  [green]Experience led to success: {improved}[/green] ({improved/total*100:.1f}%)")
            console.print(f"  [red]Experience led to failure: {degraded}[/red] ({degraded/total*100:.1f}%)")
        console.print(f"  Both successful: {both_success} (efficiency improved in {efficiency_improved})")
        console.print(f"  Both failed: {both_fail}")

        # Show high-value samples.
        high_value = [s for s in samples if s.effectiveness_score >= 40]
        if high_value:
            console.print(f"\n  [bold]High-value samples (score >= 40): {len(high_value)}[/bold]")

            table = Table(box=box.SIMPLE, show_header=True)
            table.add_column("CVE ID", style="cyan")
            table.add_column("Experience Config")
            table.add_column("Baseline", justify="center")
            table.add_column("Experiment", justify="center")
            table.add_column("Step Delta", justify="right")
            table.add_column("Score", justify="right")

            for s in high_value[:10]:
                base_status = "[green]✓[/green]" if s.baseline_success else "[red]✗[/red]"
                exp_status = "[green]✓[/green]" if s.exp_success else "[red]✗[/red]"

                step_change = f"{s.step_reduction:+d}"
                if s.step_reduction > 0:
                    step_change = f"[green]{step_change}[/green]"
                elif s.step_reduction < 0:
                    step_change = f"[red]{step_change}[/red]"

                table.add_row(
                    s.cve_id,
                    s.experience_config,
                    base_status,
                    exp_status,
                    step_change,
                    f"{s.effectiveness_score:.1f}"
                )

            console.print(table)


def generate_markdown_report(
    groups: Dict[str, ExperimentGroup],
    all_paired: Dict[str, List[PairedSample]],
    output_path: str,
    args
) -> str:
    """Generate Markdown report."""
    lines = []

    # Title.
    lines.append("# Experience Database Effectiveness Analysis Report\n")
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Analysis directory: `{args.base_dir}`\n")
    lines.append(f"Model prefixes: {', '.join(args.model_prefix) if args.model_prefix else 'all'}\n")
    lines.append(f"Analyzed turns: Turn 1 ~ Turn {args.max_turn}\n")
    if args.sort_by_success:
        lines.append("Sort mode: by repaired count\n")

    # Overview table.
    lines.append("\n## 1. Experiment Statistics Overview\n")

    for model_prefix, group in groups.items():
        lines.append(f"\n### {model_prefix}\n")
        lines.append("| k | n | Experience Config | CVE Count | Successes | Success Rate | Avg Steps | Avg Cost | Total Cost |")
        lines.append("|---|---|---------|-------|--------|--------|----------|----------|--------|")

        # Sort: no-experience first, then by k/n (or by repaired count).
        if args.sort_by_success:
            sorted_exps = sorted(
                group.experiments.values(),
                key=lambda e: (e.has_experience, -e.successful_cves, e.exp_retrieve_k, e.exp_return_n)
            )
        else:
            sorted_exps = sorted(
                group.experiments.values(),
                key=lambda e: (e.has_experience, e.exp_retrieve_k, e.exp_return_n, e.experience_config)
            )

        for exp in sorted_exps:
            config = exp.experience_config
            if not exp.has_experience:
                config = f"{config} (baseline)"

            # Display k and n.
            k_display = str(exp.exp_retrieve_k) if exp.exp_retrieve_k > 0 else "-"
            n_display = str(exp.exp_return_n) if exp.exp_return_n > 0 else "-"

            lines.append(
                f"| {k_display} | {n_display} | {config} | {exp.total_cves} | {exp.successful_cves} | "
                f"{exp.success_rate_str} | {exp.avg_steps:.1f} | "
                f"${exp.avg_cost:.4f} | ${exp.total_cost:.2f} |"
            )

        # Additional: detailed efficiency metrics table.
        lines.append("\n#### Detailed Efficiency Metrics\n")
        lines.append("| Experience Config | Avg Steps (Successful Cases) | First Success Count | First Success Rate | Cost/Step (All) | Cost/Step (Successful) |")
        lines.append("|---------|----------------|-----------|-----------|---------------|---------------|")

        baseline = group.get_baseline()

        for exp in sorted_exps:
            config = exp.experience_config
            if not exp.has_experience:
                config = f"{config} (baseline)"

            success_steps = f"{exp.success_avg_steps:.1f}" if exp.successful_cves > 0 else "-"
            first_submit_count = str(exp.first_submit_success_count) if exp.successful_cves > 0 else "-"
            first_submit_rate = exp.first_submit_success_rate_str if exp.successful_cves > 0 else "-"
            cost_per_step = f"${exp.avg_cost_per_step:.5f}"
            success_cost_per_step = f"${exp.success_avg_cost_per_step:.5f}" if exp.successful_cves > 0 else "-"

            # Add baseline delta marker.
            if baseline and exp.has_experience and baseline.avg_cost_per_step > 0:
                diff_pct = (exp.avg_cost_per_step - baseline.avg_cost_per_step) / baseline.avg_cost_per_step * 100
                cost_per_step = f"{cost_per_step} ({diff_pct:+.1f}%)"

            lines.append(
                f"| {config} | {success_steps} | {first_submit_count} | "
                f"{first_submit_rate} | {cost_per_step} | {success_cost_per_step} |"
            )

    # Paired analysis.
    lines.append("\n## 2. Paired Sample Analysis\n")

    for model_prefix, samples in all_paired.items():
        if not samples:
            continue

        lines.append(f"\n### {model_prefix}\n")

        total = len(samples)
        improved = sum(1 for s in samples if s.success_change > 0)
        degraded = sum(1 for s in samples if s.success_change < 0)

        lines.append(f"- Total paired samples: {total}")
        if total > 0:
            lines.append(f"- Experience led to success: {improved} ({improved/total*100:.1f}%)")
            lines.append(f"- Experience led to failure: {degraded} ({degraded/total*100:.1f}%)")

        # High-value samples.
        high_value = [s for s in samples if s.effectiveness_score >= 40]
        if high_value:
            lines.append(f"\n#### High-value samples (score >= 40): {len(high_value)}\n")
            lines.append("| CVE ID | Experience Config | Baseline Result | Experiment Result | Step Delta | Cost Delta | Score |")
            lines.append("|--------|---------|----------|----------|----------|----------|------|")

            for s in high_value[:30]:
                base_res = "Success" if s.baseline_success else "Failure"
                exp_res = "Success" if s.exp_success else "Failure"

                lines.append(
                    f"| {s.cve_id} | {s.experience_config} | {base_res} | {exp_res} | "
                    f"{s.step_reduction:+d} | ${s.cost_reduction:.4f} | {s.effectiveness_score:.1f} |"
                )

    # Suggested review list.
    lines.append("\n## 3. Suggested Cases for Review\n")

    # Collect all type-1 samples (experience directly enabled success).
    all_type1 = []
    for samples in all_paired.values():
        all_type1.extend([s for s in samples if s.success_change > 0])

    if all_type1:
        all_type1.sort(key=lambda x: x.effectiveness_score, reverse=True)
        lines.append("\n### Type 1: Experience directly enabled repair success\n")
        lines.append("| CVE ID | Model | Experience Config | Baseline Steps | Experiment Steps | Score |")
        lines.append("|--------|------|---------|----------|----------|------|")

        for s in all_type1[:20]:
            lines.append(
                f"| {s.cve_id} | {s.model_prefix} | {s.experience_config} | "
                f"{s.baseline_steps} | {s.exp_steps} | {s.effectiveness_score:.1f} |"
            )

    # Type 2: Significant efficiency improvement.
    all_type2 = []
    for samples in all_paired.values():
        all_type2.extend([s for s in samples
                         if s.baseline_success and s.exp_success
                         and s.step_reduction_pct >= 0.3])

    if all_type2:
        all_type2.sort(key=lambda x: x.step_reduction_pct, reverse=True)
        lines.append("\n### Type 2: Significant efficiency improvement (step reduction >= 30%)\n")
        lines.append("| CVE ID | Model | Experience Config | Baseline Steps | Experiment Steps | Reduction Ratio |")
        lines.append("|--------|------|---------|----------|----------|----------|")

        for s in all_type2[:20]:
            lines.append(
                f"| {s.cve_id} | {s.model_prefix} | {s.experience_config} | "
                f"{s.baseline_steps} | {s.exp_steps} | {s.step_reduction_pct:.1%} |"
            )

    # Write report to file.
    report_content = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return output_path


# ============================================================================
# Main function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Experience database effectiveness analysis - basic statistical comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all experiments with prefix gpt-5-mini, auto-detect max turn
  %(prog)s --base-dir ./output --model-prefix gpt-5-mini

  # Analyze multiple models up to turn 3
  %(prog)s --base-dir ./output --model-prefix gpt-5-mini deepseek-v3.2 --max-turn 3

  # Sort by repaired count and specify output directory
  %(prog)s --base-dir ./output --model-prefix gpt-5-mini --sort-by-success --output-dir ./results

  # Disable console output and only generate report
  %(prog)s --base-dir ./output --model-prefix gpt-5-mini --no-console
        """
    )

    parser.add_argument(
        '--base-dir', '-d',
        required=True,
        help='Base directory for experiment outputs'
    )

    parser.add_argument(
        '--model-prefix', '-m',
        nargs='+',
        help='Model prefixes to analyze (can provide multiple); analyze all if omitted'
    )

    parser.add_argument(
        '--max-turn', '-t',
        type=int,
        default=None,
        help='Maximum turn; accumulate data from turn_1 to turn_N (default: auto-detect max turn)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Report output directory (default: analysis_output beside base-dir)'
    )

    parser.add_argument(
        '--sort-by-success', '-s',
        action='store_true',
        help='Sort by repaired count (default: sort by config name)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output mode'
    )

    parser.add_argument(
        '--no-console',
        action='store_true',
        help='Disable console output; only generate report'
    )

    args = parser.parse_args()

    # Set output directory.
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.base_dir), 'analysis_output')
    os.makedirs(args.output_dir, exist_ok=True)

    # Discover experiments.
    console.print(Panel.fit("[bold]Experience Database Effectiveness Analysis - Basic Statistical Comparison[/bold]", border_style="blue"))
    console.print(f"Base directory: {args.base_dir}")
    console.print(f"Model prefixes: {args.model_prefix or 'all'}")

    # Auto-detect max turn or use explicit value.
    if args.max_turn is None:
        args.max_turn = detect_max_turn(args.base_dir)
        console.print(f"Auto-detected maximum turn: Turn {args.max_turn}")
    else:
        console.print(f"Specified analysis turns: Turn 1 ~ Turn {args.max_turn}")

    discovered = discover_experiments(args.base_dir, args.model_prefix)

    if not discovered:
        console.print("[red]No matching experiment directories found[/red]")
        return

    console.print(f"Discovered {len(discovered)} model groups and {sum(len(v) for v in discovered.values())} experiments")

    # Analyze experiments.
    groups: Dict[str, ExperimentGroup] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing experiments...", total=sum(len(v) for v in discovered.values()))

        for model_prefix, exp_dirs in discovered.items():
            group = ExperimentGroup(model_prefix=model_prefix)

            for exp_dir in exp_dirs:
                exp_name = os.path.basename(exp_dir)
                progress.update(task, description=f"Analyzing {exp_name}...")

                stats = analyze_experiment(exp_dir, args.max_turn)
                if stats and stats.total_cves > 0:
                    group.experiments[exp_name] = stats

                progress.advance(task)

            if group.experiments:
                groups[model_prefix] = group

    if not groups:
        console.print("[red]No valid experiment data found[/red]")
        return

    # Paired analysis.
    console.print("\n[cyan]Running paired analysis...[/cyan]")
    all_paired: Dict[str, List[PairedSample]] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Paired analysis...", total=len(groups))

        for model_prefix, group in groups.items():
            progress.update(task, description=f"Paired analysis for {model_prefix}...")
            paired = create_paired_samples(group)
            all_paired[model_prefix] = paired
            progress.advance(task)

    # Console output.
    if not args.no_console:
        print_experiment_summary(groups, args.verbose, args.sort_by_success)
        print_paired_analysis(all_paired)

    # Generate report.
    console.print("\n[cyan]Generating Markdown report...[/cyan]")
    report_path = os.path.join(args.output_dir, 'stats_comparison_report.md')
    generate_markdown_report(groups, all_paired, report_path, args)
    console.print(f"[green]✓ Report saved: {report_path}[/green]")

    # Save paired samples CSV.
    console.print("[cyan]Saving paired sample data...[/cyan]")
    csv_path = os.path.join(args.output_dir, 'paired_samples.csv')
    all_samples = []
    for samples in all_paired.values():
        all_samples.extend(samples)

    if all_samples:
        import csv
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Saving CSV...", total=len(all_samples))

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'cve_id', 'model', 'experience_config',
                    'baseline_success', 'baseline_steps', 'baseline_cost',
                    'exp_success', 'exp_steps', 'exp_cost',
                    'success_change', 'step_reduction', 'cost_reduction',
                    'effectiveness_score'
                ])
                for s in all_samples:
                    writer.writerow([
                        s.cve_id, s.model_prefix, s.experience_config,
                        s.baseline_success, s.baseline_steps, s.baseline_cost,
                        s.exp_success, s.exp_steps, s.exp_cost,
                        s.success_change, s.step_reduction, s.cost_reduction,
                        s.effectiveness_score
                    ])
                    progress.advance(task)

        console.print(f"[green]✓ Paired samples saved: {csv_path}[/green]")

    # Summary.
    console.print("\n[bold green]✓ Analysis complete[/bold green]")
    console.print(f"[cyan]Found {len(groups)} model groups and {sum(len(g.experiments) for g in groups.values())} experiments[/cyan]")
    console.print(f"[cyan]Total paired samples: {len(all_samples)}[/cyan]")


if __name__ == '__main__':
    main()
