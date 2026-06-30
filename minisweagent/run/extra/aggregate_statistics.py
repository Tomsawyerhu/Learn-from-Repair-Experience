#!/usr/bin/env python3
"""
Aggregate statistics script for multi-turn experiments.

Collects metadata_exp-level statistics and produces:
- JSON summaries for persistence
- PNG charts for visualization
- Markdown analysis reports
"""

import json
import glob
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def load_cve_language_map(input_json_path: Optional[Path]) -> Dict[str, str]:
    """
    Load the CVE-to-language mapping from input.json for fallback handling
    
    Args:
        input_json_path: Path to the input.json file
    
    Returns:
        {cve_id: language} mapping dictionary
    """
    if not input_json_path or not input_json_path.exists():
        return {}
    
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cve_lang_map = {}
        if isinstance(data, list):
            for item in data:
                cve_id = item.get("cve_id")
                # Note: the field name is programing_language (missing one m)
                lang = item.get("programing_language") or item.get("programming_language")
                if cve_id and lang:
                    cve_lang_map[cve_id] = lang
        
        console.print(f"[cyan]Loaded {len(cve_lang_map)} CVE language mappings for fallback handling[/cyan]")
        return cve_lang_map
    
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to load language mapping: {e}[/yellow]")
        return {}


def load_evaluation_summary(turn_dir: Path) -> Optional[Dict]:
    """Load evaluation results (evaluation/summary.json)"""
    summary_path = turn_dir / "evaluation" / "summary.json"
    if not summary_path.exists():
        return None
    
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[red]Error loading {summary_path}: {e}[/red]")
        return None


def load_trajectory_costs(turn_dir: Path) -> Dict:
    """Load cost and API-call statistics (averages and totals only)"""
    raw_traj_dir = turn_dir / "raw_traj"
    if not raw_traj_dir.exists():
        return {"mean": 0, "total": 0, "api_calls_mean": 0}
    
    costs = []
    api_calls = []
    
    for traj_file in raw_traj_dir.rglob("*.traj.json"):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            info = data.get("info", {})
            stats = info.get("model_stats") or data.get("model_stats", {})
            
            cost = stats.get("instance_cost")
            calls = stats.get("api_calls")
            
            if cost is not None:
                costs.append(cost)
            if calls is not None:
                api_calls.append(calls)
        except Exception:
            continue
    
    return {
        "mean": float(np.mean(costs)) if costs else 0.0,
        "total": float(np.sum(costs)) if costs else 0.0,
        "api_calls_mean": float(np.mean(api_calls)) if api_calls else 0.0
    }


def load_experience_statistics(turn_dir: Path, cve_lang_map: Optional[Dict[str, str]] = None) -> Optional[Dict]:
    """
    Load experience-injection statistics (run_statistics.json) with language fallback
    
    Args:
        turn_dir: Turn directory
        cve_lang_map: Optional CVE-to-language mapping used for fallback
    
    Returns:
        Experience statistics, including language distribution
    """
    stats_path = turn_dir / "run_statistics.json"
    if not stats_path.exists():
        return None
    
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Recompute even if summary exists so cold-start categories are handled correctly (older summaries may not include them)
        # if "summary" in data: ... (Removed)
        
        success_exp_counts = []
        failure_exp_counts = []
        cold_start_exp_counts = []  # new: cold-start experience
        cross_lang_counts = []
        quality_scores = []
        similarity_scores = []
        
        # Language distribution statistics
        lang_distribution = defaultdict(lambda: {"count": 0, "similarities": []})
        
        # Determine the records to process
        cve_records = []
        if "cve_records" in data and isinstance(data["cve_records"], list):
            cve_records = data["cve_records"]
        else:
            # Try handling the payload as {cve_id: data}, excluding non-dict items like "summary" or entries without experiences
            for k, v in data.items():
                if isinstance(v, dict) and "retrieved_experiences" in v:
                    # Ensure programming_language is present; if missing from v, fallback handling is required
                    cve_records.append(v)

        cve_count = 0
        for cve_data in cve_records:
            cve_count += 1
            current_cve_lang = cve_data.get("programming_language") or cve_data.get("programing_language", "")
            retrieved_exps = cve_data.get("retrieved_experiences", [])
            
            if not retrieved_exps:
                continue
            
            success_count = 0
            failure_count = 0
            cold_start_count = 0
            cross_lang_count = 0
            
            
            for exp in retrieved_exps:
                if not isinstance(exp, dict):
                    continue
                
                # Resolve the experience language with fallback handling
                exp_lang = exp.get("programming_language") or exp.get("programing_language", "")
                if not exp_lang and cve_lang_map:
                    exp_cve_id = exp.get("cve_id", "")
                    exp_lang = cve_lang_map.get(exp_cve_id, "")
                
                # Check whether this is cold-start experience
                tags = exp.get("tag", []) or exp.get("tags", [])
                
                is_cold_start = False
                if isinstance(tags, list):
                    # Check for exactly "cold_start" or case-insensitive match if needed
                    if "cold_start" in tags or "Cold_Start" in tags:
                        is_cold_start = True
                
                if is_cold_start:
                    cold_start_count += 1
                else:
                    # Count successful and failed experiences
                    repair_result = exp.get("repair_result", "")
                    if "success" in repair_result.lower():
                        success_count += 1
                    else:
                        failure_count += 1
                
                # Check whether the experience is cross-language
                if exp_lang and current_cve_lang and exp_lang != current_cve_lang:
                    cross_lang_count += 1
                
                # Collect quality and similarity scores
                quality_scores.append(exp.get("quality_score", 0))
                similarity = exp.get("similarity_score", 0.0)
                similarity_scores.append(similarity)
                
                # Language distribution statistics
                if exp_lang:
                    lang_distribution[exp_lang]["count"] += 1
                    lang_distribution[exp_lang]["similarities"].append(similarity)
            
            success_exp_counts.append(success_count)
            failure_exp_counts.append(failure_count)
            cold_start_exp_counts.append(cold_start_count)
            cross_lang_counts.append(cross_lang_count)
        
        # Compute average similarity by language
        final_lang_dist = {}
        for lang, stats in lang_distribution.items():
            sims = stats["similarities"]
            final_lang_dist[lang] = {
                "count": stats["count"],
                "avg_similarity": round(sum(sims) / len(sims), 4) if sims else 0.0
            }
        
        if cve_count == 0:
            return None
        
        return {
            "avg_success_exp": round(sum(success_exp_counts) / len(success_exp_counts), 2) if success_exp_counts else 0.0,
            "avg_failure_exp": round(sum(failure_exp_counts) / len(failure_exp_counts), 2) if failure_exp_counts else 0.0,
            "avg_cold_start_exp": round(sum(cold_start_exp_counts) / len(cold_start_exp_counts), 2) if cold_start_exp_counts else 0.0,
            "avg_cross_language_exp": round(sum(cross_lang_counts) / len(cross_lang_counts), 2) if cross_lang_counts else 0.0,
            "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 2) if quality_scores else 0.0,
            "avg_similarity_score": round(sum(similarity_scores) / len(similarity_scores), 4) if similarity_scores else 0.0,
            "language_distribution": final_lang_dist
        }
    
    except Exception as e:
        console.print(f"[yellow]Warning: load_experience_statistics failed for {turn_dir}: {e}[/yellow]")
        return None


def load_language_stats(root_dir: Path) -> Dict:
    """Load language baseline data (language_stats.json)

    Priority:
    1. Load from root_dir/language_stats.json
    2. Compute dynamically from data/patcheval_dataset.jsonl
    3. Fall back to hard-coded defaults as a last resort
    """
    lang_stats_path = root_dir / "language_stats.json"
    if lang_stats_path.exists():
        try:
            with open(lang_stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                lang_baseline = data.get("poc_instances_by_language", {})
                lang_baseline["total"] = sum(lang_baseline.values())
                return lang_baseline
        except Exception as e:
            console.print(f"[red]Error loading language stats: {e}[/red]")

    # Try computing baselines dynamically from the dataset
    dataset_path = root_dir.parent / "data" / "patcheval_dataset.jsonl"
    if not dataset_path.exists():
        # Try other possible paths
        alt_paths = [
            Path(__file__).parent.parent.parent / "data" / "patcheval_dataset.jsonl",
            root_dir / ".." / "data" / "patcheval_dataset.jsonl",
        ]
        for p in alt_paths:
            if p.exists():
                dataset_path = p
                break

    if dataset_path.exists():
        try:
            lang_counts = defaultdict(int)
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        lang = item.get('programing_language', 'Unknown')
                        lang_counts[lang] += 1

            lang_baseline = dict(lang_counts)
            lang_baseline["total"] = sum(lang_counts.values())
            console.print(f"[cyan]Computed language baselines dynamically from the dataset: {lang_baseline}[/cyan]")
            return lang_baseline
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to compute language baselines from the dataset: {e}[/yellow]")

    # Finally use hard-coded defaults (already corrected)
    console.print(f"[yellow]Warning: language_stats.json not found at {lang_stats_path}, using defaults[/yellow]")
    return {"Python": 70, "Go": 83, "JavaScript": 77, "total": 230}  # default values (already corrected)


def count_turn_instances(turn_dir: Path) -> int:
    """Count the total number of instances in the current turn (traj.json files under raw_traj)"""
    raw_traj_dir = turn_dir / "raw_traj"
    if not raw_traj_dir.exists():
        return 0
    
    return len(list(raw_traj_dir.rglob("*.traj.json")))


def count_execution_statuses(turn_dir: Path) -> Dict:
    """Count execution statuses"""
    raw_traj_dir = turn_dir / "raw_traj"
    if not raw_traj_dir.exists():
        return {"submitted": 0, "limits_exceeded": 0, "timeout": 0}
    
    submitted = 0
    limits_exceeded = 0
    timeout = 0
    
    for traj_file in raw_traj_dir.rglob("*.traj.json"):
        try:
            with open(traj_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info = data.get("info", {})
            exit_status = info.get("exit_status") or data.get("exit_status")
            submission = info.get("submission") or data.get("submission", "")
            
            if exit_status == "Submitted":
                submitted += 1
            elif exit_status == "LimitsExceeded":
                limits_exceeded += 1
            elif exit_status == "Error" and "timed out after" in str(submission):
                timeout += 1
        except Exception:
            continue
    
    return {
        "submitted": submitted,
        "limits_exceeded": limits_exceeded,
        "timeout": timeout
    }


def aggregate_all_turns(root_dir: Path, language_baseline: Dict, cve_lang_map: Optional[Dict[str, str]] = None) -> Dict:
    """Aggregate data across turns and compute cumulative pass rates"""
    turns_data = {}
    cumulative_success_cves = set()
    # Aggregate deduplicated successful CVEs by language
    lang_cumulative_cves = {"Python": set(), "Go": set(), "JavaScript": set()}
    
    # Find all turn directories
    turn_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("turn_")],
                      key=lambda x: int(x.name.split("_")[-1]))
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task(f"[cyan]Processing {len(turn_dirs)} turns...", total=len(turn_dirs))
        
        for turn_dir in turn_dirs:
            turn_name = turn_dir.name
            console.print(f"\n[bold blue]Processing {turn_name}...[/bold blue]")
            
            # Load data
            eval_summary = load_evaluation_summary(turn_dir)
            costs = load_trajectory_costs(turn_dir)
            experience = load_experience_statistics(turn_dir, cve_lang_map)  # pass the language mapping
            total_instances = count_turn_instances(turn_dir)
            execution = count_execution_statuses(turn_dir)
            
            if not eval_summary:
                console.print(f"[yellow]Skipping {turn_name}: no evaluation summary[/yellow]")
                progress.advance(task)
                continue
            
            strict_eval = eval_summary.get("strict_evaluation", {})
            total_cases = strict_eval.get("total_cases", 0)
            total_success = strict_eval.get("total_success", 0)
            pass_rate = strict_eval.get("pass_rate", "0%")
            success_breakdown = strict_eval.get("success_breakdown", {})
            
            # Parse pass_rate, which may be a string like "57.61%" or a number
            if isinstance(pass_rate, str):
                pass_rate = float(pass_rate.replace("%", ""))
            else:
                pass_rate = float(pass_rate)
            
            # Compute per-turn relative pass rate
            pass_rate_vs_instances = (total_success / total_instances * 100) if total_instances > 0 else 0.0
            
            # Compute pass rate among submitted patches
            submitted = execution["submitted"]
            pass_rate_vs_submitted = (total_success / submitted * 100) if submitted > 0 else 0.0
            
            # Update cumulative successful CVEs
            successful_cves_list = strict_eval.get("successful_cves", {})
            for lang, lang_cves in successful_cves_list.items():
                if isinstance(lang_cves, list):
                    # Update the overall cumulative count
                    cumulative_success_cves.update(lang_cves)
                    # Update per-language cumulative counts
                    if lang in lang_cumulative_cves:
                        lang_cumulative_cves[lang].update(lang_cves)
            
            cumulative_count = len(cumulative_success_cves)
            cumulative_rate = (cumulative_count / language_baseline["total"] * 100) if language_baseline["total"] > 0 else 0.0
            
            # Compute pass rates by language
            language_pass_rates = {}
            for lang in ["Python", "Go", "JavaScript"]:
                success = success_breakdown.get(lang, 0)
                total = language_baseline.get(lang, 1)
                lang_pass_rate = (success / total * 100) if total > 0 else 0.0
                language_pass_rates[lang] = {
                    "success": success,
                    "total": total,
                    "pass_rate": round(lang_pass_rate, 2)
                }
            
            # Build per-turn data
            turns_data[turn_name] = {
                "turn_name": turn_name,
                "total_instances": total_instances,
                "evaluation": {
                    "strict": {
                        "total_cases": total_cases,
                        "total_success": total_success,
                        "pass_rate": round(pass_rate, 2),
                        "pass_rate_vs_instances": round(pass_rate_vs_instances, 2),
                        "pass_rate_vs_submitted": round(pass_rate_vs_submitted, 2),
                        "success_breakdown": success_breakdown
                    },
                    "language_pass_rates": language_pass_rates,
                    "cumulative": {
                        "count": cumulative_count,
                        "rate": round(cumulative_rate, 2)
                    }
                },
                "costs": costs,
                "execution": execution,
                "experience": experience
            }
            
            progress.advance(task)
    
    # Compute summary data
    total_cost = sum(turn["costs"]["total"] for turn in turns_data.values())
    
    # Identify the best-performing language based on deduplicated cumulative successful CVEs
    best_lang = None
    best_lang_success = 0
    best_lang_rate = 0.0
    
    # Compute the cumulative pass rate for each language
    for lang in ["Python", "Go", "JavaScript"]:
        total = language_baseline.get(lang, 1)
        cumulative_success = len(lang_cumulative_cves[lang])
        rate = (cumulative_success / total * 100) if total > 0 else 0.0
        if rate > best_lang_rate:
            best_lang = lang
            best_lang_success = cumulative_success
            best_lang_rate = rate
    
    # Add cumulative successful counts by language to summary_data
    lang_cumulative_stats = {
        lang: {
            "cumulative_success": len(lang_cumulative_cves[lang]),
            "total": language_baseline.get(lang, 0),
            "cumulative_rate": (len(lang_cumulative_cves[lang]) / language_baseline.get(lang, 1) * 100) if language_baseline.get(lang, 0) > 0 else 0.0
        }
        for lang in ["Python", "Go", "JavaScript"]
    }
    
    summary_data = {
        "total_turns": len(turns_data),
        "cumulative_success": {
            turn_name: {
                "strict": turn_data["evaluation"]["strict"]["total_success"],
                "cumulative_count": turn_data["evaluation"]["cumulative"]["count"],
                "cumulative_rate": turn_data["evaluation"]["cumulative"]["rate"]
            }
            for turn_name, turn_data in turns_data.items()
        },
        "total_cost": round(total_cost, 2),
        "best_language": {
            "language": best_lang,
            "total_success": best_lang_success,
            "pass_rate": round(best_lang_rate, 2)
        } if best_lang else None,
        "language_cumulative_stats": lang_cumulative_stats
    }
    
    return {
        "generated_at": datetime.now().isoformat(),
        "language_baseline": language_baseline,
        "turns": turns_data,
        "summary": summary_data
    }


def generate_repair_effectiveness_chart(stats: Dict, output_dir: Path) -> Path:
    """Generate the repair-effectiveness trend chart (2x2 subplots)"""
    turns = stats["turns"]
    turn_names = sorted(turns.keys(), key=lambda x: int(x.split("_")[-1]))
    turn_labels = [f"Turn {x.split('_')[-1]}" for x in turn_names]
    
    # Extract data
    strict_pass_rates = [turns[t]["evaluation"]["strict"]["pass_rate"] for t in turn_names]
    instance_pass_rates = [turns[t]["evaluation"]["strict"]["pass_rate_vs_instances"] for t in turn_names]
    cumulative_counts = [turns[t]["evaluation"]["cumulative"]["count"] for t in turn_names]
    cumulative_rates = [turns[t]["evaluation"]["cumulative"]["rate"] for t in turn_names]
    submitted_pass_rates = [turns[t]["evaluation"]["strict"]["pass_rate_vs_submitted"] for t in turn_names]
    submitted_counts = [turns[t]["execution"]["submitted"] for t in turn_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Repair Effectiveness Across Turns / Repair Effectiveness Across Turns', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Subplot (a): strict pass-rate line chart
    ax1.plot(turn_labels, strict_pass_rates, marker='o', linewidth=2.5, markersize=8, 
             color='#2ecc71', label='Strict Pass Rate')
    ax1.set_title('(a) Strict Pass Rate Trend / Strict pass rate trend', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pass Rate (%)', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(strict_pass_rates) * 1.2)
    # Annotate each point with its value
    for i, (x, y) in enumerate(zip(turn_labels, strict_pass_rates)):
        ax1.text(i, y + 1, f'{y:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot (b): per-turn relative pass-rate line chart
    ax2.plot(turn_labels, instance_pass_rates, marker='s', linewidth=2.5, markersize=8, 
             color='#3498db', label='Pass Rate vs Instances')
    ax2.set_title('(b) Pass Rate vs Turn Instances / Pass rate vs turn instances', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pass Rate (%)', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    # Annotate turn instance counts
    for i, (x, y, t) in enumerate(zip(turn_labels, instance_pass_rates, turn_names)):
        instances = turns[t]["total_instances"]
        ax2.text(i, y + 1, f'{y:.1f}%\n(n={instances})', ha='center', va='bottom', fontsize=8)
    
    # Subplot (c): dual-axis chart for cumulative success count and rate
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(turn_labels, cumulative_counts, marker='o', linewidth=2.5, markersize=8, 
                     color='#e74c3c', label='Cumulative Count')
    line2 = ax3_twin.plot(turn_labels, cumulative_rates, marker='^', linewidth=2.5, markersize=8, 
                          color='#9b59b6', label='Cumulative Rate', linestyle='--')
    ax3.set_title('(c) Cumulative Success / Cumulative success statistics', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Count / Cumulative count', fontsize=10, color='#e74c3c')
    ax3_twin.set_ylabel('Cumulative Rate (%) / Cumulative pass rate', fontsize=10, color='#9b59b6')
    ax3.tick_params(axis='y', labelcolor='#e74c3c')
    ax3_twin.tick_params(axis='y', labelcolor='#9b59b6')
    ax3.grid(True, alpha=0.3, linestyle='--')
    # Annotate the final cumulative values
    final_count = cumulative_counts[-1]
    final_rate = cumulative_rates[-1]
    ax3.text(len(turn_labels)-1, final_count + 2, f'{final_count}', 
             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#e74c3c')
    ax3_twin.text(len(turn_labels)-1, final_rate + 1, f'{final_rate:.1f}%', 
                  ha='center', va='bottom', fontsize=9, fontweight='bold', color='#9b59b6')
    # Merge legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # Subplot (d): pass rate among submitted patches
    bars = ax4.bar(turn_labels, submitted_pass_rates, color='#f39c12', alpha=0.7, edgecolor='black')
    ax4.set_title('(d) Pass Rate vs Submitted / Pass Rate vs Submitted', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Pass Rate (%)', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    # Annotate submitted-success counts
    for i, (bar, rate, count) in enumerate(zip(bars, submitted_pass_rates, submitted_counts)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 1, 
                f'{rate:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    chart_path = output_dir / 'repair_effectiveness.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path


def generate_language_analysis_chart(stats: Dict, output_dir: Path) -> Path:
    """Generate the language-specific analysis chart (1x2 subplots)"""
    turns = stats["turns"]
    turn_names = sorted(turns.keys(), key=lambda x: int(x.split("_")[-1]))
    turn_labels = [f"Turn {x.split('_')[-1]}" for x in turn_names]
    languages = ["Python", "Go", "JavaScript"]
    lang_colors = {"Python": '#3498db', "Go": '#2ecc71', "JavaScript": '#f39c12'}
    
    # Extract successful counts by language and turn
    lang_success_data = {lang: [] for lang in languages}
    lang_pass_rates = {lang: [] for lang in languages}
    lang_totals = stats["language_baseline"]
    
    for t in turn_names:
        for lang in languages:
            success = turns[t]["evaluation"]["language_pass_rates"][lang]["success"]
            pass_rate = turns[t]["evaluation"]["language_pass_rates"][lang]["pass_rate"]
            lang_success_data[lang].append(success)
            lang_pass_rates[lang].append(pass_rate)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Language-Specific Analysis / Language-Specific Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Subplot (a): grouped bar chart of successful counts by language and turn
    x = np.arange(len(turn_labels))
    width = 0.25
    
    for i, lang in enumerate(languages):
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, lang_success_data[lang], width, 
                      label=lang, color=lang_colors[lang], alpha=0.8)
    
    ax1.set_title('(a) Success Count by Language / Success Count by Language', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Count / Success count', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(turn_labels)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Subplot (b): pass-rate line chart by language
    for lang in languages:
        ax2.plot(turn_labels, lang_pass_rates[lang], marker='o', linewidth=2.5, 
                markersize=8, label=f'{lang} (n={lang_totals[lang]})', color=lang_colors[lang])
    
    ax2.set_title('(b) Pass Rate by Language / Pass Rate by Language\n(relative to the total number of cases in each language)', 
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('Pass Rate (%) / pass rate', fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    chart_path = output_dir / 'language_analysis.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path


def generate_cost_efficiency_chart(stats: Dict, output_dir: Path) -> Path:
    """Generate the cost-efficiency chart (1x3 subplots)"""
    turns = stats["turns"]
    turn_names = sorted(turns.keys(), key=lambda x: int(x.split("_")[-1]))
    turn_labels = [f"Turn {x.split('_')[-1]}" for x in turn_names]
    
    # Extract data
    avg_costs = [turns[t]["costs"]["mean"] for t in turn_names]
    total_costs = [turns[t]["costs"]["total"] for t in turn_names]
    avg_api_calls = [turns[t]["costs"]["api_calls_mean"] for t in turn_names]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Cost and Efficiency Metrics / Cost and Efficiency Metrics', 
                 fontsize=16, fontweight='bold')
    
    # Subplot (a): average-cost trend line chart
    ax1.plot(turn_labels, avg_costs, marker='o', linewidth=2.5, markersize=8, color='#e74c3c')
    ax1.set_title('(a) Average Cost per CVE / Average Cost per CVE', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cost (USD) / Cost', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    # Annotate average values
    overall_avg = np.mean(avg_costs)
    ax1.axhline(y=overall_avg, color='gray', linestyle='--', alpha=0.5, label=f'Overall Avg: ${overall_avg:.4f}')
    ax1.legend(loc='upper right', fontsize=9)
    for i, (x, y) in enumerate(zip(turn_labels, avg_costs)):
        ax1.text(i, y + max(avg_costs)*0.02, f'${y:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Subplot (b): total-cost bar chart
    bars = ax2.bar(turn_labels, total_costs, color='#3498db', alpha=0.7, edgecolor='black')
    ax2.set_title('(b) Total Cost per Turn / Total Cost per Turn', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Cost (USD) / Total cost', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    # Annotate per-turn total cost
    for bar, cost in zip(bars, total_costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + max(total_costs)*0.02, 
                f'${cost:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Subplot (c): average API-call line chart
    ax3.plot(turn_labels, avg_api_calls, marker='s', linewidth=2.5, markersize=8, color='#9b59b6')
    ax3.set_title('(c) Average API Calls / Average API Calls', fontsize=12, fontweight='bold')
    ax3.set_ylabel('API Calls / API calls', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    # Annotate the highest and lowest values
    max_idx = np.argmax(avg_api_calls)
    min_idx = np.argmin(avg_api_calls)
    ax3.text(max_idx, avg_api_calls[max_idx] + max(avg_api_calls)*0.02, 
            f'Max: {avg_api_calls[max_idx]:.1f}', ha='center', va='bottom', 
            fontsize=8, color='red', fontweight='bold')
    ax3.text(min_idx, avg_api_calls[min_idx] - max(avg_api_calls)*0.05, 
            f'Min: {avg_api_calls[min_idx]:.1f}', ha='center', va='top', 
            fontsize=8, color='blue', fontweight='bold')
    
    plt.tight_layout()
    chart_path = output_dir / 'cost_efficiency.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path


def generate_summary_dashboard(stats: Dict, output_dir: Path) -> Path:
    """Generate the summary dashboard (card-style)"""
    summary = stats["summary"]
    total_turns = summary["total_turns"]
    final_turn = sorted(stats["turns"].keys(), key=lambda x: int(x.split("_")[-1]))[-1]
    
    cumulative_count = stats["turns"][final_turn]["evaluation"]["cumulative"]["count"]
    cumulative_rate = stats["turns"][final_turn]["evaluation"]["cumulative"]["rate"]
    total_cost = summary["total_cost"]
    best_lang = summary["best_language"]
    
    # Identify the best turn (highest single-turn strict pass rate)
    best_turn = None
    best_turn_rate = 0
    for t, data in stats["turns"].items():
        rate = data["evaluation"]["strict"]["pass_rate"]
        if rate > best_turn_rate:
            best_turn = t
            best_turn_rate = rate
    
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Summary Dashboard / Summary Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3, 
                          left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Card style
    card_color = '#ecf0f1'
    value_color = '#2c3e50'
    label_color = '#7f8c8d'
    
    # Card 1: total successes / total pass rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.6, facecolor=card_color, edgecolor='#bdc3c7', linewidth=2))
    ax1.text(0.5, 0.65, f'{cumulative_count}', ha='center', va='center', 
             fontsize=60, fontweight='bold', color=value_color)
    ax1.text(0.5, 0.38, f'Cumulative Success', ha='center', va='center', 
             fontsize=14, color=label_color)
    ax1.text(0.5, 0.28, f'{cumulative_count} / {stats["language_baseline"]["total"]} CVEs ({cumulative_rate:.1f}%)', 
             ha='center', va='center', fontsize=12, color=label_color)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Card 2: total cost
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.6, facecolor=card_color, edgecolor='#bdc3c7', linewidth=2))
    ax2.text(0.5, 0.65, f'${total_cost:.2f}', ha='center', va='center', 
             fontsize=50, fontweight='bold', color=value_color)
    ax2.text(0.5, 0.35, f'Total Cost Across All Turns', ha='center', va='center', 
             fontsize=14, color=label_color)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Card 3: best language
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    ax3.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.6, facecolor=card_color, edgecolor='#bdc3c7', linewidth=2))
    if best_lang:
        ax3.text(0.5, 0.65, f'{best_lang["language"]}', ha='center', va='center', 
                fontsize=50, fontweight='bold', color=value_color)
        ax3.text(0.5, 0.38, f'Best Language', ha='center', va='center', 
                fontsize=14, color=label_color)
        ax3.text(0.5, 0.28, f'{best_lang["pass_rate"]:.1f}% Pass Rate', 
                ha='center', va='center', fontsize=12, color=label_color)
    else:
        ax3.text(0.5, 0.5, 'N/A', ha='center', va='center', 
                fontsize=40, color=label_color)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # Card 4: best turn
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.add_patch(plt.Rectangle((0.1, 0.2), 0.8, 0.6, facecolor=card_color, edgecolor='#bdc3c7', linewidth=2))
    ax4.text(0.5, 0.65, f'Turn {best_turn.split("_")[-1]}', ha='center', va='center', 
             fontsize=50, fontweight='bold', color=value_color)
    ax4.text(0.5, 0.38, f'Best Turn (Highest Pass Rate)', ha='center', va='center', 
             fontsize=14, color=label_color)
    ax4.text(0.5, 0.28, f'{best_turn_rate:.1f}% Strict Pass Rate', 
             ha='center', va='center', fontsize=12, color=label_color)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    chart_path = output_dir / 'summary_dashboard.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path


def generate_experience_impact_chart(stats: Dict, output_dir: Path) -> Path:
    """Generate the experience-injection impact chart (2x2 subplots)"""
    turns = stats["turns"]
    turn_names = sorted(turns.keys(), key=lambda x: int(x.split("_")[-1]))
    
    # Filter turns that contain experience data (Turn 2+)
    exp_turn_names = [tn for tn in turn_names if turns[tn]["experience"] is not None]
    
    if not exp_turn_names:
        # No experience data available; create an empty placeholder chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No Experience Data Available\nExperience data unavailable', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        chart_path = output_dir / 'experience_impact.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        return chart_path
    
    exp_turn_labels = [f"Turn {tn.split('_')[-1]}" for tn in exp_turn_names]
    
    # Extract data
    avg_success_exp = [turns[tn]["experience"]["avg_success_exp"] for tn in exp_turn_names]
    avg_failure_exp = [turns[tn]["experience"]["avg_failure_exp"] for tn in exp_turn_names]
    avg_cold_start_exp = [turns[tn]["experience"].get("avg_cold_start_exp", 0.0) for tn in exp_turn_names]  # Added
    avg_cross_lang_exp = [turns[tn]["experience"]["avg_cross_language_exp"] for tn in exp_turn_names]
    avg_quality = [turns[tn]["experience"]["avg_quality_score"] for tn in exp_turn_names]
    avg_similarity = [turns[tn]["experience"]["avg_similarity_score"] for tn in exp_turn_names]
    
    # Corresponding pass-rate data
    pass_rates = [turns[tn]["evaluation"]["strict"]["pass_rate"] for tn in exp_turn_names]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experience Injection Impact Analysis / Experience Injection Impact Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Subplot (a): trend of successful, failed, and cold-start experiences
    x = np.arange(len(exp_turn_labels))
    width = 0.25  # Adjust width to fit three bars
    
    bars1 = ax1.bar(x - width, avg_cold_start_exp, width, label='Cold-start experience', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x, avg_success_exp, width, label='Successful experience', color='#2ecc71', alpha=0.8)
    bars3 = ax1.bar(x + width, avg_failure_exp, width, label='Failed experience', color='#e74c3c', alpha=0.8)
    
    ax1.set_title('(a) Average Experience Count / Average Experience Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Avg experience count / Avg Count', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_turn_labels)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Annotate values
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Annotate only values greater than 0 to reduce clutter
                ax1.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Subplot (b): cross-language experience trend
    ax2.plot(exp_turn_labels, avg_cross_lang_exp, marker='o', linewidth=2.5, 
             markersize=8, color='#9b59b6', label='Cross-language experience')
    ax2.set_title('(b) Cross-Language Experience / Cross-Language Experience', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average count / Avg Count', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper left', fontsize=10)
    # Annotate values
    for i, (x, y) in enumerate(zip(exp_turn_labels, avg_cross_lang_exp)):
        ax2.text(i, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Subplot (c): experience quality scores and similarity
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(exp_turn_labels, avg_quality, marker='s', linewidth=2.5, 
                     markersize=8, color='#f39c12', label='Quality score')
    line2 = ax3_twin.plot(exp_turn_labels, avg_similarity, marker='^', linewidth=2.5, 
                          markersize=8, color='#3498db', label='Similarity', linestyle='--')
    ax3.set_title('(c) Quality & Similarity / Quality & Similarity', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Quality score (1-5) / Quality Score', fontsize=10, color='#f39c12')
    ax3_twin.set_ylabel('Similarity (0-1) / Similarity', fontsize=10, color='#3498db')
    ax3.tick_params(axis='y', labelcolor='#f39c12')
    ax3_twin.tick_params(axis='y', labelcolor='#3498db')
    ax3.grid(True, alpha=0.3, linestyle='--')
    # Merge legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    # Subplot (d): experience count vs pass rate
    # Compute the total number of experiences, including cold-start cases
    total_exp = [c + s + f for c, s, f in zip(avg_cold_start_exp, avg_success_exp, avg_failure_exp)]
    ax4_twin = ax4.twinx()
    
    # Use an x array to ensure the input is an array rather than a scalar
    x_indices = list(range(len(exp_turn_labels)))
    line3 = ax4.bar(x_indices, total_exp, alpha=0.5, color='#95a5a6', label='Total experience count')
    line4 = ax4_twin.plot(x_indices, pass_rates, marker='o', linewidth=2.5, 
                          markersize=8, color='#e74c3c', label='Strict pass rate')
    ax4.set_title('(d) Experience vs Pass Rate / Experience vs Pass Rate', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Total experience count / Total Experience', fontsize=10, color='#95a5a6')
    ax4_twin.set_ylabel('Pass rate (%) / Pass Rate', fontsize=10, color='#e74c3c')
    ax4.set_xticks(x_indices)
    ax4.set_xticklabels(exp_turn_labels)
    ax4.tick_params(axis='y', labelcolor='#95a5a6')
    ax4_twin.tick_params(axis='y', labelcolor='#e74c3c')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    # Annotate pass rates
    for i, (label, rate) in enumerate(zip(exp_turn_labels, pass_rates)):
        ax4_twin.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', 
                     fontsize=9, fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    chart_path = output_dir / 'experience_impact.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return chart_path


def generate_markdown_report(stats: Dict, output_dir: Path) -> Path:
    """Generate a Markdown analysis report in an academic style"""
    from datetime import datetime
    
    report_lines = []
    turns = stats["turns"]
    summary = stats["summary"]
    lang_baseline = stats["language_baseline"]
    turn_names = sorted(turns.keys(), key=lambda x: int(x.split("_")[-1]))
    
    # Title and generation time
    report_lines.append("# Statistical Analysis Report for Multi-Turn Experiments\n\n")
    report_lines.append(f"**Generated at**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report_lines.append("---\n\n")
    
    # Executive Summary
    report_lines.append("## 1. Executive Summary\n\n")
    final_turn = turn_names[-1]
    final_cumulative = turns[final_turn]["evaluation"]["cumulative"]
    initial_cumulative = turns[turn_names[0]]["evaluation"]["cumulative"]
    improvement = final_cumulative["rate"] - initial_cumulative["rate"]
    
    report_lines.append("> [!IMPORTANT]\n")
    report_lines.append(f"> **Key findings:**\n")
    report_lines.append(f"> - Cumulative success rate reached **{final_cumulative['rate']:.1f}%** ({final_cumulative['count']}/{lang_baseline['total']} CVEs)\n")
    report_lines.append(f"> - From Turn 1 to Turn {turn_names[-1].split('_')[-1]} improved by **{improvement:.2f} percentage points**\n")
    report_lines.append(f"> - Total cost across all turns: **${summary['total_cost']:.2f}**\n")
    if summary["best_language"]:
        bl = summary["best_language"]
        report_lines.append(f"> - Best-performing language: **{bl['language']}** ({bl['pass_rate']:.1f}% pass rate)\n")
    report_lines.append(">\n\n")
    
    # Repair Effectiveness Analysis
    report_lines.append("## 2. Repair Effectiveness Analysis\n\n")
    report_lines.append("![Figure 1: Repair Effectiveness](repair_effectiveness.png)\n\n")
    report_lines.append("**Figure 1**: Shows repair-effectiveness trends across turns, including (a) Strict pass rate、(b) Pass rate vs turn instances、(c) cumulative success progress、(d) pass rate among submitted patches。\n\n")
    
    # Table 1: Pass Rate Summary
    report_lines.append("### Table 1: Pass Rate Summary\n\n")
    report_lines.append("| Turn | Instances | Strict pass rate | Cumulative successes | Cumulative pass rate |\n")
    report_lines.append("|------|--------|---------------|-----------|----------|\n")
    for tn in turn_names:
        t_data = turns[tn]
        instances = t_data["total_instances"]
        strict_rate = t_data["evaluation"]["strict"]["pass_rate"]
        cum_count = t_data["evaluation"]["cumulative"]["count"]
        cum_rate = t_data["evaluation"]["cumulative"]["rate"]
        report_lines.append(f"| {tn.replace('turn_', 'Turn ')} | {instances} | {strict_rate:.1f}% | {cum_count} | {cum_rate:.1f}% |\n")
    report_lines.append("\n*Note: strict pass rate means passing both PoC and unit-test validation; cumulative pass rate is measured against the total of 230 CVEs.*\n\n")
    
    # Language-Specific Analysis
    report_lines.append("## 3. Language Analysis\n\n")
    report_lines.append("![Figure 2: Language Analysis](language_analysis.png)\n\n")
    report_lines.append(
        "**Figure 2**: Language analysis showing (a) the success count for each language in each turn and "
        "(b) the pass rate by language.\n\n"
    )
    
    # Table 2: Language Pass Rates
    report_lines.append("### Table 2: Pass Rate by Language\n\n")
    report_lines.append("| Language | Baseline total | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | Overall pass rate |\n")
    report_lines.append("|------|---------|--------|--------|--------|--------|--------|----------|\n")
    
    for lang in ["Python", "Go", "JavaScript"]:
        total = lang_baseline.get(lang, 0)
        rates = [turns[tn]["evaluation"]["language_pass_rates"][lang]["pass_rate"] for tn in turn_names]
        # Get cumulative success counts from summary
        lang_stats = summary.get("language_cumulative_stats", {}).get(lang, {})
        cumulative_success = lang_stats.get("cumulative_success", 0)
        overall_rate = lang_stats.get("cumulative_rate", 0.0)
        rate_str = " | ".join([f"{r:.1f}%" for r in rates])
        report_lines.append(f"| {lang} | {total} | {rate_str} | {overall_rate:.1f}% |\n")
    report_lines.append("\n*Note: pass rates are measured against the total number of CVEs in each language.*\n\n")
    
    # Cost and Efficiency Metrics
    report_lines.append("## 4. Cost and Efficiency Metrics\n\n")
    report_lines.append("![Figure 3: Cost Efficiency](cost_efficiency.png)\n\n")
    report_lines.append(
        "**Figure 3**: Cost and efficiency metrics showing (a) average cost, (b) total cost per turn, and "
        "(c) the average number of API calls.\n\n"
    )
    
    # Table 3: Cost Summary
    report_lines.append("### Table 3: Cost and Efficiency Summary\n\n")
    report_lines.append("| Turn | Average cost (USD) | Total cost (USD) | Average API calls |\n")
    report_lines.append("|------|---------------|-------------|-------------|\n")
    for tn in turn_names:
        costs = turns[tn]["costs"]
        report_lines.append(f"| {tn.replace('turn_', 'Turn ')} | ${costs['mean']:.4f} | ${costs['total']:.2f} | {costs['api_calls_mean']:.1f} |\n")
    report_lines.append(f"\n**Total cost across all turns**: ${summary['total_cost']:.2f}\n\n")
    
    # Experience Impact (Turn 2+)
    has_experience = any(turns[tn]["experience"] is not None for tn in turn_names[1:])
    if has_experience:
        report_lines.append("## 5. Experience Injection Impact\n\n")
        report_lines.append("![Figure 5: Experience Injection Impact](experience_impact.png)\n\n")
        report_lines.append(
            "**Figure 5**: Experience injection impact analysis showing (a) counts of successful and failed "
            "experiences, (b) cross-language experience usage, (c) quality and similarity, and (d) the "
            "relationship between experience volume and pass rate.\n\n"
        )
        report_lines.append("### Table 4: Experience Usage Statistics\n\n")
        report_lines.append("| Turn | Cold-start experience | Average successful experiences | Average failed experiences | Average cross-language experiences | Average quality score | Average similarity |\n")
        report_lines.append("|------|-----------|-------------|-------------|----------------|-----------|----------|\n")
        for tn in turn_names[1:]:  # Skip Turn 1
            exp = turns[tn]["experience"]
            if exp:
                report_lines.append(f"| {tn.replace('turn_', 'Turn ')} | ")
                report_lines.append(f"{exp.get('avg_cold_start_exp', 0):.2f} | ")
                report_lines.append(f"{exp.get('avg_success_exp', 0):.2f} | ")
                report_lines.append(f"{exp.get('avg_failure_exp', 0):.2f} | ")
                report_lines.append(f"{exp.get('avg_cross_language_exp', 0):.2f} | ")
                report_lines.append(f"{exp.get('avg_quality_score', 0):.2f} | ")
                report_lines.append(f"{exp.get('avg_similarity_score', 0):.4f} |\n")
        report_lines.append("\n*Note: Turn 1 has no experience injection and acts as the baseline.*\n\n")
    
    # Summary Dashboard
    report_lines.append("## 6. Summary Dashboard\n\n")
    report_lines.append("![Figure 4: Summary Dashboard](summary_dashboard.png)\n\n")
    report_lines.append("**Figure 4**: Summary dashboard showing the key aggregate metrics across all turns.\n\n")
    
    # Save report
    report_path = output_dir / "report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    return report_path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Aggregate statistics from multi-turn experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_statistics.py /path/to/metadata_exp
  python aggregate_statistics.py f:\\software\\projects\\VulAgentEvolver\\metadata_exp
        """
    )
    parser.add_argument("root_dir", type=str, help="Root directory containing exp_turn_* subdirectories")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory (default: <root_dir>/statistics)")
    parser.add_argument("--input-json", type=str, default=None, 
                       help="Path to input.json for CVE language mapping (optional, for fallback)")
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        console.print(f"[red]Error: Directory not found: {root_dir}[/red]")
        sys.exit(1)
    
    output_dir = Path(args.output) if args.output else (root_dir / "statistics")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold green]Aggregate Statistics Analysis[/bold green]")
    console.print(f"Root directory: {root_dir}")
    console.print(f"Output directory: {output_dir}\n")
    
    # Load the CVE-to-language mapping for fallback handling
    input_json_path = Path(args.input_json) if args.input_json else None
    if not input_json_path:
        # Try locating input.json automatically
        potential_paths = [
            root_dir.parent / "expbank" / "datasets" / "input.json",
            root_dir / ".." / "expbank" / "datasets" / "input.json",
        ]
        for p in potential_paths:
            if p.exists():
                input_json_path = p
                break
    
    cve_lang_map = load_cve_language_map(input_json_path)
    
    # Load language baseline data
    language_baseline = load_language_stats(root_dir)
    console.print(f"[cyan]Language baseline: {language_baseline}[/cyan]\n")
    
    # Aggregate all turn data
    stats = aggregate_all_turns(root_dir, language_baseline, cve_lang_map)
    
    # Save JSON statistics
    stats_file = output_dir / "statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    console.print(f"\n[bold green]✓ Statistics saved to {stats_file}[/bold green]")
    
    # Generate charts
    console.print(f"\n[bold blue]Generating charts...[/bold blue]")
    chart1 = generate_repair_effectiveness_chart(stats, output_dir)
    console.print(f"[green]✓ {chart1.name}[/green]")
    
    chart2 = generate_language_analysis_chart(stats, output_dir)
    console.print(f"[green]✓ {chart2.name}[/green]")
    
    chart3 = generate_cost_efficiency_chart(stats, output_dir)
    console.print(f"[green]✓ {chart3.name}[/green]")
    
    chart4 = generate_summary_dashboard(stats, output_dir)
    console.print(f"[green]✓ {chart4.name}[/green]")
    
    chart5 = generate_experience_impact_chart(stats, output_dir)
    console.print(f"[green]✓ {chart5.name}[/green]")
    
    # Generate the report
    console.print(f"\n[bold blue]Generating Markdown report...[/bold blue]")
    report = generate_markdown_report(stats, output_dir)
    console.print(f"[green]✓ {report.name}[/green]")
    
    console.print(f"\n[bold green]✓ Analysis complete![/bold green]")
    console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
    console.print(f"\n[bold]Generated files:[/bold]")
    console.print(f"  - statistics.json")
    console.print(f"  - repair_effectiveness.png")
    console.print(f"  - language_analysis.png")
    console.print(f"  - cost_efficiency.png")
    console.print(f"  - summary_dashboard.png")
    console.print(f"  - experience_impact.png")
    console.print(f"  - report.md")


if __name__ == "__main__":
    main()
