
<h1 align="center">
  Learning from Repair Experience: Self-Evolving Agents for Automated Vulnerability Repair
</h1>

<p align="center">
  <img src="assets/overview.png" alt="Overview" width="75%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.30105">Paper</a> •
  <a href="#-overview">Overview</a> •
  <a href="#-set-up">Set Up</a> •
  <a href="#-usage">Usage</a> •
  <a href="#-repository-guide">Repository Guide</a> •
  <a href="#-additional-documentation">Documentation</a>
</p>

* * *

## 📰 News
* **[Oct. 21, 2026]**: 🎉🎉🎉 EvoRepair has been accepted to [ICSE 2027](https://conf.researchr.org/track/icse-2027) (Top 5%, 119/2375).

## 👋 Overview



EvoRepair is a self-evolving agent framework for vulnerability repair. The repository combines
multi-turn repair orchestration, experience retrieval and indexing, and benchmark-specific runners
for several repair tracks.
All repair experiment scripts in this repository are adapted to the EvoRepair workflow. EvoRepair
can be integrated with mainstream code-agent frameworks.

The repository currently exposes three main execution paths:

1. `PatchEval`: multi-turn vulnerability repair on the PatchEval dataset.
2. `SEC`: multi-turn vulnerability repair on SEC-bench.
3. `Vul4J`: single-round Java vulnerability repair on Vul4J.

Core framework code lives under `minisweagent/`. Experience generation and retrieval components
live under `expbank/`. In particular, the local changes in
[`minisweagent/agents/enhanced.py`](minisweagent/agents/enhanced.py)
adapt Mini-SWE-Agent to support model-driven tool calling in the EvoRepair workflow.

## 🚀 Supported Workflows

| Track | Entry Point | Summary |
| --- | --- | --- |
| `PatchEval` | `python multi_turn_runner.py` | Multi-turn vulnerability repair with experience generation, retrieval, and statistics aggregation. |
| `SEC` | `python multi_turn_runner_secb.py` | Multi-turn SEC-bench repair pipeline with SEC-specific retrieval and rerun behavior. |
| `Vul4J` | `python -m minisweagent.run.extra.vul4j` | Single-round Java vulnerability repair with optional retrieval and evaluation. |

## 🚀 Set Up

Install the project in editable mode with development dependencies:

```bash
pip install -e ".[dev,full]" --no-build-isolation
```

Prepare the embedding model by execute:

```bash
cd expbank
python script/download_model.py
```

If you plan to run `PatchEval` patch validation, install the PatchEval-specific dependencies and
pull the 230 Docker images used for sandboxed evaluation:

```bash
pip install -r PatchEval/requirements.txt
cd PatchEval/scripts
python download_images.py
```

Notes:

1. The image list is defined in `PatchEval/scripts/images.txt`, which currently contains 230 images.
2. These images are used by PatchEval to reproduce sandbox environments for PoC and unit-test-based validation.
3. Make sure Docker is installed and reserve enough disk space before pulling them. PatchEval's own documentation recommends at least 500 GB of free storage.

If you plan to run the default `SEC` pipeline, you can pre-pull the SEC evaluation images used by
`multi_turn_runner_secb.py`:

```bash
python SEC-bench/scripts/pull_secb_images.py
```

Notes:

1. The script reads `minisweagent/data/secb_official_cve201_dataset.jsonl` by default and resolves 201 SEC images.
2. Image naming matches the SEC runner: `hwiwonlee/secb.eval.x86_64.<instance_id>:patch`.
3. Use `python SEC-bench/scripts/pull_secb_images.py --dry-run` to inspect the full pull list first.
4. Reserve generous disk space before pulling. A practical estimate for the full SEC image set is roughly 250 GB to 450 GB.

## 🧪 Usage

### Configuration

Before running any workflow, make sure you have configured both the experience-extraction
models and the benchmark runner models:

1. Experience extraction and experience generation configs live under `expbank/src/config/model/`,
   mainly `experience_extraction_config.yaml` and `exp_extraction_no_example.yaml`.
   Update fields such as `base_url`, `api_key`, and `model_name` according to your provider.
2. Benchmark runner configs live under `minisweagent/config/extra/`, including
   `patcheval_exp.yaml`, `patcheval_exp_live_swe_agent.yaml`, `secb_exp.yaml`, and `vul4j.yaml`.
   These files control the repair model, prompt template, and runtime behavior for each track.
3. For experience extraction, the recommended model is `qwen3-max`.
4. Keep the provider/model settings in these two config groups consistent with the workflows you plan to run.

### PatchEval

Run the main multi-turn PatchEval workflow with:

```bash
python multi_turn_runner.py
```

This runner aggregates:

1. vulnerability repair with `patcheval_exp.py`
2. experience generation and indexing with `pipeline.py`
3. statistics aggregation with `aggregate_statistics.py`

Common commands:

```bash
# Run 5 turns in retrieve-only mode
python multi_turn_runner.py -n 5 -o output

# Run all turns in baseline mode
python multi_turn_runner.py -n 5 -o output --run-mode baseline

# Run retrieve-only mode with an existing experience DB
python multi_turn_runner.py -n 5 -o output_with_exp \
  --exp-db /path/to/vec_DB

# Full example
python multi_turn_runner.py \
  -d minisweagent/data/patcheval_dataset.jsonl \
  -o output \
  -n 5 \
  --run-mode retrieve-only \
  --filter-turns 1 \
  --workers 32 \
  --eval-workers 32
```

Notes:

1. `--run-mode baseline` disables retrieval injection for all turns.
2. In `baseline` mode, turn 1 generates a cold-start experience database under `turn_1/cold-start/vec_DB`.
3. This cold-start DB is typically reused as the `--exp-db` input when you later run `retrieve-only`.
4. `retrieve-only` uses the configured experience DB and updates it through the pipeline after each turn.
5. Omitting `-n/--num-turns` enables auto-run mode until no new fixed CVEs are found.

### SEC

Run the SEC-bench multi-turn pipeline with:

```bash
python multi_turn_runner_secb.py
```

This runner uses:

1. `secb_exp` for repair
2. `pipeline_sec` for experience generation and indexing
3. `aggregate_statistics.py` for summary statistics

Common commands:

```bash
# Run 5 turns in retrieve-only mode
python multi_turn_runner_secb.py -n 5 -o output_sec

# Run all turns in baseline mode
python multi_turn_runner_secb.py -n 5 -o output_sec --run-mode baseline

# Run retrieve-only mode with an existing SEC experience DB
python multi_turn_runner_secb.py -n 5 -o output_sec_with_exp \
  --exp-db /path/to/770_vec_DB

# Full example
python multi_turn_runner_secb.py \
  -d minisweagent/data/secb_official_cve201_dataset.jsonl \
  -o output_sec \
  -n 5 \
  --run-mode retrieve-only \
  --filter-turns 1 \
  --workers 32 \
  --eval-workers 32
```

SEC-specific notes:

1. If `-o/--output` is omitted, the script auto-generates an output directory under `minisweagent/output/sec/`.
2. In `baseline` mode, turn 1 generates a cold-start experience database under `turn_1/cold-start/vec_DB`.
3. This cold-start DB is typically reused as the `--exp-db` input when you later run `retrieve-only`.
4. The `--live/--no-live` flag controls whether the SEC prompt includes live tool-creation guidance.
5. From turn 2 onward, the runner uses previous turn summaries to rerun only instances that failed before.
6. To pre-pull the default SEC evaluation images, run `python SEC-bench/scripts/pull_secb_images.py`.

### Vul4J

Run the Vul4J Java repair workflow with:

```bash
python -m minisweagent.run.extra.vul4j
```

This runner supports optional experience retrieval and optional evaluation.

Common commands:

```bash
# Run Vul4J repair in retrieve-only mode
python -m minisweagent.run.extra.vul4j \
  --mode retrieve-only \
  --workers 4

# Run Vul4J repair in baseline mode
python -m minisweagent.run.extra.vul4j \
  --mode baseline \
  --workers 4

# Run repair with evaluation after completion
python -m minisweagent.run.extra.vul4j \
  --mode retrieve-only \
  --workers 4 \
  --run-eval \
  --eval-workers 12

# Evaluate an existing output directory only
python -m minisweagent.run.extra.vul4j \
  -o /path/to/output \
  --eval-only
```

Notes:

1. If `-o/--output` is omitted, the runner writes to `minisweagent/output/vul4j/`.
2. The default dataset is `minisweagent/data/vul4j_dataset.csv`.
3. The default mode is `retrieve-only`.

## 📚 Repository Guide

1. Framework code: `minisweagent/`
2. Experience bank and indexing pipeline: `expbank/`
3. Patch evaluation tooling: `PatchEval/`
4. SEC-bench resources: `SEC-bench/`

## 📖 Related Repositories

The EvoRepair workflow builds on or integrates with the following upstream repositories:

1. [Mini-SWE-Agent](https://github.com/SWE-agent/Mini-SWE-Agent): lightweight agent framework that serves as the core execution substrate for this repository.
2. [PatchEval](https://github.com/bytedance/PatchEval): benchmark and Dockerized evaluation environment for real-world vulnerability patching.
3. [Vul4J](https://github.com/tuhh-softsec/vul4j): Java vulnerability repair benchmark and evaluation framework.
4. [SEC-bench](https://github.com/SEC-bench/SEC-bench): security-oriented benchmark for evaluating LLM agents on vulnerability-related software tasks.
5. [live-swe-agent](https://github.com/OpenAutoCoder/live-swe-agent): live and interactive SWE-agent style workflow for repository repair and tool-augmented agent execution.
