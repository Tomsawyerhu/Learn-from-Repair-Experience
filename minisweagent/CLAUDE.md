# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mini-SWE-agent** (v1.16.0) is an autonomous agent framework for solving software engineering tasks. It is adapted here for vulnerability remediation using AI agents.

The system uses a modular architecture with pluggable components:
- **Agents**: Different autonomous agent implementations (default, interactive, textual UI)
- **Models**: Language model interfaces (LiteLLM, Anthropic, Portkey, Requesty)
- **Environments**: Execution environments (local, Docker, Singularity)
- **Tools**: Extensible tool system for agents to interact with code

### Key Innovation: Experience-Enhanced Agents

The project integrates an **ExpBank** experience retrieval system (`run/extra/patcheval_exp.py`). This system:
- Retrieves relevant historical experiences from similar vulnerability fixes
- Injects these experiences into agent prompts to improve remediation accuracy
- Maintains and evolves an experience bank for future reference
- Bridges the gap between isolated agents and collective learning across CVEs

## Architecture

### Core Components

#### 1. Agents (`agents/`)
- **`default.py`**: Base `DefaultAgent` class implementing the core control loop
  - Implements `run()` (main loop) → `step()` → `query()` → `get_observation()`
  - Uses Jinja2 templates for prompt rendering
  - Handles message history and exception handling
  - Parses model output via regex against configured `action_regex`
  - Exception hierarchy: `NonTerminatingException` vs `TerminatingException`

- **`interactive.py`**: Human-in-the-loop wrapper with REPL-style interface
- **`interactive_textual.py`**: TUI interface using Textual library
- **`enhanced.py`**: Extended agent that supports tool-based interactions
  - Integrates with `ToolManager` for structured tool calling
  - Handles JSON-based tool parameters instead of plain text regex parsing
  - Used in vulnerability remediation workflows (`patcheval_exp.py`)

**Control Flow Understanding** (from `default.py` docstring):
```
run() → loop until TerminatingException
  ↓
step() → query() + get_observation()
  ├─ query(): Call LM with message history
  ├─ parse_action(): Extract command from LM response
  └─ execute_action(): Run command in environment, capture output
```

#### 2. Models (`models/`)
- **`litellm_model.py`** [default]: Wrapper for LiteLLM library supporting most LLMs
- **`anthropic.py`**: Anthropic-specific optimizations
- **`litellm_tool_model.py`**: Variant supporting tool/function calling
- **`portkey_model.py`**: Portkey.ai provider
- **`requesty_model.py`**: Requesty provider
- **`test_models.py`**: Deterministic mock models for testing

All models implement the `Model` protocol:
- `query(messages: list[dict])` → response dict
- `get_template_vars()` → additional template variables
- Track `cost` and `n_calls` for limits

#### 3. Environments (`environments/`)
- **`local.py`**: Direct subprocess execution
- **`docker.py`**: Container execution (Docker/Podman)
- **`singularity.py`**: Singularity/Apptainer containers

All implement the `Environment` protocol:
- `execute(command: str, cwd: str)` → output dict
- `get_template_vars()` → environment info

#### 4. Tools System (`tools/`)
- **`annotation.py`**: `@tool_schema` decorator for defining tools
- **`base.py`**: `ToolManager` for auto-discovering and calling decorated methods
- **`evolve_tools.py`**: Vulnerability-specific tools for `EvolveToolsManager`

Used primarily by `EnhancedAgent` for structured interactions.

#### 5. Configuration (`config/`)
- **Core configs**: `mini.yaml`, `default.yaml`
- **Specialized configs**: `extra/patcheval.yaml`, `extra/patcheval_exp.yaml`

Config structure includes:
```yaml
agent:
  system_template: str        # System prompt
  instance_template: str      # Per-task prompt (with {{placeholders}})
  action_regex: str          # Regex to extract action (default: bash triple-backticks)
  step_limit: int            # Max steps
  cost_limit: float          # Max API cost
  timeout_template: str      # Timeout message
  format_error_template: str # Format error message
```

### Key Entry Points

#### Local Interactive Usage
- **`mini.py`**: Main entry point for local problem-solving
  - Options: `-m/--model` (model name), `-v/--visual` (TUI mode)
  - Prompts for task, initializes InteractiveAgent/TextualAgent
  - Saves trajectory to `~/.config/mini-swe-agent/last_mini_run.traj.json`

- **`hello_world.py`**: Minimal example showing agent usage

#### Vulnerability Remediation (PatchEval)
- **`run/extra/patcheval_exp.py`** [Enhanced with ExpBank]:
  - Processes CVE/vulnerability datasets
  - Uses `EnhancedAgent` with tool-based interactions
  - Integrates `ExperienceRetriever` from ExpBank
  - Outputs: patches, evaluation logs, statistics
  - Config: `config/extra/patcheval_exp.yaml`

#### Benchmarking
- **`run/extra/swebench.py`**: Benchmark agent on SWE-bench dataset

## Development Commands

### Environment Setup
```bash
# Configure API keys and settings
# Creates ~/.config/mini-swe-agent/.env on first run
mini --help  # or any command to trigger first-time setup

# Set environment variables:
export MSWEA_MODEL_NAME="claude-3-5-sonnet-20241022"  # or gpt-4, etc.
export MSWEA_MINI_CONFIG_PATH="/path/to/config.yaml"   # Custom config
export MSWEA_GLOBAL_CONFIG_DIR="/path/to/config"       # Config directory
export MSWEA_SILENT_STARTUP=1                          # Suppress startup banner
```

### Running Agents Locally
```bash
# Simple REPL interface
python -m minisweagent.run.mini -t "Write a Python function that..."

# Visual/pager interface (Textual)
python -m minisweagent.run.mini -v -t "..."

# With specific model
python -m minisweagent.run.mini -m gpt-4 -t "..."

# Minimal example
python minisweagent/run/hello_world.py
```

### Running on Vulnerability Datasets
```bash
# PatchEval with ExpBank experience injection
python minisweagent/run/extra/patcheval_exp.py \
  --cve-list data/cve_list.json \
  --mode baseline  # or 'retrieve-only', 'evolve'

# Original PatchEval (no experience)
python minisweagent/run/extra/patcheval.py --cve-list data/cve_list.json
```

### Testing and Inspection
```bash
# Run with test/mock models (deterministic)
python -c "from minisweagent.models.test_models import TestModel; ..."

# Inspect trajectories
python minisweagent/run/inspector.py output/trajectory.json
```

## Common Workflows

### Adding a New Agent Type

1. Create agent class in `agents/` extending `DefaultAgent` or `EnhancedAgent`
2. Implement `query()` and/or `get_observation()` if needed
3. Override exception handling if desired
4. Create corresponding config in `config/`
5. Create entry point in `run/` using the new agent

### Adding a New Model

1. Create model class in `models/` implementing the `Model` protocol
2. Implement: `query()`, `get_template_vars()`, track `cost` and `n_calls`
3. Register in `models/__init__.py` `get_model()` function
4. Reference in configs and CLI

### Adding Domain-Specific Tools

1. Create `ToolManager` subclass in `tools/`
2. Define methods with `@tool_schema` decorator
3. Pass to `EnhancedAgent` via `tool_manager` parameter
4. Agent will auto-discover and use tools

### Extending Prompting Strategy

Modify config files to change prompt behavior:
- **`system_template`**: Change system instructions
- **`instance_template`**: Change per-task prompt (use `{{ }}` for variables like `experience_context`, `cve_description`)
- **`action_regex`**: Change how actions are parsed from model output
- Add new template variables via agent's `extra_template_vars` or model's `get_template_vars()`

Example of experience injection (from `patcheval_exp.yaml`):
```yaml
{% if experience_context %}
<historical_experiences>
...{{ experience_context }}...
</historical_experiences>
{% endif %}
```

## Important Implementation Details

### Template Variable Sources

Three sources feed into Jinja2 template rendering:
1. **Agent config** (`AgentConfig` fields): `system_template`, `instance_template`, etc.
2. **Environment** (`get_template_vars()`): Environment-specific info (paths, etc.)
3. **Model** (`get_template_vars()`): Model-specific info (token limits, etc.)
4. **Extra vars**: `agent.extra_template_vars` and `task`-specific kwargs

### Exception Hierarchy

- **`NonTerminatingException`**: Recoverable errors (format errors, timeouts) → agent adds error message and continues
  - `FormatError`: Action parsing failed
  - `ExecutionTimeoutError`: Command timed out
  - `InvalidParamError`: Tool parameters invalid

- **`TerminatingException`**: Terminal conditions → agent returns with exit code
  - `Submitted`: Agent declares task complete (output first line is magic string)
  - `LimitsExceeded`: Cost or step limit reached

### Action Parsing

Default regex: `` r"```bash\s*\n(.*?)\n```" ``
- Expects actions in bash triple-backticks
- Extracts content for `env.execute()`
- Use `FormatError` if regex doesn't match

Enhanced agent uses structured JSON tool calls instead.

### Message History Format

Messages follow OpenAI chat format:
```python
{"role": "system", "content": "..."}
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
```

Additional fields allowed (implementation-dependent):
```python
agent.add_message("assistant", content="...", tool_calls=[...])
```

## ExpBank Integration (patcheval_exp.py)

The experience-enhanced vulnerability remediation workflow:

1. **Initialization**: Load CVE list, initialize `ExperienceRetriever` from ExpBank
2. **Per-CVE Loop**:
   - Retrieve similar historical fixes via `ExperienceRetriever`
   - Inject experiences into `instance_template` as `experience_context`
   - Run `EnhancedAgent` with enhanced prompt
   - Collect fixes and tool usage patterns
3. **Experience Evolution**:
   - Optionally update ExpBank with new successful fixes
   - Track what patterns worked for similar vulnerabilities

Key files:
- Config: `config/extra/patcheval_exp.yaml` (defines templates with `experience_context`)
- Entry point: `run/extra/patcheval_exp.py`
- Statistics: `run/extra/run_statistics.py` (success tracking, reporting)

## File Organization Notes

- **`output/`**: Generated trajectories, patches, evaluation logs
- **`config/`**: YAML configuration files
- **`data/`**: CVE lists, test data
- **`agents/`**: Agent implementations
- **`models/`**, **`environments/`**: Pluggable implementations
- **`run/`**: Entry point scripts
- **`tools/`**: Tool system and tool managers
- **`utils/`**: Logging and utilities
