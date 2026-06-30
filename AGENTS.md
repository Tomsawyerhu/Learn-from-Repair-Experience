# 🤖 Agent Guidelines for mini-swe-agent

Welcome! If you are an AI assistant or agentic coding tool (like Antigravity, SWE-agent, Copilot, or Cursor) working in this repository, please adhere strictly to the rules and conventions outlined in this document. 

This file serves as the definitive source of truth for repository structure, style guidelines, and build/test instructions to help you operate safely and effectively.

---

## 📚 1. Project Overview

**Project Name**: `vulagent` (built on `mini-swe-agent`)
**Type**: A self-evolving agent for vulnerability repair
**Language**: Python
**Minimum Version**: Python >= 3.10
**Build System**: `setuptools` (defined in `pyproject.toml`)

**Note on Architecture**: The `mini-swe-agent` component in this repository acts primarily as a carrier/vehicle for the agent interactions. The core functionality and primary entry point of this project is vulnerability repair and evaluation, orchestrated by `multi_turn_runner.py`.

---

## 🚀 2. Core Workflows: Vulnerability Repair

The primary operation in this repository is running the multi-turn vulnerability repair pipeline. The `multi_turn_runner.py` script aggregates three main stages:
1. **Vulnerability Repair (`patcheval_exp.py`)**
2. **Experience Generation & Indexing (`pipeline.py`)**
3. **Statistical Analysis (`aggregate_statistics.py`)**

### Running the Pipeline (`multi_turn_runner.py`)

Here are the standard commands for operating the repair pipeline:

- **Run 5 turns using the default `retrieve-only` mode (with experience injection):**
  ```bash
  python multi_turn_runner.py -n 5 -o output
  ```

- **Run all turns in `baseline` mode (no experience injection):**
  *Note: Baseline mode automatically generates a cold-start experience database for turn 1, stored in `turn_1/cold-start/vec_DB`.*
  ```bash
  python multi_turn_runner.py -n 5 -o output --run-mode baseline
  ```

- **Run `retrieve-only` mode using a pre-generated cold-start experience database:**
  ```bash
  python multi_turn_runner.py -n 5 -o output_with_exp \
      --exp-db /path/to/baseline_output/turn_1/cold-start/vec_DB
  ```

- **Full execution example with custom parameters:**
  ```bash
  python multi_turn_runner.py \
      -d /path/to/dataset.jsonl \
      -o /path/to/output \
      -n 5 \
      --run-mode retrieve-only \
      --filter-turns 1 \
      --workers 32 \
      --eval-workers 32
  ```

---

## 🧪 3. Build, Lint, and Test Commands

Whenever you make changes to the codebase, you must ensure that your code adheres to our linting rules and does not break existing tests. 

### Dependencies & Setup
To install the project in editable mode with development dependencies:
```bash
pip install -e .[dev,full]
```

### Running Tests (Pytest)
This project uses `pytest` along with `pytest-asyncio` for testing async functions.

- **Run all tests**:
  ```bash
  pytest
  ```
- **Run a specific test file**:
  ```bash
  pytest path/to/test_file.py
  ```
- **Run a single test function inside a file**:
  ```bash
  pytest path/to/test_file.py::test_function_name
  ```
- **Run tests matching a specific string**:
  ```bash
  pytest -k "keyword"
  ```

### Linting & Formatting (Ruff)
Ruff is our primary tool for linting and formatting.
- **Check for linting errors**:
  ```bash
  ruff check .
  ```
- **Auto-fix fixable linting errors**:
  ```bash
  ruff check --fix .
  ```
- **Format code**:
  ```bash
  ruff format .
  ```

---

## 💅 4. Code Style Guidelines

Please respect the existing codebase style to ensure consistency.

### General Python Standards
- **Line Length**: Max 120 characters.
- **Indentation**: 4 spaces (no tabs).
- **String Quotes**: Use double quotes (`"`) everywhere possible, consistent with Black formatting. Single quotes (`'`) should only be used if double quotes would require excessive escaping.
- **Trailing Commas**: Magic trailing commas are respected for multi-line formatting.

### Imports
- Let `ruff` handle import sorting (the `I001` rule is enabled).
- Avoid `*` imports. Import explicitly what you need.
- Group imports logically: standard library, third-party (e.g., `litellm`, `requests`), and local `minisweagent` modules.

### Typing
- Since the project targets Python >= 3.10, use modern typing syntax (e.g., `list[str]`, `dict[str, Any]`, and `int | None` instead of `Union[int, None]` or `Optional[int]`).
- Be generous with type hints on function arguments and return types to keep the code clear and maintainable.

### Naming Conventions
- **Classes**: `PascalCase`
- **Functions & Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with a single underscore (e.g., `_helper_function`). Unused variables should also be prefixed with `_`.

### Error Handling & Exceptions
- Always raise specific, descriptive exceptions (e.g., `ValueError`, `KeyError`) rather than a generic `Exception`.
- Never use bare `except:` blocks; always catch specific exceptions to avoid masking unrelated errors.
- Do not use string literals, f-strings, or `.format()` directly inside exception instantiation without a proper Exception class (per `flake8-errmsg` rules).

### Docstrings & Comments
- While missing docstring warnings are disabled (`missing-function-docstring`, `missing-class-docstring`), you should add concise docstrings to complex logic, newly created classes, and public API methods.
- Avoid writing comments explaining *what* the code does if it is obvious. Instead, explain *why* it does it.
- Keep comments up to date with code changes.

---

## 🛠 5. Repository Structure & Key Components

Understanding the layout will help you navigate and propose accurate changes:

- `minisweagent/`: The core source code for the agent, including models and runners.
- `expbank/`: Holds experiment configurations, evaluation code, and scripts for auto-evaluations.
- `PatchEval/`: Tooling for evaluating patches generated by the models.
- `pyproject.toml`: Contains all dependency definitions, project metadata, and tool configurations (like Ruff and Pytest).
- `analyze_errors.py` / `check_bad_request_tokens.py`: Scripts used for debugging and analyzing model failure cases.

---

## 🚀 6. Agent Operation Workflows

If you are an agent modifying this repository, follow these best practices:

1. **Understand Before Acting**: Use your `read` and `glob`/`grep` tools to explore the codebase. Do not assume standard commands work without checking `pyproject.toml`.
2. **Context First**: Always read the existing tests and files related to your task. Make sure you understand the import structure and naming conventions in the neighboring files.
3. **Plan Your Implementation**: Identify which files you need to touch. Keep your edits minimal and surgical. Prefer using the `edit` tool rather than overwriting an entire file unless necessary.
4. **Self-Correction & Verification**: 
    - After writing code, **always** run the relevant test using the single-test command.
    - Run `ruff check .` to ensure your changes adhere to the 120-character line limit and double-quote rules.
    - If a test fails, read the error output and correct your implementation.
5. **No Hallucinations**: Do not assume external libraries are installed unless they appear in `pyproject.toml` or `requirements.txt`.
6. **Pathing**: Ensure you use absolute paths when reading or writing files, prefixed with the repository root.

Thank you for contributing responsibly to the `mini-swe-agent` project!
