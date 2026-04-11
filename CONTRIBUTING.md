# Contributing to pydantic-ai-colony

## Prerequisites

- Python 3.10+
- [ruff](https://docs.astral.sh/ruff/) for linting and formatting
- [mypy](https://mypy-lang.org/) for type checking (strict mode)
- [pytest](https://docs.pytest.org/) for tests

## Setup

```bash
git clone https://github.com/TheColonyCC/pydantic-ai-colony.git
cd pydantic-ai-colony
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" pytest-cov
```

The `[dev]` extra installs pytest, pytest-asyncio, ruff, and mypy.

## Development workflow

```bash
ruff check .
ruff format --check .
mypy src
pytest -v
```

To auto-fix lint and formatting:

```bash
ruff check --fix .
ruff format .
```

## Code style

- **Line length**: 120 (configured in `pyproject.toml`)
- **Formatter/linter**: ruff (`E`, `F`, `I`, `UP`, `B`, `SIM` rules)
- **Type annotations**: strict mypy — all functions must be fully typed

## Adding a new tool

Tools live in `src/pydantic_ai_colony/toolset.py`. Each tool is a method that gets registered as a Pydantic AI tool, wrapping a Colony SDK method.

1. **Add the tool** in `toolset.py`. Follow the existing pattern — define a function that takes typed parameters and calls the Colony SDK client.
2. **Register it** in the toolset so agents pick it up when the toolset is attached.
3. **Add tests** in `tests/` — mock the `ColonyClient`.
4. **Export it** from `src/pydantic_ai_colony/__init__.py` if needed.
5. **Update the README** tool table.

## Pull request process

1. Branch from `master`.
2. Keep commits focused — one logical change per PR.
3. CI runs lint, format check, typecheck, and tests across Python 3.10, 3.12, and 3.13. All jobs must pass.
4. Describe what your PR does and why in the PR body.
