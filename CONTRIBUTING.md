# Contributing to Bantz v2

Bantz'a katkıda bulunmak istiyorsan — hoş geldin! 🦌

## Development Setup

```bash
git clone git@github.com:miclaldogan/bantzv2.git
cd bantzv2
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Prerequisites

- Python 3.11+
- Ollama running locally (`ollama serve`)
- A model pulled (e.g., `ollama pull qwen2.5:7b`)

## Project Structure

```
src/bantz/
├── core/          # Brain, memory, habits, places, schedule, etc.
├── llm/           # Ollama + Gemini adapters
├── tools/         # Tool implementations (shell, gmail, calendar, etc.)
├── ui/            # Textual TUI components
├── i18n/          # Translation (MarianMT)
├── auth/          # Google OAuth
├── integrations/  # Telegram bot
├── data/          # Static data files
├── config.py      # Pydantic settings
├── app.py         # Textual app entry
└── __main__.py    # CLI entry point
```

## Branch Naming

- `feat/XX-description` — new features (XX = issue number)
- `fix/XX-description` — bug fixes
- `chore/description` — maintenance, docs, CI

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add habit learning engine (#31)
fix(places): manual coordinate input (#46)
chore: add security policy and license
```

## Pull Requests

1. Create a branch from `main`
2. Make your changes
3. Ensure no lint errors: `python -m pyright src/`
4. Push and open a PR against `main`
5. Reference the issue number: `Closes #XX`

## Code Style

- Python 3.11+ features welcome (match/case, `X | Y` unions, etc.)
- Type hints on all public functions
- Docstrings on modules and classes
- Turkish user-facing strings, English code/comments
- `from __future__ import annotations` in every file

## Adding a New Tool

1. Create `src/bantz/tools/your_tool.py`
2. Inherit from `BaseTool` — implement `schema()` and `execute()`
3. Import in `src/bantz/tools/__init__.py` so it auto-registers
4. The brain will auto-discover it via the tool registry

## Reporting Issues

- Use GitHub Issues with clear reproduction steps
- For security issues, see [SECURITY.md](SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
