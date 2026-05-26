---
name: feedback-python-env
description: Always activate .venv and use plain python, not uv run python
metadata:
  type: feedback
---

Always use `uv run python` to invoke Python — never `python` or `python3` directly.

**Why:** uv manages the project venv (.venv, Python 3.12) and ensures the correct environment is used regardless of what the shell has activated. The VIRTUAL_ENV mismatch warning is harmless and should be ignored.

**How to apply:** All Bash tool calls that invoke Python must use `uv run python <script>`.
