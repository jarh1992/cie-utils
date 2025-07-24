# Guide to contributing to cie-utils

Thank you for your interest in contributing to cie-utils! 🚀

This repository uses a workflow based on **forks and Pull Requests**.
Any contribution is welcome, be it code, documentation, examples or suggestions.

## 🚦 General rules

- No direct *pushes* to `main` are allowed.
- Any changes must go in a descriptively named branch (e.g. `feature/xyz`, `bugfix/abc`).
- All changes must be accompanied by NumPy style docstrings.
- It is recommended to use `black` for code formatting.
- Tests should be included where applicable.
- Changes must be reviewed and approved by at least one person.

## 🧩 How to contribute

1. Make a fork of this repository.
2. Create a new branch from `master`:
   ```bash
   git checkout -b feature/mi-feature
   ```
3. Make your changes.
4. Commit with descriptive messages.
5. Push your fork and open a Pull Request following the [pull request template](.github/pull_request_template.md) or the [issue template](.github/ISSUE_TEMPLATE/bug_report.md).

## 🧪 Tests
To run the tests locally:

   ```bash
   pip install -e .
   pip install pytest
   pytest
   ```
