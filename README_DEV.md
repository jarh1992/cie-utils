# Development Tools Configuration

## Prerequisites

Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
# Or on Windows (PowerShell):
# (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

## Install development tools

```bash
# Install all dependencies (including dev dependencies)
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

## Run tools manually
```bash
# Format code with Black
poetry run black .

# Sort imports with isort
poetry run isort .

# Run linting with flake8
poetry run flake8 .

# Run all pre-commit hooks
poetry run pre-commit run --all-files
```

## How it works
 - `Poetry` manages dependencies and virtual environments.
 - `Black` formats code to consistent style.
 - `isort` sorts imports.
 - `pre-commit` ensures all checks pass before each commit.
