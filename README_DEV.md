# Development Tools Configuration

## Install development tools

```bash
pip install -r requirements.txt
pre-commit install
```

## Run tools manually
```bash
black .
isort .
```

## How it works
 - `Black` formats code to consistent style.
 - `isort` sorts imports.
 - `pre-commit` ensures all checks pass before each commit.
