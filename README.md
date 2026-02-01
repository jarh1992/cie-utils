# 🌈 cie-utils

> Advanced image processing in Python using the **CIELAB** color space.

[![CI](https://github.com/jarh1992/cie-utils/actions/workflows/ci.yml/badge.svg)](https://github.com/jarh1992/cie-utils/actions/workflows/ci.yml)
[![PyPI](https://github.com/jarh1992/cie-utils/actions/workflows/publish.yml/badge.svg)](https://github.com/jarh1992/cie-utils/actions/workflows/publish.yml)
[![Docs](https://github.com/jarh1992/cie-utils/actions/workflows/deploy_docs.yml/badge.svg)](https://jarh1992.github.io/cie-utils/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![pre-commit](https://img.shields.io/badge/code%20style-pre--commit%20%7C%20black%2C%20isort%2C%20flake8-blueviolet?logo=pre-commit)](https://pre-commit.com/)

---

`cie-utils` is a Python library for **image preprocessing and color analysis** using the perceptually uniform CIELAB color space.

---

## ✨ Features

✅ RGB ➡️ LAB / Lch conversion \
✅ CLAHE, PCA, false color filters \
✅ Background removal via pixel std deviation \
✅ Cluster-based segmentation: KMeans, GaussianMixture, Agglomerative \
✅ Histogram, PDF, and scatter plot visualization \
✅ CSV export of processed image stats \
✅ Clean and modular structure for reuse

---

## 📦 Installation

**From PyPI**

```bash
pip install cie-utils
```

**From GitHub**

```bash
pip install git+https://github.com/jarh1992/cie-utils.git
```

**From local source (for development)**

```bash
# Using Poetry (recommended for development)
poetry install

# Or using pip
pip install .
```

---

## 🧪 Example Usage

```python
from cie-utils import normalize_img, rgb2lab

lab_img = rgb2lab(normalize_img(img, ref_img))
```

---

## 🧰 Development & Code Style

This project uses [Poetry](https://python-poetry.org/) for dependency management:

- 🖤 [`black`](https://github.com/psf/black) for code formatting
- 🔠 [`isort`](https://github.com/PyCQA/isort) for import sorting
- 🧼 [`flake8`](https://github.com/pycqa/flake8) for linting
- ✅ [`pytest`](https://docs.pytest.org/) for testing
- 🧪 `pre-commit` hooks enabled in `.pre-commit-config.yaml`

To install dev dependencies:

```bash
poetry install
poetry run pre-commit install
```

Run tests:

```bash
poetry run pytest
```

---

## 📚 Documentation

Documentation is automatically generated with **Sphinx** and hosted via **GitHub Pages**:

👉 https://jarh1992.github.io/cie-utils/

Documentation is automatically rebuilt and deployed on every release.

---

## 🚀 Release Process

Releases are automated via GitHub Actions. To create a new release:

```bash
./release_workflow.sh [patch|minor|major]
```

This will:
1. Bump the version in `pyproject.toml`
2. Commit and push a version tag
3. GitHub Actions automatically:
   - Runs tests on Python 3.11 and 3.12
   - Publishes the package to PyPI
   - Deploys updated documentation to GitHub Pages

---

## 📝 License

MIT License. See [LICENSE](LICENSE) for details.
