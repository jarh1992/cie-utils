# 📚 Documentation for `cie-utils`

This directory contains the documentation source files for the [`cie-utils`](https://github.com/jarh1992/cie-utils) Python package — a toolkit for image processing based on the CIELAB color space.

## 📄 Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Generating the Documentation Locally](#generating-the-documentation-locally)
- [Publishing to GitHub Pages](#publishing-to-github-pages)
- [Notes](#notes)

---

## 🧭 Overview

The documentation is written using [Sphinx](https://www.sphinx-doc.org/en/master/) and the [Furo theme](https://pradyunsg.me/furo/), and is primarily based on the docstrings of the `cie-utils` functions and modules.

Documentation is auto-generated from your source code and hosted via GitHub Pages on the `gh-pages` branch.

---

## 🔧 Requirements

Ensure you have the following tools installed:

- Python ≥ 3.10
- `pip`
- Virtual environment activated (`venv`, `conda`, etc.)
- Sphinx and required extensions installed:

```bash
pip install -r docs/requirements.txt
```

If no `requirements.txt` exists yet, you can install directly:

```bash
pip install sphinx furo
```

---

## 🛠️ Generating the Documentation Locally

From the root of the project, run:

```bash
sphinx-build -b html docs docs/_build/html
```

Then open the following file in your browser to preview:

```
docs/_build/html/index.html
```

---

## 🚀 Publishing to GitHub Pages

Use the provided script to build and deploy the documentation:

```bash
./publish_docs.sh
```

This script:

1. Builds the HTML documentation into `docs/_build/html`
2. Creates or updates the `gh-pages` branch
3. Pushes the contents to GitHub Pages

📌 The GitHub Pages site will be accessible at:

```
https://jarh1992.github.io/cie-utils/
```

---

## 📝 Notes

- The `docs/_build/` folder is ignored via `.gitignore`.
- Only the contents of `docs/_build/html` are published via the `gh-pages` branch.
- The root `docs/` folder **must not** be deleted, as it contains the configuration (`conf.py`) and source files for Sphinx.

---

## 💬 Contributing to Docs

Feel free to improve the documentation by editing `.rst` or `.md` files in the `docs/` directory, or improving docstrings in the main source code. Then run `sphinx-build` or use the script to update the published site.
