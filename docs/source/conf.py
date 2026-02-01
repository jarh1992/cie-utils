import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

# Read version from pyproject.toml
pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject = tomllib.load(f)
    release = pyproject["tool"]["poetry"]["version"]

project = "cie-utils"
author = "Jesus Alberto Rodriguez Henao"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Para docstrings estilo NumPy / Google
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- HTML output -------------------------------------------------------------
html_theme = "furo"

html_static_path = ["_static"]

html_css_files = ["custom.css"]

# html_theme_options = {
#     "light_logo": "logo-light.png",  # opcional si tienes logo claro
#     "dark_logo": "logo-dark.png",    # opcional si tienes logo oscuro
#     "sidebar_hide_name": False,
# }
