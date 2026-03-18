"""Sphinx configuration for timedatamodel."""

import json
import re
from pathlib import Path

import timedatamodel as tdm

_REPO_ROOT = Path(__file__).parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_EXAMPLES_OUT_DIR = Path(__file__).parent / "examples"
_NB_PREFIX = re.compile(r"^nb_\d+_(.+\.ipynb)$")


def _strip_outputs(nb: dict) -> dict:
    """Return a deep-copied notebook dict with all cell outputs cleared."""
    nb = json.loads(json.dumps(nb))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return nb


def _copy_notebooks(app) -> None:
    """Copy example notebooks into docs/examples/, stripping the nb_XX_ prefix."""
    _EXAMPLES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in sorted(_EXAMPLES_DIR.glob("*.ipynb")):
        m = _NB_PREFIX.match(src.name)
        dest_name = m.group(1) if m else src.name
        with src.open(encoding="utf-8") as f:
            nb = json.load(f)
        with (_EXAMPLES_OUT_DIR / dest_name).open("w", encoding="utf-8") as f:
            json.dump(_strip_outputs(nb), f, indent=1, ensure_ascii=False)


def setup(app) -> None:
    app.connect("builder-inited", _copy_notebooks)

project = "timedatamodel"
author = "Rebase Energy"
release = tdm.__version__
copyright = "2024, Rebase Energy"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

# -- nbsphinx settings -----------------------------------------------------
nbsphinx_execute = "auto"

# -- MyST settings ----------------------------------------------------------
myst_enable_extensions = ["colon_fence"]

# -- Autodoc settings -------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "undoc-members": False,
    "exclude-members": ", ".join([
        # repr internals
        "_repr_html_",
        "_repr_meta_lines",
        "_repr_data_rows",
        # TimeSeries internals
        "_clone",
        "_validate_table",
        # TimeSeriesTable internals
        "_clone_df",
    ]),
}

# -- sphinx-autodoc-typehints ------------------------------------------------
# Suppress warnings for forward-ref strings like "pd.DataFrame" that cannot
# be resolved at import time (pandas/polars are optional dependencies).
suppress_warnings = ["sphinx_autodoc_typehints.forward_reference"]

# -- Intersphinx ------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# -- Theme -------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "timedatamodel"
