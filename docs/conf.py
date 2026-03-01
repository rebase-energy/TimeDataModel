"""Sphinx configuration for timedatamodel."""

import timedatamodel as tdm

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
nbsphinx_execute = "never"

# -- MyST settings ----------------------------------------------------------
myst_enable_extensions = ["colon_fence"]

# -- Autodoc settings -------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "undoc-members": False,
    "exclude-members": ", ".join([
        # _TimeSeriesBase internals
        "_from_float_array",
        "_fmt_value",
        "_fmt_location",
        "_infer_freq_tz",
        # TimeSeries internals
        "_repr_html_",
        "_repr_meta_lines",
        "_repr_data_rows",
        "_apply_scalar",
        "_apply_binary",
        "_apply_comparison",
        "_to_float_array",
        "_validate_alignment",
        "_convert_other_values",
        "_meta_kwargs",
        "_coverage_masks",
        # TimeSeriesTable internals
        "_get_attr",
        "_list_meta_kwargs",
        "_clone_with",
        # TimeSeriesCube internals
        "_get_dim",
        "_dim_index",
        "_maybe_collapse",
        # TimeSeriesCollection internals
        "_item_summary",
        "_rebin_to_global",
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
