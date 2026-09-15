"""Sphinx configuration for the RubycGW documentation website."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

project = "RubycGW"
author = "RubycGW developers"
copyright = "2026, RubycGW developers"
release = "0.1.0"
version = release

master_doc = "index"
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
add_module_names = False

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "substitution",
]

myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "RubycGW Docs"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False

html_context = {
    "github_user": "Echoxiangmega",
    "github_repo": "RubycGW",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "logo": {"text": "RubycGW"},
    "collapse_navigation": False,
    "show_toc_level": 2,
    "header_links_before_dropdown": 6,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["search-button", "theme-switcher", "navbar-icon-links"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Echoxiangmega/RubycGW",
            "icon": "fa-brands fa-github",
        }
    ],
}

html_sidebars = {
    "index": [],
}

exclude_patterns = [
    "_build",
    "README.md",
    "research_notes/**",
]

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

nitpicky = False
