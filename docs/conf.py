#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information



project = 'Quark'
copyright = '2023, Advanced Micro Devices, Inc. All rights reserved'
author = 'Quark team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "recommonmark", "sphinx_markdown_tables", "sphinx.ext.coverage", "sphinx.ext.autosummary", "sphinx_md",
    "sphinx.ext.napoleon", "sphinx.ext.githubpages", "autoapi.extension"
]
autoapi_dirs = ['../../quark']
autoapi_keep_files = True
autoapi_add_toctree_entry = False
autosummary_generate = True
autoapi_options = ["members", "show-module-summary"]
autoapi_ignore = []

templates_path = ['_templates']
exclude_patterns = []

source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# def skip_submodules(app, what, name, obj, skip, options):
#     if what == "module":
#         skip = True
#     return skip

# def setup(sphinx):
#     sphinx.connect("autoapi-skip-member", skip_submodules)
