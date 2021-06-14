# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../../banditpylib'))

"""Patching m2r2"""
import m2r2

current_m2r2_setup = m2r2.setup

def patched_m2r2_setup(app):
  try:
    return current_m2r2_setup(app)
  except (AttributeError):
    app.add_source_suffix(".md", "markdown")
    app.add_source_parser(m2r2.M2RParser)
  return dict(
    version=m2r2.__version__, parallel_read_safe=True, parallel_write_safe=True,
  )

m2r2.setup = patched_m2r2_setup

# -- Project information -----------------------------------------------------

project = 'banditpylib'
copyright = '2020, Chester Holtz, Chao Tao, Guangyu Xi'
author = 'Chester Holtz, Chao Tao, Guangyu Xi'

# The full version, including alpha/beta/rc tags
release = '0.9.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'm2r2',
    'sphinx.ext.autodoc',
    'sphinx.ext.inheritance_diagram',
    'autoapi.sphinx'
]

autoapi_modules = {
  'banditpylib': {
    'prune': True,
    'override': True,
    'output': 'auto'
  }
}

# source_suffix = '.rst'
source_suffix = ['.rst', '.md']

bibtex_bibfiles = ['references.bib']

todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
