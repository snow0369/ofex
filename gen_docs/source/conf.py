# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys

sys.path.insert(0, os.path.abspath('../../'))

project = 'ofex'
copyright = '2024, Gwonhak Lee'
author = 'Gwonhak Lee'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
"""
extensions = [
    'sphinx.ext.autodoc',  # Enables the automodule directive
    'sphinx.ext.napoleon', # For Google and NumPy-style docstrings
    'sphinx.ext.viewcode', # Adds links to source code
    'sphinx.ext.mathjax',  # Enables math rendering via MathJax
    'sphinx.ext.imgmath',  # Enables math rendering via LaTeX for PDF or images
    'sphinx.ext.autosummary',
]
templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    'members': True,        # Include all class members (methods and attributes)
    'undoc-members': False, # Do not include undocumented members
    'private-members': False, # Exclude private members
    'noindex-members': False,  # Ignore the index of all class attributes and methods
}

toc_object_entries = False
"""
extensions = [
    'sphinx.ext.napoleon', # For Google and NumPy-style docstrings
    'sphinx.ext.viewcode', # Adds links to source code
    'sphinx.ext.mathjax',  # Enables math rendering via MathJax
    'sphinx.ext.imgmath',  # Enables math rendering via LaTeX for PDF or images
]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


# -- Options for math output -------------------------------------------------
mathjax_config = {
    'TeX': {
        'Macros': {
            'vec': ['\\mathbf{#1}', 1],  # Custom LaTeX macros (optional)
        },
    },
}

imgmath_latex_preamble = r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{braket}
'''

