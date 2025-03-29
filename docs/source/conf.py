# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Include your package directory

project = 'M2 Coursework'
copyright = '2025, Jacob Tutt'
author = 'Jacob Tutt'
release = '1.0.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'sphinx.ext.mathjax', 
]

autodoc_default_options = {
    'members': True,             # Include all public members
    'undoc-members': True,       # Include members without docstrings
    'private-members': True,     # Include members starting with _
    'special-members': '__init__',  # Include special methods (like __init__)
    'show-inheritance': True,    # Show class inheritance
    'alphabetical': False,       # To maintain source order (optional)
    'member-order': 'bysource',  # To maintain source order (optional)
}
autodoc_mock_imports = [
  "transformers",
  "torch",
  "accelerate",
  "tqdm",
  "numpy",
  "h5py",
  "jupyter",
  "ipykernel",
  "torchinfo",
  "wandb",
  "matplotlib",
  "ipython",
  "pandas"
]
html_theme = 'sphinx_rtd_theme'
# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add your project root directory to sys.path (assuming docs/ is two levels deep)
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'M2 Coursework'
author = 'Jacob Tutt'
copyright = '2025, Jacob Tutt'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',           # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',           # Add links to highlighted source code
    'sphinx.ext.mathjax',            # Render math using MathJax
    'sphinx_autodoc_typehints',      # Automatically document type hints
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

exclude_patterns = []

# Mock imports for modules not needed during doc build
autodoc_mock_imports = [
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    "seaborn",
    "iminuit",
    "sweights",
    "tqdm",
    "tabulate",
    "jax",
    "jaxlib",
    "numpyro",
    "optax",
    "arviz",
    "corner",
    "torch",
    "transformers",
    "h5py",
    "wandb",
    "ipykernel",
    "torchinfo",
    "accelerate",
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'show-inheritance': True,
    'member-order': 'bysource',
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']