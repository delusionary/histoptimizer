# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from pallets_sphinx_themes import get_version
from pallets_sphinx_themes import ProjectLink

project = 'Histoptimizer'
copyright = '2023, Kelly Joyner'
author = 'Kelly Joyner'
release = '0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'jinja'
html_static_path = ['_static']

html_theme_options = {"index_sidebar_logo": True}

html_context = {
    "project_links": [
        ProjectLink("PyPI Releases",
                    "https://pypi.org/manage/project/histoptimizer/releases/"),
        ProjectLink("Source Code",
                    "https://github.com/delusionary/histoptimizer/"),
        ProjectLink("Issue Tracker",
                    "https://github.com/delusionary/histoptimizer/issues/"),
        ProjectLink("Website", "https://histoptimizer.org"),
    ]
}

html_sidebars = {
    "index": ["project.html",
              "globaltoc.html",
              "localtoc.html",
              "searchbox.html",
              "relations.html"],

    "**": ["project.html",
           "globaltoc.html",
           "localtoc.html",
           "searchbox.html",
           "relations.html"],
}
html_static_path = ["_static"]
html_favicon = "_static/histoptimizer-favicon.png"
html_logo = "_static/histoptimizer-spirit-animal.png"
html_title = f"Histoptimizer Documentation ({release})"
html_show_sourcelink = False

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinxcontrib.programoutput',
    'nbsphinx',
    'pallets_sphinx_themes'
]
