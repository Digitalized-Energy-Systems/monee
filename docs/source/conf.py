# -- Project information -----------------------------------------------------
project = "monee"
author = "monee contributors"
copyright = "2025, monee contributors"
release = "latest"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.duration",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosectionlabel",
]

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autoclass_content = "both"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_attr_annotations = True

typehints_fully_qualified = True
typehints_use_rtype = False
typehints_use_signature = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "attrs_block",
    "attrs_inline",
]

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": True,
    "source_branch": "main",
    "source_directory": "docs/source/",
    "source_repository": "https://github.com/Digitalized-Energy-Systems/monee/",
    "top_of_page_buttons": ["view", "edit"],
}
