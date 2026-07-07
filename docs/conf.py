from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "NeuralBD"
author = "Robert Jarolim"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/neuralbd_logo.png"
html_theme_options = {
    "logo": {
        "image_light": "_static/neuralbd_logo.png",
        "image_dark": "_static/neuralbd_logo.png",
    },
}
autodoc_typehints = "description"
