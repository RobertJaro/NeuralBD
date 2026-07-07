# Installation

Install the package in editable mode for development:

```bash
pip install -e ".[dev,docs]"
```

For the local `nf2` conda environment used during development:

```bash
conda run -n nf2 python -m pytest
```
