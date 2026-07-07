# Development

Run tests:

```bash
conda run -n nf2 python -m pytest
conda run -n nf2 ruff check src/neuralbd tests examples docs/conf.py
```

Build docs:

```bash
LC_ALL=C LANG=C MPLCONFIGDIR=/tmp/matplotlib conda run -n nf2 sphinx-build -b html docs docs/_build/html
```
