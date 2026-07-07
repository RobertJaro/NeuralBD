# Development

Run tests:

```bash
conda run -n nf2 python -m pytest
```

Build docs:

```bash
LC_ALL=C LANG=C MPLCONFIGDIR=/tmp/matplotlib conda run -n nf2 sphinx-build -b html docs docs/_build/html
```
