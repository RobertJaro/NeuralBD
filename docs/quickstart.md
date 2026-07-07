# Quickstart

Train standard NeuralBD on a numpy burst:

```bash
python examples/create_synthetic_burst.py
nbd-train --config examples/configs/standard_numpy.yaml
```

Train the spatially varying method:

```bash
nbd-train --config examples/configs/spatial_numpy.yaml
```
