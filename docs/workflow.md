# Workflow

NeuralBD reconstructs a latent high-resolution image from a burst of degraded observations.
The workflow is organized around explicit stages: data preparation,
model configuration, optional pretraining, blind deconvolution, validation, and export.

## 1. Prepare the burst

The training pipeline expects a burst in channels-last form:

```text
height, width, frames
```

or:

```text
height, width, frames, channels
```

All images should be strictly positive. The default image model uses a positive
`softplus` output activation, which matches the expected data domain.

Instrument-specific loaders convert supported source formats into this standard layout.
The core training code does not expose configurable axis ordering; loaders are responsible
for returning channels-last arrays.

## 2. Select the working data

Use the `data` section to choose the input path, normalization, frame subset, channel
subset, and optional subframe:

```yaml
data:
  type: numpy
  path: examples/data/synthetic_burst.npy
  normalization: minmax
  frame_indices: [0, 1, 2, 3]
  channels: [0]
  subframe:
    x: 128
    y: 128
    size: 128
```

Subframes are useful for fast experiments, memory control, and progressive development of
new instrument loaders.

## 3. Choose the method

Use the standard method when a burst can be modeled with one PSF per frame:

```yaml
method: standard
```

Use the spatial method when the PSF changes across the field of view:

```yaml
method: spatial
```

The spatial method uses a SIREN PSF field that evaluates `PSF(x, y, px, py)`.

## 4. Configure the PSF

Standard NeuralBD can use direct learnable PSF parameters:

```yaml
model:
  psf:
    type: default
    representation: parameters
    size: 65
    channel_mode: shared
```

or a continuous SIREN representation:

```yaml
model:
  psf:
    type: default
    representation: siren
    size: 65
    channel_mode: per_channel
    permute_samples: true
```

`channel_mode: shared` learns one PSF per frame and applies it to all channels.
`channel_mode: per_channel` learns a separate PSF per frame and channel.

## 5. Progressive warm start

Progressive training can replace a separate image pretraining stage by starting with a
`1x1` PSF. The first stage behaves like an identity observation model, then later stages
grow by one pixel on each side every 100 epochs while reducing the point count and learning
rate. The generated schedule follows `1x1`, `3x3`, `5x5`, and so on, while keeping the
approximate sample budget `sampling_points * psf_area` constant:

```yaml
training:
  progressive:
    enabled: true
    start_psf_size: 1
    increase_every_n_epochs: 100
    sampling_points: 16384
    fixed_epoch_size: true
    epoch_iterations: 1000
    learning_rate_start: 3.0e-4
    learning_rate_end: 3.0e-5
```

This schedule is especially useful when the target PSF support is large. The fixed epoch
size keeps 1000 batches per epoch even though the number of sampled points per batch changes
across stages. Once the full PSF is reached, the remaining epochs continue at the final PSF
size.

## 6. Optional pretraining

Pretraining is still available if you want to fit the image SIREN to a specific reference
target before blind deconvolution:

```yaml
pretraining:
  enabled: true
  epochs: 500
  learning_rate: 1.0e-4
  target: first_frame
```

Available targets are `first_frame`, `frame`, and `mean`.

## 7. Validate and export

Validation runs every 10 epochs by default and writes sampled diagnostic figures and
optional NumPy arrays. The default figures include learned PSFs, input versus reconstruction,
predicted observations versus reference observations, and optional comparison with a
ground-truth reconstruction.

After training, export the latent image with:

```bash
nbd-reconstruct --checkpoint runs/standard_numpy/neuralbd.nbd --out reconstruction.npy
```

The checkpoint stores model state and reconstruction metadata, not the full input burst.
