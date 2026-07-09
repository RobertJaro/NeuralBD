# Configuration

The primary fields are:

- `base_dir`: checkpoint location
- `work_dir`: local working files such as W&B runs and prepared data
- `method`: `standard` or `spatial`
- `data`: burst location, normalization, batch size, and coordinate scaling
- `model.image`: SIREN image model settings
- `model.psf`: default or spatial PSF settings
- `pretraining`: optional image-model warm start before blind deconvolution
- `training`: Lightning trainer and learning-rate settings
- `logging`: Weights & Biases experiment logging
- `outputs`: validation diagnostic figures

If `work_dir` is omitted, it defaults to `base_dir` for backward compatibility.
Training datasets are kept in memory; `work_dir` is used for W&B files and explicitly
prepared data from workflows.

## Data selection

Loaders return the common `(height, width, frames, channels)` representation. The number
of image channels is inferred from the loaded data, so users do not need to set it in
`model.image`. Use these fields to define the training subset, reproduce preprocessing
choices, and control memory use:

```yaml
data:
  type: numpy
  path: examples/data/synthetic_burst.npy
  frame_indices: [0, 1, 2, 3]
  channels: [0, 1]
  subframe:
    x: 128
    y: 128
    size: 128
```

`subframe` also accepts `x_range: [start, stop]` and `y_range: [start, stop]`.

Model coordinates are centered on the image: `(0, 0)` is the image center, positive
`x` points right, and positive `y` points up. `pixel_per_ds` scales these centered
pixel coordinates before they are passed to the SIREN models.

## PSF channels

The default PSF is shared across all image channels:

```yaml
model:
  psf:
    type: default
    representation: parameters
    channel_mode: shared
```

Use `per_channel` when each image channel should learn an independent PSF per frame:

```yaml
model:
  psf:
    channel_mode: per_channel
```

## PSF representation

Standard NeuralBD supports either a direct learnable PSF parameter grid or a SIREN PSF
field:

```yaml
model:
  psf:
    type: default
    representation: parameters
    target_size: 65
```

```yaml
model:
  psf:
    type: default
    representation: siren
    permute_samples: true
```

For `type: default`, the SIREN represents `PSF(px, py)`. For `type: spatial`, the
SIREN represents `PSF(x, y, px, py)`. Spatial NeuralBD therefore always uses
`type: spatial` and `representation: siren`.

`target_size` defines the full learned PSF support. Progressive training may start from a
smaller active support and grow toward this target without changing the underlying model
parameters during distributed training.

## Progressive training

Progressive training starts from a small active PSF support, typically `1x1`, and many
training points per batch. This first stage behaves like an identity observation model, giving
the image model a stable warm start without a separate pretraining phase. Later stages increase
the PSF size by one pixel on each side every `increase_every_n_epochs` epochs, reduce the
sampling points per batch, and keep training with the final PSF after the full support is reached.
Generated growth steps keep the approximate sample budget constant:

```text
sampling_points * psf_width * psf_height
```

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

The generated growth uses odd PSF sizes between `start_psf_size` and the target PSF support,
for example `1x1`, `3x3`, `5x5`, and so on. Each growth step adds one pixel on every side.
If `model.psf.target_size: 65`, a `1x1` stage with `16384` sampling points ends near `4`
sampling points at `65x65`.
With `fixed_epoch_size: true` and `epoch_iterations: 1000`, progressive training keeps each
epoch at 1000 batches even as `sampling_points` changes.
For full manual control, define explicit stages:

```yaml
training:
  progressive:
    enabled: true
    stages:
      - epochs: 200
        psf_size: 3
        training_points: 8192
        learning_rate: 3.0e-4
      - epochs: 400
        psf_size: 17
        training_points: 4096
        learning_rate: 1.0e-4
      - epochs: 800
        psf_size: 65
        training_points: 1024
        learning_rate: 3.0e-5
```

## Optional pretraining

Pretraining fits the image SIREN to one reference target before the full NeuralBD
optimization starts. This gives the latent reconstruction a stable initialization while
keeping the PSF untouched. Prefer `1x1` progressive PSF growth for most runs; use this
separate pretraining stage when you want to fit a specific reference target before blind
deconvolution.

```yaml
pretraining:
  enabled: true
  epochs: 500
  learning_rate: 1.0e-4
  target: first_frame
  frame_index: 0
```

`target` can be `first_frame`, `frame`, or `mean`.

## Learned frame shifts

Frame-shift learning is disabled by default. Enable it when observed frames have relative
translations that should be modeled separately from the centered PSF:

```yaml
model:
  registration:
    enabled: true
    sample_frames: true
    anchor: first_frame
    max_pixels: 5
    regularization: 1.0e-4
```

With `sample_frames: true`, training samples one frame per coordinate and optimizes only
that frame's PSF and shift for the sampled point. This avoids evaluating every frame for
every coordinate in large bursts. Validation still evaluates the full frame stack.

## Weights & Biases logging

Install the visualization extra to log scalar training metrics, learning-rate values, and
sampled validation figures to the `NeuralBD` project. W&B logging is enabled by default:

```bash
pip install -e ".[viz]"
```

```yaml
logging:
  wandb: true
  project: NeuralBD
  name: neuralbd
```

Local W&B files are written to `work_dir/wandb`.

## Validation figures

Validation runs every 10 epochs by default. Override the interval under `training`:

```yaml
training:
  validation_every_n_epochs: 10
```

Validation logs representative figures to W&B by default, rather than rendering every
frame and channel in large bursts:

```yaml
outputs:
  sample_count: 5
  figure_subregion_size: 512
  figure_subregion: null
  reference_path: null
  reference_array_key: null
```

Figures include learned PSFs, input versus latent reconstruction, convolved prediction
versus observed reference, and optional reconstruction versus ground truth. Image
figures use a centered `figure_subregion_size` crop for logging speed; set it to
`null` to render the full validation frame. To log an explicit centered pixel-coordinate
window, use:

```yaml
outputs:
  figure_subregion:
    x: [100, 200]
    y: [300, 400]
```
