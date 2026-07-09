# GREGOR/HIFI Workflow

This workflow prepares a GREGOR/HIFI `.fits` or `.fts` burst, trains NeuralBD, exports
the latent reconstruction, and summarizes the run outputs.

Run the full default workflow with:

```bash
scripts/gregor_workflow.sh
```

By default the script writes:

- prepared burst: `/glade/derecho/scratch/rjarolim/neuralbd/work/gregor_hifi/data/processed_burst.npz`
- runtime config: `/glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi/gregor_hifi.runtime.yaml`
- work directory: `/glade/derecho/scratch/rjarolim/neuralbd/work/gregor_hifi`
- checkpoint: `/glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi/neuralbd.nbd`
- reconstruction: `/glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi/reconstruction.npy`

## Prepare the burst

The preparation step reads the FITS file, converts the data to the NeuralBD
channels-last convention, applies optional frame/channel/subframe selection, and writes
a compressed `.npz` file. The file always contains:

- `images`: `float32` data with shape `(height, width, frames, channels)`
- `metadata`: JSON text with the source path, output layout, shape, dtype, and preparation options

The full workflow uses the same preparation defaults as the standalone command:
`GREGOR_INPUT`, `PREPARED_BURST`, `AXIS_ORDER`, and `SUBFRAME_SIZE`.

For an initial smoke test, restrict the data volume:

```bash
WORK_DIR=/glade/derecho/scratch/rjarolim/neuralbd/work/gregor_hifi_smoke \
PREPARED_BURST=/glade/derecho/scratch/rjarolim/neuralbd/work/gregor_hifi_smoke/data/processed_burst.npz \
RUN_DIR=/glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi_smoke \
N_IMAGES=16 \
SUBFRAME_SIZE=256 \
scripts/gregor_workflow.sh
```

The `images` array has shape:

```text
height, width, frames, channels
```

## FITS layouts

`--axis-order auto` handles the common cases where the FITS data is either a stack of
2D image HDUs, a `(frames, y, x)` cube, or a `(y, x, frames)` cube.

If the file uses a known cube layout, pass it explicitly:

```bash
--axis-order fyx
--axis-order yxf
--axis-order fyxc
--axis-order yxfc
```

The letters mean `f` for frame, `y` and `x` for image axes, and `c` for channel.

## Configure Training

The workflow generates a runtime config from `examples/configs/gregor_hifi.yaml` and
patches `base_dir`, `work_dir`, and `data.path` to match `RUN_DIR`, `WORK_DIR`, and
`PREPARED_BURST`. By default, prepared data is stored below `WORK_DIR/data`.

To use a custom YAML file directly, set `CONFIG`:

```bash
CONFIG=/path/to/gregor_hifi.yaml scripts/gregor_workflow.sh
```

If you manage the YAML yourself, the data section should reference the prepared `.npz`
file:

```yaml
base_dir: /glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi
work_dir: /glade/derecho/scratch/rjarolim/neuralbd/work/gregor_hifi

data:
  type: npz
  path: /glade/derecho/scratch/rjarolim/neuralbd/work/gregor_hifi/data/processed_burst.npz
  channels: [0]

model:
  psf:
    target_size: 65
```

The prepared GREGOR/HIFI burst keeps the two interleaved source channels. Select the
single channel used for training with `data.channels` in the YAML.

The full example config is available at `examples/configs/gregor_hifi.yaml`.

## Train and Export

The example configuration enables:

- positive `minmax` normalization
- standard NeuralBD with a shared PSF across channels
- progressive PSF growth from `1x1` to `65x65` with area-matched training point reduction
- validation figures logged to W&B

Switch to one PSF per channel in the YAML with:

```yaml
model:
  psf:
    channel_mode: per_channel
```

Switch to spatially varying NeuralBD in the YAML with:

```yaml
method: spatial
model:
  psf:
    representation: siren
```

After training, the workflow runs:

```bash
nbd-reconstruct \
  --checkpoint /glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi/neuralbd.nbd \
  --out /glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi/reconstruction.npy
```

and then summarizes the output arrays with:

```bash
nbd-evaluate --run /glade/derecho/scratch/rjarolim/neuralbd/runs/gregor_hifi
```

## Run Selected Steps

Each phase can be skipped:

```bash
RUN_PREPARE=0 scripts/gregor_workflow.sh
RUN_TRAIN=0 scripts/gregor_workflow.sh
RUN_RECONSTRUCT=0 scripts/gregor_workflow.sh
RUN_EVALUATE=0 scripts/gregor_workflow.sh
```
