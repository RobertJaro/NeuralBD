# Data

NeuralBD uses one public data convention throughout the training pipeline. Input loaders
must return a burst array with shape:

```text
height, width, frames
```

or:

```text
height, width, frames, channels
```

Instrument-specific loaders should be implemented as thin adapters that return this standard representation.
The training pipeline always uses channels last. Instrument adapters may convert legacy
source layouts internally, but the core modules do not expose configurable axis options.

All image values should be strictly positive. The default image model uses a positive
`softplus` output activation, and the default `minmax` normalization maps the burst into
a positive reconstruction domain.

The built-in loader types are:

- `numpy`: `.npy` or `.npz`
- `npz`: explicit `.npz`
- `fits`: FITS HDU stack, requires `neuralbd[io]`
- `gregor`: GREGOR-style FITS HDU stack, requires `neuralbd[io]`
- `dkist`: `.npz` input with `cobs` by default
- `kso`: `.npy` or `.npz`

Use `frame_indices`, `frame_slice`, `channels`, and `subframe` in the configuration to
load only the working subset needed for training or validation.

## Recommended preprocessing

Typical preprocessing before NeuralBD training includes:

- dark/flat correction and instrument calibration outside NeuralBD
- alignment or registration when required by the observing setup
- selection of useful frames from the burst
- optional channel selection for multi-channel observations
- optional subframe extraction for initial experiments
- normalization to a positive numeric range

NeuralBD keeps these steps explicit in the configuration so workflows remain reproducible.

## GREGOR/HIFI

Use `nbd-prepare-gregor` for GREGOR/HIFI `.fits` or `.fts` bursts. The command only
writes a processed channels-last `.npz` file with two keys: `images` and `metadata`.

```bash
nbd-prepare-gregor \
  --input /glade/work/cschirninger/data/hifi_20220602_095015_sd.fts \
  --output /glade/derecho/scratch/rjarolim/neuralbd/work/gregor_hifi/data/processed_burst.npz
```

Then create or edit a normal NeuralBD YAML config that points to the processed data and
train with:

```bash
nbd-train --config examples/configs/gregor_hifi.yaml
```

See the GREGOR/HIFI workflow page for layout options, smoke-test settings, spatial PSF
configuration, and export commands.
