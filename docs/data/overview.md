# Data

The stable public data interface is a burst array with shape:

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

The built-in loader types are:

- `numpy`: `.npy` or `.npz`
- `npz`: explicit `.npz`
- `fits`: FITS HDU stack, requires `neuralbd[io]`
- `gregor`: GREGOR-style FITS HDU stack, requires `neuralbd[io]`
- `dkist`: `.npz` input with `cobs` by default
- `kso`: `.npy` or `.npz`

Use `frame_indices`, `frame_slice`, `channels`, and `subframe` in the configuration to
load only the working subset needed for training or validation.
