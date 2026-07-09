# Standard NeuralBD

The standard method models a burst with one latent image and one PSF per observed frame.
The image model predicts the sharp reconstruction at continuous coordinates, and the PSF
model describes how that reconstruction is blurred into each degraded frame.

Use the standard method when the degradation can be approximated as spatially invariant
across the selected field of view.

## PSF options

The PSF can be represented as direct learnable parameters:

```yaml
model:
  psf:
    type: default
    representation: parameters
    size: 65
```

or as a SIREN field evaluated at PSF sample coordinates:

```yaml
model:
  psf:
    type: default
    representation: siren
    size: 65
    permute_samples: true
```

For multi-channel bursts, use `channel_mode: shared` for one PSF per frame or
`channel_mode: per_channel` for independent PSFs per frame and channel.

## Learned frame shifts

If relative frame translations should be learned jointly with the PSFs, enable the optional
registration model:

```yaml
model:
  registration:
    enabled: true
    sample_frames: true
    max_pixels: 5
```

The default is `enabled: false`. When enabled with frame sampling, each training point is
paired with one frame index so the image model is not evaluated for every frame in the
burst at every coordinate.

## Training strategy

For larger PSFs, enable progressive training. This starts with a restricted support, then
grows toward the target size while adjusting the number of training points and learning
rate.
