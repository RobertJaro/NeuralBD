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
    representation: parameters
    size: 65
```

or as a SIREN field evaluated at PSF sample coordinates:

```yaml
model:
  psf:
    representation: siren
    size: 65
    permute_samples: true
```

For multi-channel bursts, use `channel_mode: shared` for one PSF per frame or
`channel_mode: per_channel` for independent PSFs per frame and channel.

## Training strategy

For larger PSFs, enable progressive training. This starts with a restricted support, then
grows toward the target size while adjusting the number of training points and learning
rate.
