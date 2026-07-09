# Spatial NeuralBD

The spatial method models PSFs as a function of image position and local PSF sample
coordinate. This allows the degradation to vary across the field of view.

The spatial PSF model is a SIREN representation of:

```text
PSF(x, y, px, py)
```

where `(x, y)` is the image coordinate and `(px, py)` is the local PSF sample coordinate.

Use the spatial method when a single PSF per frame is not sufficient, for example when
anisoplanatic effects or field-dependent aberrations are visible across the selected
subframe.

## Configuration

Spatial NeuralBD uses the SIREN PSF representation:

```yaml
method: spatial

model:
  psf:
    type: spatial
    representation: siren
    size: 65
    dim: 128
    n_layers: 4
    permute_samples: true
```

Progressive PSF growth is recommended for large target supports.
