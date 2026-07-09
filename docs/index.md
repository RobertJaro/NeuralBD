# NeuralBD

![NeuralBD logo](_static/neuralbd_logo.png)

NeuralBD is a neural blind deconvolution framework for reconstructing solar images from
bursts of degraded short-exposure observations. It learns a continuous latent image and
the point spread functions that explain the observed burst through differentiable
convolution.

The package supports standard burst deconvolution with one PSF per frame, spatially
varying PSFs, multi-channel image bursts, configurable PSF representations, progressive
PSF growth schedules, pretraining, validation diagnostics, and reconstruction export.

Start with the workflow and quickstart pages if you are setting up a new reconstruction.

```{toctree}
:maxdepth: 2

installation
quickstart
workflow
configuration
methods/standard
methods/spatial
data/overview
data/gregor_hifi
api/models
api/train
api/data
development
```
