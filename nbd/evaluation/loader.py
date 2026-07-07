import numpy as np
import sys
import torch
from torch import nn
from tqdm import tqdm


def _register_numpy_pickle_aliases():
    # NumPy 2 pickles may reference numpy._core, while NumPy 1 exposes numpy.core.
    if not hasattr(np, "_core"):
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        sys.modules.setdefault("numpy._core.umath", np.core.umath)


class NBDOutput:

    def __init__(self, model_path, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _register_numpy_pickle_aliases()
        state = torch.load(model_path, map_location=self.device, weights_only=False)

        self.image_coords = state['image_coords']
        self.model = state['image_model']
        self.model = nn.DataParallel(self.model)
        self.model.eval()

    def load(self, coords, batch_size=2048, progress=True):
        batch_size = batch_size * torch.cuda.device_count() if torch.cuda.is_available() else 1024
        coordinates_tensor = torch.from_numpy(coords).float().view(-1, 2)

        n_batches = int(np.ceil(coordinates_tensor.shape[0] / batch_size))
        iter_ = tqdm(range(n_batches)) if progress else range(n_batches)
        output_image = []
        for i in iter_:
            batch_coords = coordinates_tensor[i * batch_size:(i + 1) * batch_size].to(self.device)

            pred = self.model(batch_coords)

            output_image += [pred.detach().cpu().numpy()]
        output_image = np.concatenate(output_image, axis=0).reshape((coords.shape[0], coords.shape[1], 2))

        return output_image

    def load_reconstructed_img(self, **kwargs):
        return self.load(self.image_coords)


class NBDSVOutput:

    def __init__(self, model_path, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _register_numpy_pickle_aliases()
        state = torch.load(model_path, map_location=self.device, weights_only=False)

        self.image_coords = state['image_coords']
        self.img_model = state['image_model']
        self.img_model = nn.DataParallel(self.img_model)
        self.img_model.eval()
        self.psf_model = state['psf_model']
        self.psf_model = nn.DataParallel(self.psf_model)
        self.psf_model.eval()


    def load_img(self, coords, batch_size=2048, progress=True):
        batch_size = batch_size * torch.cuda.device_count() if torch.cuda.is_available() else 1024
        coordinates_tensor = torch.from_numpy(coords).float().view(-1, 2)

        n_batches = int(np.ceil(coordinates_tensor.shape[0] / batch_size))
        iter_ = tqdm(range(n_batches)) if progress else range(n_batches)
        output_image = []
        for i in iter_:
            batch_coords = coordinates_tensor[i * batch_size:(i + 1) * batch_size].to(self.device)

            pred = self.img_model(batch_coords)

            output_image += [pred.detach().cpu().numpy()]
        output_image = np.concatenate(output_image, axis=0).reshape((coords.shape[0], coords.shape[1], 2))

        return output_image

    def load_psfs(self, coords):
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords).float()
        elif not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)
        else:
            coords = coords.float()

        coords = coords.to(self.device)

        psf_module = self.psf_model.module if isinstance(self.psf_model, nn.DataParallel) else self.psf_model
        d_in_features = psf_module.d_in.in_features

        # Compatibility with checkpoints trained with encoded PSF inputs.
        if coords.shape[-1] != d_in_features:
            if coords.shape[-1] == 2 and hasattr(psf_module, "posenc"):
                encoded = psf_module.posenc(coords)
                if encoded.shape[-1] == d_in_features:
                    coords = encoded
                else:
                    raise RuntimeError(
                        f"PSF input mismatch: got {coords.shape[-1]} features, "
                        f"d_in expects {d_in_features}, and posenc gives {encoded.shape[-1]}."
                    )
            else:
                raise RuntimeError(
                    f"PSF input mismatch: got {coords.shape[-1]} features, "
                    f"but d_in expects {d_in_features}."
                )

        log_psfs = self.psf_model(coords)  # --> batch, x, y, n_images
        psfs = torch.exp(log_psfs)  # --> batch, x, y, n_images
        psfs = psfs[0, ...]

        # Normalize PSFs
        norm = psfs.sum(dim=(0, 1), keepdim=True)  # --> 1, 1, n_images

        return psfs / (norm + 1e-8)

    def load_reconstructed_img(self, **kwargs):
        return self.load_img(self.image_coords)
