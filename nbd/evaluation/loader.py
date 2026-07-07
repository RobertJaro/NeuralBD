import numpy as np
import torch

from nbd.processing import image_coordinates
from nbd.train.module import NeuralBDModule


class NeuralBDOutput:
    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.metadata = self.state.get("metadata", {})
        self.image_coords = self._default_coords()
        self.module = self._load_module()
        self.image_model = self.module.image_model
        self.psf_model = self.module.psf_model

    def _load_module(self):
        if "model_state_dict" in self.state:
            config = self.state["config"]
            images_shape = self.metadata["images_shape"]
            module = NeuralBDModule.from_config(config, images_shape=images_shape)
            model_state = self.state["model_state_dict"]
            if "image" in model_state and "psf" in model_state:
                module.image_model.load_state_dict(model_state["image"])
                module.psf_model.load_state_dict(model_state["psf"])
            else:
                module.load_state_dict(model_state)
            return module.to(self.device).eval()

        module = torch.nn.Module()
        module.image_model = self.state["image_model"].to(self.device).eval()
        module.psf_model = self.state.get("psf_model")
        if module.psf_model is not None:
            module.psf_model = module.psf_model.to(self.device).eval()
        return module

    def _default_coords(self):
        if "images_shape" in self.metadata:
            pixel_per_ds = self.metadata.get("pixel_per_ds", 1.0)
            return image_coordinates(self.metadata["images_shape"], pixel_per_ds=pixel_per_ds)
        return self.state.get("image_coords")

    def reconstruct(self, coords=None, batch_size=8192):
        coords = self.image_coords if coords is None else coords
        if coords is None:
            raise ValueError("No coordinates provided and checkpoint does not contain image_coords")
        coords_tensor = torch.as_tensor(coords, dtype=torch.float32, device=self.device).reshape(-1, 2)
        chunks = []
        with torch.no_grad():
            for idx in range(0, coords_tensor.shape[0], batch_size):
                chunks.append(self.image_model(coords_tensor[idx:idx + batch_size]).detach().cpu().numpy())
        return np.concatenate(chunks, axis=0).reshape(coords.shape[0], coords.shape[1], -1)
