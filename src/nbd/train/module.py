import torch
import torch.distributed as dist
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import ExponentialLR

from nbd.models import FixedPSFModel, ImageSirenModel, NeuralBDConvolution, SirenPSFModel, SpatialPSFModel


class NeuralBDModule(LightningModule):
    def __init__(
        self,
        images_shape,
        method="standard",
        pixel_per_ds=1.0,
        image_config=None,
        psf_config=None,
        lr_config=None,
        validation_mapping=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.images_shape = tuple(images_shape)
        self.method = method
        self.n_frames = self.images_shape[2]
        self.n_channels = self.images_shape[3] if len(self.images_shape) == 4 else 1
        self.pixel_per_ds = pixel_per_ds
        self.lr_config = lr_config or {"start": 1e-4, "end": 1e-4, "iterations": 100000}

        image_config = dict(image_config or {})
        image_config.pop("type", None)
        if image_config.get("n_channels") is None:
            image_config["n_channels"] = self.n_channels
        image_channels = int(image_config["n_channels"])
        if image_channels != self.n_channels:
            raise ValueError(
                f"model.image.n_channels={image_channels} does not match data channels={self.n_channels}"
            )
        psf_config = dict(psf_config or {})
        psf_type = psf_config.pop("type", "fixed")
        representation = psf_config.pop("representation", "parameters")
        psf_size = tuple(psf_config.pop("size", (29, 29)))
        jitter = bool(psf_config.pop("jitter", False))
        permute_psf_samples = bool(psf_config.pop("permute_samples", False))
        channel_mode = psf_config.pop("channel_mode", "shared")

        self.image_model = ImageSirenModel(**image_config)
        if psf_type == "fixed" and representation == "parameters":
            for key in ("dim", "n_layers", "w0", "w0_init"):
                psf_config.pop(key, None)
            self.psf_model = FixedPSFModel(
                n_frames=self.n_frames,
                n_channels=self.n_channels,
                channel_mode=channel_mode,
                psf_size=psf_size,
                **psf_config,
            )
        elif psf_type == "fixed" and representation == "siren":
            psf_config.pop("sigma", None)
            self.psf_model = SirenPSFModel(
                n_frames=self.n_frames,
                n_channels=self.n_channels,
                channel_mode=channel_mode,
                **psf_config,
            )
        elif psf_type == "spatial" and representation == "siren":
            psf_config.pop("sigma", None)
            self.psf_model = SpatialPSFModel(
                n_frames=self.n_frames,
                n_channels=self.n_channels,
                channel_mode=channel_mode,
                **psf_config,
            )
        elif psf_type == "spatial" and representation == "parameters":
            raise ValueError("Spatially varying PSFs require model.psf.representation='siren'")
        else:
            raise ValueError(f"Unknown PSF model type or representation: {psf_type}/{representation}")

        self.convolution = NeuralBDConvolution(
            image_model=self.image_model,
            psf_model=self.psf_model,
            psf_size=psf_size,
            pixel_per_ds=pixel_per_ds,
            jitter=jitter,
            permute_psf_samples=permute_psf_samples,
        )
        self.validation_batches = {}
        self.validation_outputs = {}
        self.validation_mapping = validation_mapping or {0: "validation"}

    @classmethod
    def from_config(cls, config, images_shape):
        return cls(
            images_shape=images_shape,
            method=config["method"],
            pixel_per_ds=config["data"]["pixel_per_ds"],
            image_config=config["model"]["image"],
            psf_config=config["model"]["psf"],
            lr_config=config["training"]["learning_rate"],
            validation_mapping=config.get("validation_mapping"),
        )

    def forward(self, coords):
        return self.convolution(coords)

    def set_psf_size(self, psf_size):
        if isinstance(psf_size, int):
            psf_size = (psf_size, psf_size)
        psf_size = tuple(int(size) for size in psf_size)
        if hasattr(self.psf_model, "resize"):
            self.psf_model.resize(psf_size)
        self.convolution.set_psf_size(psf_size)

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            return batch["images"], batch["coords"], batch.get("indices")
        if len(batch) == 3:
            return batch
        images, coords = batch
        return images, coords, None

    def training_step(self, batch, batch_idx):
        images, coords, _indices = self._unpack_batch(batch)
        pred = self(coords)
        loss = torch.mean((pred - images) ** 2)
        self.log("train.loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        lr_start = self.lr_config["start"]
        lr_end = self.lr_config["end"]
        iterations = self.lr_config["iterations"]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr_start)
        scheduler = ExponentialLR(optimizer, gamma=(lr_end / lr_start) ** (1 / iterations))
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
            self.log("train.learning_rate", scheduler.get_last_lr()[0])

    def on_validation_epoch_start(self):
        self.validation_batches = {}
        self.validation_outputs = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, coords, indices = self._unpack_batch(batch)
        pred = self(coords)
        image_pred = self.image_model(coords)
        loss = torch.mean((pred - images) ** 2)
        output = {
            "coords": coords.detach(),
            "convolved_true": images.detach(),
            "convolved_pred": pred.detach(),
            "image_pred": image_pred.detach(),
            "loss": loss.detach().reshape(1),
            "images_shape": torch.tensor(self.images_shape, device=coords.device, dtype=torch.long),
        }
        if indices is not None:
            output["indices"] = indices.detach()
            output["dataset_idx"] = indices[:1].detach()
        return output

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs is None:
            return
        cpu_out = {key: value.detach().cpu() for key, value in outputs.items()}
        self.validation_batches.setdefault(dataloader_idx, []).append(cpu_out)

    def on_validation_epoch_end(self):
        outputs_by_loader = self.validation_batches
        if not outputs_by_loader:
            return

        if dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
            if rank == 0:
                gathered = [None] * world
                dist.gather_object(outputs_by_loader, gathered, dst=0)
                merged = {}
                for rank_outputs in gathered:
                    for loader_idx, outputs in rank_outputs.items():
                        merged.setdefault(loader_idx, []).extend(outputs)
                outputs_by_loader = merged
            else:
                dist.gather_object(outputs_by_loader, None, dst=0)
                return

        metadata_keys = {"images_shape"}
        for loader_idx, outputs in outputs_by_loader.items():
            outputs = self._sort_validation_outputs(outputs)
            keys = outputs[0].keys()
            merged = {}
            for key in keys:
                if key in metadata_keys:
                    merged[key] = outputs[0][key].reshape(-1)
                else:
                    merged[key] = torch.cat([out[key] for out in outputs], dim=0)
            merged.pop("dataset_idx", None)
            ds_name = self.validation_mapping.get(loader_idx, loader_idx)
            self.validation_outputs[ds_name] = merged
            self.log(f"val.{ds_name}.loss", merged["loss"].mean(), prog_bar=True)

    @staticmethod
    def _sort_validation_outputs(outputs):
        indexed = []
        seen = set()
        for idx, output in enumerate(outputs):
            dataset_idx = output.get("dataset_idx")
            if dataset_idx is None:
                indexed.append((idx, idx))
                continue
            if dataset_idx.ndim > 0:
                dataset_idx = dataset_idx.reshape(-1)[0]
            dataset_idx = int(dataset_idx)
            if dataset_idx in seen:
                continue
            seen.add(dataset_idx)
            indexed.append((dataset_idx, idx))
        return [outputs[idx] for _dataset_idx, idx in sorted(indexed, key=lambda item: item[0])]
