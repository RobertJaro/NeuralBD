import math
from pathlib import Path

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from neuralbd.data import load_numpy_burst
from neuralbd.io import save_model_state, save_validation_outputs
from neuralbd.io.outputs import restore_image_order
from neuralbd.models import build_psf_grid


def _odd_size(value):
    value = int(round(value))
    return value if value % 2 == 1 else value + 1


def _as_pair(value):
    if isinstance(value, int):
        return [value, value]
    return [int(value[0]), int(value[1])]


def _log_interpolate(start, end, fraction):
    start = float(start)
    end = float(end)
    if start <= 0 or end <= 0:
        return start + (end - start) * fraction
    return math.exp(math.log(start) + (math.log(end) - math.log(start)) * fraction)


class ProgressiveTrainingCallback(Callback):
    def __init__(self, config):
        self.config = config
        self.progressive_cfg = config["training"].get("progressive", {})
        self.stages = []
        self.active_stage_idx = None

    def setup(self, trainer, pl_module, stage=None):
        self.stages = self._build_stages(trainer.max_epochs)

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.progressive_cfg.get("enabled", False):
            return
        stage_idx = self._stage_index(trainer.current_epoch)
        if stage_idx == self.active_stage_idx:
            return
        self.active_stage_idx = stage_idx
        stage = self.stages[stage_idx]
        pl_module.set_psf_size(stage["psf_size"])
        self._refresh_optimizer_parameters(trainer, pl_module)
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None and stage.get("training_points") is not None:
            datamodule.set_train_batch_size(stage["training_points"])
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        if optimizer is not None and stage.get("learning_rate") is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(stage["learning_rate"])
        pl_module.log("progressive.psf_size", float(stage["psf_size"][0]), prog_bar=True)
        if stage.get("training_points") is not None:
            pl_module.log("progressive.training_points", float(stage["training_points"]))
        if stage.get("learning_rate") is not None:
            pl_module.log("progressive.learning_rate", float(stage["learning_rate"]))

    def _build_stages(self, max_epochs):
        if not self.progressive_cfg.get("enabled", False):
            return []
        explicit_stages = self.progressive_cfg.get("stages")
        if explicit_stages:
            cursor = 0
            stages = []
            for stage in explicit_stages:
                epochs = int(stage["epochs"])
                stages.append({
                    "start_epoch": cursor,
                    "end_epoch": cursor + epochs,
                    "psf_size": _as_pair(stage["psf_size"]),
                    "training_points": stage.get("training_points") or stage.get("batch_size"),
                    "learning_rate": stage.get("learning_rate"),
                })
                cursor += epochs
            return stages

        n_stages = max(1, int(self.progressive_cfg.get("n_stages", 1)))
        start_size = int(self.progressive_cfg.get("start_psf_size", 3))
        target_size = int(self.progressive_cfg.get("target_psf_size", self.config["model"]["psf"]["size"][0]))
        start_points = (
            self.progressive_cfg.get("training_points_start")
            or self.progressive_cfg.get("batch_size_start")
            or self.config["data"]["batch_size"]
        )
        end_points = (
            self.progressive_cfg.get("training_points_end")
            or self.progressive_cfg.get("batch_size_end")
            or start_points
        )
        lr_cfg = self.config["training"]["learning_rate"]
        start_lr = self.progressive_cfg.get("learning_rate_start") or lr_cfg["start"]
        end_lr = self.progressive_cfg.get("learning_rate_end") or lr_cfg["end"]

        stages = []
        for idx in range(n_stages):
            fraction = idx / max(n_stages - 1, 1)
            size = _odd_size(start_size + (target_size - start_size) * fraction)
            points = int(round(_log_interpolate(start_points, end_points, fraction)))
            learning_rate = _log_interpolate(start_lr, end_lr, fraction)
            start_epoch = round(max_epochs * idx / n_stages)
            end_epoch = round(max_epochs * (idx + 1) / n_stages)
            stages.append({
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "psf_size": [size, size],
                "training_points": max(1, points),
                "learning_rate": learning_rate,
            })
        return stages

    def _stage_index(self, epoch):
        for idx, stage in enumerate(self.stages):
            if stage["start_epoch"] <= epoch < stage["end_epoch"]:
                return idx
        return max(len(self.stages) - 1, 0)

    @staticmethod
    def _refresh_optimizer_parameters(trainer, pl_module):
        if not trainer.optimizers:
            return
        optimizer = trainer.optimizers[0]
        existing = {param for group in optimizer.param_groups for param in group["params"]}
        current = list(pl_module.parameters())
        if all(param in existing for param in current):
            return
        optimizer.param_groups[0]["params"] = current


class NeuralBDOutputCallback(Callback):
    def __init__(
        self,
        base_dir,
        config=None,
        save_validation_arrays=True,
        save_validation_figures=True,
        sample_count=5,
    ):
        self.base_dir = Path(base_dir)
        self.config = config
        self.save_validation_arrays = save_validation_arrays
        self.save_validation_figures = save_validation_figures
        self.sample_count = int(sample_count)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        save_model_state(self.base_dir / "neuralbd.nbd", pl_module, config=self.config)
        outputs = self._primary_validation_outputs(pl_module)
        if self.save_validation_arrays and outputs is not None:
            output_dir = self.base_dir / "outputs"
            save_validation_outputs(
                output_dir,
                outputs,
                images_shape=pl_module.images_shape,
            )
        if self.save_validation_figures and outputs is not None:
            self._save_validation_figures(trainer, pl_module, outputs)

    @staticmethod
    def _primary_validation_outputs(pl_module):
        if not pl_module.validation_outputs:
            return None
        if "validation" in pl_module.validation_outputs:
            return pl_module.validation_outputs["validation"]
        first_key = next(iter(pl_module.validation_outputs))
        return pl_module.validation_outputs[first_key]

    def _save_validation_figures(self, trainer, pl_module, outputs):
        figures = self._build_figures(pl_module, outputs)
        if not figures:
            return

        figure_dir = self.base_dir / "outputs" / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        for name, figure in figures.items():
            figure.savefig(figure_dir / f"{name}.png", dpi=150, bbox_inches="tight")
            self._log_wandb(trainer, name, figure)
            figure.clf()

    def _build_figures(self, pl_module, outputs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return {}

        images_shape = tuple(int(v) for v in outputs.get("images_shape", pl_module.images_shape))
        n_pixels = images_shape[0] * images_shape[1]
        if outputs["convolved_true"].shape[0] < n_pixels:
            return {}
        indices = outputs.get("indices")
        if indices is not None:
            indices = indices.numpy().astype("int64")

        if indices is not None:
            convolved_true = restore_image_order(outputs["convolved_true"].numpy(), indices, images_shape, "convolved_true")
            convolved_pred = restore_image_order(outputs["convolved_pred"].numpy(), indices, images_shape, "convolved_pred")
            image_pred = restore_image_order(outputs["image_pred"].numpy(), indices, images_shape, "image_pred")
        else:
            convolved_true = outputs["convolved_true"][:n_pixels].reshape(images_shape).numpy()
            convolved_pred = outputs["convolved_pred"][:n_pixels].reshape(images_shape).numpy()
            image_pred = outputs["image_pred"][:n_pixels].reshape(
                images_shape[0], images_shape[1], images_shape[-1]
            ).numpy()
        sample_frames = list(range(min(self.sample_count, images_shape[2])))
        sample_channels = list(range(min(self.sample_count, images_shape[3])))

        figures = {
            "psfs": self._plot_psfs(plt, pl_module, sample_frames, sample_channels),
            "input_vs_reconstruction": self._plot_input_vs_reconstruction(
                plt, convolved_true, image_pred, sample_frames, sample_channels
            ),
            "predicted_vs_reference": self._plot_predicted_vs_reference(
                plt, convolved_true, convolved_pred, sample_frames, sample_channels
            ),
        }
        reference = self._load_reference()
        if reference is not None:
            figures["reconstruction_vs_ground_truth"] = self._plot_ground_truth(
                plt, image_pred, reference, sample_channels
            )
        return {key: value for key, value in figures.items() if value is not None}

    def _plot_psfs(self, plt, pl_module, sample_frames, sample_channels):
        psf_model = pl_module.psf_model
        if hasattr(psf_model, "channel_mode") and psf_model.channel_mode == "per_channel":
            n_cols = len(sample_frames) * len(sample_channels)
        else:
            n_cols = len(sample_frames)
        if n_cols == 0:
            return None

        if hasattr(psf_model, "model"):
            coords = pl_module.convolution.psf_coords.new_tensor([[0.0, 0.0]])
            psf_coords, area = build_psf_grid(
                pl_module.convolution.psf_size,
                pixel_per_ds=pl_module.convolution.pixel_per_ds,
                device=coords.device,
                dtype=coords.dtype,
            )
            psf = psf_model(coords, psf_coords[None, ...], area[None, ...])[0].detach().cpu().numpy()
        else:
            psf = psf_model().detach().cpu().numpy()

        fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.2), squeeze=False)
        col = 0
        for frame in sample_frames:
            if psf.ndim == 4:
                for channel in sample_channels:
                    axes[0, col].imshow(psf[:, :, frame, channel], origin="lower", cmap="magma")
                    axes[0, col].set_title(f"f{frame} c{channel}")
                    axes[0, col].axis("off")
                    col += 1
            else:
                axes[0, col].imshow(psf[:, :, frame], origin="lower", cmap="magma")
                axes[0, col].set_title(f"f{frame}")
                axes[0, col].axis("off")
                col += 1
        fig.suptitle("Learned PSFs")
        return fig

    def _plot_input_vs_reconstruction(self, plt, observed, reconstructed, sample_frames, sample_channels):
        cols = min(self.sample_count, len(sample_frames) * len(sample_channels))
        if cols == 0:
            return None
        fig, axes = plt.subplots(2, cols, figsize=(2.4 * cols, 4.8), squeeze=False)
        for col, (frame, channel) in enumerate(self._sample_pairs(sample_frames, sample_channels, cols)):
            axes[0, col].imshow(observed[:, :, frame, channel], origin="lower", cmap="gray")
            axes[0, col].set_title(f"input f{frame} c{channel}")
            axes[1, col].imshow(reconstructed[:, :, channel], origin="lower", cmap="gray")
            axes[1, col].set_title(f"recon c{channel}")
            axes[0, col].axis("off")
            axes[1, col].axis("off")
        return fig

    def _plot_predicted_vs_reference(self, plt, reference, predicted, sample_frames, sample_channels):
        cols = min(self.sample_count, len(sample_frames) * len(sample_channels))
        if cols == 0:
            return None
        fig, axes = plt.subplots(2, cols, figsize=(2.4 * cols, 4.8), squeeze=False)
        for col, (frame, channel) in enumerate(self._sample_pairs(sample_frames, sample_channels, cols)):
            axes[0, col].imshow(predicted[:, :, frame, channel], origin="lower", cmap="gray")
            axes[0, col].set_title(f"pred f{frame} c{channel}")
            axes[1, col].imshow(reference[:, :, frame, channel], origin="lower", cmap="gray")
            axes[1, col].set_title(f"ref f{frame} c{channel}")
            axes[0, col].axis("off")
            axes[1, col].axis("off")
        return fig

    def _plot_ground_truth(self, plt, reconstructed, reference, sample_channels):
        if reference.ndim == 3:
            reference = reference[..., None]
        cols = min(self.sample_count, len(sample_channels), reconstructed.shape[-1], reference.shape[-1])
        if cols == 0:
            return None
        fig, axes = plt.subplots(2, cols, figsize=(2.4 * cols, 4.8), squeeze=False)
        for col, channel in enumerate(sample_channels[:cols]):
            axes[0, col].imshow(reconstructed[:, :, channel], origin="lower", cmap="gray")
            axes[0, col].set_title(f"recon c{channel}")
            axes[1, col].imshow(reference[:, :, channel], origin="lower", cmap="gray")
            axes[1, col].set_title(f"gt c{channel}")
            axes[0, col].axis("off")
            axes[1, col].axis("off")
        return fig

    def _sample_pairs(self, frames, channels, limit):
        pairs = [(frame, channel) for frame in frames for channel in channels]
        return pairs[:limit]

    def _load_reference(self):
        output_cfg = (self.config or {}).get("outputs", {})
        reference_path = output_cfg.get("reference_path")
        if reference_path is None:
            return None
        return load_numpy_burst(reference_path, array_key=output_cfg.get("reference_array_key"))

    def _log_wandb(self, trainer, name, figure):
        logger = getattr(trainer, "logger", None)
        experiment = getattr(logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        try:
            import wandb
        except ImportError:
            return
        experiment.log({f"validation/{name}": wandb.Image(figure)}, step=trainer.global_step)
