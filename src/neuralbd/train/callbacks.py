import math
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from neuralbd.data import load_numpy_burst
from neuralbd.io import save_model_state
from neuralbd.models import build_psf_grid


def _odd_size(value):
    value = int(round(value))
    return value if value % 2 == 1 else value + 1


def _log_interpolate(start, end, fraction):
    start = float(start)
    end = float(end)
    if start <= 0 or end <= 0:
        return start + (end - start) * fraction
    return math.exp(math.log(start) + (math.log(end) - math.log(start)) * fraction)


def _psf_area(psf_size):
    return int(psf_size[0]) * int(psf_size[1])


def _scheduled_training_points(start_points, start_psf_size, psf_size):
    sample_budget = int(start_points) * _psf_area(start_psf_size)
    return max(1, int(round(sample_budget / _psf_area(psf_size))))


def progressive_sampling_points(config, psf_size=None):
    progressive_cfg = config["training"].get("progressive", {})
    start_points = progressive_cfg.get("sampling_points")
    if start_points is None:
        raise ValueError("training.progressive.sampling_points is required")
    return progressive_stage_points(progressive_cfg, start_points, psf_size=psf_size)


def progressive_validation_sampling_points(config, psf_size=None):
    progressive_cfg = config["training"].get("progressive", {})
    start_points = progressive_cfg.get("validation_sampling_points")
    if start_points is None:
        start_points = min(int(progressive_cfg.get("sampling_points") or 16384), 16384)
    if start_points is None:
        raise ValueError("training.progressive.sampling_points is required")
    return progressive_stage_points(progressive_cfg, start_points, psf_size=psf_size)


def progressive_stage_points(progressive_cfg, start_points, psf_size=None):
    start_size = _odd_size(progressive_cfg.get("start_psf_size", 1))
    start_psf_size = [start_size, start_size]
    if psf_size is None:
        psf_size = start_psf_size
    elif isinstance(psf_size, int):
        psf_size = [psf_size, psf_size]
    return _scheduled_training_points(start_points, start_psf_size, psf_size)


class ProgressiveTrainingCallback(Callback):
    def __init__(self, config, resume=False):
        self.config = config
        self.progressive_cfg = config["training"].get("progressive", {})
        self.resume = bool(resume)
        self.stages = []
        self.active_stage_idx = None
        self.fixed_epoch_size_set = False

    def setup(self, trainer, pl_module, stage=None):
        self.stages = self._build_stages(trainer.max_epochs)
        if self.progressive_cfg.get("enabled", False) and self.stages and not self._is_resuming(trainer):
            self._apply_current_stage(trainer, pl_module)

    def on_fit_start(self, trainer, pl_module):
        if self.progressive_cfg.get("enabled", False) and not self._is_resuming(trainer):
            self._apply_current_stage(trainer, pl_module)

    def on_sanity_check_start(self, trainer, pl_module):
        if self.progressive_cfg.get("enabled", False):
            self._apply_current_stage(trainer, pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.progressive_cfg.get("enabled", False):
            self._apply_current_stage(trainer, pl_module)
            stage_idx = self._stage_index(trainer.current_epoch)
            stage = self.stages[stage_idx]
            if getattr(rank_zero_only, "rank", 0) == 0:
                print(
                    "[NeuralBD:progressive] "
                    f"validation epoch={trainer.current_epoch} "
                    f"stage={stage_idx + 1}/{len(self.stages)} "
                    f"psf={stage['psf_size'][0]}x{stage['psf_size'][1]}",
                    flush=True,
                )

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.progressive_cfg.get("enabled", False):
            return
        stage_idx = self._stage_index(trainer.current_epoch)
        if stage_idx == self.active_stage_idx:
            return
        self.active_stage_idx = stage_idx
        stage = self.stages[stage_idx]
        self._apply_stage(trainer, pl_module, stage)
        self._refresh_optimizer_parameters(trainer, pl_module)
        optimizer = trainer.optimizers[0] if trainer.optimizers else None
        if optimizer is not None and stage.get("learning_rate") is not None:
            for group in optimizer.param_groups:
                group["lr"] = float(stage["learning_rate"])
        if getattr(rank_zero_only, "rank", 0) == 0:
            pl_module.log("progressive.psf_size", float(stage["psf_size"][0]), prog_bar=True)
            if stage.get("training_points") is not None:
                pl_module.log("progressive.training_points", float(stage["training_points"]))
            datamodule = getattr(trainer, "datamodule", None)
            if datamodule is not None and hasattr(datamodule.train_dataset, "unsharded_epoch_size"):
                pl_module.log("progressive.epoch_size", float(datamodule.train_dataset.unsharded_epoch_size))
            if stage.get("learning_rate") is not None:
                pl_module.log("progressive.learning_rate", float(stage["learning_rate"]))
            print(
                f"Progressive PSF stage {stage_idx + 1}/{len(self.stages)}: "
                f"psf={stage['psf_size'][0]}x{stage['psf_size'][1]}, "
                f"sampling_points={stage.get('training_points')}"
            )

    def _apply_current_stage(self, trainer, pl_module):
        if not self.stages:
            return
        stage = self.stages[self._stage_index(trainer.current_epoch)]
        self._apply_stage(trainer, pl_module, stage)

    def _apply_stage(self, trainer, pl_module, stage):
        pl_module.set_psf_size(stage["psf_size"])
        datamodule = getattr(trainer, "datamodule", None)
        if (
            datamodule is not None
            and self.progressive_cfg.get("fixed_epoch_size", True)
            and not self.fixed_epoch_size_set
        ):
            epoch_iterations = self.progressive_cfg.get("epoch_iterations")
            if hasattr(datamodule, "fix_train_epoch_size"):
                if epoch_iterations is None:
                    datamodule.fix_train_epoch_size()
                else:
                    datamodule.set_train_epoch_size(int(epoch_iterations))
            elif hasattr(datamodule, "train_dataset") and hasattr(datamodule.train_dataset, "set_epoch_size"):
                if epoch_iterations is None:
                    epoch_iterations = datamodule.train_dataset.unsharded_epoch_size
                datamodule.train_dataset.set_epoch_size(int(epoch_iterations))
            self.fixed_epoch_size_set = True
        if datamodule is not None and stage.get("training_points") is not None:
            if hasattr(datamodule, "set_train_sampling_points"):
                datamodule.set_train_sampling_points(stage["training_points"])
            else:
                datamodule.set_train_batch_size(stage["training_points"])
        if datamodule is not None and getattr(datamodule, "valid_dataset", None) is not getattr(
            datamodule, "train_dataset", None
        ):
            validation_points = progressive_validation_sampling_points(self.config, psf_size=stage["psf_size"])
            validation_points = int(validation_points)
            if hasattr(datamodule, "set_validation_sampling_points"):
                datamodule.set_validation_sampling_points(validation_points)
            elif hasattr(datamodule, "valid_dataset") and hasattr(datamodule.valid_dataset, "set_batch_size"):
                datamodule.valid_dataset.set_batch_size(validation_points)

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
                    "psf_size": stage["psf_size"],
                    "training_points": stage.get("training_points"),
                    "learning_rate": stage.get("learning_rate"),
                })
                cursor += epochs
            return stages

        start_size = int(self.progressive_cfg.get("start_psf_size", 1))
        target_size = int(self.progressive_cfg.get("target_psf_size", self.config["model"]["psf"]["size"][0]))
        increase_every = max(1, int(self.progressive_cfg.get("increase_every_n_epochs", 100)))
        start_size = _odd_size(start_size)
        target_size = _odd_size(target_size)
        if start_size >= target_size:
            sizes = [target_size]
        else:
            sizes = list(range(start_size, target_size + 1, 2))
            if sizes[-1] != target_size:
                sizes.append(target_size)
        start_psf_size = [start_size, start_size]
        start_points = self.progressive_cfg.get("sampling_points")
        if start_points is None:
            raise ValueError("training.progressive.sampling_points is required")
        lr_cfg = self.config["training"]["learning_rate"]
        start_lr = self.progressive_cfg.get("learning_rate_start") or lr_cfg["start"]
        end_lr = self.progressive_cfg.get("learning_rate_end") or lr_cfg["end"]

        stages = []
        for idx, size in enumerate(sizes):
            fraction = idx / max(len(sizes) - 1, 1)
            psf_size = [size, size]
            points = _scheduled_training_points(start_points, start_psf_size, psf_size)
            learning_rate = _log_interpolate(start_lr, end_lr, fraction)
            start_epoch = idx * increase_every
            end_epoch = (idx + 1) * increase_every
            stages.append({
                "start_epoch": start_epoch,
                "end_epoch": end_epoch,
                "psf_size": psf_size,
                "training_points": max(1, points),
                "learning_rate": learning_rate,
            })
        if stages:
            stages[-1]["end_epoch"] = max(max_epochs, stages[-1]["end_epoch"])
        return stages

    def _stage_index(self, epoch):
        for idx, stage in enumerate(self.stages):
            if stage["start_epoch"] <= epoch < stage["end_epoch"]:
                return idx
        return max(len(self.stages) - 1, 0)

    def _is_resuming(self, trainer):
        return self.resume or getattr(trainer, "ckpt_path", None) is not None

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


class NeuralBDCheckpointCallback(Callback):
    def __init__(self, base_dir, config=None):
        self.base_dir = Path(base_dir)
        self.config = config

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        save_model_state(self.base_dir / "neuralbd.nbd", pl_module, config=self.config)


class NeuralBDOutputCallback(Callback):
    def __init__(
        self,
        *,
        config=None,
        sample_count=5,
        figure_subregion_size=None,
        figure_subregion=None,
    ):
        self.config = config
        self.sample_count = int(sample_count)
        output_cfg = (config or {}).get("outputs", {})
        if figure_subregion_size is None:
            figure_subregion_size = output_cfg.get("figure_subregion_size")
        self.figure_subregion_size = None if figure_subregion_size is None else int(figure_subregion_size)
        self.figure_subregion = output_cfg.get("figure_subregion") if figure_subregion is None else figure_subregion

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        outputs = self._primary_validation_outputs(pl_module)
        if outputs is None:
            raise RuntimeError("Validation finished without outputs for figure logging.")
        self._log_validation_figures(trainer, pl_module, outputs)

    @staticmethod
    def _primary_validation_outputs(pl_module):
        if not pl_module.validation_outputs:
            return None
        if "validation" in pl_module.validation_outputs:
            return pl_module.validation_outputs["validation"]
        first_key = next(iter(pl_module.validation_outputs))
        return pl_module.validation_outputs[first_key]

    @rank_zero_only
    def _log_validation_figures(self, trainer, pl_module, outputs):
        figures = self._build_figures(pl_module, outputs)
        if not figures:
            print("[NeuralBD:validation] No validation figures were built.", flush=True)
            return
        import matplotlib.pyplot as plt

        for name, figure in figures.items():
            try:
                logged = self._log_wandb(trainer, name, figure)
                psf_size = tuple(int(v) for v in getattr(pl_module.convolution, "psf_size", ()))
                print(
                    f"[NeuralBD:validation] figure={name} psf={psf_size} wandb_logged={logged}",
                    flush=True,
                )
            finally:
                plt.close(figure)

    def _build_figures(self, pl_module, outputs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return {}

        images_shape = tuple(int(v) for v in outputs.get("images_shape", pl_module.images_shape))
        n_pixels = images_shape[0] * images_shape[1]
        if outputs["convolved_true"].shape[0] < n_pixels:
            raise RuntimeError(
                f"Validation figures require all {n_pixels} pixels, "
                f"but only {outputs['convolved_true'].shape[0]} were available after merging validation batches."
            )
        indices = outputs.get("indices")
        if indices is not None:
            indices = indices.numpy().astype("int64")

        sample_frames = list(range(min(self.sample_count, images_shape[2])))
        sample_channels = list(range(min(self.sample_count, images_shape[3])))
        convolved_true = self._restore_convolved_samples(
            outputs["convolved_true"],
            indices,
            images_shape,
            sample_frames,
            sample_channels,
        )
        convolved_pred = self._restore_convolved_samples(
            outputs["convolved_pred"],
            indices,
            images_shape,
            sample_frames,
            sample_channels,
        )
        first_frame = self._restore_convolved_samples(
            outputs["convolved_true"],
            indices,
            images_shape,
            [0],
            sample_channels,
        )
        image_pred = self._restore_image_samples(outputs["image_pred"], indices, images_shape, sample_channels)
        height, width = images_shape[:2]
        if self.figure_subregion is None:
            if self.figure_subregion_size is None:
                crop = (slice(0, height), slice(0, width))
            else:
                size = int(self.figure_subregion_size)
                if size <= 0:
                    raise ValueError("outputs.figure_subregion_size must be positive or null")
                crop_h = min(height, size)
                crop_w = min(width, size)
                y0 = (height - crop_h) // 2
                x0 = (width - crop_w) // 2
                crop = (slice(y0, y0 + crop_h), slice(x0, x0 + crop_w))
        else:
            region = self.figure_subregion
            x_range = region.get("x_range", region.get("x"))
            y_range = region.get("y_range", region.get("y"))
            x_slice = slice(0, width)
            y_slice = slice(0, height)
            x_extent = (-width / 2, width / 2)
            y_extent = (-height / 2, height / 2)
            if x_range is not None:
                x_extent = (float(x_range[0]), float(x_range[1]))
                x_start = int(np.floor(x_extent[0] + width / 2))
                x_stop = int(np.ceil(x_extent[1] + width / 2))
                if x_start < 0 or x_stop > width or x_start >= x_stop:
                    print(
                        "Invalid outputs.figure_subregion.x "
                        f"{list(x_extent)} for image width {width}; valid range is "
                        f"[{-width / 2}, {width / 2}]. Skipping validation figures."
                    )
                    return {}
                x_slice = slice(x_start, x_stop)
            if y_range is not None:
                y_extent = (float(y_range[0]), float(y_range[1]))
                y_start = int(np.floor(height / 2 - y_extent[1]))
                y_stop = int(np.ceil(height / 2 - y_extent[0]))
                if y_start < 0 or y_stop > height or y_start >= y_stop:
                    print(
                        "Invalid outputs.figure_subregion.y "
                        f"{list(y_extent)} for image height {height}; valid range is "
                        f"[{-height / 2}, {height / 2}]. Skipping validation figures."
                    )
                    return {}
                y_slice = slice(y_start, y_stop)
            crop = (y_slice, x_slice)
            image_extent = (*x_extent, *y_extent)

        if self.figure_subregion is None:
            image_extent = (
                crop[1].start - width / 2,
                crop[1].stop - width / 2,
                height / 2 - crop[0].stop,
                height / 2 - crop[0].start,
            )
        convolved_true = convolved_true[crop[0], crop[1]]
        convolved_pred = convolved_pred[crop[0], crop[1]]
        first_frame = first_frame[crop[0], crop[1]]
        image_pred = image_pred[crop[0], crop[1]]
        if self.figure_subregion is not None:
            convolved_true = convolved_true[::-1, ...]
            convolved_pred = convolved_pred[::-1, ...]
            first_frame = first_frame[::-1, ...]
            image_pred = image_pred[::-1, ...]

        figures = {
            "psfs": self._plot_psfs(plt, pl_module, sample_frames, sample_channels),
            "input_vs_reconstruction": self._plot_input_vs_reconstruction(
                plt,
                first_frame,
                image_pred,
                sample_channels,
                pl_module.pixel_per_ds,
                image_extent=image_extent,
            ),
            "predicted_vs_reference": self._plot_predicted_vs_reference(
                plt,
                convolved_true,
                convolved_pred,
                sample_frames,
                sample_channels,
                pl_module.pixel_per_ds,
                image_extent=image_extent,
            ),
        }
        reference = self._load_reference()
        if reference is not None:
            reference = reference[crop[0], crop[1]]
            if self.figure_subregion is not None:
                reference = reference[::-1, ...]
            figures["reconstruction_vs_ground_truth"] = self._plot_ground_truth(
                plt,
                image_pred,
                reference,
                sample_channels,
                pl_module.pixel_per_ds,
                image_extent=image_extent,
            )
        return {key: value for key, value in figures.items() if value is not None}

    @staticmethod
    def _restore_convolved_samples(value, indices, images_shape, frames, channels):
        height, width = images_shape[:2]
        n_pixels = height * width
        array = value.numpy()
        samples = array[:, frames, :][:, :, channels]
        if indices is None:
            return samples[:n_pixels].reshape(height, width, len(frames), len(channels))
        restored = np.full((n_pixels, len(frames), len(channels)), np.nan, dtype=array.dtype)
        restored[indices] = samples
        return restored.reshape(height, width, len(frames), len(channels))

    @staticmethod
    def _restore_image_samples(value, indices, images_shape, channels):
        height, width = images_shape[:2]
        n_pixels = height * width
        array = value.numpy()
        samples = array[:, channels]
        if indices is None:
            return samples[:n_pixels].reshape(height, width, len(channels))
        restored = np.full((n_pixels, len(channels)), np.nan, dtype=array.dtype)
        restored[indices] = samples
        return restored.reshape(height, width, len(channels))

    def _plot_psfs(self, plt, pl_module, sample_frames, sample_channels):
        from matplotlib.colors import LogNorm

        psf_model = pl_module.psf_model
        if hasattr(psf_model, "channel_mode") and psf_model.channel_mode == "per_channel":
            n_cols = len(sample_frames) * len(sample_channels)
        else:
            n_cols = len(sample_frames)
        if n_cols == 0:
            return None

        pixel_per_ds = pl_module.convolution.pixel_per_ds
        if hasattr(psf_model, "model"):
            coords = pl_module.convolution.psf_coords.new_tensor([[0.0, 0.0]])
            psf_coords, area = build_psf_grid(
                pl_module.convolution.psf_size,
                pixel_per_ds=pixel_per_ds,
                device=coords.device,
                dtype=coords.dtype,
            )
            psf = psf_model(coords, psf_coords[None, ...], area[None, ...])[0].detach().cpu().numpy()
        else:
            psf = psf_model().detach().cpu().numpy()
        psf = psf / (float(pixel_per_ds) ** 2)

        psf_panels = []
        for frame in sample_frames:
            if psf.ndim == 4:
                for channel in sample_channels:
                    psf_panels.append(psf[:, :, frame, channel])
            else:
                psf_panels.append(psf[:, :, frame])
        positive = np.concatenate([panel[np.isfinite(panel) & (panel > 0)].reshape(-1) for panel in psf_panels])
        if positive.size == 0:
            return None
        vmax = float(np.nanmax(positive))
        vmin = max(float(np.nanmin(positive)), vmax * 1e-8)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        fig, axes = plt.subplots(1, n_cols, figsize=(2.5 * n_cols, 2.8), squeeze=False)
        col = 0
        image = None
        for frame in sample_frames:
            if psf.ndim == 4:
                for channel in sample_channels:
                    panel = psf[:, :, frame, channel]
                    image = axes[0, col].imshow(
                        panel,
                        origin="lower",
                        cmap="inferno",
                        norm=norm,
                        extent=self._centered_extent(panel.shape, pixel_per_ds),
                    )
                    axes[0, col].set_title(f"f{frame} c{channel}")
                    axes[0, col].set_xlabel("dx [px]", fontsize=8)
                    axes[0, col].set_ylabel("dy [px]", fontsize=8)
                    axes[0, col].tick_params(axis="both", labelsize=7, length=2, pad=1)
                    col += 1
            else:
                panel = psf[:, :, frame]
                image = axes[0, col].imshow(
                    panel,
                    origin="lower",
                    cmap="inferno",
                    norm=norm,
                    extent=self._centered_extent(panel.shape, pixel_per_ds),
                )
                axes[0, col].set_title(f"f{frame}")
                axes[0, col].set_xlabel("dx [px]", fontsize=8)
                axes[0, col].set_ylabel("dy [px]", fontsize=8)
                axes[0, col].tick_params(axis="both", labelsize=7, length=2, pad=1)
                col += 1
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.08, pad=0.18)
        colorbar.set_label("PSF density [px^-2]", fontsize=8, labelpad=1)
        colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
        fig.suptitle("Learned PSFs")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.26, top=0.82, wspace=0.35)
        return fig

    def _plot_input_vs_reconstruction(
        self,
        plt,
        observed,
        reconstructed,
        sample_channels,
        pixel_per_ds,
        image_extent=None,
    ):
        cols = min(self.sample_count, len(sample_channels), reconstructed.shape[-1], observed.shape[-1])
        if cols == 0:
            return None
        fig, axes = plt.subplots(2, cols, figsize=(2.9 * cols, 5.2), squeeze=False)
        extent = image_extent or self._image_extent(observed.shape[:2], pixel_per_ds)
        shown = []
        for channel in sample_channels[:cols]:
            shown.extend([observed[:, :, 0, channel], reconstructed[:, :, channel]])
        vmin, vmax = self._finite_limits(*shown)
        image = None
        for col, channel in enumerate(sample_channels[:cols]):
            image = axes[0, col].imshow(
                observed[:, :, 0, channel],
                origin="lower",
                cmap="gray",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            axes[1, col].imshow(
                reconstructed[:, :, channel],
                origin="lower",
                cmap="gray",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, col].set_title(f"first frame c{channel}")
            axes[1, col].set_title(f"recon c{channel}")
            for ax in axes[:, col]:
                ax.set_xlabel("x [px]", fontsize=8)
                ax.set_ylabel("y [px]", fontsize=8)
                ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.045, pad=0.12)
        colorbar.set_label("intensity [normalized]", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
        fig.suptitle("Reconstruction with first-frame reference")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.9, wspace=0.35, hspace=0.35)
        return fig

    def _plot_predicted_vs_reference(
        self,
        plt,
        reference,
        predicted,
        sample_frames,
        sample_channels,
        pixel_per_ds,
        image_extent=None,
    ):
        cols = min(self.sample_count, len(sample_frames) * len(sample_channels))
        if cols == 0:
            return None
        fig, axes = plt.subplots(2, cols, figsize=(2.9 * cols, 5.2), squeeze=False)
        extent = image_extent or self._image_extent(reference.shape[:2], pixel_per_ds)
        pairs = self._sample_pairs(sample_frames, sample_channels, cols)
        shown = []
        for frame, channel in pairs:
            shown.extend([predicted[:, :, frame, channel], reference[:, :, frame, channel]])
        vmin, vmax = self._finite_limits(*shown)
        image = None
        for col, (frame, channel) in enumerate(pairs):
            image = axes[0, col].imshow(
                predicted[:, :, frame, channel],
                origin="lower",
                cmap="gray",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            axes[1, col].imshow(
                reference[:, :, frame, channel],
                origin="lower",
                cmap="gray",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, col].set_title(f"pred f{frame} c{channel}")
            axes[1, col].set_title(f"ref f{frame} c{channel}")
            for ax in axes[:, col]:
                ax.set_xlabel("x [px]", fontsize=8)
                ax.set_ylabel("y [px]", fontsize=8)
                ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.045, pad=0.12)
        colorbar.set_label("intensity [normalized]", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
        fig.suptitle("Predicted vs reference")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.9, wspace=0.35, hspace=0.35)
        return fig

    def _plot_ground_truth(self, plt, reconstructed, reference, sample_channels, pixel_per_ds, image_extent=None):
        if reference.ndim == 3:
            reference = reference[..., None]
        cols = min(self.sample_count, len(sample_channels), reconstructed.shape[-1], reference.shape[-1])
        if cols == 0:
            return None
        fig, axes = plt.subplots(2, cols, figsize=(2.9 * cols, 5.2), squeeze=False)
        extent = image_extent or self._image_extent(reconstructed.shape[:2], pixel_per_ds)
        shown = []
        for channel in sample_channels[:cols]:
            shown.extend([reconstructed[:, :, channel], reference[:, :, channel]])
        vmin, vmax = self._finite_limits(*shown)
        image = None
        for col, channel in enumerate(sample_channels[:cols]):
            image = axes[0, col].imshow(
                reconstructed[:, :, channel],
                origin="lower",
                cmap="gray",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            axes[1, col].imshow(
                reference[:, :, channel],
                origin="lower",
                cmap="gray",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
            axes[0, col].set_title(f"recon c{channel}")
            axes[1, col].set_title(f"gt c{channel}")
            for ax in axes[:, col]:
                ax.set_xlabel("x [px]", fontsize=8)
                ax.set_ylabel("y [px]", fontsize=8)
                ax.tick_params(axis="both", labelsize=7, length=2, pad=1)
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.045, pad=0.12)
        colorbar.set_label("intensity [normalized]", fontsize=8, labelpad=2)
        colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
        fig.suptitle("Reconstruction vs ground truth")
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.9, wspace=0.35, hspace=0.35)
        return fig

    @staticmethod
    def _finite_limits(*images):
        finite = np.concatenate([np.asarray(image)[np.isfinite(image)].reshape(-1) for image in images])
        if finite.size == 0:
            return 0.0, 1.0
        if finite.size >= 1000:
            vmin, vmax = np.nanpercentile(finite, [0.5, 99.5])
            vmin = float(vmin)
            vmax = float(vmax)
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        if vmin == vmax:
            pad = 0.5 if vmin == 0 else abs(vmin) * 0.05
            return vmin - pad, vmax + pad
        return vmin, vmax

    @staticmethod
    def _image_extent(shape, pixel_per_ds):
        height, width = shape
        return (-width / 2, width / 2, -height / 2, height / 2)

    @staticmethod
    def _centered_extent(shape, pixel_per_ds):
        height, width = shape
        x_centers = np.linspace(-(width // 2), width // 2, width, dtype="float64")
        y_centers = np.linspace(-(height // 2), height // 2, height, dtype="float64")
        x_spacing = (x_centers[1] - x_centers[0]) if width > 1 else 1.0
        y_spacing = (y_centers[1] - y_centers[0]) if height > 1 else 1.0
        return (
            float(x_centers[0] - x_spacing / 2),
            float(x_centers[-1] + x_spacing / 2),
            float(y_centers[0] - y_spacing / 2),
            float(y_centers[-1] + y_spacing / 2),
        )

    @staticmethod
    def _psf_density_per_pixel(image, pixel_per_ds):
        return image / (float(pixel_per_ds) ** 2)

    def _sample_pairs(self, frames, channels, limit):
        pairs = [(frame, channel) for frame in frames for channel in channels]
        return pairs[:limit]

    def _load_reference(self):
        output_cfg = (self.config or {}).get("outputs", {})
        reference_path = output_cfg.get("reference_path")
        if reference_path is None:
            return None
        return load_numpy_burst(reference_path, array_key=output_cfg.get("reference_array_key"))

    @rank_zero_only
    def _log_wandb(self, trainer, name, figure):
        key = f"validation/{name}"
        try:
            import wandb
        except ImportError:
            print("[NeuralBD:validation] wandb is not importable; skipping W&B figure logging.", flush=True)
            return False
        for logger in self._iter_loggers(trainer):
            experiment = getattr(logger, "experiment", None)
            if experiment is None or not hasattr(experiment, "log"):
                continue
            experiment.log({key: wandb.Image(figure)}, step=trainer.global_step)
            return True
        print("[NeuralBD:validation] No W&B logger experiment found; saved figures locally only.", flush=True)
        return False

    @staticmethod
    def _iter_loggers(trainer):
        loggers = getattr(trainer, "loggers", None)
        if loggers is not None:
            return list(loggers)
        logger = getattr(trainer, "logger", None)
        return [] if logger is None else [logger]
