import torch
from lightning.pytorch import LightningModule, Trainer


def _pretraining_target(images, target="first_frame", frame_index=0):
    if target in {"first_frame", "frame"}:
        return images[:, int(frame_index), :]
    if target == "mean":
        return images.mean(dim=1)
    raise ValueError("target must be 'first_frame', 'frame', or 'mean'")


def pretrain_image_model(module, datamodule, config):
    pre_cfg = config.get("pretraining", {})
    epochs = int(pre_cfg.get("epochs", 0))
    if not pre_cfg.get("enabled", False) or epochs <= 0:
        return []

    device = next(module.parameters()).device
    module.train()
    optimizer = torch.optim.Adam(module.image_model.parameters(), lr=float(pre_cfg.get("learning_rate", 1e-4)))
    history = []
    for _epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for batch in datamodule.train_dataloader():
            images, coords = ImagePretrainingModule._unpack_batch(batch)
            images = images.to(device=device)
            coords = coords.to(device=device)
            target = _pretraining_target(
                images,
                target=pre_cfg.get("target", "first_frame"),
                frame_index=pre_cfg.get("frame_index", 0),
            )
            pred = module.image_model(coords)
            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            n_batches += 1
            del loss, pred, target, images, coords
        history.append(running_loss / max(n_batches, 1))
    return history


class ImagePretrainingModule(LightningModule):
    def __init__(self, image_model, config):
        super().__init__()
        self.image_model = image_model
        self.config = config
        self.pre_cfg = config.get("pretraining", {})

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            return batch["images"], batch["coords"]
        if len(batch) == 3:
            images, coords, _indices = batch
            return images, coords
        return batch

    def training_step(self, batch, batch_idx):
        images, coords = self._unpack_batch(batch)
        target = _pretraining_target(
            images,
            target=self.pre_cfg.get("target", "first_frame"),
            frame_index=self.pre_cfg.get("frame_index", 0),
        )
        pred = self.image_model(coords)
        loss = torch.mean((pred - target) ** 2)
        self.log("pretrain.loss", loss.detach(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.image_model.parameters(),
            lr=float(self.pre_cfg.get("learning_rate", 1e-4)),
        )


def fit_pretraining_stage(module, datamodule, config, logger=False):
    pre_cfg = config.get("pretraining", {})
    epochs = int(pre_cfg.get("epochs", 0))
    if not pre_cfg.get("enabled", False) or epochs <= 0:
        return None

    trainer_cfg = config.get("training", {})
    trainer = Trainer(
        max_epochs=epochs,
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        precision=trainer_cfg.get("precision", "32-true"),
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
    )
    trainer.fit(ImagePretrainingModule(module.image_model, config), train_dataloaders=datamodule.train_dataloader())
    return trainer
