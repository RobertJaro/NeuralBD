import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from neuralbd.processing import image_coordinates, normalize


def ensure_burst_shape(images, n_images=None, duplicate_channels=False):
    images = np.asarray(images, dtype="float32")
    if images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4:
        raise ValueError("images must have shape (height, width, frames) or (height, width, frames, channels)")
    if n_images is not None:
        images = images[:, :, : int(n_images), :]
    if duplicate_channels and images.shape[-1] == 1:
        images = np.repeat(images, 2, axis=-1)
    return images


class BurstDataset(Dataset):
    def __init__(
        self,
        images,
        coords,
        batch_size=2048,
        shuffle=True,
        epoch_size=None,
        sample_frames=False,
    ):
        self.images = np.asarray(images, dtype="float32")
        self.coords = np.asarray(coords, dtype="float32")
        self.images_shape = self.images.shape
        self.image_coordinates = self.coords
        self.shuffle = shuffle
        self.sample_frames = bool(sample_frames)
        self.epoch_size = None

        self.image_tensor = torch.from_numpy(self.images).reshape(-1, self.images.shape[2], self.images.shape[3])
        self.coord_tensor = torch.from_numpy(self.coords).reshape(-1, 2)
        self.index_tensor = torch.arange(self.image_tensor.shape[0], dtype=torch.long)
        if shuffle:
            order = torch.randperm(self.image_tensor.shape[0])
            self.image_tensor = self.image_tensor[order]
            self.coord_tensor = self.coord_tensor[order]
            self.index_tensor = self.index_tensor[order]
        self.set_batch_size(batch_size)
        if epoch_size is not None:
            self.set_epoch_size(epoch_size)

    def set_batch_size(self, batch_size):
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.image_batches = torch.split(self.image_tensor, self.batch_size, dim=0)
        self.coord_batches = torch.split(self.coord_tensor, self.batch_size, dim=0)
        self.index_batches = torch.split(self.index_tensor, self.batch_size, dim=0)

    def set_epoch_size(self, epoch_size):
        if epoch_size is None:
            self.epoch_size = None
            return
        self.epoch_size = int(epoch_size)
        if self.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")

    def set_sample_frames(self, sample_frames):
        self.sample_frames = bool(sample_frames)

    @property
    def unsharded_epoch_size(self):
        return self.epoch_size or len(self.image_batches)

    def __len__(self):
        return self.unsharded_epoch_size

    def __getitem__(self, index):
        index = int(index) % len(self.image_batches)
        images = self.image_batches[index]
        if self.sample_frames:
            frame_indices = torch.randint(images.shape[1], (images.shape[0],), dtype=torch.long)
            images = images[torch.arange(images.shape[0]), frame_indices]
            return {
                "images": images,
                "coords": self.coord_batches[index],
                "indices": self.index_batches[index],
                "frame_indices": frame_indices,
            }
        return {
            "images": images,
            "coords": self.coord_batches[index],
            "indices": self.index_batches[index],
        }

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    @classmethod
    def from_numpy(
        cls,
        images,
        pixel_per_ds=1.0,
        n_images=None,
        batch_size=2048,
        shuffle=True,
        normalization="minmax",
        duplicate_channels=False,
        epoch_size=None,
        sample_frames=False,
    ):
        images = ensure_burst_shape(images, n_images=n_images, duplicate_channels=duplicate_channels)
        images = normalize(images, normalization)
        coords = image_coordinates(images.shape, pixel_per_ds=pixel_per_ds)
        return cls(
            images=images,
            coords=coords,
            batch_size=batch_size,
            shuffle=shuffle,
            epoch_size=epoch_size,
            sample_frames=sample_frames,
        )


class BurstPointDataset(Dataset):
    def __init__(self, images, coords):
        self.images = np.asarray(images, dtype="float32")
        self.coords = np.asarray(coords, dtype="float32")
        self.images_shape = self.images.shape
        self.image_coordinates = self.coords
        self.image_tensor = torch.from_numpy(self.images).reshape(-1, self.images.shape[2], self.images.shape[3])
        self.coord_tensor = torch.from_numpy(self.coords).reshape(-1, 2)
        self.index_tensor = torch.arange(self.image_tensor.shape[0], dtype=torch.long)
        self.batch_size = None

    def __len__(self):
        return int(self.image_tensor.shape[0])

    def __getitem__(self, index):
        index = int(index)
        return {
            "images": self.image_tensor[index],
            "coords": self.coord_tensor[index],
            "indices": self.index_tensor[index],
            "dataset_idx": self.index_tensor[index],
        }

    def set_batch_size(self, batch_size):
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @classmethod
    def from_numpy(
        cls,
        images,
        pixel_per_ds=1.0,
        n_images=None,
        normalization="minmax",
        duplicate_channels=False,
    ):
        images = ensure_burst_shape(images, n_images=n_images, duplicate_channels=duplicate_channels)
        images = normalize(images, normalization)
        coords = image_coordinates(images.shape, pixel_per_ds=pixel_per_ds)
        return cls(images=images, coords=coords)

    @classmethod
    def from_burst_dataset(cls, dataset):
        return cls(images=dataset.images, coords=dataset.coords)


class BurstDataModule(LightningDataModule):
    def __init__(self, train_dataset, valid_dataset=None, num_workers=0, validation_batch_size=2048):
        super().__init__()
        self.train_dataset = train_dataset
        if valid_dataset is None:
            valid_dataset = BurstPointDataset.from_burst_dataset(train_dataset)
        elif isinstance(valid_dataset, BurstDataset):
            valid_dataset = BurstPointDataset.from_burst_dataset(valid_dataset)
        self.valid_dataset = valid_dataset
        self.num_workers = num_workers
        self.validation_batch_size = int(validation_batch_size)
        self.valid_dataset.set_batch_size(self.validation_batch_size)
        self.img_coords = self.valid_dataset.image_coordinates
        self.images_shape = self.valid_dataset.images_shape

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=None, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.validation_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def set_train_batch_size(self, batch_size):
        self.train_dataset.set_batch_size(batch_size)

    def set_train_sampling_points(self, sampling_points):
        self.train_dataset.set_batch_size(sampling_points)

    def set_validation_sampling_points(self, sampling_points):
        self.validation_batch_size = int(sampling_points)
        self.valid_dataset.set_batch_size(self.validation_batch_size)

    def set_train_epoch_size(self, epoch_size):
        self.train_dataset.set_epoch_size(epoch_size)

    def set_train_sample_frames(self, sample_frames):
        self.train_dataset.set_sample_frames(sample_frames)

    def fix_train_epoch_size(self):
        self.set_train_epoch_size(self.train_dataset.unsharded_epoch_size)
