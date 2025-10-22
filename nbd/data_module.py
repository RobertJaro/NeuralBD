import multiprocessing as mp
import os
import random

import numpy as np
import torch
from astropy.io import fits
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, Dataset, RandomSampler
from tqdm import tqdm

from nbd.data.editor import get_KL_basis, get_KL_wavefront, generate_PSFs, ReadSimulationEditor, get_convolution, \
    cutout, get_filtered, optimize_shift, shift_image, compute_rms_contrast


class NeuralBDDataModule(LightningDataModule):
    def __init__(self, num_workers=4, psf_size=(51, 51), **dataset_config):
        super().__init__()
        self.num_workers = num_workers

        self.psf_size = psf_size

        data_set_type = dataset_config.pop('type')
        if data_set_type.upper() == 'GREGOR':
            self.train_dataset = GREGORDataset(**dataset_config, psf_size = psf_size)
            self.valid_dataset = GREGORDataset(**dataset_config, psf_size = psf_size, patch_size=(128, 128))

            self.contrast_weights = self.train_dataset.image_contrast
            self.img_coords = self.train_dataset.image_coordinates
            self.speckle = self.train_dataset.fits_array_speckle

        elif data_set_type.upper() == 'MURAM':
            self.train_dataset = MURAMDataset(**dataset_config, shuffle=True)
            self.valid_dataset = MURAMDataset(**dataset_config, shuffle=False)

            self.muram = self.train_dataset.high_quality
            self.psfs = self.train_dataset.kl_psfs
            self.img_coords = self.train_dataset.image_coordinates

        elif data_set_type.upper() == 'DKIST':
            self.train_dataset = DKISTDataset(**dataset_config, shuffle=True)
            self.valid_dataset = DKISTDataset(**dataset_config, shuffle=False)

            self.img_coords = self.train_dataset.image_coordinates

        else:
            raise ValueError('Unknown data type')

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=128, num_workers=self.num_workers)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.valid_dataset, batch_size=1, num_workers=self.num_workers, sampler=RandomSampler(num_samples=1, data_source=self.valid_dataset)) #sampler=RandomSampler(num_samples=1, data_source=self.valid_dataset)
        return loader


class MURAMDataset(TensorDataset):

    def __init__(self, data_path, n_images, pixel_per_ds, n_modes=44, psf_size=29, coef_range=2, x_crop=300, y_crop=300,
                 crop_size=512, shuffle=True, batch_size=1024, noise=True, **kwargs):

        # Generate PSFs
        kl_basis = get_KL_basis(n_modes_max=n_modes, size=psf_size)
        kl_wavefront = get_KL_wavefront(kl_basis, n_modes, n_images, coef_range=coef_range)
        self.kl_psfs = generate_PSFs(kl_wavefront, n_images)

        # Load data
        muram = ReadSimulationEditor().call(data_path)
        sim_array = muram.data

        if noise:
            mu, sigma = sim_array.mean(), sim_array.std()
            noise = np.random.normal(mu, sigma, sim_array.shape)
            sim_array += noise
        else:
            pass

        vmin_sim, vmax_sim = sim_array.min(), sim_array.max()
        sim_array = (sim_array - vmin_sim) / (vmax_sim - vmin_sim)

        sim_tensor = torch.tensor(sim_array, dtype=torch.float32).unsqueeze(-1)
        self.high_quality = cutout(sim_tensor[..., None], x_crop, y_crop, crop_size)

        # Convolve images
        images = get_convolution(self.high_quality, self.kl_psfs, n_images, noise=False)

        # Normalize images
        # vmin, vmax = images.min(), images.max()
        # images = (images - vmin) / (vmax - vmin)

        # Create dataset
        image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1]], -1)
        self.image_coordinates = image_coordinates / pixel_per_ds

        coordinates_tensor = torch.from_numpy(self.image_coordinates).float().view(-1, 2)
        image_tensor = torch.from_numpy(images).float().reshape(-1, n_images, 2)

        if shuffle:
            r = torch.randperm(len(image_tensor))
            image_tensor = image_tensor[r]
            coordinates_tensor = coordinates_tensor[r]

        # split into batches
        image_tensor = image_tensor.view(-1, batch_size, n_images, 2)
        coordinates_tensor = coordinates_tensor.view(-1, batch_size, 2)

        super().__init__(image_tensor, coordinates_tensor)


class GREGORDataset(Dataset):

    def __init__(self, data_path, n_images, pixel_per_ds, psf_size, x_crop=None, y_crop=None, crop_size=None, filter=False,
                 cutoff_freq=None, patch_size=(64, 64), random_sampling=True, n_patches=10000, **kwargs):

        # Load data
        fits_array = []
        fits_header = []
        for i in range(1, 201):  # <-- 2022: 0, 200 ; 2022-->: 1,201
            fits_array.append(fits.getdata(data_path, i))
            fits_header.append(fits.getheader(data_path, i))
        h, w = fits_array[0].shape
        fits_array = np.stack(fits_array, -1).reshape((h, w, 100, 2))
        fits_array = fits_array[..., 0]  # remove the second channel

        # Apply low-pass filter
        if filter:
            lowpass_img = [get_filtered(fits_array[:, :, i], cutoff_freq) for i in range(fits_array.shape[-1])]
            fits_array = np.stack(lowpass_img, -1)
            fits_array = fits_array[0]

        # apply shift to align images
        max_shift = 300

        print('Aligning images...')
        with mp.Pool(mp.cpu_count()) as pool:
            shifts = list(tqdm(pool.starmap(optimize_shift,
                                            [(fits_array[x_crop - max_shift:x_crop + crop_size + max_shift,
                                            y_crop - max_shift:y_crop + crop_size + max_shift, 0],
                                              fits_array[x_crop - max_shift:x_crop + crop_size + max_shift,
                                              y_crop - max_shift:y_crop + crop_size + max_shift, i])
                                             for i in range(n_images)]), total=n_images))

        fits_array = np.stack([shift_image(fits_array[x_crop - max_shift:x_crop + crop_size + max_shift,
        y_crop - max_shift:y_crop + crop_size + max_shift, i],
                                           shifts[i][0][1], shifts[i][0][0]) for i in range(n_images)], -1)

        # Load speckle data if available
        if os.path.exists(data_path.split('.')[-2] + '_speckle.fts'):
            fits_array_speckle = []
            for i in range(2):
                fits_array_speckle.append(fits.getdata(data_path.split('.')[-2] + '_speckle.fts', i))
            fits_array_speckle = np.stack(fits_array_speckle, -1)
            fits_array_speckle = cutout(fits_array_speckle[:, :, :, None], x_crop, y_crop, crop_size)
            fits_array_speckle = fits_array_speckle[:, :, 1]
            self.fits_array_speckle = np.stack([fits_array_speckle, fits_array_speckle], -1)
            # self.fits_array_speckle = fits_array_speckle
        else:
            self.fits_array_speckle = None

        # Crop images
        fits_array = cutout(fits_array[..., None], max_shift, max_shift, crop_size)
        # fits_array = cutout(fits_array, x_crop, y_crop, crop_size)

        # Normalize images
        vmin, vmax = fits_array.min(), fits_array.max()
        fits_array = (fits_array - vmin) / (vmax - vmin)

        # Compute contrast and sort images
        rms = [compute_rms_contrast(fits_array[:, :, i]) for i in range(n_images)]
        highest_indices = [index for index, value in
                           sorted(enumerate(rms), key=lambda x: x[1], reverse=True)[:n_images]]
        self.image_contrast = [rms[i] for i in highest_indices]
        fits_array = np.stack([fits_array[:, :, i] for i in highest_indices], -1)

        # workaround to use the same channel
        # fits_array[..., 1] = fits_array[..., 0]
        fits_array = np.stack([fits_array, fits_array], -1)

        images = fits_array

        # Create dataset
        self.psf_pad_x = psf_size[0] // 2
        self.psf_pad_y = psf_size[1] // 2
        image_coordinates = np.stack(np.mgrid[-self.psf_pad_x:images.shape[0] + self.psf_pad_x, -self.psf_pad_y:images.shape[1] + self.psf_pad_y], -1)
        image_coordinates = image_coordinates / pixel_per_ds

        # Save processed data
        self.images = images  # shape [x, y, n_images, 2]
        self.image_coordinates = image_coordinates  # shape [x, y, 2]
        self.patch_size = patch_size
        self.image_shape = images.shape[:2]

        self.random_sampling = random_sampling
        self.n_patches = n_patches  # only used if random_sampling=True

        patch_coords = [(i,j)
                        for i in range(self.image_shape[0] - patch_size[0])
                        for j in range(self.image_shape[1] - patch_size[1])]
        self.patch_coords = patch_coords

        super().__init__()

    def __len__(self):
        return len(self.patch_coords)

    def __getitem__(self, index):
        px, py = self.patch_size[0], self.patch_size[1]
        ix, iy = self.patch_coords[index]
        psf_pad_x = self.psf_pad_x
        psf_pad_y = self.psf_pad_y

        # Extract patch
        image_patch = self.images[ix:ix + px, iy:iy + py, :, :]  # [px, py, n_images, 2]
        coord_patch = self.image_coordinates[ix:ix + px + psf_pad_x * 2, iy:iy + py + psf_pad_y * 2, :]  # [px, py, 2]

        return torch.tensor(image_patch, dtype=torch.float32), torch.tensor(coord_patch, dtype=torch.float32)


class DKISTDataset(TensorDataset):

    def __init__(self, data_path, n_images, pixel_per_ds, x_crop=None, y_crop=None, crop_size=None, filter=False,
                 cutoff_freq=None,
                 shuffle=True, batch_size=1024, **kwargs):
        # Load data
        dkist = np.load(data_path)
        dkist_array = dkist['cobs']  # [n_images, h, w]
        dkist_array = dkist_array.transpose(1, 2, 0)  # [h, w, n_images]

        # apply shift to align images
        max_shift = 600

        print('Aligning images...')
        with mp.Pool(mp.cpu_count()) as pool:
            shifts = list(tqdm(pool.starmap(optimize_shift,
                                            [(dkist_array[x_crop - max_shift:x_crop + crop_size + max_shift,
                                            y_crop - max_shift:y_crop + crop_size + max_shift, 0],
                                              dkist_array[x_crop - max_shift:x_crop + crop_size + max_shift,
                                              y_crop - max_shift:y_crop + crop_size + max_shift, i])
                                             for i in range(n_images)]), total=n_images))

        dkist_array = np.stack([shift_image(dkist_array[x_crop - max_shift:x_crop + crop_size + max_shift,
        y_crop - max_shift:y_crop + crop_size + max_shift, i],
                                            shifts[i][0][1], shifts[i][0][0]) for i in range(n_images)], -1)

        # Crop images
        dkist_array = cutout(dkist_array[..., None], max_shift, max_shift, crop_size)
        # fits_array = cutout(fits_array, x_crop, y_crop, crop_size)

        # Normalize images
        vmin, vmax = dkist_array.min(), dkist_array.max()
        dkist_array = (dkist_array - vmin) / (vmax - vmin)

        # Compute contrast and sort images
        rms = [compute_rms_contrast(dkist_array[:, :, i]) for i in range(n_images)]
        highest_indices = [index for index, value in
                           sorted(enumerate(rms), key=lambda x: x[1], reverse=True)[:n_images]]
        self.image_contrast = [rms[i] for i in highest_indices]
        dkist_array = np.stack([dkist_array[:, :, i] for i in highest_indices], -1)

        # workaround to use the same channel
        # fits_array[..., 1] = fits_array[..., 0]
        dkist_array = np.stack([dkist_array, dkist_array], -1)

        images = dkist_array

        # Create dataset
        image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1]], -1)
        self.image_coordinates = image_coordinates / pixel_per_ds

        # apply binning and cropping
        # x_start, x_end, y_start, y_end = crop if crop else (0, -1, 0, -1)
        # image_coordinates = image_coordinates[x_start:x_end:bin, y_start:y_end:bin]
        coordinates_tensor = torch.from_numpy(self.image_coordinates).float().view(-1, 2)
        image_tensor = torch.from_numpy(images).float().reshape(-1, n_images, 2)

        if shuffle:
            r = torch.randperm(len(image_tensor))
            image_tensor = image_tensor[r]
            coordinates_tensor = coordinates_tensor[r]

        # split into batches
        image_tensor = image_tensor.view(-1, batch_size, n_images, 2)
        coordinates_tensor = coordinates_tensor.view(-1, batch_size, 2)

        super().__init__(image_tensor, coordinates_tensor)
