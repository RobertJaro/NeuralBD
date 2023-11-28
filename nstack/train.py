import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.nddata import block_reduce
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# 60 arcsec FOV - ref
# step size of 30 arcsec --> overlap of 50%
# pixel size of 1/20 arcsec --> 600 pixels
base_path = '/home/fs71254/schirni/neural_stacking/run2'
data_path = '/gpfs/data/fs71254/schirni/Level1_Files/Level1_Files_GBand/hifi_20170618_082414_sd.fts'
os.makedirs(base_path, exist_ok=True)

model_path = os.path.join(base_path, f'model.pt')

offset = 200
fits_array = []

for i in range(0, 200):
    fits_array.append(fits.getdata(data_path, i))

h, w = fits_array[0].shape
fits_array = np.stack(fits_array, -1).reshape((h, w, 100, 2))

#fits_array = fits_array[:, :, :]
fits_array = fits_array[700:1212, 1200:1712, :, :]
#fits_array = fits_array[70:2100, 70:2100, :, :]
vmin, vmax = 0, np.percentile(fits_array, 99)

# positions = np.array(positions)
images = fits_array / vmax


ref_image = images[:, :, 0]
# plot first two channels
fig, axs = plt.subplots(1, 2, figsize=(20, 10))
axs[0].imshow(ref_image[..., 0], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
axs[1].imshow(ref_image[..., 1], cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
plt.savefig(os.path.join(base_path, 'images.jpg'))
plt.close(fig)

def rms_contrast(image):
    return torch.sqrt((image - np.mean(image))**2)


#images = block_reduce(images, block_size=(4, 4, 1, 1), func=np.mean)

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding of the input coordinates.

    encodes x to (..., sin(2^k x), cos(2^k x), ...)
    k takes "num_freqs" number of values equally spaced between [0, max_freq]
    """

    def __init__(self, max_freq, num_freqs):
        """
        Args:
            max_freq (int): maximum frequency in the positional encoding.
            num_freqs (int): number of frequencies between [0, max_freq]
        """
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs)  # (num_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (batch, in_features)
        Outputs:
            out: (batch, 2*num_freqs*in_features)
        """
        x_proj = x[:, None, :] * self.freqs[None, :, None]  # (batch, num_freqs, in_features)
        x_proj = x_proj.reshape(x.shape[0], -1)  # (batch, num_freqs*in_features)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)],
                        dim=-1)  # (batch, 2*num_freqs*in_features)
        return out


class ImageModel(nn.Module):

    def __init__(self, dim, n_channels=2):
        super().__init__()
        posenc = PositionalEncoding(8, 20)
        d_in = nn.Linear(2 * 40, dim)
        self.d_in = nn.Sequential(posenc, d_in)

        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, n_channels)
        self.activation = Sine()

    def forward(self, coords):
        x = self.activation(self.d_in(coords))

        for l in self.layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        return x

class PSFModel(nn.Module):

    def __init__(self, dim, n_images):
        super().__init__()
        d_in = nn.Linear(2, dim)
        self.d_in = d_in

        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, n_images)
        self.activation = Sine()

    def forward(self, coords):
        x = self.activation(self.d_in(coords))

        for l in self.layers:
            x = self.activation(l(x))
        x = torch.sigmoid(self.d_out(x))
        return x

class Transformer(nn.Module):

    def __init__(self, n_images):
        super().__init__()
        #self.activation = Sine()
        #self.d_in = nn.Linear(coord_dim, dim)
        #lin = [nn.Linear(dim, dim) for _ in range(4)]
        #self.layers = nn.ModuleList(lin)
        #self.d_out = nn.Linear(dim, coord_dim)
        self.transform_matrix = nn.Parameter(torch.ones(n_images - 1, 2, 2, dtype=torch.float32), requires_grad=True)

    def forward(self, coords):
        #x = self.activation(self.d_in(coords))
        #for l in self.layers:
        #    x = self.activation(l(x))
        #x = self.d_out(x)
        first_img_coords = coords[:, 0:1]
        coords = coords[:, 1:]

        #extended_coords  = torch.cat([coords, torch.ones_like(coords)[..., 0:1]], -1)
        # (batch, n_images - 1, 3)
        #extended_coords = torch.einsum('nij,bnj->bni', self.transform_matrix, extended_coords)
        #coords = extended_coords[..., :2]
        coords = torch.einsum('nij,bnj->bni', self.transform_matrix, coords)

        coords = torch.cat([first_img_coords, coords], -2)
        return coords

class ImageStackModel(nn.Module):

    def __init__(self, n_images, dim):
        # images = (w, h, n, c)
        super().__init__()
        self.n_images = n_images
        self.image_models = nn.ModuleList([ImageModel(dim) for _ in range(self.n_images)])
        self.coord_transform = Transformer(n_images)

    def get_transformed_images(self, coords):
        transformed_coords = self.transform_coords(coords)
        image_stack = torch.stack([model(transformed_coords[:, i]) for i, model in enumerate(self.image_models)], -2)
        return image_stack

    def get_images(self, coords):
        image_stack = torch.stack([model(coords[:, i]) for i, model in enumerate(self.image_models)], -2)
        return image_stack

    def transform_coords(self, coords):
        transformed_coords = self.coord_transform(coords)
        return transformed_coords

    def forward(self, coords, transform=False):
        if transform:
            coords = self.transform_coords(coords)
        image_stack = self.get_images(coords)
        return image_stack

class PSFStackModel(nn.Module):

    def __init__(self, n_images, dim):
        # images = (w, h, n, c)
        super().__init__()
        self.n_images = n_images
        self.image_model = ImageModel(dim)
        self.psf_model = PSFModel(dim, n_images)
        self.psf_width = nn.Parameter(torch.tensor(0.05, dtype=torch.float32), requires_grad=True)

    def get_transformed_images(self, coords):
        transformed_coords = self.transform_coords(coords)
        image_stack = torch.stack([model(transformed_coords[:, i]) for i, model in enumerate(self.image_models)], -2)
        return image_stack

    def get_images(self, coords):
        n_random_sample = 500
        random_sampling = torch.randn(coords.shape[0], n_random_sample, 2, device=coords.device) * self.psf_width
        random_sampling_coords = coords[:, None, :] + random_sampling
        #
        condition = (random_sampling_coords[..., 0] >= -1) & (random_sampling_coords[..., 0] <= 1) & \
                    (random_sampling_coords[..., 1] >= -1) & (random_sampling_coords[..., 1] <= 1)
        #
        psf = self.psf_model(random_sampling.reshape(-1, 2))
        psf = psf.reshape(coords.shape[0], n_random_sample, self.n_images)
        psf = psf * condition[..., None]
        sampling_image = self.image_model(random_sampling_coords.reshape(-1, 2))
        sampling_image = sampling_image.reshape(coords.shape[0], n_random_sample, 2)
        convolved_images = torch.einsum('bsc,bsn->bnc', sampling_image, psf) / torch.sum(condition, -1)[:, None, None]
        image = self.image_model(coords)
        return image, convolved_images


    def forward(self, coords):
        return self.get_images(coords)

# optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_images = images.shape[-2]
model = PSFStackModel(n_images, 128)
parallel_model = nn.DataParallel(model)
parallel_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# loss
loss_fn = torch.nn.MSELoss()

# training
epochs = 10000
batch_size = 1024 * torch.cuda.device_count() if torch.cuda.is_available() else 1024

im_scaling = (images.shape[0] - 1) / 2
im_shift = (images.shape[0] - 1) / 2

shift_range = 0.02

image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1]], -1)
coordinates_tensor = torch.from_numpy(image_coordinates).float().view(-1, 2)

coordinates_tensor = (coordinates_tensor - im_shift) / im_scaling

# check coordinates
print(f'MIN;  {coordinates_tensor[..., 0].min()} , {coordinates_tensor[..., 1].min()}')
print(f'MAX;  {coordinates_tensor[..., 0].max()} , {coordinates_tensor[..., 1].max()}')

# image_tensor = torch.from_numpy(images).float().view(-1, 1)
image_tensor = torch.from_numpy(images).float().reshape(-1, n_images, 2)
print(f'Image tensor: SHAPE {image_tensor.shape}; MIN {image_tensor.min()}; MAX {image_tensor.max()}')

test_coords = coordinates_tensor

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path)['model'].state_dict())
    print('model restored')



for epoch in range(epochs):
    r = torch.randperm(len(image_tensor))
    image_tensor = image_tensor[r]
    coordinates_tensor = coordinates_tensor[r]

    total_loss = []
    for i in range(np.ceil(len(coordinates_tensor) / batch_size).astype(int)):
        batch_coordinates = coordinates_tensor[i * batch_size:(i + 1) * batch_size]
        batch_image = image_tensor[i * batch_size:(i + 1) * batch_size]
        batch_coordinates = batch_coordinates.to(device)
        batch_image = batch_image.to(device)

        img, convolved_imgs = parallel_model(batch_coordinates)
        loss = loss_fn(convolved_imgs, batch_image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += [loss.detach().cpu().numpy()]

    with torch.no_grad():
        # print loss
        print(f'epoch: {epoch + 1}; loss: {np.mean(total_loss)}')
        # print pretty
        if (epoch + 1) % 5 != 0:
            continue

        output_image = []
        output_convolved_image = []
        for i in range(np.ceil(len(test_coords) / batch_size).astype(int)):
            batch_coordinates = test_coords[i * batch_size:(i + 1) * batch_size]
            batch_coordinates = batch_coordinates.to(device)

            output_img, convolved_imgs = parallel_model(batch_coordinates)

            output_image += [output_img.detach().cpu().numpy()]
            output_convolved_image += [convolved_imgs.detach().cpu().numpy()]

        output_image = np.concatenate(output_image, 0).reshape((*image_coordinates.shape[:-1], 2))
        output_convolved_image = np.concatenate(output_convolved_image, 0).reshape((*image_coordinates.shape[:-1], n_images, 2))

        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        axs[0, 0].imshow(output_image[..., 0], cmap='gray', vmin=0, origin='lower')
        axs[0, 1].imshow(output_image[..., 1], cmap='gray', vmin=0, origin='lower')
        axs[1, 0].imshow(output_convolved_image[..., 1, 0], cmap='gray', vmin=0, origin='lower')
        axs[1, 1].imshow(output_convolved_image[..., 1, 1], cmap='gray', vmin=0, origin='lower')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f'images_{epoch + 1:04d}.jpg'), dpi=300)
        plt.close(fig)

        torch.save({'model': model}, model_path)
