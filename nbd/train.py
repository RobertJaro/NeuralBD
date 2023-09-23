import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from nbd.util import aperture, psf_scale

base_path = '/gpfs/gpfs0/robert.jarolim/neuralbd/train_v1'
data_path = '/gpfs/gpfs0/robert.jarolim/data/gregor/hifi_20160928_081718_sd.fts'
os.makedirs(base_path, exist_ok=True)

model_path = os.path.join(base_path, f'model.pt')

n_channels = 2

fits_array = []
for c in range(n_channels):
    channel_frames = []
    for i in range(c, 200, n_channels):
        channel_frames.append(fits.getdata(data_path, i)[0:1024, 0:1024])
    fits_array.append(channel_frames)

fits_array = np.array(fits_array)  # (n_channels, n_images, 1024, 1024)
fits_array = fits_array.T  # (1024, 1024, n_images, n_channels)

wavelenghts = [fits.getheader(data_path, 0)['WAVELNTH'], fits.getheader(data_path, 1)['WAVELNTH']]
pixel_sizes = [0.0253, 0.0253]

vmin, vmax = np.min(fits_array), np.max(fits_array)

# plot images
image_stack = np.array(fits_array, dtype=np.float32) / vmax
# image_stack = block_reduce(image_stack, block_size=(4, 4, 1), func=np.mean)

n_images = image_stack.shape[2]
im_scaling = (image_stack.shape[1] - 1) / 2
im_shift = (image_stack.shape[1] - 1) / 2

fig, axs = plt.subplots(5, 2, figsize=(20, 4))
for row, idx in zip(axs, range(0, image_stack.shape[2], image_stack.shape[2] // 5)):
    row[0].imshow(image_stack[:, :, idx, 0], cmap='gray', origin='lower', vmin=0, vmax=1)
    row[1].imshow(image_stack[:, :, idx, 1], cmap='gray', origin='lower', vmin=0, vmax=1)
    row[0].set_axis_off()
    row[1].set_axis_off()

plt.savefig(os.path.join(base_path, 'images.png'))
plt.close()

# print(f'STACK RMSC: {((image_stack - image_stack.mean((0, 1))) ** 2).mean((0, 1)) ** 0.5}')

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


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class ImageModel(nn.Module):

    def __init__(self, dim, n_images, n_channels):
        super().__init__()
        posenc = PositionalEncoding(8, 20)
        d_in = nn.Linear(2 * 2 * 20, dim)
        self.d_in = nn.Sequential(posenc, d_in)

        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, n_images * n_channels)
        self.activation = Sine()

        self.n_images = n_images
        self.n_channels = n_channels

    def forward(self, coords):
        x = self.activation(self.d_in(coords))
        for l in self.linear_layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        x = x.view(-1, self.n_images, self.n_channels)
        return x


# optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image_model = ImageModel(128, n_images, n_channels)
S_model = ImageModel(128, n_images, 2)
O_model = ImageModel(128, 1, 2)

# if os.path.exists(model_path):
#     state = torch.load(model_path)
#     # image_model.load_state_dict(state['image_model'].state_dict())
#     S_model.load_state_dict(state['s_model'].state_dict())
#     O_model.load_state_dict(state['o_model'].state_dict())
#     print('model restored')

parallel_S_model = nn.DataParallel(S_model)
parallel_S_model.to(device)
parallel_O_model = nn.DataParallel(O_model)
parallel_O_model.to(device)

optimizer = torch.optim.Adam(list(parallel_S_model.parameters()) + list(parallel_O_model.parameters()), lr=1e-3)
scheduler = ExponentialLR(optimizer, 0.9999)

# loss
loss_fn = torch.nn.MSELoss()

# training
epochs = 1000
batch_size = 2048 * torch.cuda.device_count() if torch.cuda.is_available() else 1024

image_coordinates = np.stack(np.mgrid[:image_stack.shape[0], :image_stack.shape[1]], -1).astype(np.float32)
coordinates_tensor = torch.from_numpy(image_coordinates).view(-1, 2)

coordinates_tensor[:, 0] = (coordinates_tensor[:, 0] - im_shift) / im_scaling
coordinates_tensor[:, 1] = (coordinates_tensor[:, 1] - im_shift) / im_scaling

# check coordinates
print(f'MIN;  {coordinates_tensor[:, 0].min()} , {coordinates_tensor[:, 1].min()}')
print(f'MAX;  {coordinates_tensor[:, 0].max()} , {coordinates_tensor[:, 1].max()}')

A = np.stack([
    aperture(image_stack.shape[0], 0, 0, psf_scale(wl, 100, pixel_size)) for wl, pixel_size in
    zip(wavelenghts, pixel_sizes)], -1)

# FFT transform image stack
image_stack_ft = np.fft.fftshift(np.fft.fft2(image_stack, axes=(0, 1)), axes=(0, 1))
image_stack_ft = np.abs(image_stack_ft)
image_stack_ft = np.log(image_stack_ft + 1e-10)
image_stack_ft = image_stack_ft / image_stack_ft.max()

image_tensor_ft = torch.from_numpy(image_stack_ft).float().reshape((-1, n_images, n_channels))
print(f'Image tensor: SHAPE {image_tensor_ft.shape}; MIN {image_tensor_ft.min()}; MAX {image_tensor_ft.max()}')

A_tensor = torch.from_numpy(A).float().reshape((-1, n_channels))

test_coordinate_tensor = coordinates_tensor

for epoch in range(epochs):
    # shuffle batches
    r = torch.randperm(image_tensor_ft.shape[0])
    image_tensor_ft = image_tensor_ft[r]
    coordinates_tensor = coordinates_tensor[r]
    #
    total_loss = []
    for i in tqdm(range(np.ceil(len(image_tensor_ft) / batch_size).astype(int))):
        batch = image_tensor_ft[i * batch_size:(i + 1) * batch_size]
        batch_coordinates = coordinates_tensor[i * batch_size:(i + 1) * batch_size]
        batch_coordinates.requires_grad = True

        batch_A = A_tensor[i * batch_size:(i + 1) * batch_size]

        batch = batch.to(device)
        batch_coordinates = batch_coordinates.to(device)

        # forward images
        optimizer.zero_grad()

        # construct wavefront
        O = parallel_O_model(batch_coordinates)
        S = parallel_S_model(batch_coordinates)

        I = O * (S / S.sum(keepdim=True, dim=1))

        loss = (batch - I).pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += [loss.detach().cpu().numpy()]
    print(f'EPOCH {epoch + 1}; LOSS: {np.mean(total_loss)}')
    with torch.no_grad():
        if (epoch + 1) % 1 != 0:
            continue
        torch.save({
            's_model': S_model,
            'o_model': O_model,
        }, model_path)
        # plot full image
        fft_output_image = []
        for i in range(np.ceil(len(test_coordinate_tensor) / batch_size).astype(int)):
            batch_coordinates = test_coordinate_tensor[i * batch_size:(i + 1) * batch_size]
            batch_coordinates = batch_coordinates.to(device)

            O = parallel_O_model(batch_coordinates)

            fft_output_image += [O.detach().cpu().numpy()]

        fft_output_image = np.concatenate(fft_output_image, 0)
        fft_output_image = fft_output_image.reshape((image_stack.shape[0], image_stack.shape[1], n_channels))
        output_image = np.fft.ifft2(np.fft.ifftshift(fft_output_image, axes=(0, 1)), axes=(0, 1))
        output_image = (np.conj(output_image) * output_image).real

        fig, axs = plt.subplots(1, n_channels, figsize=(20, 6))
        for i in range(n_channels):
            axs[i].imshow(output_image[:, :, i], cmap='gray', origin='lower')
            axs[i].axis('off')
        plt.tight_layout()

        plt.savefig(os.path.join(base_path, f'epoch_{epoch + 1:04d}.jpg'), dpi=150)
        plt.close()
