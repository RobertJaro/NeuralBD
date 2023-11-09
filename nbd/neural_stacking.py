import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from astropy.nddata import block_reduce

from imreg_dft import similarity, transform_img_dict
from matplotlib.colors import Normalize
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

# 60 arcsec FOV - ref
# step size of 30 arcsec --> overlap of 50%
# pixel size of 1/20 arcsec --> 600 pixels
from mlpc.train.model import PositionalEncoding

base_path = '/gpfs/gpfs0/robert.jarolim/data/mlpc/neural_stack_v1'
data_path = ''
os.makedirs(base_path, exist_ok=True)

model_path = os.path.join(base_path, f'model.pt')

fits_files = sorted(glob.glob(os.path.join(data_path, '*.fits')))
mosaic_shape = (3, 3)
offset = 200

fits_array = []
for i in range(mosaic_shape[0]):
    files = fits_files[i * mosaic_shape[1]:(i + 1) * mosaic_shape[1]]
    files = files if (i % 2) == 0 else list(reversed(files))
    fits_array.append(files)

fits_array = np.array(fits_array)#np.flip(np.array(fits_array).T, 0)

ref_image = fits.getdata(fits_array[mosaic_shape[0] // 2, mosaic_shape[1] // 2])[0]
ref_image = ref_image[offset:-offset, offset:-offset]
vmin, vmax = np.min(ref_image), np.max(ref_image)

# plot images
images = []
positions = []
for i in range(mosaic_shape[0] * mosaic_shape[1]):
    plt.subplot(mosaic_shape[0], mosaic_shape[1], i + 1)
    pos = (i // mosaic_shape[0], i % mosaic_shape[0])
    image = fits.getdata(fits_array.flatten()[i])[0][offset:-offset, offset:-offset]
    plt.imshow(image, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
    images.append(image)
    positions.append(pos)
plt.savefig(os.path.join(base_path, 'mosaic.jpg'))
plt.close()

positions = np.array(positions)

images = np.array(images) / vmax
images = block_reduce(images, block_size=(1, 4, 4), func=np.mean)

class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class ImageModel(nn.Module):

    def __init__(self, images, dim, ref_image_id=None):
        super().__init__()
        self.ref_image_id = ref_image_id if ref_image_id is not None else len(images) // 2
        self.embedding = nn.Embedding(len(images), 3)
        # lin = [nn.Linear(dim, dim) for _ in range(4)]
        # coord_out = nn.Linear(dim, 3)
        # self.coordinate_transform = nn.Sequential(*lin, coord_out)

        posenc = PositionalEncoding(8, 20)
        d_in = nn.Linear(2 * 40, dim)
        self.d_in = nn.Sequential(posenc, d_in)

        lin = [nn.Linear(dim, dim) for _ in range(8)]
        self.linear_layers = nn.ModuleList(lin)
        self.d_out = nn.Linear(dim, 1)
        self.activation = Sine()

    def forward(self, image_coordinates, image_position, image_range):
        shift, theta = self.get_tranform(image_coordinates[:, 0], image_range)

        #apply rotation
        coords = image_coordinates[:, 1:]
        x = coords[:, 0] * torch.cos(theta) - coords[:, 1] * torch.sin(theta)
        y = coords[:, 0] * torch.sin(theta) + coords[:, 1] * torch.cos(theta)
        coords = torch.stack([x, y], -1)
        # # apply shift
        coords = image_position + coords + shift
        x = self.get_image(coords)
        return x, shift, theta

    def get_image(self, coords):
        x = self.activation(self.d_in(coords))
        for l in self.linear_layers:
            x = self.activation(l(x))
        x = self.d_out(x)
        return x

    def get_tranform(self, image_id, shift_range):
        transform = self.embedding(image_id.long())
        # transform = self.coordinate_transform(x)
        transform[image_id.long() == self.ref_image_id] = 0 # static reference image at index 0

        shift = torch.tanh(transform[:, :2]) * shift_range
        theta = torch.tanh(transform[:, 2]) * np.pi * 0 # TODO

        return shift, theta

# optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageModel(images, 128)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path)['model'].state_dict())
    print('model restored')
parallel_model = nn.DataParallel(model)
parallel_model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ExponentialLR(optimizer, 0.9999)

# loss
loss_fn = torch.nn.MSELoss()

# training
epochs = 10000
batch_size = 1024 * torch.cuda.device_count() if torch.cuda.is_available() else 1024

im_scaling = (images.shape[1] - 1) / 2
im_shift = (images.shape[1] - 1) / 2
max_range = 2.2
shift_range = 0.

image_coordinates = np.stack(np.mgrid[:images.shape[0], :images.shape[1], :images.shape[2]], -1)
coordinates_tensor = torch.from_numpy(image_coordinates).float().view(-1, 3)

coordinates_tensor[:, 1] = (coordinates_tensor[:, 1] - im_shift) / im_scaling
coordinates_tensor[:, 2] = (coordinates_tensor[:, 2] - im_shift) / im_scaling

positions_coordinates = np.ones((images.shape[0], images.shape[1], images.shape[2], 2)) * positions[:, None, None, :]
positions_coordinates = (positions_coordinates - mosaic_shape[0] // 2) / (mosaic_shape[0] // 2) * (max_range - 1)
positions_tensor = torch.from_numpy(positions_coordinates).float().view(-1, 2)

# check coordinates
print(f'MIN;  {coordinates_tensor[:, 0].min()} , {coordinates_tensor[:, 1].min()} , {coordinates_tensor[:, 2].min()}')
print(f'MAX;  {coordinates_tensor[:, 0].max()} , {coordinates_tensor[:, 1].max()} , {coordinates_tensor[:, 2].max()}')

image_tensor = torch.from_numpy(images).float().view(-1, 1)
print(f'Image tensor: SHAPE {image_tensor.shape}; MIN {image_tensor.min()}; MAX {image_tensor.max()}')

r = torch.randperm(len(image_tensor))
image_tensor = image_tensor[r]
coordinates_tensor = coordinates_tensor[r]

for epoch in range(epochs):
    total_loss = []
    for i in range(np.ceil(len(image_tensor) / batch_size).astype(int)):
        batch = image_tensor[i * batch_size:(i + 1) * batch_size]
        batch_coordinates = coordinates_tensor[i * batch_size:(i + 1) * batch_size]
        batch_positions = positions_tensor[i * batch_size:(i + 1) * batch_size]
        batch = batch.to(device)
        batch_coordinates = batch_coordinates.to(device)
        batch_positions = batch_positions.to(device)
        # forward
        optimizer.zero_grad()
        output, shift, theta = parallel_model(batch_coordinates, batch_positions, shift_range)
        weight = torch.ones_like(output)
        weight[batch_coordinates[:, 0] == model.ref_image_id] = 10
        loss = ((output - batch).pow(2)).mean() + 1e-3 * theta.pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += [loss.detach().cpu().numpy()]

    with torch.no_grad():
        image_ids = torch.tensor(list(range(len(images)))).to(device)
        shift, theta = model.get_tranform(image_ids, shift_range)
        shift, theta = shift.detach().cpu().numpy(), theta.detach().cpu().numpy()
        pos = positions_coordinates[:, 0, 0, :]
        # print loss
        print(f'epoch: {epoch + 1}; loss: {np.mean(total_loss)}')
        # print pretty
        if (epoch + 1) % 10 != 0:
            continue
        torch.save({'model': model}, model_path)
        # log shift and theta to file
        with open(os.path.join(base_path, f'tranformation_{epoch + 1:04d}.txt'), 'w') as f:
            for i, (s, t) in enumerate(zip(shift, theta)):
                print(f'image {i}: shift: {s[0]}, {s[1]}; theta: {t}', file=f)

        # plot full image
        test_coords = np.meshgrid(np.linspace(-max_range, max_range, 2048), np.linspace(-max_range, max_range, 2048), indexing='ij')
        test_tensor = torch.from_numpy(np.stack(test_coords, -1)).float().view(-1, 2)

        outputs = []
        for i in range(np.ceil(len(test_tensor) / batch_size).astype(int)):
            batch_coordinates = test_tensor[i * batch_size:(i + 1) * batch_size]
            batch_coordinates = batch_coordinates.to(device)
            output = model.get_image(batch_coordinates)
            output = output.detach().cpu().numpy()
            outputs += [output]
        output = np.concatenate(outputs).reshape(2048, 2048)

        plt.figure(figsize=(10, 10))
        plt.imshow(output, origin='lower', cmap='gray', extent=(-max_range, max_range, -max_range, max_range))
        plt.scatter(pos[:, 0] + shift[:, 0], pos[:, 0] + shift[:, 1], c='r', s=10)
        # add label and quiver to dots
        for i, (s, t, p) in enumerate(zip(shift, theta, pos)):
            plt.quiver(p[0] + s[0], p[1] + s[1], np.sin(t), np.cos(t), color='r', scale=10)
            plt.text(p[0] + s[0], p[1] + s[1], str(i), fontsize=16, color='b')
        plt.savefig(os.path.join(base_path, f'epoch_{epoch + 1:04d}.jpg'))
        plt.close()



