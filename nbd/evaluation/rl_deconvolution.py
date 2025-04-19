import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import restoration

from nbd.data.editor import get_KL_basis, get_KL_wavefront, generate_PSFs, ReadSimulationEditor, cutout, get_convolution

# Workaround until training is finished
n_modes = 44
psf_size = 11
n_images = 50
coef_range = 2
data_path = '/gpfs/data/fs71254/schirni/nstack/data/I_out_med.468000'
x_crop, y_crop = 332, 332
crop_size = 256

# Retrieve data
image_pred = np.load('/gpfs/data/fs71254/schirni/NeuralBD_muram/image_pred.npy')
psfs_pred = np.load('/gpfs/data/fs71254/schirni/NeuralBD_muram/psfs_pred.npy')
convolved_true = np.load('/gpfs/data/fs71254/schirni/NeuralBD_muram/convolved_true.npy')
convolved_pred = np.load('/gpfs/data/fs71254/schirni/NeuralBD_muram/convolved_pred.npy')

kl_basis = get_KL_basis(n_modes_max=n_modes, size=11)
kl_wavefront = get_KL_wavefront(kl_basis, n_modes, n_images, coef_range=coef_range)
kl_psfs = generate_PSFs(kl_wavefront, n_images)

muram = ReadSimulationEditor().call(data_path)
sim_array = muram.data
vmin, vmax = 0, sim_array.max()
sim_array = (sim_array - vmin) / (vmax - vmin)
sim_array = np.stack([sim_array, sim_array], -1)

high_quality = cutout(sim_array, x_crop, y_crop, crop_size)
images = get_convolution(high_quality, torch.tensor(psfs_pred), n_images, noise=False)

deconvolve_rl = restoration.richardson_lucy(images[:, :, 0, 0], psfs_pred[:, :, 0], num_iter=200)

fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
axs[0].imshow(high_quality[:, :, 0], cmap='gray', origin='lower')
axs[0].set_title('MuRAM', fontsize=20)
axs[0].axis('off')
axs[1].imshow(image_pred[:, :, 0], cmap='gray', origin='lower')
axs[1].set_title('NBD Deconvoution', fontsize=20)
axs[1].axis('off')
plt.tight_layout()
plt.savefig('/gpfs/data/fs71254/schirni/NeuralBD_muram/deconvolve-muram.jpg')

fig, axs = plt.subplots(2, 5, figsize=(2 * 5, 5), dpi=300)
for c in range(2):
    for i in range(5):
        ax = axs[0, i]
        ax.imshow(convolved_true[:, :, i, c], cmap='gray', origin='lower')
        # divider1 = make_axes_locatable(axs[0, n_samples-1])
        # cax1 = divider1.append_axes("right", size="5%", pad="2%")
        # fig.colorbar(im1, ax=cax1)
        ax = axs[1, i]
        ax.imshow(convolved_pred[:, :, i, c], cmap='gray', origin='lower')
        # divider2 = make_axes_locatable(axs[1, n_samples-1])
        # cax2 = divider2.append_axes("right", size="5%", pad="2%")
        # fig.colorbar(im2, ax=cax2)
        axs[0, 0].set_ylabel('True')
        axs[1, 0].set_ylabel('Prediction')
        [axs[0, i].set_title(f'Frame {i:01d}') for i in range(5)]
        axs[0, i].axis('off')
        axs[1, i].axis('off')
plt.savefig('/gpfs/data/fs71254/schirni/NeuralBD_muram/convolution-muram.jpg')