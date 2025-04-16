import numpy as np
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

high_quality = cutout(sim_array, x_crop, y_crop, crop_size)
images = get_convolution(sim_array, kl_psfs, n_images, noise=False)

deconvolve_rl = restoration.richardson_lucy(images, kl_psfs, num_iter=50)