import argparse
import sys
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm

from nbd.data.editor import ReadSimulationEditor, get_KL_basis


def convolve_pixel(args):
    x, y, padded_image, psf_generator, r = args

    # Extract patch from padded image
    patch = padded_image[y:y + 2*r + 1, x:x + 2*r + 1]

    # Compute normalized pixel coordinates (in original image space)
    H = padded_image.shape[0] - 2*r
    W = padded_image.shape[1] - 2*r
    xn = x / (W - 1)
    yn = y / (H - 1)

    psf = psf_generator.get_PSF(xn, yn)

    # PSF and patch should always match now
    val = np.sum(psf * patch)
    return y, x, val, psf


class SpatiallyVaryingPSFGenerator:
    def __init__(self, n_modes_max, psf_size, coef_range=2.0):
        #
        # Random coefficients
        self.coef1 = np.random.uniform(-coef_range, coef_range, n_modes_max)
        self.coef2 = np.random.uniform(-coef_range, coef_range, n_modes_max)
        self.coef3 = np.random.uniform(-coef_range, coef_range, n_modes_max)
        #
        # KL basis: shape (n_modes, psf_size, psf_size)
        self.kl_basis = get_KL_basis(n_modes_max=n_modes_max, size=psf_size)
    #
    def get_PSF(self, x, y):
        """
        x, y are normalized between [0,1]
        """
        c = self.coef1 * (1 - x) * (1 - y) + \
            self.coef2 * x + \
            self.coef3 * y

        #
        KL_wavefront = np.einsum('kij,k->ij', self.kl_basis, c)
        complex_pupil = np.exp(1j * KL_wavefront)
        PSF = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(complex_pupil)))
        PSF = np.abs(PSF)**2
    #
        return PSF / np.sum(PSF)


class SpatiallyVaryingConvolution:
    def __init__(self, image, psf_generator):
        self.image = image
        self.psf_generator = psf_generator

    def convolve_full(self):
        H, W = self.image.shape
        psf_size = self.psf_generator.kl_basis.shape[-1]
        r = psf_size // 2

        # Pad image to handle borders
        padded = np.pad(self.image, r, mode="reflect")

        output = np.zeros((H, W), dtype=float)
        psfs_all = np.zeros((psf_size, psf_size, H * W), dtype=float)

        # Build list of tasks (in original coords)
        tasks = [
            (x, y, padded, self.psf_generator, r)
            for y in range(H)
            for x in range(W)
        ]

        print(f"Dispatching {len(tasks)} pixels to {cpu_count()} CPU cores...")

        # Multiprocessing pool
        with Pool(cpu_count()) as pool:
            for y, x, val, psfs in pool.imap_unordered(convolve_pixel, tasks, chunksize=400):
                output[y, x] = val
                psfs_all[:, :, y * W + x] = psfs

        return output, psfs_all

class Convolver:
    def __init__(self, image, psf_size=(29, 29), n_modes=44, coef_range=2.0, variation_range=64):
        self.image = image

        self.psf_size = psf_size
        self.n_modes = n_modes
        self.coef_range = coef_range
        self.variation_range = variation_range

        coefs_x_coord = np.arange(0, image.shape[0] + variation_range, variation_range)
        coefs_y_coord = np.arange(0, image.shape[1] + variation_range, variation_range)
        coefs_x = [self.get_random_coefficients() for _ in coefs_x_coord]
        coefs_y = [self.get_random_coefficients() for _ in coefs_y_coord]

        coefs = np.zeros((len(coefs_x_coord), len(coefs_y_coord), self.n_modes)) * np.nan
        coefs[0, :, :] = coefs_x
        coefs[:, 0, :] = coefs_y
        for i in range(1, len(coefs_x_coord)):
            for j in range(1, len(coefs_y_coord)):
                coefs[i, j, :] = coefs[i - 1, j, :] + coefs[i, j - 1, :]
        self.coefs = coefs

        self.kl_basis = get_KL_basis(n_modes_max=n_modes, size=psf_size[0])

    def get_random_coefficients(self):
        return np.random.uniform(-self.coef_range, self.coef_range, self.n_modes)

    def get_psf(self, x, y):
        coefs_i = x // self.variation_range
        coefs_j = y // self.variation_range
        coef1 = self.coefs[coefs_i, coefs_j, :]
        coef2 = self.coefs[coefs_i + 1, coefs_j, :]
        coef3 = self.coefs[coefs_i, coefs_j + 1, :]

        px = (x % self.variation_range) / self.variation_range
        py = (y % self.variation_range) / self.variation_range
        c = coef1 * (1 - px) * (1 - py) + \
            coef2 * px + \
            coef3 * py

        KL_wavefront = np.einsum('kij,k->ij', self.kl_basis, c)
        complex_pupil = np.exp(1j * KL_wavefront)
        psf = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(complex_pupil)))
        psf = np.abs(psf) ** 2
        return psf / np.sum(psf)

    def get_convolved_pixel(self, pix):
        x, y = pix
        psf = self.get_psf(x, y)
        psf_size = psf.shape
        psf_shape_x, psf_shape_y = psf_size
        # Extract patch from padded image
        patch = self.image[x - psf_shape_x // 2 :x + psf_shape_x // 2 + 1, y - psf_shape_y // 2:y + psf_shape_y // 2 + 1]

        convolved = np.sum(psf * patch) # convolution at pixel (x,y)
        return convolved

    def get_convolve(self):
        x = np.arange(self.psf_size[0] // 2, self.image.shape[0] - self.psf_size[0] // 2)
        y = np.arange(self.psf_size[1] // 2, self.image.shape[1] - self.psf_size[1] // 2)
        pixels = np.stack(np.meshgrid(x, y, indexing='ij'), axis=-1)
        with Pool(cpu_count()) as pool:
            convolved_pixels = [r for r in tqdm(pool.imap(self.get_convolved_pixel, pixels.reshape(-1, 2)), total=pixels.size // 2)]
        convolved_image = np.array(convolved_pixels).reshape(pixels.shape[:-1])
        return convolved_image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str)
    args, overwrite_args = parser.parse_known_args()

    muram = ReadSimulationEditor().call('/gpfs/data/fs71254/schirni/nstack/data/I_out_med.468000')
    image = muram.data

    vmin_sim, vmax_sim = image.min(), image.max()
    image = (image - vmin_sim) / (vmax_sim - vmin_sim)

    # noise
    #mu, sigma = image.mean(), image.std()
    #noise = np.random.normal(mu, sigma, image.shape)
    #image += noise

    # convolver = Convolver(image)
    # convolved_image = convolver.get_convolve()

    n_realizations = 100

    convolved_list = []
    for i in range(n_realizations):
        print(f"Generating {i + 1}/{n_realizations}")
        convolver = Convolver(image)
        convolved_list.append(convolver.get_convolve())

    convolved_stack = np.stack(convolved_list, axis=-1)

    np.save(args.out_file, convolved_stack)