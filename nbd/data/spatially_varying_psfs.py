import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from nbd.data.editor import get_KL_basis, ReadSimulationEditor


# ============================================================
# PSF GENERATOR
# ============================================================

class SpatiallyVaryingPSFGenerator:
    def __init__(
        self,
        n_modes=44,
        psf_size=29,
        fft_pad=8,
        variation_range=64,
        seed=1234,
        interpolate=False,
        wavelength=500e-9,      # meters
        telescope_diameter=1.5, # meters
        r0=0.15                 # Fried parameter [m] at wavelength
    ):
        self.psf_size = psf_size
        self.fft_pad = fft_pad
        self.variation_range = variation_range
        self.interpolate = interpolate

        self.lambda_ = wavelength
        self.D = telescope_diameter
        self.r0 = r0

        rng = np.random.default_rng(seed)

        kl = get_KL_basis(n_modes_max=n_modes, size=psf_size)

        # Remove tip/tilt
        self.kl_basis = kl[2:, :, :]
        self.n_modes = self.kl_basis.shape[0]

        # Kolmogorov eigenvalues
        k = np.arange(1, self.n_modes + 1)
        self.eigenvalues = k ** (-11.0 / 6.0)

        # physical scaling
        self.phase_scale = (self.D / self.r0) ** (5.0 / 6.0)

        x = np.linspace(-1, 1, psf_size)
        xx, yy = np.meshgrid(x, x)
        self.pupil = (xx ** 2 + yy ** 2) <= 1.0

        self.rng = rng
        self.coefficients = None

    def initialize_field(self, H, W):
        nx = H // self.variation_range + 2
        ny = W // self.variation_range + 2

        self.coefficients = self.rng.normal(
            scale=np.sqrt(self.eigenvalues),
            size=(nx, ny, self.n_modes)
        )

    def _interp_coeffs(self, y, x):
        i = y // self.variation_range
        j = x // self.variation_range

        if not self.interpolate:
            return self.coefficients[i, j]

        py = (y % self.variation_range) / self.variation_range
        px = (x % self.variation_range) / self.variation_range

        c00 = self.coefficients[i, j]
        c10 = self.coefficients[i + 1, j]
        c01 = self.coefficients[i, j + 1]
        c11 = self.coefficients[i + 1, j + 1]

        return (
            c00 * (1 - px) * (1 - py) +
            c10 * px       * (1 - py) +
            c01 * (1 - px) * py +
            c11 * px       * py
        )

    def get_psf(self, y, x):
        c = self._interp_coeffs(y, x)
        phase = np.einsum("kij,k->ij", self.kl_basis, c)
        phase *= self.phase_scale

        complex_pupil = self.pupil * np.exp(1j * phase)

        N = self.fft_pad * self.psf_size
        if N % 2 == 0:
            N += 1

        padded = np.zeros((N, N), dtype=complex)
        c0 = N // 2
        r = self.psf_size // 2
        padded[c0 - r:c0 + r + 1, c0 - r:c0 + r + 1] = complex_pupil

        psf = np.abs(np.fft.fftshift(np.fft.fft2(padded)))**2

        c = N // 2
        r = self.psf_size // 2
        psf = psf[c - r:c + r + 1, c - r:c + r + 1]
        psf /= psf.sum()

        return psf


# ============================================================
# CONVOLUTION
# ============================================================

def convolve_pixel(args):
    y, x, padded, psf_gen = args
    psf = psf_gen.get_psf(y, x)
    r = psf.shape[0] // 2

    patch = padded[y:y + 2*r + 1, x:x + 2*r + 1]
    return y, x, np.sum(psf * patch)


class SpatiallyVaryingConvolution:
    def __init__(self, image, psf_gen):
        self.image = image
        self.psf_gen = psf_gen

    def convolve(self):
        H, W = self.image.shape
        self.psf_gen.initialize_field(H, W)

        test_psf = self.psf_gen.get_psf(0, 0)
        r = test_psf.shape[0] // 2
        padded = np.pad(self.image, r, mode="reflect")

        out = np.zeros_like(self.image)

        tasks = [(y, x, padded, self.psf_gen)
                 for y in range(H)
                 for x in range(W)]

        with Pool(cpu_count()) as pool:
            for y, x, val in tqdm(
                pool.imap_unordered(convolve_pixel, tasks, chunksize=200),
                total=len(tasks)
            ):
                out[y, x] = val

        return out

# ============================================================
# MAIN
# ============================================================

def main(args):

    muram = ReadSimulationEditor().call(
        "/gpfs/data/fs71254/schirni/nstack/data/I_out_med.468000"
    )
    image = muram.data.astype(np.float32)
    image -= image.min()
    image /= image.max()

    results = []

    for i in range(args.n_realizations):
        print(f"Realization {i + 1}/{args.n_realizations}")

        psf_gen = SpatiallyVaryingPSFGenerator(
            n_modes=44,
            psf_size=29,
            fft_pad=1,
            variation_range=64,
            seed=30 + i,
            interpolate=False,
            wavelength=450.6e-9,
            telescope_diameter=1.44,
            r0=0.2
        )

        convolver = SpatiallyVaryingConvolution(image, psf_gen)
        results.append(convolver.convolve())

    stack = np.stack(results, axis=-1)
    np.save(args.out_file, stack)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--n_realizations", type=int)
    args = parser.parse_args()

    main(args)



#from nbd.data.spatially_varying_psfs import SpatiallyVaryingPSFGenerator
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#psf_gen = SpatiallyVaryingPSFGenerator(n_modes=44,psf_size=29,fft_pad=1,variation_range=256,seed=31,interpolate=False,wavelength=450.6e-9,telescope_diameter=1.44,r0=0.2)
#psf_gen.initialize_field(1024, 1024)
#psf = psf_gen.get_psf(0, 0)
#plt.figure(figsize=(8,8), dpi=100)
#plt.imshow(psf, norm=LogNorm(vmin=psf.min(), vmax=psf.max()), origin='lower')
#plt.savefig('/gpfs/data/fs71254/schirni/nstack/psf_02_1_1_00.jpg')
#plt.close()