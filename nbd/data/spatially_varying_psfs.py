import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from nbd.data.editor import get_KL_basis, ReadSimulationEditor


# ============================================================
# PHYSICAL KL PHASE SCREEN PSF GENERATOR
# ============================================================

class PhysicalKLPSFGenerator:
    def __init__(
        self,
        n_modes=44,
        psf_size=29,
        fft_pad=1,
        seed=1234,
        wavelength=500e-9,
        telescope_diameter=1.5,
        r0=0.15,
        correlation_length=64  # spatial smoothness control
    ):
        self.psf_size = psf_size
        self.fft_pad = fft_pad
        self.correlation_length = correlation_length

        self.lambda_ = wavelength
        self.D = telescope_diameter
        self.r0 = r0

        self.rng = np.random.default_rng(seed)

        # KL basis (remove tip/tilt)
        kl = get_KL_basis(n_modes_max=n_modes + 2, size=psf_size)
        self.kl_basis = kl[2:, :, :]
        self.n_modes = min(n_modes, self.kl_basis.shape[0])

        # Kolmogorov eigenvalues ~ k^(-11/3)
        k = np.arange(1, self.n_modes + 1)
        self.eigenvalues = k ** (-11.0 / 6.0)

        # circular pupil
        x = np.linspace(-1, 1, psf_size)
        xx, yy = np.meshgrid(x, x)
        self.pupil = (xx**2 + yy**2) <= 1.0

        self.coeff_fields = None


    # --------------------------------------------------------
    # Generate smooth periodic coefficient fields
    # --------------------------------------------------------
    def initialize_field(self, H, W, L0=np.inf):
        """
        Generate KL coefficient fields with true Kolmogorov
        spatial statistics (von Kármán spectrum).
        """

        self.H = H
        self.W = W

        ky = np.fft.fftfreq(H)
        kx = np.fft.fftfreq(W)
        kx, ky = np.meshgrid(kx, ky)
        k2 = kx ** 2 + ky ** 2

        # Outer scale handling
        if np.isinf(L0):
            k0 = 0.0
        else:
            k0 = 1.0 / L0

        # Avoid singularity at DC
        k2[0, 0] = 1.0

        # von Kármán spectrum
        power = (k2 + k0 ** 2) ** (-11.0 / 6.0)
        power[0, 0] = 0.0

        power = power[:, :, None]  # broadcast to modes

        # Generate complex white noise for all modes for random turbulence realization
        noise = (
                self.rng.normal(size=(H, W, self.n_modes)) +
                1j * self.rng.normal(size=(H, W, self.n_modes))
        )

        # Apply Kolmogorov spectrum
        field_ft = noise * np.sqrt(power)

        # Batched inverse FFT
        coeff = np.fft.ifft2(field_ft, axes=(0, 1)).real

        # Normalize each mode
        coeff /= coeff.std(axis=(0, 1), keepdims=True)

        # Apply KL eigenvalue scaling
        coeff *= np.sqrt(self.eigenvalues)[None, None, :]

        # Physical amplitude scaling
        coeff *= (self.D / self.r0) ** (5.0 / 6.0)

        self.coeff_fields = coeff.astype(np.float32)

    # --------------------------------------------------------
    # Compute PSF from KL expansion at (y,x)
    # --------------------------------------------------------
    def get_psf(self, y, x):

        coeffs = self.coeff_fields[y, x, :]

        phase = np.einsum("kij,k->ij", self.kl_basis[:self.n_modes], coeffs)

        complex_pupil = self.pupil * np.exp(1j * phase)

        N = self.fft_pad * self.psf_size
        if N % 2 == 0:
            N += 1

        padded = np.zeros((N, N), dtype=np.complex64)

        c0 = N // 2
        r = self.psf_size // 2

        padded[c0 - r:c0 + r + 1, c0 - r:c0 + r + 1] = complex_pupil

        psf = np.abs(np.fft.fftshift(np.fft.fft2(padded)))**2

        psf = psf[c0 - r:c0 + r + 1, c0 - r:c0 + r + 1]
        psf /= psf.sum()

        return psf.astype(np.float32)


# ============================================================
# SPACE-VARIANT CONVOLUTION (UNCHANGED PHYSICS)
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

        out = np.zeros_like(self.image, dtype=np.float32)

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

    muram = ReadSimulationEditor().call("/gpfs/data/fs71254/schirni/nstack/data/I_out_med.468000")
    image = muram.data.astype(np.float32)
    image -= image.min()
    image /= image.max()
    image = image[256:-256, 256:-256]

    results = []

    for i in range(args.n_realizations):
        print(f"Realization {i + 1}/{args.n_realizations}")

        psf_gen = PhysicalKLPSFGenerator(
            n_modes=150,
            psf_size=29,
            fft_pad=1,
            seed=30 + i,
            wavelength=450.6e-9,
            telescope_diameter=1.44,
            r0=0.25,
            correlation_length=64
        )

        convolver = SpatiallyVaryingConvolution(image, psf_gen)
        results.append(convolver.convolve())

    stack = np.stack(results, axis=-1)
    np.save(args.out_file, stack)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--n_realizations", type=int, default=1)
    args = parser.parse_args()

    main(args)



#from nbd.data.spatially_varying_psfs import PhysicalKLPSFGenerator
#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#psf_gen = PhysicalKLPSFGenerator(n_modes=150, psf_size=29, fft_pad=1, seed=30, wavelength=450.6e-9, telescope_diameter=1.44, r0=0.15, correlation_length=128)
#psf_gen.initialize_field(512, 512)
#psf = psf_gen.get_psf(0, 0)
#plt.figure(figsize=(8,8), dpi=100)
#plt.imshow(psf, norm=LogNorm(vmin=psf.min(), vmax=psf.max()), origin='lower')
#plt.savefig('/glade/u/home/cschirninger/psf1.jpg')
#plt.close()