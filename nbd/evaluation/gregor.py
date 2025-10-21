import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.io import fits

from nbd.data.editor import cutout
from nbd.evaluation.loader import NBDOutput
from nbd.evaluation.psd import power_spectrum, azimuthal_power_spectrum

parser = argparse.ArgumentParser(description='Create evaluation plots for NBD and Gregor data')
parser.add_argument('--base_path', type=str, help='the path to the base directory')
parser.add_argument('--speckle', type=str, help='the path to the speckle data')

args = parser.parse_args()


base_path = args.base_path
plot_path = base_path + '/plots_crop2'
os.makedirs(plot_path, exist_ok=True)

cdelt = 0.0276 # arcsec/pixel

model_path = base_path+'/neuralbd.nbd'
neuralbd = NBDOutput(model_path)

# load reconstructed images
reconstructed_pred = neuralbd.load_reconstructed_img()
# speckle = neuralbd.speckle
fits_array_speckle = []
for i in range(2):
    fits_array_speckle.append(fits.getdata(args.speckle, i))
fits_array_speckle = np.stack(fits_array_speckle, -1)
fits_array_speckle_crop = cutout(fits_array_speckle[:, :, :, None], 962, 964, reconstructed_pred.shape[0])
speckle = fits_array_speckle_crop[..., 0]
vmin, vmax = speckle.min(), speckle.max()
speckle = (speckle - vmin) / (vmax - vmin)

# load convolved images
convolved_pred = np.load(base_path+'/conv_pred.npy')
convolved_true = np.load(base_path+'/conv_true.npy')

# load psfs
psfs_pred = np.load(base_path+'/psfs_pred.npy')

# crop
#reconstructed_pred = reconstructed_pred[50:-50, 50:-50, :] # 50:-50, 50:-50
#speckle = speckle[48:-52, 50:-50]
#convolved_true = convolved_true[50:-50, 50:-50, :, :]
#convolved_pred = convolved_pred[50:-50, 50:-50, :, :]

reconstructed_pred = reconstructed_pred[120:220, 30:130, :]
speckle = speckle[118:218, 30:130]
convolved_true = convolved_true[120:220, 30:130, :, :]
convolved_pred = convolved_pred[120:220, 30:130, :, :]

reconstructed_pred = (reconstructed_pred - reconstructed_pred.mean()) + speckle.mean()

# calculate power spectral density
k_frame, psd_frame = power_spectrum(convolved_pred[:, :, 0, 0] + 1e-10)  # add small value to avoid division by zero
k_muram, psd_muram = power_spectrum(speckle)
k_nbd, psd_nbd = power_spectrum(reconstructed_pred[:, :, 0])


def _plot_hists(x, y, bins, title_x=None, title_y=None, name=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)
    ax[0].hist(x.flatten(), bins=bins, histtype='step', color='blue', alpha=0.5)
    ax[0].set_title(title_x, fontsize=30, fontweight='bold')
    ax[1].hist(y.flatten(), bins=bins, histtype='step', color='red', alpha=0.5)
    ax[1].set_title(title_y, fontsize=30, fontweight='bold')
    [axs.set_xlabel('normalized Intensity') for axs in ax]
    [axs.set_ylabel('Counts') for axs in ax]
    [axs.set_xlim(0, 1) for axs in ax]
    plt.tight_layout()
    plt.savefig(plot_path+f'/hists_{name}.jpg') if name else plt.savefig(plot_path+'/hists.jpg')
    plt.close()

def _plot_hist(speckle, nbd, bins, name=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=300)
    ax.hist(speckle.flatten(), bins=bins, histtype='step', color='blue', alpha=0.5, label='Speckle')
    ax.hist(nbd.flatten(), bins=bins, histtype='step', color='red', alpha=0.5, label='NBD')
    ax.set_xlabel('normalized Intensity', fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.legend(fontsize=15, loc='upper right')
    plt.savefig(plot_path+f'/hist_{name}.jpg') if name else plt.savefig(plot_path+'/hist.jpg')
    plt.close()

def _plot_image(x, y, name=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300) # cmap='yohkohsxtal'

    # Display images and keep references for colorbars
    im0 = ax[0].imshow(x, cmap='yohkohsxtal', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt], vmin=0, vmax=1)
    im1 = ax[1].imshow(y, cmap='yohkohsxtal', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt], vmin=0, vmax=1)

    # Labels and titles
    [axs.set_xlabel('Distance [arcsec]', fontsize=20) for axs in ax]
    ax[0].set_ylabel('Distance [arcsec]', fontsize=20)
    ax[1].set_yticks([])
    ax[0].set_title('Speckle', fontsize=20, fontweight='bold')
    ax[1].set_title('NBD', fontsize=20, fontweight='bold')

    # Ticks
    for axs in ax:
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        axs.yaxis.set_major_locator(MaxNLocator(integer=True))
        axs.tick_params(axis='both', which='major', labelsize=20)

    # Add colorbar to first image
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('normalized Intensity', fontsize=20)
    cbar0.ax.tick_params(labelsize=18)

    # Add colorbar to second image
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('normalized Intensity', fontsize=20)
    cbar1.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(plot_path + f'/reconstructed_image_{name}.jpg' if name else plot_path + '/reconstructed_image.jpg')
    plt.close()


def _plot_conv_reconstructed(conv, nbd, speckle):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), dpi=300)

    # Show each image and keep the handle for colorbar
    im0 = ax[0].imshow(conv, cmap='yohkohsxtal', origin='lower',
                       extent=[0, conv.shape[0] * cdelt, 0, conv.shape[0] * cdelt], vmin=0, vmax=1)
    im1 = ax[1].imshow(nbd, cmap='yohkohsxtal', origin='lower',
                       extent=[0, nbd.shape[0] * cdelt, 0, nbd.shape[0] * cdelt], vmin=0, vmax=1)
    im2 = ax[2].imshow(speckle, cmap='yohkohsxtal', origin='lower',
                       extent=[0, speckle.shape[0] * cdelt, 0, speckle.shape[0] * cdelt], vmin=0, vmax=1)

    # Set axis labels and titles
    [axs.set_xlabel('Distance [arcsec]', fontsize=20) for axs in ax]
    ax[0].set_ylabel('Distance [arcsec]', fontsize=20)
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[0].set_title('Convolved 1 Frame', fontsize=20, fontweight='bold')
    ax[1].set_title('NBD', fontsize=20, fontweight='bold')
    ax[2].set_title('Speckle', fontsize=20, fontweight='bold')

    # Format ticks
    for axs in ax:
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        axs.yaxis.set_major_locator(MaxNLocator(integer=True))
        axs.tick_params(axis='both', which='major', labelsize=20)

    # Add individual colorbars
    for i, (image, axis) in enumerate(zip([im0, im1, im2], ax)):
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label('normalized Intensity', fontsize=20)
        cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(plot_path + '/conv_reconstructed.jpg')
    plt.close()

def _plot_difference_map(x, y, name=None):
    diff = (x - y) * 100
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    im = ax.imshow(diff, cmap='seismic', origin='lower', vmin=-100, vmax=100,
                   extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Difference [%]', fontsize=20)
    ax.set_xlabel('Distance [arcsec]', fontsize=20)
    ax.set_ylabel('Distance [arcsec]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_path + f'/reconstructed_difference_{name}.jpg') if name else plt.savefig(plot_path + '/reconstructed_difference.jpg')
    plt.close()

def _plot_convolved(convolved_true, convolved_pred):
    n_channels = convolved_true.shape[-1]
    n_images = convolved_true.shape[-2]
    n_samples = min(5, n_images)
    fig, axs = plt.subplots(2, n_samples, figsize=(2 * n_samples, 5), dpi=300)
    for c in range(n_channels):
        for i in range(n_samples):
            ax = axs[0, i]
            ax.imshow(convolved_true[:, :, i, c], cmap='yohkohsxtal', origin='lower', vmin=0, vmax=1, extent=[0, convolved_true.shape[1] * cdelt, 0, convolved_true.shape[0] * cdelt])
            ax.tick_params(axis='both', which='major', labelsize=15)
            [axis.set_yticks([]) for axis in axs[0, 1:i+1]]
            ax = axs[1, i]
            ax.imshow(convolved_pred[:, :, i, c], cmap='yohkohsxtal', origin='lower', vmin=0, vmax=1, extent=[0, convolved_true.shape[1] * cdelt, 0, convolved_true.shape[0] * cdelt])
            ax.tick_params(axis='both', which='major', labelsize=15)
            [axis.set_yticks([]) for axis in axs[1, 1:i+1]]
            [axs[0, i].set_title(f'Frame {i:01d}') for i in range(n_samples)]
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.2)
    # Manually add centered labels
    fig.text(0.5, 0.04, 'X [arcsec]', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Y [arcsec]', ha='center', va='center', rotation='vertical', fontsize=20)
    plt.savefig(plot_path+'/convolved.jpg')
    plt.close()


def _plot_psfs(psfs):
    n_images = psfs.shape[-1]
    n_samples = min(5, n_images)
    fig, axs = plt.subplots(1, n_samples, figsize=(2.2 * n_samples, 4), dpi=300,
                            sharex=True, sharey=True)
    if n_samples == 1:
        axs = [axs]
    # Use log normalization for color scale
    norm = LogNorm(vmin=psfs[psfs > 0].min(), vmax=psfs.max())
    for i in range(n_samples):
        ax = axs[i]
        im = ax.imshow(psfs[:, :, i], origin='lower', norm=norm, cmap='viridis')
        # ax.set_title(f'PSF {i:02d}', fontsize=12)
        ax.set_xlabel('X [pix]', fontsize=18)
        axs[0].set_ylabel('Y [pix]', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=16)
    # Add colorbar below all subplots (full width)
    cbar_ax = fig.add_axes([0.1, 0.1, 0.85, 0.07])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
    plt.savefig(plot_path + '/psfs_pred.jpg', bbox_inches='tight')
    plt.close()

def _plot_psd(k1, psd1, k2, psd2, k3, psd3):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    axs.semilogy(k1 / cdelt, psd1 / psd1[0], label='Frame', color='green')
    axs.semilogy(k2 / cdelt, psd2 / psd2[0], label='Speckle', color='black')
    axs.semilogy(k3 / cdelt, psd3 / psd3[0], label='NBD', color='red')
    axs.set_xlabel('Spatial frequency [1/arcsec]', fontsize=17)
    axs.set_ylabel('Azimuthal PSD', fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=15)
    axs.legend(fontsize=15, loc='upper right')
    axs.set_xlim(0, 25)
    plt.tight_layout()
    plt.savefig(plot_path + '/psd.jpg')

# Test plotting
fig, axs = plt.subplots(1, 1, figsize=(10,10), dpi=300)
norm = LogNorm(vmin=psfs_pred[psfs_pred > 0].min(), vmax=psfs_pred.max())
axs.imshow(psfs_pred[:, :, 4], origin='lower', norm=norm, cmap='viridis')
axs.set_xlabel('X [pix]', fontsize=48)
axs.set_ylabel('Y [pix]', fontsize=48)
axs.tick_params(axis='both', which='major', labelsize=44)
axs.tick_params(axis='both', which='major', labelsize=44)
# add colorbar
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(axs.images[0], cax=cax)
cbar.ax.tick_params(labelsize=42)
plt.tight_layout()
plt.savefig(plot_path + '/psf_pred_4.jpg')
plt.close()

if __name__ == '__main__':
    bins = np.linspace(speckle.min(), speckle.max(), 100)
    _plot_hists(speckle, reconstructed_pred[:, :, 0], bins=bins, title_x='Speckle', title_y='NBD')
    _plot_hist(speckle, reconstructed_pred[:, :, 0], bins=bins)
    _plot_difference_map(speckle, reconstructed_pred[:, :, 0], name='speckle-nbd')
    _plot_image(speckle, reconstructed_pred[:, :, 0])
    _plot_convolved(convolved_true, convolved_pred)
    _plot_conv_reconstructed(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0], speckle)
    _plot_psfs(psfs_pred)

    # plot psd
    _plot_psd(k_frame, psd_frame, k_muram, psd_muram, k_nbd, psd_nbd)
