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

parser = argparse.ArgumentParser(description='Create evaluation plots for NBD and KSO data')
parser.add_argument('--base_path', type=str, help='the path to the base directory')

args = parser.parse_args()


base_path = args.base_path
plot_path = base_path + '/plots/crop7'
os.makedirs(plot_path, exist_ok=True)

cdelt = 1.04 # arcsec/pixel

model_path = base_path+'/neuralbd.nbd'
neuralbd = NBDOutput(model_path)

# load reconstructed images
reconstructed_pred = neuralbd.load_reconstructed_img()

# load convolved images
convolved_pred = np.load(base_path+'/conv_pred.npy')
convolved_true = np.load(base_path+'/conv_true.npy')

# load psfs
psfs_pred = np.load(base_path+'/psfs_pred.npy')

# crop
#reconstructed_pred = reconstructed_pred[700:-900, 1100:1548, :]
#convolved_true = convolved_true[700:-900, 1100:1548, :, :]
#convolved_pred = convolved_pred[700:-900, 1100:1548, :, :]

# crop2 ar
#reconstructed_pred = reconstructed_pred[600:856, 980:1236, :]
#convolved_true = convolved_true[600:856, 980:1236, :, :]
#convolved_pred = convolved_pred[600:856, 980:1236, :, :]

# crop3 ar
#reconstructed_pred = reconstructed_pred[1300:1478, 850:1078, :]
#convolved_true = convolved_true[1300:1478, 850:1078, :, :]
#convolved_pred = convolved_pred[1300:1478, 850:1078, :, :]

# crop4 ar
#reconstructed_pred = reconstructed_pred[900:1350, 1600:2050, :]
#convolved_true = convolved_true[900:1350, 1600:2050, :, :]
#convolved_pred = convolved_pred[900:1350, 1600:2050, :, :]

# crop5 ar
#reconstructed_pred = reconstructed_pred[700:800, 980:1080, :]
#convolved_true = convolved_true[700:800, 980:1080, :, :]
#convolved_pred = convolved_pred[700:800, 980:1080, :, :]

# crop6 ar
#reconstructed_pred = reconstructed_pred[600:856, 750:1006, :]
#convolved_true = convolved_true[600:856, 750:1006, :, :]
#convolved_pred = convolved_pred[600:856, 750:1006, :, :]

# crop7 ar
reconstructed_pred = reconstructed_pred[700:800, 780:880, :]
convolved_true = convolved_true[700:800, 780:880, :, :]
convolved_pred = convolved_pred[700:800, 780:880, :, :]


# crop3
#reconstructed_pred = reconstructed_pred[900:950, 1250:1300, :]
#convolved_true = convolved_true[900:950, 1250:1300, :, :]
#convolved_pred = convolved_pred[900:950, 1250:1300, :, :]

# crop limb
#reconstructed_pred = reconstructed_pred[800:-800, :-1500, :] # 50:-50, 50:-50
#convolved_true = convolved_true[800:-800, :-1500, :, :]
#convolved_pred = convolved_pred[800:-800, :-1500, :, :]

# crop limb2
#reconstructed_pred = reconstructed_pred[1000:1100, 50:150, :] # 50:-50, 50:-50
#convolved_true = convolved_true[1000:1100, 50:150, :, :]
#convolved_pred = convolved_pred[1000:1100, 50:150, :, :]

# crop limb ar
#reconstructed_pred = reconstructed_pred[750:1050, 50:350, :] # 50:-50, 50:-50
#convolved_true = convolved_true[750:1050, 50:350, :, :]
#convolved_pred = convolved_pred[750:1050, 50:350]

# calculate power spectral density
k_frame, psd_frame = power_spectrum(convolved_true[:, :, 0, 0] + 1e-10)  # add small value to avoid division by zero
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
    ax.hist(speckle.flatten(), bins=bins, histtype='step', color='blue', alpha=0.5, label='Original')
    ax.hist(nbd.flatten(), bins=bins, histtype='step', color='red', alpha=0.5, label='NeuralBD')
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
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    # Display images and keep references for colorbars
    im0 = ax[0].imshow(x, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt], vmin=vmin, vmax=vmax)
    im1 = ax[1].imshow(y, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt], vmin=vmin, vmax=vmax)

    # Labels and titles
    [axs.set_xlabel('Distance [arcsec]', fontsize=16) for axs in ax]
    ax[0].set_ylabel('Distance [arcsec]', fontsize=16)
    ax[1].set_yticks([])
    ax[0].set_title('Original', fontsize=20, fontweight='bold')
    ax[1].set_title('NeuralBD', fontsize=20, fontweight='bold')

    # Ticks
    for axs in ax:
        axs.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axs.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        axs.tick_params(axis='both', which='major', labelsize=14)

    # Add colorbar to first image
    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    cbar0 = fig.colorbar(im0, cax=cax0)
    cbar0.set_label('normalized Intensity', fontsize=20)
    cbar0.ax.tick_params(labelsize=12)

    # Add colorbar to second image
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('normalized Intensity', fontsize=20)
    cbar1.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(plot_path + f'/reconstructed_image_{name}.jpg' if name else plot_path + '/reconstructed_image.jpg')
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
            ax.imshow(convolved_true[:, :, i, c], cmap='gray', origin='lower', vmin=0, vmax=1, extent=[0, convolved_true.shape[1] * cdelt, 0, convolved_true.shape[0] * cdelt])
            ax.tick_params(axis='both', which='major', labelsize=15)
            [axis.set_yticks([]) for axis in axs[0, 1:i+1]]
            ax = axs[1, i]
            ax.imshow(convolved_pred[:, :, i, c], cmap='gray', origin='lower', vmin=0, vmax=1, extent=[0, convolved_true.shape[1] * cdelt, 0, convolved_true.shape[0] * cdelt])
            ax.tick_params(axis='both', which='major', labelsize=15)
            [axis.set_yticks([]) for axis in axs[1, 1:i+1]]
            [axs[0, i].set_title(f'Frame {i:01d}') for i in range(n_samples)]
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.2)
    # Manually add centered labels
    fig.text(0.5, 0.04, 'X [arcsec]', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Y [arcsec]', ha='center', va='center', rotation='vertical', fontsize=20)
    plt.savefig(plot_path+'/convolved.jpg')
    plt.close()


def _plot_psfs(psfs, name=None):
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
    plt.savefig(plot_path + f'/psfs_{name}.jpg', bbox_inches='tight') if name else plt.savefig(plot_path + '/psfs.jpg', bbox_inches='tight')
    plt.close()

def _plot_psd(k1, psd1, k2, psd2):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    axs.semilogy(k1 / cdelt, psd1 / psd1[0], label='Frame', color='green')
    axs.semilogy(k2 / cdelt, psd2 / psd2[0], label='NeuralBD', color='red')
    axs.set_xlabel('Spatial frequency [1/arcsec]', fontsize=17)
    axs.set_ylabel('Azimuthal PSD', fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=15)
    axs.legend(fontsize=15, loc='upper right')
    # axs.set_xlim(0, 30)
    plt.tight_layout()
    plt.savefig(plot_path + '/psd.jpg')


if __name__ == '__main__':
    bins = np.linspace(convolved_true[:, :, 0, 0].min(), convolved_true[:, :, 0, 0].max(), 100)
    _plot_hists(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0], bins=bins, title_x='Original', title_y='NeuralBD')
    _plot_hist(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0], bins=bins)
    _plot_difference_map(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0], name='original-nbd')
    _plot_image(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0])
    _plot_convolved(convolved_true, convolved_pred)
    _plot_psfs(psfs_pred)

    # plot psd
    _plot_psd(k_frame, psd_frame, k_nbd, psd_nbd)
