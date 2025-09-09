import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nbd.data.editor import cutout
from nbd.evaluation.loader import NBDOutput
from nbd.evaluation.psd import power_spectrum

parser = argparse.ArgumentParser(description='Create evaluation plots for NBD and DKIST data')
parser.add_argument('--base_path', type=str, help='the path to the base directory')

args = parser.parse_args()

base_path = args.base_path
plot_path = base_path + '/plots'
os.makedirs(plot_path, exist_ok=True)

cdelt = 0.011 # arcsec/pixel

model_path = base_path+'/neuralbd.nbd'
neuralbd = NBDOutput(model_path)

# load reconstructed images
reconstructed_pred = neuralbd.load_reconstructed_img()
nbd_mean = reconstructed_pred.mean()

dkist_rec = np.load('/gpfs/data/fs71254/schirni/DKIST/recon.1370.npz')
dkist_rec = dkist_rec['rec']
dkist_rec = cutout(dkist_rec[:, :, None, None], 1500, 2500, 512) # ss: 3000, 3000; penumbra: 1500, 2500
vmin, vmax = dkist_rec.min(), dkist_rec.max()
dkist_rec = (dkist_rec - vmin) / (vmax - vmin)  # normalize to [0, 1]
dkist_mean = dkist_rec.mean()

# shift mean
reconstructed_pred = reconstructed_pred - nbd_mean + dkist_mean

# load convolved images
convolved_pred = np.load(base_path+'/conv_pred.npy')
convolved_true = np.load(base_path+'/conv_true.npy')

# load psfs
psfs_pred = np.load(base_path+'/psfs_pred.npy')

# crop
#reconstructed_pred = reconstructed_pred[230:290, 230:290, :] # 50:-50, 50:-50
#dkist_rec = dkist_rec[225:285, 215:275, :]
#convolved_true = convolved_true[230:290, 230:290, :, :]
#convolved_pred = convolved_pred[100:420, 100:420, :, :]

# calculate power spectral density
k_frame, psd_frame = power_spectrum(convolved_true[:, :, 0, 0] + 1e-10)  # add small value to avoid division by zero
k_nbd, psd_nbd = power_spectrum(reconstructed_pred[:, :, 0])
k_speckle, psd_speckle = power_spectrum(dkist_rec[:, :, 0] + 1e-10)


def _plot_image(x, y, name=None, title1=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    # Display images
    im0 = ax[0].imshow(x, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt],
                       vmin=0, vmax=1)
    im1 = ax[1].imshow(y, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt],
                       vmin=0, vmax=1)

    # Axis labels and titles
    [axs.set_xlabel('Distance [arcsec]', fontsize=20) for axs in ax]
    ax[0].set_ylabel('Distance [arcsec]', fontsize=20)
    ax[1].set_yticks([])
    ax[0].set_title(title1, fontsize=20, fontweight='bold')
    ax[1].set_title('NBD', fontsize=20, fontweight='bold')

    # Ticks formatting
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

def _plot_frame_nbd_speckle(x, y, z, name=None, title1="Frame", title2="NBD", title3="Reconstruction"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

    # Display images
    im0 = ax[0].imshow(x, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, x.shape[1] * cdelt],
                       vmin=0, vmax=1)
    im1 = ax[1].imshow(y, cmap='gray', origin='lower',
                       extent=[0, y.shape[0] * cdelt, 0, y.shape[1] * cdelt],
                       vmin=0, vmax=1)
    im2 = ax[2].imshow(z, cmap='gray', origin='lower',
                       extent=[0, z.shape[0] * cdelt, 0, z.shape[1] * cdelt],
                       vmin=0, vmax=1)

    # Axis labels and titles
    for axs in ax:
        axs.set_xlabel('Distance [arcsec]', fontsize=20)
        axs.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axs.yaxis.set_major_locator(MaxNLocator(nbins=5))
        axs.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))  # round to 1 decimal
        axs.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axs.tick_params(axis='both', which='major', labelsize=20)

    ax[0].set_ylabel('Distance [arcsec]', fontsize=20)
    ax[1].set_yticks([])
    ax[2].set_yticks([])

    ax[0].set_title(title1, fontsize=20, fontweight='bold')
    ax[1].set_title(title2, fontsize=20, fontweight='bold')
    ax[2].set_title(title3, fontsize=20, fontweight='bold')

    # Add colorbars
    for axs, im in zip(ax, [im0, im1, im2]):
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('normalized Intensity', fontsize=20)
        cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(plot_path + f'/frame_nbd_spckl_{name}.jpg' if name else plot_path + '/frame_nbd_spckl.jpg')
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
        ax.set_title(f'PSF {i:02d}', fontsize=12)
        ax.set_xlabel('X [pix]', fontsize=10)
        axs[0].set_ylabel('Y [pix]', fontsize=10)
    # Add colorbar below all subplots (full width)
    cbar_ax = fig.add_axes([0.1, 0.1, 0.85, 0.07])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
    plt.savefig(plot_path + '/psfs_pred.jpg', bbox_inches='tight')
    plt.close()

def _plot_psd(k1, psd1, k2, psd2, k3, psd3):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    axs.semilogy(k1 / cdelt, psd1 / psd1[0], label='Frame', color='green')
    axs.semilogy(k2 / cdelt, psd2 / psd2[0], label='Speckle', color='black')
    axs.semilogy(k3 / cdelt, psd3 / psd3[0], label='NBD', color='red')
    axs.set_xlabel('Spatial frequency [1/Mm]', fontsize=17)
    axs.set_ylabel('Azimuthal PSD', fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=15)
    axs.legend(fontsize=15, loc='upper right')
    axs.set_xlim(0, 63)
    plt.tight_layout()
    plt.savefig(plot_path + '/psd.jpg')

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

if __name__ == '__main__':

    _plot_image(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0])
    _plot_image(dkist_rec[:, :, 0], reconstructed_pred[:, :, 0], name='dkist_vs_nbd', title1='SPECKLE')
    _plot_frame_nbd_speckle(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0], dkist_rec[:, :, 0],)
    _plot_convolved(convolved_true, convolved_pred)
    _plot_psfs(psfs_pred)
    _plot_hist(dkist_rec[:, :, 0], reconstructed_pred[:, :, 0], bins=50)

    # plot psd
    _plot_psd(k_frame, psd_frame, k_speckle, psd_speckle, k_nbd, psd_nbd)