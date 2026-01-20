import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from nbd.data.editor import cutout, ReadSimulationEditor
from nbd.evaluation.loader import NBDOutput
from nbd.evaluation.psd import power_spectrum
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import richardson_lucy

parser = argparse.ArgumentParser(description='Create evaluation plots for NBD and Muram data')
parser.add_argument('--base_path', type=str, help='the path to the base directory')

args = parser.parse_args()


base_path = args.base_path

plot_path = base_path + '/plots'
os.makedirs(plot_path, exist_ok=True)

cdelt = 96 # km/pixel
cdelt = cdelt / 1000 # Mm/pixel

model_path = base_path+'/neuralbd.nbd'
neuralbd = NBDOutput(model_path)

# load reconstructed images
reconstructed_pred = neuralbd.load_reconstructed_img()


muram = ReadSimulationEditor().call('/gpfs/data/fs71254/schirni/nstack/data/I_out_med.468000')
sim_array = muram.data

vmin, vmax = np.min(sim_array), np.max(sim_array)
sim_array = (sim_array - vmin) / (vmax - vmin)

sim_array = np.stack([sim_array, sim_array], -1)
#sim_array = cutout(sim_array[..., None], 332, 332, reconstructed_pred.shape[0]) # 478, 349, 256 # old: 332, 332, 256

# load convolved images
convolved_pred = np.load(base_path+'/conv_pred.npy')
convolved_true = np.load(base_path+'/conv_true.npy')

# remove 10% of the intensity from sim_array from recontructed_pred
# reconstructed_pred = reconstructed_pred * 0.9
# reconstructed_pred = (reconstructed_pred - reconstructed_pred.mean()) + sim_array.mean()

# load psfs
psfs_pred = np.load(base_path+'/psfs_pred.npy')
psfs_true = np.load(base_path+'/psfs_true.npy')

# perform richardson lucy deconvolution on convolved images
# rl_reconstruction = richardson_lucy(convolved_true[:, :, 0, 0], psfs_true[:, :, 0], num_iter=1000)

# crop sim array and reconstructed and convolved images
crop_param = 120
#sim_array = sim_array[crop_param:-crop_param, crop_param:-crop_param, :]
#reconstructed_pred = reconstructed_pred[crop_param:-crop_param, crop_param:-crop_param, :]
#convolved_pred = convolved_pred[crop_param:-crop_param, crop_param:-crop_param, :, :]
#convolved_true = convolved_true[crop_param:-crop_param, crop_param:-crop_param, :, :]

# RL plot
#sim_array = sim_array[58:138, 58:138 :]
#reconstructed_pred = reconstructed_pred[60:140, 60:140, :]
#convolved_pred = convolved_pred[60:140, 60:140, :, :]
#convolved_true = convolved_true[60:140, 60:140, :, :]
# rl_reconstruction = rl_reconstruction[55:135, 55:135]

# quiet
#sim_array = sim_array[8:78, 8:78 :]
#reconstructed_pred = reconstructed_pred[10:80, 10:80, :]
#convolved_pred = convolved_pred[10:80, 10:80, :, :]
#convolved_true = convolved_true[10:80, 10:80, :, :]
#rl_reconstruction = rl_reconstruction[10:80, 10:80]

# penmumbra
#sim_array = sim_array[108:178, 108:178, :]
#reconstructed_pred = reconstructed_pred[110:180, 110:180, :]
#convolved_pred = convolved_pred[110:180, 110:180, :, :]
#convolved_true = convolved_true[110:180, 110:180, :, :]
#rl_reconstruction = rl_reconstruction[110:180, 110:180]

# umbra
#sim_array = sim_array[158:228, 158:228, :]
#reconstructed_pred = reconstructed_pred[160:230, 160:230, :]
#convolved_pred = convolved_pred[160:230, 160:230, :, :]
#convolved_true = convolved_true[160:230, 160:230, :, :]
#rl_reconstruction = rl_reconstruction[160:230, 160:230]

# calculate power spectral density
k_frame, psd_frame = power_spectrum(convolved_true[:, :, 0, 0] + 1e-10)
k_muram, psd_muram = power_spectrum(sim_array[:, :, 0])
k_nbd, psd_nbd = power_spectrum(reconstructed_pred[:, :, 0])
# k_rl, psd_rl = power_spectrum(rl_reconstruction)


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

def _plot_hist(speckle, nbd, bins, name=None, set_xlim=True):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    ax.hist(speckle.flatten(), bins=bins, histtype='step', color='blue', alpha=0.5, label='MuRAM')
    ax.hist(nbd.flatten(), bins=bins, histtype='step', color='red', alpha=0.5, label='NBD')
    ax.set_xlabel('normalized Intensity', fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    if set_xlim:
        ax.set_xlim(0, 1)
    plt.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=15, loc='upper right')
    plt.savefig(plot_path+f'/hist_{name}.jpg') if name else plt.savefig(plot_path+'/hist.jpg')
    plt.close()

def _plot_hist_conv(speckle, nbd, conv, bins, name=None, set_xlim=True):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    ax.hist(speckle.flatten(), bins=bins, histtype='step', color='blue', alpha=0.5, label='MuRAM')
    ax.hist(nbd.flatten(), bins=bins, histtype='step', color='red', alpha=0.5, label='NBD')
    ax.hist(conv.flatten(), bins=bins, histtype='step', color='green', alpha=0.5, label='Convolved')
    ax.set_xlabel('normalized Intensity', fontsize=20)
    ax.set_ylabel('Counts', fontsize=20)
    if set_xlim:
        ax.set_xlim(0, 1)
    plt.tight_layout()
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(fontsize=15, loc='upper right')
    plt.savefig(plot_path + f'/hist+conv_{name}.jpg') if name else plt.savefig(plot_path + '/hist.jpg')
    plt.close()

def _plot_image(x, y, name=None):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    # Display images and keep references for colorbars
    im0 = ax[0].imshow(x, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt], vmin=0, vmax=0.7)
    im1 = ax[1].imshow(y, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, y.shape[0] * cdelt], vmin=0, vmax=0.7)

    # Labels and titles
    [axs.set_xlabel('Distance [Mm]', fontsize=20) for axs in ax]
    ax[0].set_ylabel('Distance [Mm]', fontsize=20)
    ax[1].set_yticks([])
    ax[0].set_title('MuRAM', fontsize=20, fontweight='bold')
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

def _plot_images(x, y, z, name=None):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=300)  # removed constrained_layout=True

    # Display images and store references
    im0 = ax[0].imshow(x, cmap='gray', origin='lower',
                       extent=[0, x.shape[0] * cdelt, 0, x.shape[1] * cdelt],
                       vmin=0, vmax=1)
    im1 = ax[1].imshow(y, cmap='gray', origin='lower',
                       extent=[0, y.shape[0] * cdelt, 0, y.shape[1] * cdelt],
                       vmin=0, vmax=1)
    im2 = ax[2].imshow(z, cmap='gray', origin='lower',
                       extent=[0, z.shape[0] * cdelt, 0, z.shape[1] * cdelt],
                       vmin=0, vmax=1)

    # Labels and titles
    [axs.set_xlabel('X [Mm]', fontsize=20) for axs in ax]
    ax[0].set_ylabel('Y [Mm]', fontsize=20)
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[0].set_title('MuRAM', fontsize=20, fontweight='bold')
    ax[1].set_title('NBD', fontsize=20, fontweight='bold')
    ax[2].set_title('Richardson Lucy', fontsize=20, fontweight='bold')

    # Format ticks
    for axs in ax:
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        axs.yaxis.set_major_locator(MaxNLocator(integer=True))
        axs.tick_params(axis='both', which='major', labelsize=20)

    # Add colorbars individually
    for image, axis in zip([im0, im1, im2], ax):
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label('Intensity', fontsize=16)
        cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()  # Use this instead of constrained_layout
    if name:
        plt.savefig(plot_path + f'/reconstructed_image_{name}.jpg', bbox_inches='tight')
    else:
        plt.savefig(plot_path + '/reconstructed_image.jpg', bbox_inches='tight')
    plt.close()


def _plot_conv_reconstructed(conv, nbd, speckle, name=None):
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5), dpi=300)

    # Show images
    im0 = ax[0].imshow(conv, cmap='gray', origin='lower',
                       extent=[0, conv.shape[0] * cdelt, 0, conv.shape[0] * cdelt],
                       vmin=0, vmax=1)
    im1 = ax[1].imshow(nbd, cmap='gray', origin='lower',
                       extent=[0, nbd.shape[0] * cdelt, 0, nbd.shape[0] * cdelt],
                       vmin=0, vmax=1)
    im2 = ax[2].imshow(speckle, cmap='gray', origin='lower',
                       extent=[0, speckle.shape[0] * cdelt, 0, speckle.shape[0] * cdelt],
                       vmin=0, vmax=1)

    # Axis labels and titles
    [axs.set_xlabel('X [Mm]', fontsize=20) for axs in ax]
    ax[0].set_ylabel('Y [Mm]', fontsize=20)
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    ax[0].set_title('Convolved', fontsize=20, fontweight='bold')
    ax[1].set_title('NBD', fontsize=20, fontweight='bold')
    ax[2].set_title('MuRAM', fontsize=20, fontweight='bold')

    # Format ticks
    for axs in ax:
        axs.xaxis.set_major_locator(MaxNLocator(integer=True))
        axs.yaxis.set_major_locator(MaxNLocator(integer=True))
        axs.tick_params(axis='both', which='major', labelsize=20)

    # Add individual colorbars
    for image, axis in zip([im0, im1, im2], ax):
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label('normalized Intensity', fontsize=20)
        cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(plot_path + f'/conv_reconstructed_{name}.jpg' if name else plot_path + '/conv_reconstructed.jpg')
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
    ax.set_xlabel('X [Mm]', fontsize=20)
    ax.set_ylabel('Y [Mm]', fontsize=20)
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
            ax.imshow(convolved_true[:, :, i, c], cmap='gray', origin='lower', vmin=0, vmax=0.7, extent=[0, convolved_true.shape[1] * cdelt, 0, convolved_true.shape[0] * cdelt])
            ax.tick_params(axis='both', which='major', labelsize=15)
            [axis.set_yticks([]) for axis in axs[0, 1:i+1]]
            ax = axs[1, i]
            ax.imshow(convolved_pred[:, :, i, c], cmap='gray', origin='lower', vmin=0, vmax=0.7, extent=[0, convolved_true.shape[1] * cdelt, 0, convolved_true.shape[0] * cdelt])
            ax.tick_params(axis='both', which='major', labelsize=15)
            [axis.set_yticks([]) for axis in axs[1, 1:i+1]]
            [axs[0, i].set_title(f'Frame {i:01d}') for i in range(n_samples)]
    fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.2)
    # Manually add centered labels
    fig.text(0.5, 0.04, 'X [Mm]', ha='center', va='center', fontsize=20)
    fig.text(0.04, 0.5, 'Y [Mm]', ha='center', va='center', rotation='vertical', fontsize=20)
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
        ax.set_title(f'PSF {i:02d}', fontsize=12)
        ax.set_xlabel('Pixels', fontsize=10)
        axs[0].set_ylabel('Pixels', fontsize=10)
    # Add colorbar below all subplots (full width)
    cbar_ax = fig.add_axes([0.1, 0.1, 0.85, 0.07])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3)
    plt.savefig(plot_path + f'/psfs_{name}.jpg' if name else plot_path + '/psfs.jpg', bbox_inches='tight')
    plt.close()


def _plot_kl_psfs(kl_psfs, psfs_pred):
    n_images = psfs_pred.shape[-1]
    n_samples = min(5, n_images)
    fig, axs = plt.subplots(2, n_samples, figsize=(2 * n_samples, 4), dpi=300,
                            sharex=True, sharey=True)
    if n_samples == 1:
        axs = [axs]
    # Combine both arrays to determine global vmin for LogNorm
    # all_data = np.concatenate([kl_psfs[..., :n_samples].flatten(),
    #                            psfs_pred[..., :n_samples].flatten()])
    # all_data = all_data[all_data > 0]  # filter out zero or negative values
    # norm = LogNorm(vmin=all_data.min(), vmax=all_data.max())
    for i in range(n_samples):
        ax = axs[0, i]
        im = ax.imshow(np.sqrt(kl_psfs[:, :, i]), origin='lower', cmap='viridis')
        ax.set_title(f'PSF {i:02d}')

        ax = axs[1, i]
        im = ax.imshow(np.sqrt(psfs_pred[:, :, i]), origin='lower', cmap='viridis')
        ax.set_xlabel('Pixels', fontsize=10)
    # Add colorbar below all subplots
    cbar_ax = fig.add_axes([0.1, 0.1, 0.85, 0.07])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.31, wspace=0.3)
    plt.savefig(plot_path + '/kl_psfs_pred.jpg', bbox_inches='tight')
    plt.close()

def _plot_psd(k1, psd1, k2, psd2, k3, psd3, k4=None, psd4=None, name=None):
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
    axs.semilogy(k1 / cdelt, psd1 / psd1[0], label='Frame', color='green')
    axs.semilogy(k2 / cdelt, psd2 / psd2[0], label='MURaM', color='black')
    axs.semilogy(k3 / cdelt, psd3 / psd3[0], label='NBD', color='red')
    if k4 is not None and psd4 is not None:
        axs.semilogy(k4 / cdelt, psd4 / psd4[0], label='RL', color='blue')
    axs.set_xlabel('Spatial frequency [1/Mm]', fontsize=17)
    axs.set_ylabel('Azimuthal PSD', fontsize=17)
    axs.tick_params(axis='both', which='major', labelsize=15)
    axs.legend(fontsize=15, loc='upper right')
    axs.set_xlim(0, 7.2)
    plt.tight_layout()
    plt.savefig(plot_path + f'/psd_{name}.jpg') if name else plt.savefig(plot_path + '/psd.jpg')

def _plot_scatter(x, y, name=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    ax.scatter(x.flatten(), y.flatten(), s=2, alpha=0.5, color='purple')
    ax.set_xlabel('MURaM', fontsize=20)
    ax.set_ylabel('NBD', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_path + f'/scatter_{name}.jpg') if name else plt.savefig(plot_path + '/scatter.jpg')
    plt.close()

def _plot_2d_hist(x, y, name=None):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
    h = ax.hist2d(x.flatten(), y.flatten(), bins=100, cmap='plasma', norm=LogNorm())
    ax.set_xlabel('MURaM', fontsize=25)
    ax.set_ylabel('NBD', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=22)
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Counts', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig(plot_path + f'/2d_hist_{name}.jpg') if name else plt.savefig(plot_path + '/2d_hist.jpg')
    plt.close()


def _plot_2d_hist_combined(x, y1, y2, name=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    bins = 100
    x_flat = x.flatten()

    # Compute histograms to determine shared normalization
    H1, xedges, yedges1 = np.histogram2d(x_flat, y1.flatten(), bins=bins)
    H2, _, yedges2 = np.histogram2d(x_flat, y2.flatten(), bins=bins)

    # Shared LogNorm (avoid zeros)
    vmin = min(H1[H1 > 0].min(), H2[H2 > 0].min())
    vmax = max(H1.max(), H2.max())
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Plot histograms
    h1 = axes[0].hist2d(
        x_flat, y1.flatten(),
        bins=[xedges, yedges1],
        cmap='plasma',
        norm=norm
    )

    h2 = axes[1].hist2d(
        x_flat, y2.flatten(),
        bins=[xedges, yedges2],
        cmap='plasma',
        norm=norm
    )

    for ax in axes:
        ax.set_xlabel('MURaM', fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=22)
    axes[0].set_ylabel('NeuralBD', fontsize=30)
    axes[1].set_ylabel('Richardson-Lucy', fontsize=30)
    # Shared colorbar
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.15)

    cbar = fig.colorbar(h1[3], cax=cax)
    cbar.set_label('Counts', fontsize=28)
    cbar.ax.tick_params(labelsize=22)

    plt.tight_layout()
    plt.savefig(
        plot_path + f'/2d_hist_{name}.jpg'
        if name else plot_path + '/2d_hist.jpg'
    )
    plt.close()


def mse(x, y):
    return np.mean((x - y) ** 2)

def rmse(x, y):
    return np.sqrt(np.mean((x - y) ** 2))



if __name__ == '__main__':
    # Plot
    bins = np.linspace(sim_array.min(), sim_array.max(), 100)
    _plot_hists(sim_array[:, :, 0], reconstructed_pred[:, :, 0], bins=bins, title_x='MuRAM', title_y='NBD')
    _plot_hist(sim_array[:, :, 0], reconstructed_pred[:, :, 0], bins=bins)
    _plot_hist_conv(sim_array[:, :, 0], reconstructed_pred[:, :, 0], convolved_true[:, :, 1, 0], bins=bins, name='convolved')
    #_plot_difference_map(sim_array[:, :, 0], reconstructed_pred[:, :, 0])
    _plot_image(sim_array[:, :, 0], reconstructed_pred[:, :, 0])
    _plot_convolved(convolved_true, convolved_pred)
    _plot_conv_reconstructed(convolved_true[:, :, 0, 0], reconstructed_pred[:, :, 0], sim_array[:,:,0])
    _plot_psfs(psfs_pred, name='pred')
    _plot_psfs(psfs_true, name='true')
    _plot_kl_psfs(psfs_true, psfs_pred)

    convolved_true_sum = np.sum(convolved_true, axis=-2)
    convolved_pred_sum = np.sum(convolved_pred, axis=-2)
    # normalize
    convolved_true_sum = (convolved_true_sum - np.min(convolved_true_sum)) / (np.max(convolved_true_sum) - np.min(convolved_true_sum))
    convolved_pred_sum = (convolved_pred_sum - np.min(convolved_pred_sum)) / (np.max(convolved_pred_sum) - np.min(convolved_pred_sum))
    _plot_hist(convolved_true_sum[:, :, 0], convolved_pred_sum[:, :, 0], bins=100, name='convolved_sum', set_xlim=False)

    #_plot_difference_map(convolved_true[:, :, 0, 0], convolved_pred[:, :, 0, 0], name='convolved_diff')
    #_plot_difference_map(sim_array[:, :, 0], convolved_true[:, :, 1, 0], name='sim_convolved_diff')
    #_plot_difference_map(sim_array[:, :, 0], rl_reconstruction, name='sim_rl_diff')

    _plot_psd(k_frame, psd_frame, k_muram, psd_muram, k_nbd, psd_nbd)
    #_plot_psd(k_frame, psd_frame, k_muram, psd_muram, k_nbd, psd_nbd, k_rl, psd_rl, name='psd_rl')

    #_plot_images(sim_array[:, :, 0], reconstructed_pred[:, :, 0], rl_reconstruction, name='sim_rl_reconstruction')

    #_plot_scatter(sim_array[:, :, 0], reconstructed_pred[:, :, 0])
    #_plot_2d_hist(sim_array[:, :, 0], reconstructed_pred[:, :, 0], name='nbd')
    #_plot_2d_hist(sim_array[:, :, 0], rl_reconstruction, name='rl')
    #_plot_2d_hist_combined(sim_array[:, :, 0], reconstructed_pred[:, :, 0], rl_reconstruction, name='nbd_rl')

    #mse_reconstructed = mse(sim_array[:, :, 0], reconstructed_pred[:, :, 0])
    #rmse_reconstructed = rmse(sim_array[:, :, 0], reconstructed_pred[:, :, 0])
    #mse_baseline = mse(sim_array[:, :, 0], convolved_true[:, :, 1, 0])
    #rmse_baseline = rmse(sim_array[:, :, 0], convolved_true[:, :, 1, 0])
    #mse_rl = mse(sim_array[:, :, 0], rl_reconstruction)
    #rmse_rl = rmse(sim_array[:, :, 0], rl_reconstruction)

    #reconstructed_pred_norm = (reconstructed_pred[:, :, 0] - np.min(reconstructed_pred[:, :, 0])) / (np.max(reconstructed_pred[:, :, 0]) - np.min(reconstructed_pred[:, :, 0]))
    #sim_array_norm = (sim_array[:, :, 0] - np.min(sim_array[:, :, 0])) / (np.max(sim_array[:, :, 0]) - np.min(sim_array[:, :, 0]))
    #rl_reconstruction = (rl_reconstruction - np.min(rl_reconstruction)) / (np.max(rl_reconstruction) - np.min(rl_reconstruction))
    #convolved_true_norm = (convolved_true[:, :, 1, 0] - np.min(convolved_true[:, :, 1, 0])) / (np.max(convolved_true[:, :, 1, 0]) - np.min(convolved_true[:, :, 1, 0]))
    #ssim_reconstructed = ssim(reconstructed_pred_norm, sim_array_norm, data_range=1.0)
    #ssim_baseline = ssim(convolved_true_norm, sim_array_norm, data_range=1.0)
    #ssim_rl = ssim(rl_reconstruction, sim_array_norm, data_range=1.0)

    #print(f'MSE NBD: {mse_reconstructed:.4f}, RMSE NBD: {rmse_reconstructed:.4f}')
    #print(f'MSE Baseline: {mse_baseline:.4f}, RMSE Baseline: {rmse_baseline:.4f}')
    #print(f'MSE RL: {mse_rl:.4f}, RMSE RL: {rmse_rl:.4f}')

    #print(f'SSIM NBD: {ssim_reconstructed:.4f}, SSIM Baseline: {ssim_baseline:.4f}, SSIM RL: {ssim_rl:.4f}')
    #print(f'PSNR NBD: {20 * np.log10(1 / rmse_reconstructed):.4f}, PSNR Baseline: {20 * np.log10(1 / rmse_baseline):.4f}')
    #print(f'PSNR RL: {20 * np.log10(1 / rmse_rl):.4f}')