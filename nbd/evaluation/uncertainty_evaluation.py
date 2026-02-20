import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sunpy.visualization.colormaps

from nbd.evaluation.loader import NBDOutput
from matplotlib.colors import LogNorm

base_path = '/gpfs/data/fs71254/schirni/nstack/training/GREGOR_100imgs'
# base_path = '/gpfs/data/fs71254/schirni/nstack/training/GREGOR_wvlnth2'
base_path1 = "/gpfs/data/fs71254/schirni/nstack/training/GREGOR_run2"
# base_path1 = "/gpfs/data/fs71254/schirni/nstack/training/NeuralBD_GREGOR_256"
base_path2 = "/gpfs/data/fs71254/schirni/nstack/training/GREGOR_run3"
base_path3 = "/gpfs/data/fs71254/schirni/nstack/training/GREGOR_run4"
base_path4 = "/gpfs/data/fs71254/schirni/nstack/training/GREGOR_run5"
model_path = base_path + '/neuralbd.nbd'
model_path1 = base_path1 + '/neuralbd.nbd'
model_path2 = base_path2 + '/neuralbd.nbd'
model_path3 = base_path3 + '/neuralbd.nbd'
model_path4 = base_path4 + '/neuralbd.nbd'

neuralbd = NBDOutput(model_path)
neuralbd1 = NBDOutput(model_path1)
neuralbd2 = NBDOutput(model_path2)
neuralbd3 = NBDOutput(model_path3)
neuralbd4 = NBDOutput(model_path4)

reconstructed_pred = neuralbd.load_reconstructed_img()
reconstructed_pred1 = neuralbd1.load_reconstructed_img()
reconstructed_pred2 = neuralbd2.load_reconstructed_img()
reconstructed_pred3 = neuralbd3.load_reconstructed_img()
reconstructed_pred4 = neuralbd4.load_reconstructed_img()

psfs_pred = np.load(base_path + '/psfs_pred.npy')
psfs_pred1 = np.load(base_path1 + '/psfs_pred.npy')
psfs_pred2 = np.load(base_path2 + '/psfs_pred.npy')
psfs_pred3 = np.load(base_path3 + '/psfs_pred.npy')
psfs_pred4 = np.load(base_path4 + '/psfs_pred.npy')

cdelt = 0.0276 # arcsec/pixel

def _plot_psfs(psfs, plot_path, name=None):
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

def _plot_images(images, plot_path, name=None):
    """
    images : list of 5 numpy arrays
    titles : list of 5 strings (optional)
    """
    n = 5
    fig, ax = plt.subplots(1, n, figsize=(25, 5), dpi=300)
    ims = []
    for i in range(n):
        im = ax[i].imshow(
            images[:, :, i],
            cmap='yohkohsxtal',
            origin='lower',
            extent=[0, images[:, :, i].shape[0] * cdelt,
                    0, images[:, :, i].shape[1] * cdelt],
            vmin=0,
            vmax=1
        )
        ims.append(im)
        # Labels
        ax[i].set_xlabel('Distance [arcsec]', fontsize=20)
        if i == 0:
            ax[i].set_ylabel('Distance [arcsec]', fontsize=20)
        else:
            ax[i].set_yticks([])
        # Ticks
        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))
        ax[i].tick_params(axis='both', which='major', labelsize=20)
        # Colorbar
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label('normalized Intensity', fontsize=20)
        cbar.ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(
        plot_path + f'/reconstructed_image_{name}.jpg'
        if name else plot_path + '/reconstructed_image.jpg'
    )
    plt.close()

def _plot_std_map(x, plot_path, name=None):
    map = x * 100
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)
    im = ax.imshow(map, cmap='Reds', origin='lower', vmin=0,
                   extent=[0, x.shape[0] * cdelt, 0, x.shape[0] * cdelt])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=60)
    cbar.set_label('Uncertainty [%]', fontsize=50)
    ax.set_xlabel('X [arcsec]', fontsize=50)
    ax.set_ylabel('Y [arcsec]', fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30)
    plt.tight_layout()
    plt.savefig(plot_path + f'/{name}_std.jpg') if name else plt.savefig(plot_path + '/reconstructed_difference.jpg')
    plt.close()

def _plot_std_map_psf(x, plot_path, name=None):
    map = x * 100
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    im = ax.imshow(map, cmap='Reds', origin='lower', vmin=0, vmax=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=30)
    cbar.set_label('Uncertainty [%]', fontsize=35)
    ax.set_xlabel('X [pix]', fontsize=25)
    ax.set_ylabel('Y [pix]', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.tight_layout()
    plt.savefig(plot_path + f'/{name}_std.jpg') if name else plt.savefig(plot_path + '/reconstructed_difference.jpg')
    plt.close()

def _plot_single_psf(psf, plot_path, name=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=300)
    norm = LogNorm(vmin=psf[psf > 0].min(), vmax=psf.max())
    im = ax.imshow(psf, origin='lower', norm=norm, cmap='viridis')
    ax.set_xlabel('X [pix]', fontsize=20)
    ax.set_ylabel('Y [pix]', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Intensity', fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    plt.tight_layout()
    plt.savefig(plot_path + f'/single_psf_{name}.jpg') if name else plt.savefig(plot_path + '/single_psf.jpg')
    plt.close()

def _plot_mean_psf(psf, plot_path, name=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    norm = LogNorm(vmin=psf[psf > 0].min(), vmax=psf.max())
    im = ax.imshow(psf, cmap='Reds', origin='lower', norm=norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("Mean", fontsize=35)
    ax.set_xlabel('X [pix]', fontsize=25)
    ax.set_ylabel('Y [pix]', fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=22)
    plt.tight_layout()
    plt.savefig(plot_path + f'/{name}_mean.jpg') if name else plt.savefig(plot_path + '/mean.jpg')
    plt.close()


reconstructed_pred_stack = np.stack(
    [reconstructed_pred[:, :, 0], reconstructed_pred1[:, :, 0], reconstructed_pred2[:, :, 0], reconstructed_pred3[:, :, 0], reconstructed_pred4[:, :, 0]], axis=-1)
reconstructed_std = np.std(reconstructed_pred_stack, axis=-1)
reconstructed_std = reconstructed_std / np.max(reconstructed_pred_stack)
reconstructed_mean_std = np.mean(reconstructed_std)

psfs_stack = np.stack([psfs_pred1, psfs_pred2, psfs_pred3, psfs_pred4], axis=-1)
psfs_std = np.std(psfs_stack, axis=-1)
psfs_std = psfs_std / np.max(psfs_stack)
psfs_mean_std = np.mean(psfs_std)

reconstructed_std_mean = np.mean(reconstructed_std)
psfs_std_mean = np.std(psfs_std[:, :, 0])

_plot_psfs(psfs_pred, plot_path='/gpfs/data/fs71254/schirni/nstack', name='0')
_plot_psfs(psfs_pred1, plot_path='/gpfs/data/fs71254/schirni/nstack', name='1')
_plot_psfs(psfs_pred2, plot_path='/gpfs/data/fs71254/schirni/nstack', name='2')
_plot_psfs(psfs_pred3, plot_path='/gpfs/data/fs71254/schirni/nstack', name='3')
_plot_psfs(psfs_pred4, plot_path='/gpfs/data/fs71254/schirni/nstack', name='4')

_plot_images(reconstructed_pred_stack[:, :, 0:5], plot_path='/gpfs/data/fs71254/schirni/nstack', name='recon')
_plot_std_map(reconstructed_std, plot_path='/gpfs/data/fs71254/schirni/nstack', name='recon')
_plot_std_map_psf(psfs_std[:, :, 0], plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs0')
_plot_std_map_psf(psfs_std[:, :, 1], plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs1')
_plot_std_map_psf(psfs_std[:, :, 2], plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs2')
_plot_std_map_psf(psfs_std[:, :, 3], plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs3')
_plot_std_map_psf(psfs_std[:, :, 4], plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs4')

_plot_mean_psf(np.mean(psfs_pred, axis=-1), plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs0')
_plot_mean_psf(np.mean(psfs_pred1, axis=-1), plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs1')
#_plot_mean_psf(np.mean(psfs_pred2, axis=-1), plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs2')
#_plot_mean_psf(np.mean(psfs_pred3, axis=-1), plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs3')
#_plot_mean_psf(np.mean(psfs_pred4, axis=-1), plot_path='/gpfs/data/fs71254/schirni/nstack', name='psfs4')