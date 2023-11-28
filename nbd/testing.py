import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import scipy.ndimage as nd
from skimage.morphology import flood


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real


base_path = '/home/fs71254/schirni/nbd'
lvl1_path = '/gpfs/data/fs71254/schirni/Level1_Files/Level1_Files_GBand/hifi_20170618_082313_sd.fts'
lvl2_path = '/gpfs/data/fs71254/schirni/Level2_Files/Level2_Files_GBand/hifi_20170618_082313_sd_speckle.fts'

hdu1 = fits.open(lvl1_path)
hdu2 = fits.open(lvl2_path)
data1, header1 = hdu1[0].data, hdu1[0].header
data2, header2 = hdu2[0].data, hdu2[0].header

#I = np.fft.fft2(data1)
#O = np.fft.fft2(data2)

I = calculate_2dft(data1)
O = calculate_2dft(data2)

# I = O * S
S = I / O

#i = np.fft.ifft2(I).real
#o = np.fft.ifft2(O).real
#s = np.fft.ifft2(S).real

i, o, s = calculate_2dift(I), calculate_2dift(O), calculate_2dift(S)

Sconj_S = (np.conjugate(S) * S).real



# plot I, O, S in log scale with colorbar
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, img, title in zip(axes, [I, O, S], ['I', 'O', 'S']):
    im = ax.imshow(np.log(np.abs(img)), cmap='seismic')
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
plt.savefig(os.path.join(base_path, 'I_O_S_fourier.jpg'))


# plot i, o, s and data1, data2 below i and o
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for ax, img, title in zip(axes[0], [i, o, s], ['i', 'o', 's']):
    im = ax.imshow(img, cmap='gray')
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
for ax, img, title in zip(axes[1], [data1, data2, Sconj_S], ['data1', 'data2', 'sconj_s']):
    im = ax.imshow(img, cmap='gray')
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)
plt.savefig(os.path.join(base_path, 'i_o_s_data1_data2_sconj_s.jpg'))


