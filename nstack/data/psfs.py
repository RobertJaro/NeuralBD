import torch


def PSF(complx_pupil):
    PSF = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(complx_pupil)))
    PSF = (torch.abs(PSF))**2 #or PSF*PSF.conjugate()
    PSF = PSF/ torch.sum(PSF, dim=(0, 1))  #normalizing the PSF
    return PSF

def OTF(psf):
    otf = torch.fft.ifftshift(psf) #move the central frequency to the corner
    otf = torch.fft.fft2(otf)
    otf_max = otf[0, 0] #otf_max = otf[size/2,size/2] if max is shifted to center
    otf = otf/otf_max #normalize by the central frequency signal
    return otf

def MTF(otf):
    mtf = torch.abs(otf)
    return mtf


