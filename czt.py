#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    ：czt.py
@IDE     ：PyCharm 
@Author  ：ZJA
@Email ：zhujunan1998@163.com
@Date    ：2025/9/13 15:02
This code is used to compute the chirp-z transform using PyTorch.
"""
import math
import torch
import numpy as np


# %%
def czt(x, m=None, w=None, a=1.):
    device = x.device
    n = x.shape[-1]
    if m is None:
        m = n
    if w is None:
        w = torch.exp(-2j * math.pi / m)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.complex64)
    nfft = int(2 ** nextpow2(n + m - 1))
    k = torch.arange(-n + 1, max((m, n)), device=device) ** 2 / 2
    ww = w ** k
    aa = a ** (-torch.arange(0, n, dtype=torch.float32, device=device)) * ww[n - 1:2 * n - 1]
    y = x * aa
    fy = torch.fft.fft(y, nfft)
    fv = torch.fft.fft(1 / ww[:m + n - 1], nfft)
    fy = fy * fv
    g0 = torch.fft.ifft(fy)
    g = g0[..., n - 1:n + m - 1] * ww[n - 1:n + m - 1]
    return g


def nextpow2(n):
    '''
    nextpow2: Exponents of higher powers of 2
    Eg:
    nextpow2(2) = 1
    nextpow2(2**10+1) = 11
    nextpow2(2**20+1) = 21
    '''
    return np.ceil(np.log2(abs(n)))


def CZT2D(Inpt, XList, YList, UList, VList):
    """
    :param Inpt: input tensor
    :param XList: input axis-X
    :param YList: input axis-Y
    :param UList: output axis-U
    :param VList: output axis-V
    :return: the fourier transform results of Input tensor
    """

    my_device = Inpt.device
    LX = torch.max(XList) - torch.min(XList)
    LY = torch.max(YList) - torch.min(YList)
    NX = len(XList)
    dX = LX / (NX - 1)
    NY = len(YList)
    dY = LY / (NY - 1)

    LU = torch.max(UList) - torch.min(UList)
    LV = torch.max(VList) - torch.min(VList)
    MU = len(UList)
    dU = LU / (MU - 1)
    MV = len(VList)
    dV = LV / (MV - 1)

    indX, indY = torch.meshgrid(torch.linspace(0, NX - 1, NX), torch.linspace(0, NY - 1, NY), indexing='ij')
    indX, indY = indX.to(my_device), indY.to(my_device)

    WX = torch.exp(-1j * 2 * math.pi * dX * dU)
    WY = torch.exp(-1j * 2 * math.pi * dY * dV)
    AX = torch.exp(1j * 2 * math.pi * torch.min(XList) * dU)
    AY = torch.exp(1j * 2 * math.pi * torch.min(YList) * dV)

    BX = torch.exp(-1j * 2 * math.pi * torch.min(UList) * dX * indX)
    BY = torch.exp(-1j * 2 * math.pi * torch.min(VList) * dY * indY)
    CXU = torch.exp(1j * 2 * math.pi * torch.min(XList) * torch.min(UList))
    CYV = torch.exp(1j * 2 * math.pi * torch.min(YList) * torch.min(VList))

    return (CYV * BY) * (CXU * BX) * torch.transpose(czt(torch.transpose(czt(Inpt, NY, WY, AY), -2, -1), NX, WX, AX),
                                                     -2, -1)


def CZT1D(Inpt, XList, UList):
    my_device = Inpt.device
    LX = torch.max(XList) - torch.min(XList)
    NX = len(XList)
    dX = LX / (NX - 1)

    LU = torch.max(UList) - torch.min(UList)
    MU = len(UList)
    dU = LU / (MU - 1)

    indX = torch.linspace(0, NX - 1, NX)
    indX = indX.to(my_device)

    WX = torch.exp(-1j * 2 * math.pi * dX * dU)
    AX = torch.exp(1j * 2 * math.pi * torch.min(XList) * dU)
    BX = torch.exp(-1j * 2 * math.pi * torch.min(UList) * dX * indX)
    CXU = torch.exp(1j * 2 * math.pi * torch.min(XList) * torch.min(UList))

    return (CXU * BX) * czt(Inpt, NX, WX, AX)


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    M = 512
    dx = 16e-3  # input plane pixel size
    dfx = 4e-3  # output plane pixel size
    L = M * dx
    wvl = 532e-6  # wavelength
    f1 = 200  # focal length of Fourier lens
    # source plane
    x0 = torch.linspace(-M / 2 + 1, M / 2, M, dtype=torch.float32) * dx
    xx, yy = torch.meshgrid(x0, x0, indexing='ij')

    Fx = torch.linspace(-M / 2 + 1, M / 2, M, dtype=torch.float32) * dfx / (wvl * f1)
    Fxx, Fyy = torch.meshgrid(Fx, Fx, indexing='ij')

    Ein = torch.zeros([3,512, 512], device='cpu')
    Ein[0,216:226, 206:306] = 1
    Ein[0,286:296, 206:306] = 1
    Ein[1,251:261, 206:306] = 1
    # Ein[1,286:296, 206:306] = 1
    Ein[2,226:236, 206:306] = 1
    Ein[2,276:286, 206:306] = 1

    # 1D Fourier transform example
    # Ein1D = Ein[:, 256]
    # Eout1D = CZT1D(Ein1D, x0, Fx)
    # plt.subplot(1, 2, 1)
    # plt.plot(Ein1D, label='')
    # plt.subplot(1, 2, 2)
    # plt.plot((abs(Eout1D) / max(abs(Eout1D))) ** 2, label='')
    # plt.show()

    # 2D Fourier transform example
    Eout = CZT2D(Ein, xx, yy, Fxx, Fyy)
    plt.subplot(1, 3, 1)
    plt.imshow(abs(Ein[0,:,:]) ** 2, 'jet')
    plt.title('Ein')
    plt.subplot(1, 3, 2)
    plt.imshow(abs(Ein[1,:,:]) ** 2, 'jet')
    plt.title('Ein')
    plt.subplot(1, 3, 3)
    plt.imshow(abs(Ein[2,:,:]) ** 2, 'jet')
    plt.title('Ein')

    plt.subplot(1, 3, 1)
    plt.imshow(abs(Eout[0,:,:]) ** 2, 'jet')
    plt.title('Eout1')
    plt.subplot(1, 3, 2)
    plt.imshow(abs(Eout[1,:,:]) ** 2, 'jet')
    plt.title('Eout2')
    plt.subplot(1, 3, 3)
    plt.imshow(abs(Eout[2,:,:]) ** 2, 'jet')
    plt.title('Eout3')
    # plt.subplot(1, 3, 3)
    # plt.plot(abs(Eout[300,:]) ** 2)
    # plt.tight_layout()
    # plt.title('Eout crossline')
    plt.show()
