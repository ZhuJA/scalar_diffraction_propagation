import torch
import torch.nn as nn
import math
from czt import CZT2D

class fresnel_diff(nn.Module):
    def __init__(self, L, M, wvl, z, device):
        super(fresnel_diff, self).__init__()
        k = 2 * math.pi / wvl
        j = torch.tensor([0 + 1j], dtype=torch.complex64)
        delta_f = 1 / L
        xx = torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32) * delta_f
        FX, FY = torch.meshgrid(xx, xx, indexing='xy')
        H = torch.exp(-j * math.pi * wvl * z * (FX ** 2 + FY ** 2)) * torch.exp(j * k * z)
        self.H = torch.fft.fftshift(H, dim=(-2, -1)).to(device)

    def forward(self, Uin):
        # Uin = torch.tensor(Uin,device=self.device)
        # M, N = Uin.shape[-2:]
        U1 = torch.fft.fft2(torch.fft.fftshift(Uin, dim=(-2, -1)))
        U1 *= self.H
        Uout = torch.fft.ifftshift(torch.fft.ifft2(U1), dim=(-2, -1))
        return Uout
    


class FourierFocal(nn.Module):
    def __init__(self, L, M, fL, wvl, flens):
        super(FourierFocal, self).__init__()
        dx = L / M
        dfx = fL / M
        x0 = torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32) * dx
        self.xx, self.yy = torch.meshgrid(x0, x0, indexing='ij')

        fx0 = torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32) * dfx / (wvl * flens)
        self.fxx, self.fyy = torch.meshgrid(fx0, fx0, indexing='ij')

    def forward(self, Uin):
        # Uin = torch.tensor(Uin,device=self.device)
        # M, N = Uin.shape[-2:]
        Eout = CZT2D(Uin, self.xx, self.yy, self.fxx, self.fyy)
        return Eout
    
class ang_spec_propABCD(nn.Module):
    def __init__(self, M, wvl, d1, d2, ABCD, device):
        super(ang_spec_propABCD, self).__init__()
        A, B, C, D = ABCD
        k = 2 * math.pi / wvl
        j = torch.tensor([0 + 1j], dtype=torch.complex64)
        x1,y1 = torch.meshgrid(torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32)*d1, torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32)*d1, indexing='ij')
        r1sq = x1**2 + y1**2
        delta_f = 1 / (M * d1)
        fx,fy = torch.meshgrid(torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32) * delta_f, torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32) * delta_f, indexing='xy')
        fsq = fx**2 + fy**2
        self.m = d2/d1
        x2,y2 = torch.meshgrid(torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32) * d2, torch.linspace(-M / 2, M / 2 - 1, M, dtype=torch.float32) * d2, indexing='ij')
        r2sq = x2**2 + y2**2
        self.q1 = torch.exp(j*math.pi/(wvl*B)*(A-self.m)*r1sq).to(device)
        self.q2 = torch.fft.fftshift(torch.exp(-j*math.pi*wvl*B/self.m*fsq), dim=(-2, -1)).to(device)
        self.q3 = torch.exp(j*math.pi/(wvl*B)*A*(B*C-A*(A-self.m)/self.m)*r2sq).to(device)

    def forward(self, Uin):
        # Uin = torch.tensor(Uin,device=self.device)
        # M, N = Uin.shape[-2:]
        U1 = torch.fft.fft2(torch.fft.fftshift(self.q1 * Uin/self.m, dim=(-2, -1)))
        U1 *= self.q2
        Uout = self.q3*torch.fft.ifftshift(torch.fft.ifft2(U1), dim=(-2, -1))
        return Uout
    
class ft2(nn.Module):
    def __init__(self, delta):
        super(ft2, self).__init__()
        self.delta = delta

    def forward(self, Uin):
        # Uin = torch.tensor(Uin,device=self.device)
        # M, N = Uin.shape[-2:]
        Uout = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(Uin, dim=(-2, -1))), dim=(-2, -1))*self.delta**2
        return Uout
    

class ift2(nn.Module):
    def __init__(self, delta):
        super(ift2, self).__init__()
        self.delta = delta

    def forward(self, Uin):
        # Uin = torch.tensor(Uin,device=self.device)
        # M, N = Uin.shape[-2:]
        Uout = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.ifftshift(Uin, dim=(-2, -1))), dim=(-2, -1))*self.delta**2
        return Uout
    
if __name__ == '__main__':
    # quick test
    import matplotlib.pyplot as plt
    M = 512
    dx = 6.4e-3
    L = M * dx
    dfx = 3.45e-3
    fL = M * dfx
    wvl = 532e-6
    z = 150
    flens = 150
    device = 'cpu'
    Uin = torch.zeros((M, M), dtype=torch.complex64).to(device)
    Uin[200:312,200:312] = 1

    fresnel = fresnel_diff(L, M, wvl, z, device)
    Uout = fresnel(Uin)
    plt.imshow((Uout.abs().numpy())**2, cmap='gray')
    plt.title('Fresnel Diffraction')
    plt.show()

    prop = ang_spec_propABCD(M, wvl, dx, dx, [1,z,0,1], device)
    Uout2 = prop(Uin)
    plt.imshow((Uout2.abs().numpy())**2, cmap='gray')
    plt.title('Fresnel Diffraction')
    plt.show()

    prop = FourierFocal(L, M, fL, wvl, flens)
    Uout3 = prop(Uin)
    plt.imshow((Uout3.abs().numpy())**2, cmap='gray')
    plt.title('Fourier Diffraction')
    plt.show()

    prop = ft2(dx)
    Uout4 = prop(Uin)
    plt.imshow((Uout4.abs().numpy())**2, cmap='gray')
    plt.title('Fourier Diffraction')
    plt.show()