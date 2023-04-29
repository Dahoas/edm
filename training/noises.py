import torch
import math


class Noise:
    def __init__(self, name):
        self.name = name


class Normal(Noise):
    def __init__(self, size, mean, var, name="normal"):
        super().__init__(name)
        self.mean = mean
        self.var = var
        self.size = list(size)

    def __call__(self, y):
        return (self.var)**(1/2) * torch.randn_like(y) + self.mean


class GaussianRF(Noise):

    def __init__(self, size, alpha=2.0, tau=3.0, sigma=None, boundary="periodic", device=None, name="trace_class"):
        """
        size : ?
        alpha : ?
        tau : ?
        sigma : ?
        """

        self.device = device
        self.dim = 2

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

        k_x = wavenumers.transpose(0,1)
        k_y = wavenumers

        self.sqrt_eig = (size**2)*math.sqrt(2.0)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def __call__(self, y):

        N = y.shape[0]
        self.device = y.device
        coeff = torch.randn(N, *self.size, 2, device=self.device)*mul

        coeff[...,0] = self.sqrt_eig*coeff[...,0] #real
        coeff[...,1] = self.sqrt_eig*coeff[...,1] #imag

        ##########torch 1.7###############
        #u = torch.ifft(coeff, self.dim, normalized=False)
        #u = u[...,0]
        ##################################

        #########torch latest#############
        coeff_new = torch.complex(coeff[...,0],coeff[...,1])
        #print(coeff_new.size())
        u = torch.fft.ifft2(coeff_new, dim = (-2,-1), norm=None)
        u = u.real
        return u