# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu
from torch import nn

#----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

#----------------------------------------------------------------------------
# Fully-connected layer.

@persistence.persistent_class
class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

#----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.

@persistence.persistent_class
class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x, out_h=None, out_w=None):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        # Adjust f_pad to fit down/up size
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        out_pad = 0 if out_h is None else out_h % 2  # Odd targets need output padding


        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad, output_padding=out_pad)
            if self.down:
                # f.tile([self.in_channels, 1, ,1, 1]) has shape (self.in_channels, 1, 2, 2)
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


#--------------------------------------------------------------
# fourier layer

import tensorly as tl
from tensorly.plugins import use_opt_einsum
tl.set_backend('pytorch')

use_opt_einsum('optimal')

from tltorch.factorized_tensors.core import FactorizedTensor

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1]+rank_sym] #in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym,out_sym+rank_sym] #in, out
    factor_syms += [xs+rank_sym for xs in x_syms[2:]] #x, y, ...
    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, up=False, down=False, verbose=False):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        print("Initializing fourier layer...")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.down = down
        self.up = up

        self.verbose = verbose

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixyt,ioxyt->boxyt", input, weights)

    # Downsampling by truncating fourier modes?
    def forward(self, x, out_h=None, out_w=None):
        # TODO(dahoas): Fix hack
        print("Spec conv input, weights: ", x.shape, self.weights1.shape) if self.verbose else None
        w1 = self.weights1.to(x.dtype)
        w2 = self.weights2.to(x.dtype)
        batchsize, c, h, w = x.shape
        if out_h is None and out_w is None:
            if self.down:
                out_h = h // 2
                out_w = w // 2
            elif self.up:
                out_h = 2 * h
                out_w = 2 * w
            else:
                out_h = h
                out_w = w
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        x_ft = torch.view_as_real(x_ft)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros((batchsize, self.out_channels,  out_h, out_w//2 + 1, 2), device=x.device, dtype=x.dtype)
        #print("x device: ", x.device, "x dtype: ", x.dtype)
        #print("weights device: ", self.weights1.device, "weights dtype: ", self.weights1.dtype)
        #exit()
        print("out_ft shape: {}".format(out_ft.shape)) if self.verbose else None
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)
        # TODO(dahoas): Sampling from the end samples higher modes for larger images
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)

        #Return to physical space
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft2(out_ft, s=(out_h, out_w))
        print("Spectral out x shape: {}".format(x.shape)) if self.verbose else None
        return x

#---------------------------------------------------------------------------

class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes,
                 output_scaling_factor=None,
                 n_layers=1,
                 incremental_n_modes=None,
                 use_mlp=False, mlp=None,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
                 fno_skip='linear',
                 mlp_skip='soft-gating',
                 separable=False,
                 factorization=None,
                 rank=1.0,
                 SpectralConv=SpectralConv2d,
                 joint_factorization=False, 
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 fft_norm='forward',
                 **kwargs):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.n_dim = len(n_modes)

        if output_scaling_factor is not None:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [float(output_scaling_factor)]*len(self.n_modes)
        self.output_scaling_factor = output_scaling_factor

        self._incremental_n_modes = incremental_n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip,
        self.mlp_skip = mlp_skip,
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation

        self.convs = SpectralConv(
                self.in_channels, self.out_channels, self.n_modes, 
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes, 
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=n_layers,
            )

        self.fno_skips = nn.ModuleList([skip_connection(self.in_channels, self.out_channels, type=fno_skip, n_dim=self.n_dim) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.out_channels, 
                     hidden_channels=int(round(self.out_channels*mlp['expansion'])),
                     dropout=mlp['dropout'], n_dim=self.n_dim) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.in_channels, self.out_channels, type=mlp_skip, n_dim=self.n_dim) for _ in range(n_layers)])
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.out_channels) for _ in range(n_layers*self.n_norms)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.out_channels) for _ in range(n_layers*self.n_norms)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm(elementwise_affine=False) for _ in range(n_layers*self.n_norms)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

    def forward(self, x, index=0):
        x_skip_fno = self.fno_skips[index](x)

        x_fno = self.convs(x, index)

        x = x_fno + x_skip_fno

        if not self.preactivation and (self.mlp is not None) or (index < (self.n_layers - index)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)

        if self.mlp is not None:
            # x_skip = self.mlp_skips[index](x)
            x = self.mlp[index](x) + x_skip_mlp

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)
        return x


class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers
    
    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, 
                 n_layers=2, n_dim=2, non_linearity=F.gelu, dropout=0., **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)]) if dropout > 0. else None
        
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x


def skip_connection(in_features, out_features, n_dim=2, bias=False, type="soft-gating"):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D. 
    bias : bool, optional
        whether to use a bias, by default False
    type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if type.lower() == 'soft-gating':
        return SoftGating(in_features=in_features, out_features=out_features, bias=bias, n_dim=n_dim)
    elif type.lower() == 'linear':
        return getattr(nn, f'Conv{n_dim}d')(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=bias)
    elif type.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Got skip-connection {type=}, expected one of {'soft-gating', 'linear', 'id'}.")


class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D. 
    bias : bool, default is False
    """
    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(f"Got {in_features=} and {out_features=}"
                             "but these two must be the same for soft-gating")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,)*n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,)*n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations
        """
        if self.bias is not None:
            return self.weight*x + self.bias
        else:
            return self.weight*x

#---------------------------------------------------------------------------
class DualConv(nn.Module):
    def __init__(self, 
        in_channels, out_channels, kernel, 
        modes1, modes2,
        bias=True, up=False, down=False, resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0, 
        use_spatial=True, use_spectral=True, verbose=False):
        super(DualConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.modes1 = modes1
        self.modes2 = modes2
        self.use_spatial = use_spatial
        self.use_spectral = use_spectral
        self.spatial_conv = Conv2d(in_channels, out_channels, kernel, bias, up, down, 
                                   resample_filter, fused_resample, init_mode, init_weight, init_bias) if use_spatial else None
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes1, modes2, up, down, verbose) if use_spectral else None
        self.verbose = verbose
        self.up = up
        self.down = down

    def forward(self, x, out_h=None, out_w=None):
        spatial_out = self.spatial_conv(x, out_h=out_h, out_w=out_w) if self.use_spatial else 0
        spectral_out = self.spectral_conv(x, out_h=out_h, out_w=out_w) if self.use_spectral else 0
        print("Spatial out nan: {}, Spectral out nan: {}".format(torch.any(spatial_out.isnan()) if self.use_spatial else False, torch.any(spectral_out.isnan()) if self.use_spectral else False)) if self.verbose else None
        print("Spatial out size: {}, Spectral out size: {}".format(spatial_out.shape if self.use_spatial else None, spectral_out.shape if self.use_spectral else None)) if self.verbose else None
        # TODO(dahoas): Try other combination techniques
        return spatial_out + spectral_out

#----------------------------------------------------------------------------
# Group normalization.

@persistence.persistent_class
class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

#----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.

class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk


#----------------------------------------------------------------------------
# Unified DualFNO U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.

@persistence.persistent_class
class DualUNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels, 
        modes1, modes2,
        up=False, down=False, attention=False,
        num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
        resample_filter=[1,1], resample_proj=False, adaptive_scale=True,
        init=dict(), init_zero=dict(init_weight=0), init_attn=None, 
        use_spatial=True, use_spectral=True, verbose=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.use_spatial = use_spatial
        self.use_spectral = use_spectral
        self.verbose = verbose
        self.down = down
        self.up = up


        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = DualConv(in_channels=in_channels, out_channels=out_channels, kernel=3, up=up, down=down, resample_filter=resample_filter, **init,
                              modes1=modes1, modes2=modes2, use_spatial=use_spatial, use_spectral=use_spectral, verbose=verbose)
        self.conv1 = DualConv(in_channels=out_channels, out_channels=out_channels, kernel=3, resample_filter=resample_filter, **init_zero,
                              modes1=modes1, modes2=modes2, use_spatial=use_spatial, use_spectral=use_spectral, verbose=verbose)

        self.affine = Linear(in_features=emb_channels, out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)

        self.skip = None
        if out_channels != in_channels or up or down:
            # TODO(dahoas): This should probably be turned into a dual convolution
            kernel = 1 if resample_proj or out_channels!= in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3, kernel=1, **(init_attn if init_attn is not None else init))
            self.proj = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=1, **init_zero)

    def forward(self, x, emb, out_h=None, out_w=None):
        orig = x
        # Only need to pass out_h, out_w to the up/down sampling layer
        x = self.conv0(silu(self.norm0(x)), out_h=out_h, out_w=out_w)
        
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig, out_h=out_h, out_w=out_w) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.

@persistence.persistent_class
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

#----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.

@persistence.persistent_class
class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


#####################################UNet Architectures#########################################

#----------------------------------------------------------------------------

@persistence.persistent_class
class DualUNet(torch.nn.Module):
    def __init__(self,
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_dim           = 0,            # Number of class labels, 0 = unconditional.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_levels         = [1],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
        mode                = "dual",        # Run convs in def/fourier/dual mode
        modes1_list         = None,           # Number of fourier modes to take in first dim
        modes2_list         = None,           # Number of fourier modes to take in second dim
        dual_block_thresh   = -1,
        random_fourier_feature = None,      # Random matrix B such that we map a vector v -> cos(2piBv),sin(2piBv)
        unfrozen_layers     = None,         # Layers unfrozen for training
        verbose             = False,         # For print debugging
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        self.verbose = verbose
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, use_spatial="fourier"!=mode, use_spectral="def"!=mode,
            verbose=verbose,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)


        self.random_projection_matrix = random_fourier_feature

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            modes1, modes2 = modes1_list[level], modes2_list[level]
            block_kwargs["use_spectral"] = level <= dual_block_thresh if mode=="dual" else block_kwargs["use_spectral"]
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{level}_conv'] = DualConv(in_channels=cin, out_channels=cout, kernel=3, modes1=modes1, modes2=modes2, use_spatial=block_kwargs["use_spatial"], use_spectral=block_kwargs["use_spectral"], **init)
            else:
                self.enc[f'{level}_down'] = DualUNetBlock(in_channels=cout, out_channels=cout, down=True, modes1=modes1, modes2=modes2, **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (level in attn_levels)
                self.enc[f'{level}_block{idx}'] = DualUNetBlock(in_channels=cin, out_channels=cout, attention=attn, modes1=modes1, modes2=modes2, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            modes1, modes2 = modes1_list[level], modes2_list[level]
            #TODO(dahoas): Actually fourier layers end one level lower on encder vs. decoder because of upsampling
            # But actually this also true of the fourier layers in the encoder after down-sampling
            block_kwargs["use_spectral"] = level <= dual_block_thresh if mode=="dual" else block_kwargs["use_spectral"]
            if level == len(channel_mult) - 1:
                self.dec[f'{level}_in0'] = DualUNetBlock(in_channels=cout, out_channels=cout, attention=True, modes1=modes1, modes2=modes2, **block_kwargs)
                self.dec[f'{level}_in1'] = DualUNetBlock(in_channels=cout, out_channels=cout, modes1=modes1, modes2=modes2, **block_kwargs)
            else:
                # For dual upsampling block cannot use full set of modes yet. First must upsample!
                # This block may also only use a spatial convolution if it is at the dual_block_thresh
                block_kwargs["use_spectral"] = level+1 <= dual_block_thresh if mode=="dual" else block_kwargs["use_spectral"]
                self.dec[f'{level}_up'] = DualUNetBlock(in_channels=cout, out_channels=cout, up=True, modes1=modes1_list[level+1], modes2=modes2_list[level+1], **block_kwargs)
                block_kwargs["use_spectral"] = level <= dual_block_thresh if mode=="dual" else block_kwargs["use_spectral"]
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and level in attn_levels)
                self.dec[f'{level}_block{idx}'] = DualUNetBlock(in_channels=cin, out_channels=cout, attention=attn, modes1=modes1, modes2=modes2, **block_kwargs)
            if level == 0:
                self.dec[f'{level}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{level}_aux_conv'] = DualConv(in_channels=cout, out_channels=out_channels, kernel=3, modes1=modes1, modes2=modes2, use_spatial=block_kwargs["use_spatial"], use_spectral=block_kwargs["use_spectral"], **init_zero)

    def fourier_projection(self, v):
        B = self.random_projection_matrix
        if B == None:
            return v
        else:
            pi = torch.pi
            return torch.cat((torch.cos(2*pi*v@B),torch.sin(2*pi*v@B)))

    def forward(self, x, noise_labels, class_labels, augment_labels=None):
        x = self.fourier_projection(x)

        # Mapping.
        print("Input has shape: {}\nHas nan: {}".format(x.shape, torch.any(x.isnan()))) if self.verbose else None
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        aux = x
        resolution_levels = []  # Tracks output resolutions of ith layer
        for name, block in self.enc.items():
            # Keep track of resolution we are downsampling from
            resolution_levels.append(list(x.shape[-2:])) if block.down else None
            x = block(x, emb) if isinstance(block, DualUNetBlock) else block(x)
            print("Out shape at block {}: {}\nHas nan: {}".format(name, x.shape, torch.any(x.isnan()))) if self.verbose else None
            skips.append(x)

        print("res levels: ", resolution_levels) if self.verbose else None

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                out_h, out_w = resolution_levels.pop() if block.up else [None, None]
                print("out_h: ", out_h, "out_w: ", out_w) if self.verbose else None
                x = block(x, emb, out_h=out_h, out_w=out_w)
                print("Out shape at block {}: {}\nHas nan: {}".format(name, x.shape, torch.any(x.isnan()))) if self.verbose else None
        return aux


#####################################Preconditioning-Wrappers#####################################

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMPrecond(torch.nn.Module):
    def __init__(self,
        img_channels,                       # Number of color channels.
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'DualUNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](in_channels=img_channels, out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=x.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
