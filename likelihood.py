import torch
import numpy as np
from scipy import integrate
import tqdm


"""Likelihood function, adapted from the repo by Yang-Song
    https://github.com/yang-song/score_sde_pytorch/tree/main
"""

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(x, t, eps):
        with torch.enable_grad():
            x.requires_grad_(True)
            fn_eps = torch.sum(fn(x, t) * eps)
            grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
            x.requires_grad_(False)
        return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

    return div_fn


def prior_logdensity(z, t):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi * t) - \
        torch.sum(z ** 2, dim=(1, 2, 3)) / (2. * t**2)
    return logps


def get_likelihood_fn(sigma_min, sigma_max, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
    """Create a function to compute the unbiased log-likelihood estimate of a given data point.

    Args:
      sigma_min: Minimal noise scale
      sigma_max: Maximal noise scale
      hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
      rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
      atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
      method: A `str`. The algorithm for the black-box ODE solver.
        See documentation for `scipy.integrate.solve_ivp`.
      eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

    Returns:
      A function that a batch of data points and returns the log-likelihoods in bits/dim,
        the latent code, and the number of function evaluations cost by computation.
    """

    def drift_fn(model, x, t):
        """The drift function of the Probability flow ODE for EDM"""
        t_shape = t
        while len(t_shape.shape) < len(x.shape):
            t_shape = t_shape.unsqueeze(-1)

        return -t_shape * model(x, t)

    def div_fn(model, x, t, noise):
        return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

    def likelihood_fn(model, data):
        """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
          model: A score model.
          data: A PyTorch tensor.

        Returns:
          bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
          z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
            probability flow ODE.
          nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
        """
        pbar = tqdm.tqdm(total=sigma_max,desc="Computing Likelihood")
        with torch.no_grad():
            shape = data.shape
            if hutchinson_type == 'Gaussian':
                epsilon = torch.randn_like(data)
            elif hutchinson_type == 'Rademacher':
                epsilon = torch.randint_like(
                    data, low=0, high=2).float() * 2 - 1.
            else:
                raise NotImplementedError(
                    f"Hutchinson type {hutchinson_type} unknown.")

            def ode_func(t, x):
                # Pass a progress bar if you want to see how its doing
                pbar.set_description(f"Computing Likelihood t : {t}/{sigma_max}")

                sample = from_flattened_numpy(
                    x[:-shape[0]], shape).to(data.device).type(torch.float32)
                vec_t = torch.ones(sample.shape[0], device=sample.device) * t
                drift = to_flattened_numpy(drift_fn(model, sample, vec_t))
                logp_grad = to_flattened_numpy(
                    div_fn(model, sample, vec_t, epsilon))
                return np.concatenate([drift, logp_grad], axis=0)

        s_min = sigma_min
        s_max = sigma_max
        init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
        solution = integrate.solve_ivp(
            ode_func, (s_min, s_max), init, rtol=rtol, atol=atol, method=method)
        pbar.close()
        nfe = solution.nfev
        zp = solution.y[:, -1]
        z = from_flattened_numpy(
            zp[:-shape[0]], shape).to(data.device).type(torch.float32)
        delta_logp = from_flattened_numpy(
            zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
        prior_logp = prior_logdensity(z, s_max)
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[1:])
        bpd = bpd / N
        return bpd, z, nfe

    return likelihood_fn
