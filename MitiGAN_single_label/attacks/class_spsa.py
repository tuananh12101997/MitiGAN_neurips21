import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch


def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


def batch_clamp(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_clamp_tensor_by_vector(float_or_vector, tensor)
        return tensor
    elif isinstance(float_or_vector, float):
        tensor = clamp(tensor, -float_or_vector, float_or_vector)
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor


def linf_clamp_(dx, x, eps, clip_min, clip_max):
    """Clamps perturbation `dx` to fit L_inf norm and image bounds.
    Limit the L_inf norm of `dx` to be <= `eps`, and the bounds of `x + dx`
    to be in `[clip_min, clip_max]`.
    :param dx: perturbation to be clamped (inplace).
    :param x: the image.
    :param eps: maximum possible L_inf.
    :param clip_min: upper bound of image values.
    :param clip_max: lower bound of image values.
    :return: the clamped perturbation `dx`.
    """
    dx_clamped = batch_clamp(eps, dx)
    x_adv = clamp(x + dx_clamped, clip_min, clip_max)
    # `dx` is changed *inplace* so the optimizer will keep
    # tracking it. the simplest mechanism for inplace was
    # adding the difference between the new value `x_adv - x`
    # and the old value `dx`.
    dx += x_adv - x - dx
    return dx


def _get_batch_sizes(n, max_batch_size):
    batches = [max_batch_size for _ in range(n // max_batch_size)]
    if n % max_batch_size > 0:
        batches.append(n % max_batch_size)
    return batches


@torch.no_grad()
def spsa_grad(predict, loss_fn, x, y, delta, nb_sample, max_batch_size):
    """Uses SPSA method to apprixmate gradient w.r.t `x`.
    Use the SPSA method to approximate the gradient of `loss_fn(predict(x), y)`
    with respect to `x`, based on the nonce `v`.
    :param predict: predict function (single argument: input).
    :param loss_fn: loss function (dual arguments: output, target).
    :param x: input argument for function `predict`.
    :param y: target argument for function `loss_fn`.
    :param v: perturbations of `x`.
    :param delta: scaling parameter of SPSA.
    :param reduction: how to reduce the gradients of the different samples.
    :return: return the approximated gradient of `loss_fn(predict(x), y)`
             with respect to `x`.
    """
    with torch.no_grad():

        predict = predict.to("cuda")
        grad = torch.zeros_like(x)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        def f(xvar, yvar):
            return loss_fn(predict(xvar), yvar)

        x = x.expand(max_batch_size, *x.shape[1:]).contiguous()
        y = y.expand(max_batch_size, *y.shape[1:]).contiguous()
        v = torch.empty_like(x[:, :1, ...])

        for batch_size in _get_batch_sizes(nb_sample, max_batch_size):
            x_ = x[:batch_size]
            y_ = y[:batch_size]
            vb = v[:batch_size]
            vb = vb.bernoulli_().mul_(2.0).sub_(1.0)
            v_ = vb.expand_as(x_).contiguous()
            x_shape = x_.shape
            x_ = x_.view(-1, *x.shape[2:])
            y_ = y_.view(-1, *y.shape[2:])
            v_ = v_.view(-1, *v.shape[2:])
            df = f(x_ + delta * v_, y_) - f(x_ - delta * v_, y_)
            df = df.view(-1, *[1 for _ in v_.shape[1:]])
            grad_ = df / (2.0 * delta * v_)
            grad_ = grad_.view(x_shape)
            grad_ = grad_.sum(dim=0, keepdim=False)
            grad += grad_
        grad /= nb_sample

    return grad


def spsa_perturb(
    predict, loss_fn, x, y, eps, delta, lr, nb_iter, nb_sample, max_batch_size, clip_min=-1.0, clip_max=1.0
):
    """Perturbs the input `x` based on SPSA attack.
    :param predict: predict function (single argument: input).
    :param loss_fn: loss function (dual arguments: output, target).
    :param x: input argument for function `predict`.
    :param y: target argument for function `loss_fn`.
    :param eps: the L_inf budget of the attack.
    :param delta: scaling parameter of SPSA.
    :param lr: the learning rate of the `Adam` optimizer.
    :param nb_iter: number of iterations of the attack.
    :param nb_sample: number of samples for the SPSA gradient approximation.
    :param max_batch_size: maximum batch size to be evaluated at once.
    :param clip_min: upper bound of image values.
    :param clip_max: lower bound of image values.
    :return: the perturbated input.
    """

    dx = torch.zeros_like(x)

    dx.grad = torch.zeros_like(dx)
    dx.require_grad = True
    optimizer = torch.optim.Adam([dx], lr=lr)
    for _ in range(nb_iter):
        optimizer.zero_grad()
        dx.grad = spsa_grad(predict, loss_fn, x + dx, y, delta, nb_sample, max_batch_size)
        optimizer.step()

        dx = linf_clamp_(dx, x, eps, clip_min, clip_max)

    x_adv = x + dx

    return x_adv


def spsa_perturb_half(
    predict, loss_fn, x, y, eps, delta, lr, nb_iter, nb_sample, max_batch_size, clip_min=0.0, clip_max=1.0
):
    """Perturbs the input `x` based on SPSA attack.
    :param predict: predict function (single argument: input).
    :param loss_fn: loss function (dual arguments: output, target).
    :param x: input argument for function `predict`.
    :param y: target argument for function `loss_fn`.
    :param eps: the L_inf budget of the attack.
    :param delta: scaling parameter of SPSA.
    :param lr: the learning rate of the `Adam` optimizer.
    :param nb_iter: number of iterations of the attack.
    :param nb_sample: number of samples for the SPSA gradient approximation.
    :param max_batch_size: maximum batch size to be evaluated at once.
    :param clip_min: upper bound of image values.
    :param clip_max: lower bound of image values.
    :return: the perturbated input.
    """

    dx = torch.zeros_like(x[:, :, :, int(x.shape[3] / 2) :])
    dx.grad = torch.zeros_like(dx)
    optimizer = torch.optim.Adam([dx], lr=lr)
    for _ in range(nb_iter):
        optimizer.zero_grad()
        dx_full_size = torch.cat([torch.zeros_like(x[:, :, :, : int(x.shape[3] / 2)]), dx], axis=-1)
        dx.grad = spsa_grad(predict, loss_fn, x + dx_full_size, y, delta, nb_sample, max_batch_size)[
            :, :, :, int(x.shape[3] / 2) :
        ]
        optimizer.step()
        dx_full_size = torch.cat([torch.zeros_like(x[:, :, :, : int(x.shape[3] / 2)]), dx], axis=-1)
        dx = linf_clamp_(dx_full_size, x, eps, clip_min, clip_max)[:, :, :, int(x.shape[3] / 2) :]

    dx_full_size = torch.cat([torch.zeros_like(x[:, :, :, : int(x.shape[3] / 2)]), dx], axis=-1)
    x_adv = x + dx_full_size

    return x_adv
