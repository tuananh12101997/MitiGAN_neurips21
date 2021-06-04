"""The SPSA attack."""
import numpy as np
import torch
from torch import optim
# from cleverhans.torch.utils import clip_eta


def spsa(
    model_fn,
    x,
    eps,
    nb_iter,
    norm=2,
    clip_min=-np.inf,
    clip_max=np.inf,
    y=None,
    targeted=False,
    early_stop_loss_threshold=None,
    learning_rate=0.01,
    delta=0.0001,
    spsa_samples=16,
    spsa_iters=1,
    is_debug=False,
    sanity_checks=True,
):
    """
    This implements the SPSA adversary, as in https://arxiv.org/abs/1802.05666
    (Uesato et al. 2018). SPSA is a gradient-free optimization method, which is useful when
    the model is non-differentiable, or more generally, the gradients do not point in useful
    directions.
    :param model_fn: A callable that takes an input tensor and returns the model logits.
    :param x: Input tensor.
    :param eps: The size of the maximum perturbation, measured in the L-infinity norm.
    :param nb_iter: The number of optimization steps.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param clip_min: If specified, the minimum input value.
    :param clip_max: If specified, the maximum input value.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the
              default, will try to make the label incorrect. Targeted will instead try to
              move in the direction of being more like y.
    :param early_stop_loss_threshold: A float or None. If specified, the attack will end as
              soon as the loss is below `early_stop_loss_threshold`.
    :param learning_rate: Learning rate of ADAM optimizer.
    :param delta: Perturbation size used for SPSA approximation.
    :param spsa_samples:  Number of inputs to evaluate at a single time. The true batch size
              (the number of evaluated inputs for each update) is `spsa_samples *
              spsa_iters`
    :param spsa_iters:  Number of model evaluations before performing an update, where each
              evaluation is on `spsa_samples` different inputs.
    :param is_debug: If True, print the adversarial loss after each update.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    # print('vao cleaverhans-----------')
    if y is not None and len(x) != len(y):
        raise ValueError(
            "number of inputs {} is different from number of labels {}".format(
                len(x), len(y)
            )
        )
    if y is None:
        y = torch.argmax(model_fn(x), dim=1)
    # import pdb; pdb.set_trace()
    # The rest of the function doesn't support batches of size greater than 1,
    # so if the batch is bigger we split it up.
    if len(x) != 1:
        adv_x = []
        for x_single, y_single in zip(x, y):
            adv_x_single = spsa(
                model_fn=model_fn,
                x=x_single.unsqueeze(0),
                eps=eps,
                nb_iter=nb_iter,
                norm=norm,
                clip_min=clip_min,
                clip_max=clip_max,
                y=y_single.unsqueeze(0),
                targeted=targeted,
                early_stop_loss_threshold=early_stop_loss_threshold,
                learning_rate=learning_rate,
                delta=delta,
                spsa_samples=spsa_samples,
                spsa_iters=spsa_iters,
                is_debug=is_debug,
                sanity_checks=sanity_checks,
            )
            adv_x.append(adv_x_single)
        return torch.cat(adv_x)

    if eps < 0:
        raise ValueError(
            "eps must be greater than or equal to 0, got {} instead".format(eps)
        )
    if eps == 0:
        return x

    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    asserts = []

    # If a data range was specified, check that the input was in that range
    asserts.append(torch.all(x >= clip_min))
    asserts.append(torch.all(x <= clip_max))

    if is_debug:
        print("Starting SPSA attack with eps = {}".format(eps))

    perturbation = (torch.rand_like(x) * 2 - 1) * eps
    _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)
    optimizer = optim.Adam([perturbation], lr=learning_rate)

    for i in range(nb_iter):
        def loss_fn(pert):
            """
            Margin logit loss, with correct sign for targeted vs untargeted loss.
            """
            logits = model_fn(x + pert)
            loss_multiplier = 1 if targeted else -1
            # loss_multiplier = -1
            return loss_multiplier * _margin_logit_loss(logits, y.expand(len(pert)))
        # import pdb; pdb.set_trace()
        spsa_grad = _compute_spsa_gradient(
            loss_fn, x, delta=delta, samples=spsa_samples, iters=spsa_iters
        )
        perturbation.grad = spsa_grad
        optimizer.step()

        _project_perturbation(perturbation, norm, eps, x, clip_min, clip_max)

        loss = loss_fn(perturbation).item()
        if is_debug:
            print("Iteration {}: loss = {}".format(i, loss))
            print('per', torch.mean(perturbation))
        if early_stop_loss_threshold is not None and loss < early_stop_loss_threshold:
            break
    # import pdb; pdb.set_trace()
    adv_x = torch.clamp((x + perturbation).detach(), clip_min, clip_max)

    if norm == np.inf:
        asserts.append(torch.all(torch.abs(adv_x - x) <= eps + 1e-6))
    else:
        asserts.append(
            torch.all(
                torch.abs(
                    torch.renorm(adv_x - x, p=norm, dim=0, maxnorm=eps) - (adv_x - x)
                )
                < 1e-6
            )
        )
    asserts.append(torch.all(adv_x >= clip_min))
    asserts.append(torch.all(adv_x <= clip_max))

    if sanity_checks:
        assert np.all(asserts)

    return adv_x


def _project_perturbation(
    perturbation, norm, epsilon, input_image, clip_min=-np.inf, clip_max=np.inf
):
    """
    Project `perturbation` onto L-infinity ball of radius `epsilon`. Also project into
    hypercube such that the resulting adversarial example is between clip_min and clip_max,
    if applicable. This is an in-place operation.
    """

    clipped_perturbation = clip_eta(perturbation, norm, epsilon)
    new_image = torch.clamp(input_image + clipped_perturbation, clip_min, clip_max)

    perturbation.add_((new_image - input_image) - perturbation)


def _compute_spsa_gradient(loss_fn, x, delta, samples, iters):
    """
    Approximately compute the gradient of `loss_fn` at `x` using SPSA with the
    given parameters. The gradient is approximated by evaluating `iters` batches
    of `samples` size each.
    """

    assert len(x) == 1
    num_dims = len(x.size())

    x_batch = x.expand(samples, *([-1] * (num_dims - 1)))

    grad_list = []
    for i in range(iters):
        # import pdb; pdb.set_trace()
        delta_x = delta * torch.sign(torch.rand_like(x_batch) - 0.5)
        delta_x = torch.cat([delta_x, -delta_x])
        with torch.no_grad():
            loss_vals = loss_fn(x + delta_x)
        while len(loss_vals.size()) < num_dims:
            loss_vals = loss_vals.unsqueeze(-1)
        avg_grad = (
            torch.mean(loss_vals * torch.sign(delta_x), dim=0, keepdim=True) / delta
        )
        # avg_grad = reduce_mean(loss_vals * tf.sign(delta_x), axis=0) / deltav
        # avg_grad = torch.mean(loss_vals /delta_x , dim=0, keepdim=True)
        grad_list.append(avg_grad)

    return torch.mean(torch.cat(grad_list), dim=0, keepdim=True)


def _margin_logit_loss(logits, labels):
    """
    Computes difference between logits for `labels` and next highest logits.
    The loss is high when `label` is unlikely (targeted by default).
    """
    # import pdb; pdb.set_trace()
    # logits = logits[0]
    correct_logits = logits.gather(1, labels[:, None]).squeeze(1)

    logit_indices = torch.arange(
        logits.size()[1],
        dtype=labels.dtype,
        device=labels.device,
    )[None, :].expand(labels.size()[0], -1)
    incorrect_logits = torch.where(
        logit_indices == labels[:, None],
        torch.full_like(logits, float("-inf")),
        logits,
    )
    max_incorrect_logits, _ = torch.max(incorrect_logits, 1)

    return max_incorrect_logits - correct_logits

def clip_eta(eta, norm, eps):
    """
    PyTorch implementation of the clip_eta in utils_tf.
    :param eta: Tensor
    :param norm: np.inf, 1, or 2
    :param eps: float
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")

    avoid_zero_div = torch.tensor(1e-12, dtype=eta.dtype, device=eta.device)
    reduc_ind = list(range(1, len(eta.size())))
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    else:
        if norm == 1:
            raise NotImplementedError("L1 clip is not implemented.")
            norm = torch.max(
                avoid_zero_div, torch.sum(torch.abs(eta), dim=reduc_ind, keepdim=True)
            )
        elif norm == 2:
            norm = torch.sqrt(
                torch.max(
                    avoid_zero_div, torch.sum(eta ** 2, dim=reduc_ind, keepdim=True)
                )
            )
        factor = torch.min(
            torch.tensor(1.0, dtype=eta.dtype, device=eta.device), eps / norm
        )
        eta *= factor
    return eta

# def spsa(
#     model_fn,
#     x,
#     eps,
#     nb_iter,
#     norm=2,
#     clip_min=-np.inf,
#     clip_max=np.inf,
#     y=None,
#     targeted=False,
#     early_stop_loss_threshold=None,
#     learning_rate=0.01,
#     delta=0.001,
#     spsa_samples=16,
#     spsa_iters=1,
#     is_debug=False,
#     sanity_checks=True,
# ):