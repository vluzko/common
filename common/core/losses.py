"""Common loss functions"""
import torch


def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood loss
    Assumes output is in log probability space
    """
    return torch.mean(-torch.take_along_dim(output, target, dim=1))


def kl_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL divergence loss
    Assumes output is in log probability space.

    Arguments:
        output: The output of a neural network in log probability space
        target: The target distribution in probability space
    """
    return torch.mean(target * (torch.log(target) - output))


def bce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy loss"""
    raise NotImplementedError


def bce_loss_logits(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss

    Arguments:
        output: any tensor
        target: any tensor (shapes must match)
    """
    return torch.mean((target-output)**2)


def poisson_nll(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def huber_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def hinge_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError