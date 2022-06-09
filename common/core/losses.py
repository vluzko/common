"""Common loss functions"""
import torch


def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(-torch.take_along_dim(output, target, dim=1))
    raise NotImplementedError


def kl_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def bce_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def bce_loss_logits(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def poisson_nll(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def huber_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def hinge_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError