import torch
from torch.nn import functional
from common.core import losses


def compare_losses(x: torch.Tensor, y: torch.Tensor, mine, theirs):
    assert torch.isclose(mine(x, y), theirs(x, y))


def test_nll():
    k = 5
    batch = 3
    x = functional.log_softmax(torch.randn(batch, k), dim=1)
    y = torch.randint(0, k, (batch, 1))
    mine = losses.nll_loss(x, y)
    theirs = functional.nll_loss(x, y.view(batch))
    assert torch.isclose(mine, theirs)


def test_kl_loss():
    k = 5
    batch = 3
    x = functional.log_softmax(torch.randn(batch, k), dim=1)
    y = functional.softmax(torch.randn(batch, k))
    mine = losses.kl_loss(x, y)
    theirs =  functional.kl_div(x, y)
    assert torch.isclose(mine, theirs)


def test_mse_loss():
    k = 5
    batch = 3
    x = torch.randn(batch, k)
    y = torch.randn(batch, k)
    compare_losses(x, y, losses.mse_loss, functional.mse_loss)


def test_bce_loss():
    k = 5
    batch = 3
    x = torch.softmax(torch.randn(batch, k), dim=1)
    y = torch.empty((batch, k)).random_(2)
    compare_losses(x, y, losses.bce_loss, functional.binary_cross_entropy)