import torch
from torch import nn
from common.core import layers


def compare_layer_classes(mine, theirs):
    compare_layers(mine(), theirs())


def compare_layers(mine: nn.Module, theirs: nn.Module):
    x = torch.randn(5, 2)
    y1 = mine(x)
    y2 = theirs(x)
    assert torch.isclose(y1, y2).all()


def test_layer_norm():
    compare_layers(layers.MyLayerNorm(), nn.LayerNorm((2), elementwise_affine=False))


def test_batch_norm():

    compare_layers(layers.MyBatchNorm(), nn.BatchNorm1d(2, track_running_stats=False, affine=False))