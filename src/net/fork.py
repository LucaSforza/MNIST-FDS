from typing import Callable
from torch import nn
from functorch.dim import Tensor


class Fork(nn.Module):
    def __init__(self, id_layer: str, n: int, residuals: dict[str, dict[int, Tensor]]):
        # n is the number of clones
        self.n = n
        self.id_layer
        self.resisuals = residuals
        pass

    def forward(self, input: Tensor):
        for i in range(self.n):
            self.residuals[self.id_layer] = input.clone()
        return input


class Join(nn.Module):
    def __init__(self, join: Callable):
        pass


class Level(nn.Module):
    def __init__(self, *args):
        self.modules: tuple[nn.Module] = args
        self.outputs = tuple()

    def forward(self, x: Tensor):
        x: Tensor = x.unsqueeze(1).repeat(1, len(self.modules), 1)
        for m in self.modules:
            m(x)
        pass
