from typing import Optional

import torch.optim


class AvgMeter:
    """

    :param name:
    """

    def __init__(self, name: str = "Metric") -> None:
        self.count: Optional[int] = None
        self.sum: Optional[int] = None
        self.avg: Optional[float] = None
        self.name = name
        self.reset()

    def reset(self) -> None:
        """

        """
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val: int, count: int = 1) -> None:
        """

        :param val:
        :param count:
        :return:
        """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """

    :param optimizer:
    :return:
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
