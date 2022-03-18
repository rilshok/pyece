from itertools import product
from typing import Sequence, Tuple, Union

import numpy as np

Point = Sequence[Union[float, int]]
LikeCorners = Union[np.ndarray, Sequence[Point]]

class Corners:
    def __init__(self, corners: LikeCorners):
        corners = np.asarray(corners)
        n, p = corners.shape
        if 2**p != n:
            msg = ""
            raise ValueError(msg)
        self._corners = np.asarray(corners)

    def copy(self) -> "Corners":
        return Corners(self._corners)

    def dim(self):
        return self._corners.shape[1]

    @staticmethod
    def product(shape: Tuple) -> "Corners":
        masks = list(product((False, True), repeat=len(shape)))
        corners = [[s if c else 0 for c, s in zip(mask, shape)] for mask in masks]
        return Corners(corners)

    @property
    def сentre(self):
        return self._corners.mean(0)

    def __getitem__(self, index):
        pass
    # def __repr__(self):
    #     return
