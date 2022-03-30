from abc import ABC, abstractclassmethod, abstractmethod
from itertools import product
from typing import Sequence, Tuple, Union

import numpy as np

from .primitive import Property

Point = Sequence[Union[float, int]]
LikeCorners = Union[np.ndarray, Sequence[Point]]


class Corners(Property):
    def __init__(self, corners: LikeCorners):
        corners = np.asarray(corners)
        n, p = corners.shape
        if 2 ** p != n:
            msg = ""
            raise ValueError(msg)
        self._corners = np.asarray(corners)

    def get(self):
        return self._corners

    def copy(self) -> "Corners":
        return Corners(self._corners)

    @staticmethod
    def product(shape: Tuple) -> "Corners":
        masks = list(product((False, True), repeat=len(shape)))
        corners = [[s if c else 0 for c, s in zip(mask, shape)] for mask in masks]
        return Corners(corners)

    @property
    def dim(self) -> int:
        return self._corners.shape[1]

    @property
    def centre(self) -> np.ndarray:
        return self._corners.mean(0)

    def __getitem__(self, index):
        pass

    def __repr__(self) -> str:
        d = self.dim
        c = tuple(self.centre)
        return f"Corners{d}d[сentre={c}]"
