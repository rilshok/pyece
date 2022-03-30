from itertools import product
from typing import Tuple

import numpy as np

from .primitive import PointCloud


class Corners(PointCloud):
    def get(self) -> np.ndarray:
        corners = super().get()
        n, p = corners.shape
        if 2 ** p != n:
            msg = ""
            raise ValueError(msg)
        return corners

    @staticmethod
    def product(shape: Tuple) -> "Corners":
        masks = list(product((False, True), repeat=len(shape)))
        corners = [[s if c else 0 for c, s in zip(mask, shape)] for mask in masks]
        return Corners(corners)
