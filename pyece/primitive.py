
from abc import abstractmethod

from abc import ABC, abstractmethod
from typing import Any, Sequence, Union
import numpy as np


LikeProperty = Union["Property", float, int]
LikePoint = Sequence[LikeProperty]


class Property(ABC):
    @abstractmethod
    def get(self)->Any:
        return NotImplemented

    @staticmethod
    def value(instance: LikeProperty):
        if isinstance(instance, Property):
            return instance.get()
        return instance


class Point(Property):
    def __init__(self, point: LikePoint):
        self._point = list(point)

    def get(self) -> np.ndarray:
        return np.asanyarray([Property.value(p) for p in self._point])


class Vector:
    pass

class Angle:
    pass
