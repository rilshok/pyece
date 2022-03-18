from abc import abstractmethod

from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence, Union
import numpy as np

LikeProperty = Union["Property", float, int, str]
PropertySequence = Sequence[LikeProperty]


class Property(ABC):
    @abstractmethod
    def get(self) -> Any:
        return NotImplemented

    @property
    def value(self) -> Any:
        return self.get()


class ConstantProperty(Property):
    def __init__(self, value):
        self._value = value

    def get(self) -> Any:
        return self._value


def as_property(value: LikeProperty) -> Property:
    if isinstance(value, Property):
        return value
    return ConstantProperty(value)


class Convertible(Property):
    def __init__(self, p: LikeProperty, fn: Callable, *args, **kwargs):
        assert callable(fn)
        self._original = as_property(p)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def get(self) -> Any:
        return self._fn(self._original.value, *self._args, **self._kwargs)


class RandomUniform(Property):
    def __init__(self, low=0.0, high=1.0):
        self._low = low
        self._high = high

    def get(self):
        return np.random.uniform(self._low, self._high)


class Point(Property):
    def __init__(self, point: PropertySequence):
        self._point = [as_property(p) for p in point]

    def get(self) -> np.ndarray:
        return np.asarray([p.value for p in self._point])


class Vector:
    pass


class Angle:
    pass
