__all__ = (
    "Point",
    "PointOperation",
    "PointShift",
    "PointRotate",
    "PointInflation",
)

from abc import abstractmethod

import numpy as np

from .. import typing as tp
from ..math.rotate import Angle, rotate
from .base import LikeProperty, Operation, Property, PropertySequence, as_property

LikePoint = tp.Union["Point", PropertySequence, tp.NDArray]


class Point(Property):
    def __init__(self, point: PropertySequence):
        self._point = [as_property(p) for p in point]

    def get(self) -> np.ndarray:
        return np.asarray([p.value for p in self._point])


def as_point(value: LikePoint) -> Point:
    if isinstance(value, Point):
        return value
    return Point(value)


class PointOperation(Operation):
    def __call__(self, **params) -> tp.Callable[[LikePoint], Point]:
        def inner(point: LikePoint) -> Point:
            point_ = as_point(point).value
            value = self.operation(point_, **params)
            return Point(value)

        return inner

    @abstractmethod
    def operation(self, obj: tp.NDArray, **params) -> tp.NDArray:
        return NotImplemented


class PointShift(PointOperation):
    def __init__(self, shift: LikePoint):
        self._shift = as_point(shift)

    def __call__(self, **params) -> tp.Callable[[LikePoint], Point]:
        shift = self._shift.value
        return super().__call__(shift=shift)

    def operation(self, obj: tp.NDArray, **params) -> tp.NDArray:
        shift: tp.NDArray = params["shift"]
        return obj + shift


class PointRotate(PointOperation):
    def __init__(self, angle: LikeProperty, pivot: LikePoint = None):
        self._angle = as_property(angle)
        self._pivot = None if pivot is None else as_point(pivot)

    def __call__(self, **params) -> tp.Callable[[LikePoint], Point]:
        pivot: tp.Optional[LikePoint] = params["pivot"]
        angle = np.asarray(self._angle.value).reshape(-1) % (2 * np.pi)
        if pivot is not None:
            pivot = as_point(pivot).value
        elif self._pivot is not None:
            pivot = self._pivot.value
        else:
            raise RuntimeError
        return super().__call__(pivot=pivot, angle=angle)

    def operation(self, obj: tp.NDArray, **params) -> tp.NDArray:
        pivot: tp.NDArray = params["pivot"]
        angle: Angle = params["angle"]
        return rotate(pivot, obj, angle)


class PointInflation(PointOperation):
    def __init__(
        self,
        factor: LikeProperty,
        pivot: LikePoint = None,
    ):
        self._factor = as_property(factor)
        self._pivot = None if pivot is None else as_point(pivot)

    def __call__(self, **params) -> tp.Callable[[LikePoint], Point]:
        pivot: LikePoint = (params["pivot"],)
        factor = self._factor.value
        if pivot is not None:
            pivot = as_point(pivot).value
        elif self._pivot is not None:
            pivot = self._pivot.value
        else:
            raise RuntimeError
        return super().__call__(pivot=pivot, factor=factor)

    def operation(self, obj: tp.NDArray, **params) -> tp.NDArray:
        pivot: tp.NDArray = params["pivot"]
        factor: tp.NDArray = params["factor"]
        return pivot + (obj - pivot) * factor
