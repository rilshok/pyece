from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Sequence, Union

import numpy as np

from .math.rotate import rotate

LikeProperty = Union["Property", float, int, str]
PropertySequence = Sequence[LikeProperty]


class Property(ABC):
    @abstractmethod
    def get(self) -> Any:
        return NotImplemented

    @property
    def value(self) -> Any:
        return self.get()

    def transform(self, operation: "Operation", **kwargs) -> "Property":
        assert isinstance(operation, Operation)
        return operation(**kwargs)(self)


class ConstantProperty(Property):
    def __init__(self, value):
        self._value = value

    def get(self) -> Any:
        return self._value


def as_property(value: LikeProperty) -> Property:
    if isinstance(value, Property):
        return value
    return ConstantProperty(value)


class Source(Property):
    def __init__(self, fn: Callable, *args, **kwargs):
        assert callable(fn)
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def get(self) -> Any:
        return self._fn(*self._args, **self._kwargs)


class Convert(Source):
    def __init__(self, p: LikeProperty, fn: Callable, *args, **kwargs):
        self._original = as_property(p)
        super().__init__(fn, *args, **kwargs)

    def get(self) -> Any:
        return self._fn(self._original.value, *self._args, **self._kwargs)


class Iter(Property):
    def __init__(self, seq: Iterable):
        self._source = seq
        self._iter = iter(self._source)

    def _next(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self._source)
            return next(self._iter)

    def get(self) -> Any:
        return self._next()


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


LikePoint = Union[Point, PropertySequence]


def as_point(value: LikePoint) -> Point:
    if isinstance(value, Point):
        return value
    return Point(value)


class PointCloud(Point):
    def __init__(self, points: Sequence[LikePoint]):
        super().__init__([as_point(p) for p in points])

    def transform(self, operation: "Operation", **kwargs) -> "Point":
        if isinstance(operation, PointOperation):
            points = self.value
            if isinstance(operation, (PointRotate, PointInflation)):
                if operation._pivot is None:
                    kwargs["pivot"] = points.mean(0)
            fn = operation(**kwargs)
            return PointCloud(map(fn, points))
        return super().transform(operation, **kwargs)


LikePointCloud = Union[PointCloud, Sequence[LikePoint]]


def as_pointcloud(value: LikePointCloud) -> PointCloud:
    if isinstance(value, PointCloud):
        return value
    return PointCloud(value)


class Operation:
    def __call__(self, **params) -> Callable:
        def inner(obj):
            self.operation(obj, **params)

        return inner

    @abstractmethod
    def operation(self, obj, **params) -> Any:
        return NotImplemented


class PointOperation(Operation):
    def __call__(self, **params) -> Callable[[LikePoint], Point]:
        def inner(point: LikePoint):
            point = as_point(point).value
            value = self.operation(point, **params)
            return Point(value)

        return inner

    @abstractmethod
    def operation(self, obj: np.ndarray, **params) -> np.ndarray:
        return NotImplemented


class PointShift(PointOperation):
    def __init__(self, shift: LikePoint):
        self._shift = as_point(shift)

    def __call__(self) -> Callable[[LikePoint], Point]:
        shift = self._shift.value
        return super().__call__(shift=shift)

    def operation(self, obj: np.ndarray, shift: np.ndarray) -> np.ndarray:
        return obj + shift


class PointRotate(PointOperation):
    def __init__(self, angle: LikeProperty, pivot: LikePoint = None):
        self._angle = as_property(angle)
        self._pivot = None if pivot is None else as_point(pivot)

    def __call__(self, pivot: LikePoint = None) -> Callable[[LikePoint], Point]:
        angle = np.asarray(self._angle.value).reshape(-1) % (2 * np.pi)
        if pivot is not None:
            pivot = as_point(pivot).value
        else:
            pivot = self._pivot.value
        return super().__call__(pivot=pivot, angle=angle)

    def operation(
        self, obj: np.ndarray, pivot: np.ndarray, angle: np.ndarray
    ) -> np.ndarray:
        return rotate(pivot, obj, angle)


class PointInflation(PointOperation):
    def __init__(self, factor: LikeProperty, pivot: LikePoint = None):
        self._factor = as_property(factor)
        self._pivot = None if pivot is None else as_point(pivot)

    def __call__(self, pivot: LikePoint = None) -> Callable[[LikePoint], Point]:
        factor = self._factor.value
        if pivot is not None:
            pivot = as_point(pivot).value
        else:
            pivot = self._pivot.value
        return super().__call__(pivot=pivot, factor=factor)

    def operation(
        self, obj: np.ndarray, pivot: np.ndarray, factor: np.ndarray
    ) -> np.ndarray:
        return pivot + (obj - pivot) * factor


class Transformator:
    def __init__(self, *operations: Operation):
        assert all([isinstance(op, Operation) for op in operations])
        self._operations = operations

    def __call__(self, instance: Property) -> Property:
        for op in self._operations:
            instance = instance.transform(op)
        return instance
