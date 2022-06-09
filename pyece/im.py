import itertools

import h5py as h5  # type: ignore
import numpy as np

from .core import typing as tp
from .core.property import Corners


def linspace(start, end, count: int):
    # NOTE: np.linspace works ~ 8 times faster
    start = np.asarray(start)
    end = np.asarray(end)
    C = (end - start) / (count)
    return start + np.asarray([C * (k + 0.5) for k in range(count)])


# TODO: do it asynchronously
def meshcorners(corners: tp.NDArray, grid: tp.IntTuple):
    c = len(corners) // 2
    if c == 0:
        return corners[0]
    return linspace(
        start=meshcorners(corners[:c], grid[1:]),
        end=meshcorners(corners[c:], grid[1:]),
        count=grid[0],
    )


def cutpatch(
    data: tp.Union[tp.NDArray, h5.Dataset],
    corners: tp.NDArray,
    grid: tp.IntTuple,
    fill: tp.Any = None,
) -> np.ndarray:
    mesh = np.asarray(meshcorners(corners, grid)).round()
    idx = np.rollaxis(mesh.astype(int), -1)
    d = idx.shape[0]
    ring_idx = idx % np.reshape(data.shape[:d], (d, *[1] * d))
    if isinstance(data, h5.Dataset):
        data = data[:]
    patch = data[tuple(ring_idx)]
    if fill is not None:
        fill_idx = (idx != ring_idx).any(0)
        patch[fill_idx] = fill
    return patch


class ByPatchWrapper:
    def __init__(
        self,
        func: tp.Callable[[tp.NDArray], tp.NDArray],
        size: tp.IntTuple,
        spacing: tp.FloatTuple = None,
        grid: tp.IntTuple = None,
    ):
        assert callable(func)
        self._wrapper = func

        self._dim = len(size)
        self._size = np.asarray(size)

        self._spacing = np.asarray(spacing or [1.0] * self._dim)
        assert len(self._spacing) == self._dim

        self._volume = self._size * self._spacing

        assert (grid is None) or (len(grid) == self._dim)
        self._grid = None
        if grid is not None:
            assert len(grid) == self._dim
            self._grid = np.asarray(grid).astype(int)

    def __call__(
        self,
        data: tp.NDArray,
        spacing: tp.FloatTuple = None,
        grid: tp.IntTuple = None,
        **kwargs,
    ) -> np.ndarray:
        shape = np.asarray(data.shape[: self._dim])
        spacing = np.asarray(spacing or [1.0] * self._dim)
        volume = shape * spacing
        # NOTE: resize data to required spacing
        if not np.all(self._spacing == spacing):
            data = cutpatch(
                data=data,
                corners=Corners.product(shape).value - 0.5,
                grid=np.round(volume / self._spacing).astype(int),
            )
        grid = np.asarray(
            grid or self._grid or np.floor((volume / self._volume)),
            dtype="int",
        )

        ids = np.asarray(list(itertools.product(*map(range, grid))))
        shift = self._volume + ((volume - (grid * self._volume)) / (grid - 1))
        shifts = np.round((ids * shift)).astype(int)

        wrapped = list()
        for shift in shifts:
            idx = tuple(slice(s, e, 1) for s, e in zip(shift, shift + self._size))
            patch = data[idx]
            wrapped_patch = self._wrapper(patch, **kwargs)
            wrapped.append(wrapped_patch)

        result = np.asarray(wrapped)
        tail_shape = result.shape[1:]
        return result.reshape((*grid, *tail_shape))
