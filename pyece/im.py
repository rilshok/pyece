import typing as tp

import h5py as h5  # type: ignore
import numpy as np
from .core.property import Corners

import itertools


def linspace(start, end, count: int):
    # NOTE: np.linspace works ~ 8 times faster
    start = np.asarray(start)
    end = np.asarray(end)
    C = (end - start) / (count)
    return start + np.asarray([C * (k + 0.5) for k in range(count)])


# TODO: do it asynchronously
def meshcorners(corners: np.ndarray, grid: tp.Tuple):
    c = len(corners) // 2
    if c == 0:
        return corners[0]

    return linspace(
        meshcorners(corners[:c], grid[1:]), meshcorners(corners[c:], grid[1:]), grid[0]
    )


def cutpatch(
    data: tp.Union[np.ndarray, h5.Dataset],
    corners: np.ndarray,
    grid: tp.Tuple[int, ...],
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
        func: tp.Callable[[np.ndarray], tp.Any],
        size: tp.Union[np.ndarray, tp.Tuple[int, ...]],
        spacing: tp.Union[np.ndarray, tp.Tuple[float, ...]] = None,
        grid: tp.Union[np.ndarray, tp.Tuple[float, ...]] = None,
    ):
        assert callable(func)
        self._wrapper = func
        dim = len(size)
        size = np.asarray(size)
        spacing = np.asarray(spacing or [1.0] * dim)

        if grid is not None:
            grid = np.asarray(grid).astype(int)

        assert len(spacing) == dim
        assert (grid is None) or (len(grid) == dim)

        self._dim = dim
        self._size = volume = size
        if spacing is not None:
            volume = volume * spacing
        self._volume = volume
        self._corners = Corners.product(tuple(volume)).value

        self._grid = grid

    def __call__(
        self,
        data: np.ndarray,
        spacing: tp.Union[np.ndarray, tp.Tuple[float, ...]] = None,
        grid: tp.Union[np.ndarray, tp.Tuple[float, ...]] = None,
        **kwargs,
    ) -> np.ndarray:
        size = np.asarray(data.shape[: self._dim])
        if spacing is not None:
            size = size * np.asarray(spacing)

        grid = np.asarray(
            grid or self._grid or np.floor((size / self._volume)), dtype="int"
        )
        ids = np.asarray(list(itertools.product(*map(range, grid))))
        shift = self._volume + ((size - (grid * self._volume)) / (grid - 1))
        shifts = ids * shift

        wrapped = list()
        for shift in shifts:
            corners = self._corners + shift
            if spacing is not None:
                corners = corners / spacing
            corners = corners - 0.5
            patch = cutpatch(
                data=data,
                grid=tuple(self._size),
                corners=corners,
            ).astype(np.float32)

            wrapped_patch = self._wrapper(patch, **kwargs)
            wrapped.append(wrapped_patch)

        result = np.asarray(wrapped)
        tail_shape = result.shape[1:]
        return result.reshape((*grid, *tail_shape))
