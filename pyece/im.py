import numpy as np


def linspace(start, end, count):
    # NOTE: np.linspace works ~ 8 times faster
    start = np.asarray(start)
    end = np.asarray(end)
    C = (end - start) / (count)
    return start + np.asarray([C * (k + 0.5) for k in range(count)])


# TODO: do it asynchronously
def meshcorners(corners, grid):
    c = len(corners) // 2
    if c == 0:
        return corners[0]

    return linspace(
        meshcorners(corners[:c], grid[1:]), meshcorners(corners[c:], grid[1:]), grid[0]
    )


def cutpatch(data: np.ndarray, corners: np.ndarray, grid: tuple, fill=None):
    mesh = np.asarray(meshcorners(corners, grid)).round()
    idx = np.rollaxis(mesh.astype(int), -1)
    d = idx.shape[0]
    ring_idx = idx % np.reshape(data.shape[:d], (2, *[1] * d))
    patch = data[tuple(ring_idx)]
    if fill is not None:
        fill_idx = (idx != ring_idx).any(0)
        patch[fill_idx] = fill
    return patch
