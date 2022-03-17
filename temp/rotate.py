import numpy as np
import typing as tp


def rotate(
        pivot_point: tp.Iterable[float],
        turning_point: tp.Iterable[float],
        angle: tp.Iterable[float]
) -> np.ndarray:
    pivot_point = np.asarray(pivot_point)
    turning_point = np.asarray(turning_point)
    assert pivot_point.shape == turning_point.shape
    rotate_matrix = get_rotate_matrix(pivot_point.shape[-1], angle)

    turning_point = turning_point - pivot_point
    turning_point = rotate_matrix @ turning_point
    turning_point += np.asarray(pivot_point)
    return turning_point


def get_rotate_matrix(dim: int, angle: tp.Iterable[float]) -> np.ndarray:
    if dim == 2:
        return get_rotate_matrix_2d(angle[0])
    elif dim == 3:
        return get_rotate_matrix_3d(angle)
    else:
        raise ValueError(f"ndim should be 2 or 3, not {dim}")


def get_rotate_matrix_2d(
        angle: float
):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])


def get_rotate_matrix_3d(
        angle: tp.Iterable[float]
):
    rotate_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle[0]), -np.sin(angle[0])],
        [0, np.sin(angle[0]), np.cos(angle[0])],
    ])
    rotate_matrix_y = np.array([
        [np.cos(angle[1]), 0, np.sin(angle[1])],
        [0, 1, 0],
        [-np.sin(angle[1]), 0, np.cos(angle[1])],
    ])
    rotate_matrix_z = np.array([
        [np.cos(angle[2]), -np.sin(angle[2]), 0],
        [np.sin(angle[2]), np.cos(angle[2]), 0],
        [0, 0, 1],
    ])
    return rotate_matrix_x @ rotate_matrix_y @ rotate_matrix_z
