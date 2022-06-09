__all__ = ("rotate",)

import numpy as np

from .. import typing as tp

Angle2D = tp.Tuple[float]
Angle3D = tp.Tuple[float, float, float]
Angle = tp.Union[Angle2D, Angle3D]


def rotate(
    pivot: tp.NDArray,
    turning: tp.NDArray,
    angle: Angle,
) -> np.ndarray:
    pivot = np.asarray(pivot)
    turning = np.asarray(turning)
    assert pivot.shape == turning.shape
    matrix = get_rotate_matrix(angle)
    return matrix @ (turning - pivot) + pivot


def get_rotate_matrix(angle: Angle) -> tp.NDArray:
    if len(angle) == 1:
        return rotate_matrix_2d(angle[0])
    elif len(angle) == 3:
        return rotate_matrix_3d(angle)
    else:
        raise ValueError(f"ndim should be 2 or 3")


def rotate_matrix_2d(angle: float) -> tp.NDArray:
    return np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )


def rotate_matrix_3d(angle: Angle3D) -> tp.NDArray:
    rotate_matrix_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle[0]), -np.sin(angle[0])],
            [0, np.sin(angle[0]), np.cos(angle[0])],
        ]
    )
    rotate_matrix_y = np.array(
        [
            [np.cos(angle[1]), 0.0, np.sin(angle[1])],
            [0.0, 1.0, 0.0],
            [-np.sin(angle[1]), 0.0, np.cos(angle[1])],
        ]
    )
    rotate_matrix_z = np.array(
        [
            [np.cos(angle[2]), -np.sin(angle[2]), 0.0],
            [np.sin(angle[2]), np.cos(angle[2]), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rotate_matrix_x @ rotate_matrix_y @ rotate_matrix_z
