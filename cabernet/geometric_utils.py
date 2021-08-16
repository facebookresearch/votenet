import numpy as np

from typing import Tuple


BASE_NORMALS = {
    "xy+": np.array([0, 0, 1]),
    "xy-": np.array([0, 0, -1]),
    "xz+": np.array([0, 1, 0]),
    "xz-": np.array([0, -1, 0]),
    "yz+": np.array([1, 0, 0]),
    "yz-": np.array([-1, 0, 0]),
}


def get_plane_params(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Return the plane's normal and bias:
    ax + by + cz + d = 0 ===>
    dot(normal, V) + bias = 0
    :param points:
    :return:
    """
    if points.shape[0] < 3 or points.shape[1] != 3:
        raise RuntimeError

    v1 = points[2] - points[0]
    v2 = points[1] - points[0]
    normal = np.cross(v1, v2)
    bias = -1.0 * np.dot(normal, points[0])

    return normal, bias
