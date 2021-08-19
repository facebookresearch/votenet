import argparse
import numpy as np
import trimesh_utils as utils

from pathlib import Path
from render import render_points
from typing import List, Optional


PLANAR_ROTATIONS = {
    "xy": lambda x: np.asarray(
        [[np.cos(x), np.sin(x), 0], [-1.0 * np.sin(x), np.cos(x), 0], [0, 0, 1]], dtype=np.float64
    ),
    "xz": lambda x: np.asarray(
        [[np.cos(x), 0, np.sin(x)], [0, 1, 0], [-1.0 * np.sin(x), 0, np.cos(x)]], dtype=np.float64
    ),
    "yz": lambda x: np.asarray(
        [[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -1.0 * np.sin(x), np.cos(x)]], dtype=np.float64
    ),
}


class CabernetTransforms(object):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random if seed is None else np.random.RandomState(seed)

    def rand_alpha(self):
        return 360 * self.rng.rand(1)

    def apply_rotation(self, arrays: List[np.ndarray], alpha: float = None) -> List[np.ndarray]:
        """
        Applies rotation over the arrays
        where each array undergoes the same rotation!

        :param arrays: List of arrays to rotate. arrays are arranged in shape (n_points,3).
        :param alpha: Degree to rotate on xz-plane. Randomized when None.
        :return: List of rotated arrays of the same shape as input arrays.
        """
        if alpha is None:
            alpha = self.rand_alpha()

        cartesian_rotation_transform = np.asarray(
            [[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-1.0 * np.sin(alpha), 0, np.cos(alpha)]], dtype=np.float64
        )

        rotated_data = []
        for array in arrays:
            rotated_data.append(np.matmul(cartesian_rotation_transform, array.T).T)

        return rotated_data

    def rand_unit_vector(self):
        v = self.rng.rand(3)
        v[1] = 0  # only in xz plane
        t_vec = v / np.linalg.norm(v)

        return t_vec

    def apply_translation(
        self, arrays: List[np.ndarray], t_vec: np.ndarray = None, step_size: Optional[float] = 1.0
    ) -> List[np.ndarray]:

        if t_vec is None:
            t_vec = self.rand_unit_vector()

        t_vec *= step_size
        translated_data = []
        for array in arrays:
            translated_data.append(array + t_vec)

        return translated_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--rotate-xy", type=float, default=0.0)
    parser.add_argument("--rotate-xz", type=float, default=0.0)
    parser.add_argument("--rotate-yz", type=float, default=0.0)
    parser.add_argument("--center-to-origin", action="store_true", default=False)
    parser.add_argument("--floor-xy", action="store_true", default=False)
    parser.add_argument("--floor-xz", action="store_true", default=False)
    parser.add_argument("--floor-yz", action="store_true", default=False)
    opts = parser.parse_args()
    opts.path = opts.path.expanduser()

    points, bbox = utils.load_mesh(opts.path, sample=opts.sample)

    if opts.rotate_xy != 0.0:
        rotation = PLANAR_ROTATIONS["xy"](opts.rotate_xy)
        points = np.matmul(rotation, points.T).T
        bbox = np.matmul(rotation, bbox.T).T

    if opts.rotate_xz != 0.0:
        rotation = PLANAR_ROTATIONS["xz"](opts.rotate_xz)
        points = np.matmul(rotation, points.T).T
        bbox = np.matmul(rotation, bbox.T).T

    if opts.rotate_yz != 0.0:
        rotation = PLANAR_ROTATIONS["yz"](opts.rotate_yz)
        points = np.matmul(rotation, points.T).T
        bbox = np.matmul(rotation, bbox.T).T

    if opts.center_to_origin:
        centroid = np.mean(points, axis=0).reshape((1, 3))
        points = points - centroid
        bbox = bbox - centroid

    if opts.floor_xy:
        centroid_z = np.mean(points[:, 2].min())
        points[:, 2] = points[:, 2] - centroid_z
        bbox[:, 2] = bbox[:, 2] - centroid_z

    if opts.floor_xz:
        centroid_y = np.mean(points[:, 1].min())
        points[:, 1] = points[:, 1] - centroid_y
        bbox[:, 1] = bbox[:, 1] - centroid_y

    if opts.floor_yz:
        centroid_x = np.mean(points[:, 0].min())
        points[:, 0] = points[:, 0] - centroid_x
        bbox[:, 0] = bbox[:, 0] - centroid_x

    render_points(points, bbox=bbox)
