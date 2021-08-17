import argparse
import numpy as np
import trimesh.transformations as tf

from typing import List, Optional


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
            [[np.cos(alpha), 0, np.sin(alpha)], [0, 1, 0], [-1.0 * np.sin(alpha), 0, np.cos(alpha)],], dtype=np.float64
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
