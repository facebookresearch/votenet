import argparse
import numpy as np
import trimesh
import trimesh.transformations as tf
import trimesh_utils as utils

from pathlib import Path
from render import render_points
from transforms import PLANAR_ROTATIONS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)

    opts = parser.parse_args()
    opts.path = opts.path.expanduser()

    mesh = trimesh.load_mesh(opts.path)

    R = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    mesh = mesh.apply_transform(R)

    R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh = mesh.apply_transform(R)

    T = tf.translation_matrix([0, -mesh.vertices[:, 1].min(), 0])
    mesh = mesh.apply_transform(T)

    # render_points(mesh.vertices, bbox=mesh.bounding_box.vertices)
    obj_path = opts.path.parent.joinpath(opts.path.stem + f"_n").with_suffix(".obj")
    obj_str = trimesh.exchange.obj.export_obj(mesh)
    with open(obj_path, "w") as f:
        f.write(obj_str)
