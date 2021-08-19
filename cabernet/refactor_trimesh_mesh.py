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
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--rotate-xy-to-xz", action="store_true", default=False)

    parser.add_argument("--flip-x-coords", action="store_true", default=False)
    parser.add_argument("--flip-y-coords", action="store_true", default=False)
    parser.add_argument("--flip-z-coords", action="store_true", default=False)

    parser.add_argument("--floor-xy", action="store_true", default=False)
    parser.add_argument("--floor-xz", action="store_true", default=False)
    parser.add_argument("--floor-yz", action="store_true", default=False)

    opts = parser.parse_args()
    opts.path = opts.path.expanduser()

    mesh = trimesh.load_mesh(opts.path)

    if opts.rotate_xy_to_xz:
        R = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh = mesh.apply_transform(R)

    if opts.flip_x_coords:
        R = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        mesh = mesh.apply_transform(R)

    if opts.flip_y_coords:
        R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        mesh = mesh.apply_transform(R)

    if opts.flip_z_coords:
        R = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        mesh = mesh.apply_transform(R)

    if opts.floor_xy:
        T = tf.translation_matrix([0, 0, -mesh.vertices[:, 2].min()])
        mesh = mesh.apply_transform(T)

    if opts.floor_xz:
        T = tf.translation_matrix([0, -mesh.vertices[:, 1].min(), 0])
        mesh = mesh.apply_transform(T)

    if opts.floor_yz:
        T = tf.translation_matrix([-mesh.vertices[:, 0].min(), 0, 0])
        mesh = mesh.apply_transform(T)

    if opts.render:
        render_points(mesh.vertices, bbox=mesh.bounding_box.vertices)

    if opts.save:
        obj_path = opts.path.parent.joinpath(opts.path.stem + f"_refactored").with_suffix(".obj")
        obj_str = trimesh.exchange.obj.export_obj(mesh)
        with open(obj_path, "w") as f:
            f.write(obj_str)
