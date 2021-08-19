import argparse
import numpy as np
import trimesh
import trimesh_utils as utils

from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--save-meshes", action="store_true", default=False)
    parser.add_argument("--merge-and-save-pointcloud", action="store_true", default=False)
    parser.add_argument("--sample-meshes", type=int, default=None)
    opts = parser.parse_args()
    opts.path = opts.path.expanduser()

    meshes = utils.load_scene(opts.path)
    if opts.save_meshes:
        for i, mesh in enumerate(meshes):
            obj_path = opts.path.parent.joinpath(opts.path.stem + f"_{i}").with_suffix(".obj")
            obj_str = trimesh.exchange.obj.export_obj(mesh)
            with open(obj_path, "w") as f:
                f.write(obj_str)

    if opts.merge_and_save_pointcloud:
        vertices = []
        faces = []
        for i, mesh in enumerate(meshes):
            points = utils.sample_mesh(mesh, opts.sample_meshes)
            vertices.append(points)
        new_mesh = trimesh.PointCloud(vertices=np.concatenate(vertices, axis=0))
        obj_path = opts.path.parent.joinpath(opts.path.stem + f"_pc").with_suffix(".obj")
        obj_str = trimesh.exchange.obj.export_obj(new_mesh)
        with open(obj_path, "w") as f:
            f.write(obj_str)
