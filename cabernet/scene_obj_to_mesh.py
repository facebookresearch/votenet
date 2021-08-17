import argparse
import trimesh
import trimesh_utils as utils

from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    opts = parser.parse_args()
    opts.path = opts.path.expanduser()

    try:
        points, _ = utils.load_mesh(opts.path)
    except RuntimeError:
        meshes = utils.load_scene(opts.path)
        for i, mesh in enumerate(meshes):
            obj_path = opts.path.parent.joinpath(opts.path.stem + f"_{i}").with_suffix(".obj")
            obj_str = trimesh.exchange.obj.export_obj(mesh)
            with open(obj_path, "w") as f:
                f.write(obj_str)
