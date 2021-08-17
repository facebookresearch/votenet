import argparse
import json
import numpy as np
import random

from geometric_utils import Measures, intersect_bboxes_aligned
from pathlib import Path
from render import Render
from transforms import CabernetTransforms
from trimesh_utils import load_mesh
from typing import Dict


INPUT_PATH = Path.cwd().joinpath("meshes")
OUTPUT_PATH = Path.cwd().joinpath("scenes")


class Parser(object):
    class ShapesStr(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "shapes_dict", {s.split(":")[0]: int(s.split(":")[1]) for s in values.split(",")})

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", type=int, default=None)

        data = parser.add_argument_group("data")
        data.add_argument("--shapes-str", type=str, action=Parser.ShapesStr, required=True, help="chair:2,table:1")
        data.add_argument("--sample-shapes", type=int, default=1024)
        data.add_argument("--num-scenes", type=int, default=2)

        placement = parser.add_argument_group("placement")
        placement.add_argument("--rotate-xz", type=float, default=None)
        placement.add_argument("--translate-x", type=float, default=0.0)
        placement.add_argument("--translate-z", type=float, default=0.0)
        placement.add_argument("--tolerance", type=float, default=0.001)
        placement.add_argument("--step-size", type=float, default=1.0)
        # placement.add_argument("--save-figure", action="store_true", default=False)

        opts = parser.parse_args()
        return opts


def set_seed(seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def get_random_files_list(dataset_path: Path, shapes_dict: Dict[str, int]):
    objects_list = []
    for shape, quantity in shapes_dict.items():
        path = dataset_path.joinpath(shape)
        files = [f for f in path.iterdir() if f.suffix == ".obj"]
        sampled_files = random.sample(files, quantity)
        objects_list.extend(sampled_files)
    return objects_list


def get_path_data(path: Path) -> Path:
    with open(path, "r") as f:
        data = json.load(f)

    pcd = np.asarray(data["vertices"])
    bbox = np.asarray(data["oriented_bbox"])
    return pcd, bbox, data["centroid"]


if __name__ == "__main__":

    opts = Parser.parse()
    set_seed(opts.seed)

    if not INPUT_PATH.exists():
        raise RuntimeError

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    transforms = CabernetTransforms(seed=opts.seed)

    for i in range(opts.num_scenes):

        objects_paths = get_random_files_list(dataset_path=INPUT_PATH, shapes_dict=opts.shapes_dict)

        pointclouds, bboxes = [], []
        scene_data = []
        for object_path in objects_paths:

            points, bbox = load_mesh(object_path, sample=opts.sample_shapes)

            alpha = transforms.rand_alpha() if opts.rotate_xz is None else opts.rotate_xz
            points, bbox = transforms.apply_rotation([points, bbox], alpha=alpha)

            # Get the translation vector
            t_vec = transforms.rand_unit_vector()
            # t_vec = np.asarray([200, 0, 0], dtype=np.float64)
            # opts.step_size = 1.0

            # Try to place the object in direction
            while True:
                is_intersect = intersect_bboxes_aligned(points, bboxes, tolerance=opts.tolerance)
                if not is_intersect:
                    break
                points, bbox = transforms.apply_translation([points, bbox], t_vec=t_vec, step_size=opts.step_size)

            pointclouds.append(points)
            bboxes.append(bbox)

            bbox_centeroid = np.mean(bbox, axis=0)
            bbox_m = Measures.from_pointcloud(bbox)
            votenet_bbox = np.concatenate(
                (bbox_centeroid, np.asarray([bbox_m.size_x, bbox_m.size_y, bbox_m.size_z, np.deg2rad(alpha).item()]))
            )
            scene_data.append({"object": str(object_path), "points": points.tolist(), "bbox": votenet_bbox.tolist()})
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        for p in pointclouds:
            ax.scatter(p[:, 0], p[:, 1], p[:, 2])
        plt.show()

        data_path = OUTPUT_PATH.joinpath(str(i).zfill(5))
        with open(data_path.with_suffix(".json"), "w") as f:
            json.dump(scene_data, f, indent=4)
