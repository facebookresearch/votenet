import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import trimesh.transformations as trans

from pathlib import Path
from typing import Dict, List, Tuple, Union


class Parser(object):
    class ShapesStr(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, "shapes_dict", {s.split(":")[0]: int(s.split(":")[1]) for s in values.split(",")})

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed", type=int, default=None)

        data = parser.add_argument_group("data")
        data.add_argument("--input-dataset-path", type=Path, required=True)
        data.add_argument("--output-path", type=Path, required=True)
        data.add_argument("--mode", type=str, default="train", choices=("train", "test"))
        data.add_argument("--shapes-str", type=str, action=Parser.ShapesStr, help="chair:2,table:1")
        data.add_argument("--num-scenes", type=int, default=10)

        placement = parser.add_argument_group("placement")
        placement.add_argument("--collision-tol", type=float, default=0.05)
        placement.add_argument("--step-size", type=float, default=1.0)
        placement.add_argument("--save-figure", action="store_true", default=False)

        scale = parser.add_argument_group("scale")
        scale.add_argument("--scale-min", type=float, default=1.0, help="FIXME: CURRENTLY NOT SUPPORTED")
        scale.add_argument("--scale-max", type=float, default=1.0, help="FIXME: CURRENTLY NOT SUPPORTED")

        rotate = parser.add_argument_group("rotate")
        rotate.add_argument("--rotate-x", type=float, default=None)
        rotate.add_argument("--rotate-y", type=float, default=None)
        rotate.add_argument("--rotate-z", type=float, default=None)

        translate = parser.add_argument_group("translate")
        translate.add_argument("--translate-x", type=float, default=None)
        translate.add_argument("--translate-y", type=float, default=None)
        translate.add_argument("--translate-z", type=float, default=0.0)

        opts = parser.parse_args()
        return opts


class PointcloudIntersect(object):

    BBOX_PLANES_LAYOUT = (
        ((0, 1, 2, 3), (4, 5, 6, 7)),
        ((0, 1, 4, 5), (2, 3, 6, 7)),
        ((0, 2, 4, 6), (1, 3, 5, 7)),
    )

    @staticmethod
    def plane_coefficients(coords: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the plane's coefficients for the given coordinates.
        The plane's eq. is: a x + b y + c z + d = 0
        :param coords: a numpy array of shape (m, 3) where m >= 3.
        :return: tuple with plane's coefficients and its bias.
        """
        v1 = coords[2] - coords[0]
        v2 = coords[1] - coords[0]
        cp = np.cross(v1, v2)
        a, b, c = cp
        d = np.dot(cp, coords[2])
        return np.asarray([a, b, c]), -d

    @staticmethod
    def above_or_below_plane(pcd: np.ndarray, plane_coords):
        """
        Compute whether each point in pointcloud is above/below a given plane.
        :param plane_coords: coordinates on the plane (array of shape (m, 3), where m > 3).
        :return: array of size (self.num_points,) with values in {-1, 1} indicating above/below plane.
        """
        coeff, bias = PointcloudIntersect.plane_coefficients(plane_coords)
        above_or_below = np.sign((pcd @ coeff + bias) / np.linalg.norm(coeff))
        return above_or_below

    @staticmethod
    def is_in_bbox(pcd: np.ndarray, bbox: np.ndarray, tolerance: float = 0.1):
        num_points = pcd.shape[0]

        # Assume all points in cube
        coords_in_cube = np.ones(num_points)
        # Iterate over 3 possible planes' pairs layouts
        for layout in PointcloudIntersect.BBOX_PLANES_LAYOUT:
            plane1_above_or_below = PointcloudIntersect.above_or_below_plane(pcd, bbox[layout[0], :])
            plane2_above_or_below = PointcloudIntersect.above_or_below_plane(pcd, bbox[layout[1], :])
            coords_in_cube *= (plane1_above_or_below * plane2_above_or_below < 0)

        num_in_cube = np.sum(coords_in_cube)
        return num_in_cube / num_points >= tolerance

    @staticmethod
    def is_in_bboxes(pcd: np.ndarray, bboxes: List[np.ndarray], tolerance: float = 0.1):
        for bbox in bboxes:
            if PointcloudIntersect.is_in_bbox(pcd, bbox, tolerance):
                return True
        return False

    @staticmethod
    def render(pcd: np.ndarray, bbox: np.ndarray = None):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], marker="o")
        if bbox is not None:
            ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], marker="^")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    @staticmethod
    def render_scene(pcd_list: List[np.ndarray], bboxes: List[np.ndarray], show: bool = True, save: Union[Path, str] = None):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        for pcd, bbox in zip(pcd_list, bboxes):
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], marker="o")
            if bbox is not None:
                ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], marker="^")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if save is not None:
            plt.savefig(save)
        if show:
            plt.show()
        plt.close()


def set_seed(seed: int = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def get_random_files_list(dataset_path: Path, shapes_dict: Dict[str, int], mode: str):
    objects_list = []
    for shape, quantity in shapes_dict.items():
        path = dataset_path.joinpath(shape).joinpath(mode)
        files = [f for f in path.iterdir()]
        sampled_files = random.sample(files, quantity)
        objects_list.extend(sampled_files)
    return objects_list


def get_path_data(path: Path) -> Path:
    with open(path, "r") as f:
        data = json.load(f)

    pcd = np.asarray(data["vertices"])
    bbox = np.asarray(data["oriented_bbox"])
    return pcd, bbox, data["centroid"]


def get_rotation_matrix(opts: argparse.Namespace):
    rand_rotation = 2 * np.pi * np.random.rand(3)
    alpha, beta, gamma = [x if y is None else 0.0 for x, y in
                          zip(rand_rotation, [opts.rotate_x, opts.rotate_y, opts.rotate_z])]

    origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    Rx = trans.rotation_matrix(alpha, xaxis)
    Ry = trans.rotation_matrix(beta, yaxis)
    Rz = trans.rotation_matrix(gamma, zaxis)
    R = trans.concatenate_matrices(Rx, Ry, Rz)
    return R


def get_translation_matrix(opts: argparse.Namespace):
    rand_rotation = np.random.rand(3)
    alpha, beta, gamma = [x if y is None else 0.0 for x, y in
                          zip(rand_rotation, [opts.translate_x, opts.translate_y, opts.translate_z])]

    vector = np.asarray([alpha, beta, gamma])
    unit_vector = vector / np.linalg.norm(vector)

    T = trans.translation_matrix(unit_vector)
    return T


if __name__ == "__main__":

    opts = Parser.parse()
    set_seed(opts.seed)
    opts.output_path.mkdir(parents=True, exist_ok=True)

    for i in range(opts.num_scenes):

        objects_paths = get_random_files_list(
            dataset_path=opts.input_dataset_path, shapes_dict=opts.shapes_dict, mode=opts.mode
        )

        pointclouds, bboxes = [], []
        scene_data = []
        for object_path in objects_paths:

            # Load Pointcloud
            pcd, bbox, _ = get_path_data(object_path)

            # Rotate Pointcloud
            R = get_rotation_matrix(opts)[0:3, 0:3]
            pcd = np.matmul(R, pcd.T).T
            bbox = np.matmul(R, bbox.T).T

            # Get the direction for placing object
            T = get_translation_matrix(opts)
            translate_v = opts.step_size * T[0:3, -1]
            translate_v = translate_v[None, ...]

            # Try to place the object in direction
            while PointcloudIntersect.is_in_bboxes(pcd, bboxes, tolerance=opts.collision_tol):
                pcd += translate_v
                bbox += translate_v

            pointclouds.append(pcd)
            bboxes.append(bbox)
            scene_data.append({
                "object": str(object_path),
                "pointcloud": pcd.tolist(),
                "oriented_bbox": bbox.tolist()
            })

        data_path = opts.output_path.joinpath(str(i).zfill(5))
        pdf_path = data_path.with_suffix(".pdf") if opts.save_figure else None
        PointcloudIntersect.render_scene(pointclouds, bboxes, show=False, save=pdf_path)
        with open(data_path.with_suffix(".json"), "w") as f:
            json.dump(scene_data, f, indent=4)
