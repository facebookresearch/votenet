import argparse
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from geometric_utils import BASE_NORMALS, Measures, get_plane_params
from pathlib import Path
from transforms import CabernetTransforms
from trimesh_utils import load_mesh
from typing import Dict, Optional, Tuple


class Render(object):
    def __init__(self, figsize: Optional[Tuple[float, float]] = (10, 10)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        self.ax = ax

    @staticmethod
    def _scalar_fn_to_colors(scalar_fn: np.ndarray = None):
        if scalar_fn is None:
            return None
        norm = clr.Normalize(vmin=scalar_fn.min(), vmax=scalar_fn.max())
        return plt.cm.coolwarm(norm(scalar_fn))

    def add_pcd(self, pcd: np.ndarray, marker: Optional[str] = "o", color: Optional[np.ndarray] = None):
        self.ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], marker=marker, color=self._scalar_fn_to_colors(color))

    def add_bbox(self, bbox: np.ndarray, marker: Optional[str] = "^"):
        self.ax.scatter(bbox[:, 0], bbox[:, 1], bbox[:, 2], marker=marker)

    def _which_plane(self, normal: np.ndarray) -> str:
        for k, v in BASE_NORMALS.items():
            if np.array_equal(normal, v):
                return k
        return "any"

    def add_xy_plane(
        self, normal: np.ndarray, bias: float, measures: Measures, surface_kwargs: Dict = dict(alpha=0.5),
    ):
        xx, yy = np.meshgrid(
            np.linspace(start=measures.min_x, stop=measures.max_x, num=100),
            np.linspace(start=measures.min_y, stop=measures.max_y, num=100),
        )
        self.ax.plot_surface(xx, yy, -1.0 * bias * np.ones_like(xx), **surface_kwargs)

    def add_xz_plane(
        self, normal: np.ndarray, bias: float, measures: Measures, surface_kwargs: Dict = dict(alpha=0.5),
    ):
        xx, zz = np.meshgrid(
            np.linspace(start=measures.min_x, stop=measures.max_x, num=100),
            np.linspace(start=measures.min_z, stop=measures.max_z, num=100),
        )
        self.ax.plot_surface(xx, -1.0 * bias * np.ones_like(xx), zz, **surface_kwargs)

    def add_yz_plane(
        self, normal: np.ndarray, bias: float, measures: Measures, surface_kwargs: Dict = dict(alpha=0.5),
    ):
        yy, zz = np.meshgrid(
            np.linspace(start=measures.min_y, stop=measures.max_y, num=100),
            np.linspace(start=measures.min_z, stop=measures.max_z, num=100),
        )
        self.ax.plot_surface(-1.0 * bias * np.ones_like(yy), yy, zz, **surface_kwargs)

    def add_any_plane(
        self, normal: np.ndarray, bias: float, measures: Measures, surface_kwargs: Dict = dict(alpha=0.5),
    ):
        a, b, c = normal
        xx, yy = np.meshgrid(
            np.linspace(start=measures.min_x, stop=measures.max_x, num=100),
            np.linspace(start=measures.min_y, stop=measures.max_y, num=100),
        )
        z = -1.0 * (a * xx + b * yy + bias) / c
        self.ax.plot_surface(xx, yy, z, **surface_kwargs)

    def add_plane(
        self, points: np.ndarray, measures: Measures = None, surface_kwargs: Dict = dict(alpha=0.5),
    ):
        if measures is None:
            measures = Measures()
            measures.min_x, measures.max_x = -10.0, 10.0
            measures.min_y, measures.max_y = -10.0, 10.0
            measures.min_z, measures.max_z = -10.0, 10.0

        normal, bias = get_plane_params(points)
        plane_attr = self._which_plane(normal)
        planes_fn = {
            "xy+": self.add_xy_plane,
            "xy-": self.add_xy_plane,
            "xz+": self.add_xz_plane,
            "xz-": self.add_xz_plane,
            "yz+": self.add_yz_plane,
            "yz-": self.add_yz_plane,
            "any": self.add_any_plane,
        }[plane_attr]
        planes_fn(normal=normal, bias=bias, measures=measures, surface_kwargs=surface_kwargs)

    def save(self, path: Path):
        plt.savefig(path)

    def render(self):
        plt.show()

    def close(self):
        self.__del__()

    def __del__(self):
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--bbox", action="store_true", default=False)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--rotate-xz", type=float, default=0.0)
    parser.add_argument("--translate-x", type=float, default=0.0)
    parser.add_argument("--translate-z", type=float, default=0.0)
    opts = parser.parse_args()
    opts.path = opts.path.expanduser()

    points, bbox = load_mesh(opts.path, sample=opts.sample)

    transforms = CabernetTransforms(seed=None)
    points, bbox = transforms.apply_rotation([points, bbox], alpha=opts.rotate_xz)
    points, bbox = transforms.apply_translation(
        [points, bbox], t_vec=np.asarray([opts.translate_x, 0, opts.translate_z], dtype=np.float64)
    )

    render = Render()
    render.add_pcd(pcd=points)
    if opts.bbox:
        render.add_bbox(bbox=bbox)

    xz_plane = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    measures = Measures.from_pointcloud(points)
    render.add_plane(points=xz_plane, measures=measures)

    render.render()
