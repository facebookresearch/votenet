import argparse
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from geometric_utils import BASE_NORMALS, get_plane_params
from pathlib import Path
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
        self,
        normal: np.ndarray,
        bias: float,
        x_min: float = -10.0,
        x_max: float = 10.0,
        x_num: int = 100,
        y_min: float = -10.0,
        y_max: float = 10.0,
        y_num: int = 100,
        z_min: float = -10.0,
        z_max: float = 10.0,
        z_num: int = 100,
        surface_kwargs: Dict = dict(alpha=0.5),
    ):
        xx, yy = np.meshgrid(
            np.linspace(start=x_min, stop=x_max, num=x_num), np.linspace(start=y_min, stop=y_max, num=y_num)
        )
        self.ax.plot_surface(xx, yy, -1.0 * bias * np.ones_like(xx), **surface_kwargs)

    def add_xz_plane(
        self,
        normal: np.ndarray,
        bias: float,
        x_min: float = -10.0,
        x_max: float = 10.0,
        x_num: int = 100,
        y_min: float = -10.0,
        y_max: float = 10.0,
        y_num: int = 100,
        z_min: float = -10.0,
        z_max: float = 10.0,
        z_num: int = 100,
        surface_kwargs: Dict = dict(alpha=0.5),
    ):
        xx, zz = np.meshgrid(
            np.linspace(start=x_min, stop=x_max, num=x_num), np.linspace(start=z_min, stop=z_max, num=z_num)
        )
        self.ax.plot_surface(xx, -1.0 * bias * np.ones_like(xx), zz, **surface_kwargs)

    def add_yz_plane(
        self,
        normal: np.ndarray,
        bias: float,
        x_min: float = -10.0,
        x_max: float = 10.0,
        x_num: int = 100,
        y_min: float = -10.0,
        y_max: float = 10.0,
        y_num: int = 100,
        z_min: float = -10.0,
        z_max: float = 10.0,
        z_num: int = 100,
        surface_kwargs: Dict = dict(alpha=0.5),
    ):
        yy, zz = np.meshgrid(
            np.linspace(start=y_min, stop=y_max, num=y_num), np.linspace(start=z_min, stop=z_max, num=z_num)
        )
        self.ax.plot_surface(-1.0 * bias * np.ones_like(yy), yy, zz, **surface_kwargs)

    def add_any_plane(
        self,
        normal: np.ndarray,
        bias: float,
        x_min: float = -10.0,
        x_max: float = 10.0,
        x_num: int = 100,
        y_min: float = -10.0,
        y_max: float = 10.0,
        y_num: int = 100,
        z_min: float = -10.0,
        z_max: float = 10.0,
        z_num: int = 100,
        surface_kwargs: Dict = dict(alpha=0.5),
    ):
        a, b, c = normal
        xx, yy = np.meshgrid(
            np.linspace(start=x_min, stop=x_max, num=x_num), np.linspace(start=y_min, stop=y_max, num=y_num)
        )
        z = -1.0 * (a * xx + b * yy + bias) / c
        self.ax.plot_surface(xx, yy, z, **surface_kwargs)

    def add_plane(
        self,
        points: np.ndarray,
        x_min: float = -10.0,
        x_max: float = 10.0,
        x_num: int = 100,
        y_min: float = -10.0,
        y_max: float = 10.0,
        y_num: int = 100,
        z_min: float = -10.0,
        z_max: float = 10.0,
        z_num: int = 100,
        surface_kwargs: Dict = dict(alpha=0.5),
    ):
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
        planes_fn(
            normal=normal,
            bias=bias,
            x_min=x_min,
            x_max=x_max,
            x_num=x_num,
            y_min=y_min,
            y_max=y_max,
            y_num=y_num,
            z_min=z_min,
            z_max=z_max,
            z_num=z_num,
            surface_kwargs=surface_kwargs,
        )

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
    opts = parser.parse_args()

    opts.path = opts.path.expanduser()
    mesh = trimesh.load_mesh(opts.path)

    if opts.sample:
        sampler = (
            trimesh.sample.sample_surface if len(mesh.vertices) < opts.sample else trimesh.sample.sample_surface_even
        )
        points, _ = sampler(mesh, opts.sample)
    else:
        points = mesh.vertices

    render = Render()
    render.add_pcd(pcd=points)
    if opts.bbox:
        bbox = mesh.bounding_box.vertices
        render.add_bbox(bbox=bbox)

    xz_plane = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    render.add_plane(
        points=xz_plane,
        x_min=points[:, 0].min(), x_max=points[:, 0].max(),
        y_min=points[:, 1].min(), y_max=points[:, 1].max(),
        z_min=points[:, 2].min(), z_max=points[:, 2].max()
    )

    render.render()
