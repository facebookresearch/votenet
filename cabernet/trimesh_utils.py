import os
import numpy as np
import sys
import trimesh

from pathlib import Path
from typing import List, Tuple


class SuppressPrints(object):
    def __init__(self, stdout: bool = True, stderr: bool = True):
        self._out = stdout
        self._err = stderr

    def __enter__(self):
        if self._out:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")
        if self._err:
            self._original_stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._out:
            sys.stdout.close()
            sys.stdout = self._original_stdout
        if self._err:
            sys.stderr.close()
            sys.stderr = self._original_stderr


def load_mesh(path: Path, sample: int = None) -> Tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(path)
    # if not isinstance(mesh, trimesh.Trimesh):
    #     raise RuntimeError(f"expecting type `trimesh.Trimesh` but input has type `{type(mesh)}`")

    if sample:
        sampler = trimesh.sample.sample_surface if len(mesh.vertices) < sample else trimesh.sample.sample_surface_even
        with SuppressPrints():
            points, _ = sampler(mesh, sample)
    else:
        points = mesh.vertices
    bbox = mesh.bounding_box.vertices

    return points, bbox


def load_scene(path: Path) -> List[trimesh.Trimesh]:
    mesh = trimesh.load_mesh(path)
    if not isinstance(mesh, trimesh.Scene):
        raise RuntimeError(f"expecting type `trimesh.Scene` but input has type `{type(mesh)}`")

    return [m for m in mesh.geometry.values()]
