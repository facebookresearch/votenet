import argparse
import json
import meshio
import numpy as np
import os
import shutil
import sys
import trimesh
import zipfile

from loguru import logger
from pathlib import Path


class MeshSampler(object):
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


    def __init__(self, mesh: meshio.Mesh, center_to_origin: bool = True):
        mesh = trimesh.Trimesh(vertices=mesh.points, faces=mesh.cells_dict["triangle"])
        if center_to_origin:
            principal_inertia_transform = mesh.principal_inertia_transform
            mesh = mesh.apply_transform(principal_inertia_transform)
        self.mesh = mesh

    def __call__(self, num_points: int = 1024):
        with self.SuppressPrints():
            points, faces = trimesh.sample.sample_surface_even(self.mesh, num_points)
            if points.shape[0] < num_points:
                points, faces = trimesh.sample.sample_surface(self.mesh, num_points)
        normals = self.mesh.face_normals[faces]
        return points, normals

    def get_mesh(self):
        return self.mesh


def process_dir(source: str, target: str, num_points: int) -> None:
    """
    Process files of a single dir.
     1. read file
     2. sample num_points of the shape.
     3. rotate for later convenient.
    :param source: source dir.
    :param target: target dir.
    :param num_points: number of points to sample.
    """
    for file in source.iterdir():
        try:
            mesh = meshio.read(file)
        except meshio._exceptions.ReadError:
            continue

        mesh_sampler = MeshSampler(mesh, center_to_origin=True)
        vertices, _ = mesh_sampler(num_points)
        bbox = mesh_sampler.get_mesh().bounding_box_oriented.vertices
        centeroid = mesh_sampler.get_mesh().centroid

        with open(target.joinpath(file.stem).with_suffix(".json"), "w") as f:
            json.dump({
                "vertices": vertices.tolist(),
                "oriented_bbox": bbox.tolist(),
                "centroid": centeroid.tolist()
            }, f, indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-dataset-path", type=Path, required=True, help="Path to a zipped model-net dataset")
    parser.add_argument("--num-points", type=int, default=1024, help="#points (sampled) for each mesh")
    parser.add_argument("--force", action="store_true", help="Set to force pre-proceesing")
    opts = parser.parse_args()

    ds_path = opts.zip_dataset_path.expanduser()
    if not ds_path.exists() or not zipfile.is_zipfile(ds_path):
        raise ValueError(f"{ds_path} does not exist or not a zip file")

    with zipfile.ZipFile(ds_path, "r") as zip_ref:
        ds_path_unzip = ds_path.parent.joinpath("__unzip__")
        if not ds_path_unzip.exists():
            zip_ref.extractall(ds_path_unzip)
        ds_path_unzip_ = ds_path_unzip.joinpath(ds_path.stem)

    processed_ds_path = ds_path.parent.joinpath(f"{ds_path.stem}_{opts.num_points}")
    if opts.force and opts.processed_ds_path.exists():
        shutil.rmtree(processed_ds_path, ignore_errors=True)
    processed_ds_path.mkdir(parents=True, exist_ok=True)

    for file in ds_path_unzip_.iterdir():

        if not file.is_dir():
            continue

        logger.info(f"Processing `{file.name}`")
        for mode in ("train", "test"):
            processed_path = processed_ds_path.joinpath(file.name).joinpath(mode)
            processed_path.mkdir(parents=True, exist_ok=True)
            process_dir(file.joinpath(mode), processed_path, opts.num_points)

    shutil.rmtree(ds_path_unzip)
