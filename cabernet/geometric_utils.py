import numpy as np

from typing import List, Optional, Tuple


BASE_NORMALS = {
    "xy+": np.array([0, 0, 1]),
    "xy-": np.array([0, 0, -1]),
    "xz+": np.array([0, 1, 0]),
    "xz-": np.array([0, -1, 0]),
    "yz+": np.array([1, 0, 0]),
    "yz-": np.array([-1, 0, 0]),
}


class Measures:
    def __init(self):
        pass

    @classmethod
    def from_pointcloud(cls, pointcloud: np.ndarray):
        self = cls()

        self.min_x = pointcloud[:, 0].min().item()
        self.max_x = pointcloud[:, 0].max().item()
        self.size_x = self.max_x - self.min_x

        self.min_y = pointcloud[:, 1].min().item()
        self.max_y = pointcloud[:, 1].max().item()
        self.size_y = self.max_y - self.min_y

        self.min_z = pointcloud[:, 2].min().item()
        self.max_z = pointcloud[:, 2].max().item()
        self.size_z = self.max_z - self.min_z

        return self


def get_plane_params(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Return the plane's normal and bias:
    ax + by + cz + d = 0 ===>
    dot(normal, V) + bias = 0
    :param points:
    :return:
    """
    if points.shape[0] < 3 or points.shape[1] != 3:
        raise RuntimeError

    v1 = points[2] - points[0]
    v2 = points[1] - points[0]
    normal = np.cross(v1, v2)
    bias = -1.0 * np.dot(normal, points[0])

    return normal, bias


def intersect_bbox_aligned(pointcloud: np.ndarray, bbox: np.ndarray, tolerance: Optional[float] = 0.1) -> bool:
    """
    Checks if a given pointcloud intersects the axis-aligned bounding box defines by corner points.

    :param pointcloud: pointcloud of shape (num_points, 3).
    :param bbox: corners of axis-aligned bounding box, as an array of shape (8, 3).
    :param tolerance: tolerance of intersection.
    :return: bool indicating if the pointcloud intersects the axis-aligned bounding box.
    """
    measures = Measures.from_pointcloud(bbox)
    x_intersect = (measures.min_x <= pointcloud[:, 0]).astype(np.uint8) * (pointcloud[:, 0] <= measures.max_x).astype(
        np.uint8
    )
    y_intersect = (measures.min_y <= pointcloud[:, 1]).astype(np.uint8) * (pointcloud[:, 1] <= measures.max_y).astype(
        np.uint8
    )
    z_intersect = (measures.min_z <= pointcloud[:, 2]).astype(np.uint8) * (pointcloud[:, 2] <= measures.max_z).astype(
        np.uint8
    )
    intersection = (x_intersect * y_intersect * z_intersect).astype(np.uint8)
    return np.sum(intersection) >= tolerance * pointcloud.shape[0]


def intersect_bboxes_aligned(
    pointcloud: np.ndarray, bboxes: List[np.ndarray], tolerance: Optional[float] = 0.1
) -> bool:
    """
    Checks if a given pointcloud intersects any of the axis-aligned bounding boxes defines by corner points.

    :param pointcloud: pointcloud of shape (num_points, 3).
    :param bboxes: corners of axis-aligned bounding boxes, as a list of arrays of shape (8, 3).
    :param tolerance: tolerance of intersection.
    :return: bool indicating if the pointcloud intersects any of the axis-aligned bounding boxes.
    """
    for bbox in bboxes:
        if intersect_bbox_aligned(pointcloud=pointcloud, bbox=bbox, tolerance=tolerance):
            return True
    return False
