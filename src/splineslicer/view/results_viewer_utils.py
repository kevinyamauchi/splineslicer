from geomdl import exchange, operations
import numpy as np


def make_corners(half_width):
    """Make a simple triangular mesh of a plane."""
    min_val = -half_width
    max_val = half_width

    corners = np.array(
        [
            [0, min_val, min_val],
            [0, min_val, max_val],
            [0, max_val, max_val],
            [0, max_val, min_val]
        ]
    )

    faces = np.array(
        [
            [0, 1, 2],
            [0, 3, 2]
        ]
    )

    return corners, faces


def calculate_rotation_matrix(u, v):
    """Copmute the transformation to reotate u to v"""
    rotation_axis = np.cross(u, v)
    A = np.array([u, rotation_axis, np.cross(u, rotation_axis)]).T
    B = np.array([v, rotation_axis, np.cross(v, rotation_axis)]).T
    inv_A = np.linalg.inv(A)
    rot_mat = np.dot(B, inv_A)
    return rot_mat


def get_plane_coords(spline, u, half_width):
    corners, faces = make_corners(half_width)

    reference_plane_normal = [1, 0, 0]
    if u < 0:
        spline_length = operations.length_curve(spline)
        dist = u * spline_length
        tangent_data = operations.tangent(spline, 0)
        plane_normal = np.asarray(tangent_data[1])
        center_point = np.asarray(tangent_data[0]) + dist * plane_normal
    elif u > 1:
        spline_length = operations.length_curve(spline)
        dist = (1 - u) * spline_length
        tangent_data = operations.tangent(spline, 1)
        plane_normal = np.asarray(tangent_data[1])
        center_point = np.asarray(tangent_data[0]) - dist * plane_normal
    else:
        tangent_data = operations.tangent(spline, u)
        center_point = np.asarray(tangent_data[0])
        plane_normal = np.asarray(tangent_data[1])

    rot = calculate_rotation_matrix(reference_plane_normal, plane_normal)
    rotated_coords = corners @ rot.T

    translated_coords = rotated_coords + center_point

    return translated_coords, faces, center_point, plane_normal