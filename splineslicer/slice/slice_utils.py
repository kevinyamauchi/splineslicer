import numpy as np


def create_sampling_grid(
        im_half_width_x: int = 50,
        im_half_width_y: int = 50,
        step_x: int = 1,
        step_y: int = 1
) -> np.ndarray:
    """Create the coordinates for creating a rectangular sampling grid
    centered around (0, 0, 0) with normal (1, 0, 0).

    Parameters
    ----------
    im_half_width_x : int
        The half width of the grid in the x direction. The full width will be
        2 * im_half_width_x + 1. The default value is 50.
    im_half_width_y : int
        The half width of the grid in the y direction. The full width will be
        2 * im_half_width_x + 1. The default value is 50.
    step_x : int
        The step size of the sampling grid in the x direction.
    step_y : int
        The step size of the sampling grid in the y direction.

    Returns
    -------
    positions : np.ndarray
        The coordinates of the
    """
    # create a grid of points with z=0 , centered around (0, 0, 0)
    z_indices = [0]
    y_indices = np.arange(-im_half_width_y, im_half_width_y + 1, step_y)
    x_indices = np.arange(-im_half_width_x, im_half_width_x + 1, step_x)

    z, y, x = np.meshgrid(z_indices, y_indices, x_indices)

    positions = np.vstack([z.ravel(), y.ravel(), x.ravel()]).T

    return positions


def calculate_rotation_matrix(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Calculate the rotation matrix to transform vector v to vector u

    Parameters
    ----------
    u : np.ndarray
        The vector to be rotated to
    v : np.ndarray
        The vector to be rotated

    Returns
    -------
    rot_mat np.ndarray
        The rotation matrix. Can be applied to an (N, D) array of points by
        pts @ rot_mat.T
    """
    rotation_axis = np.cross(u, v)
    A = np.array([u, rotation_axis, np.cross(u, rotation_axis)]).T
    B = np.array([v, rotation_axis, np.cross(v, rotation_axis)]).T
    inv_A = np.linalg.inv(A)
    rot_mat = np.dot(B, inv_A)

    return rot_mat
