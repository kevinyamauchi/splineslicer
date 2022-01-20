import os
from typing import Any, Dict

from geomdl import exchange, operations, BSpline
import h5py
from magicgui import magic_factory
import numpy as np
from scipy import ndimage

from .slice_utils import calculate_rotation_matrix, create_sampling_grid


def sample_image(image: np.ndarray, transformed_sampling_grid: np.ndarray, original_sampling_grid: np.ndarray, order:int = 0, im_half_width_x: int = 50,
                 im_half_width_y: int = 50) -> np.ndarray:
    """Sample an image with a sampling grid

    Parameters
    ----------
    image : np.ndarray
        The image to sample from.
    transformed_sampling_grid : np.ndarray
        (N, D) array of coordinates for the sampling grid transformed
        to the plane to be sampled from. These are the points that are
        used for sampling.
    original_sampling_grid : np.ndarray
        (N, D) array of coordinates for the axis-alinged, untransformed
        sampling grid. This is used to index the sampled values into the
        returned sliced image.
    order : int
        The order of interpolation to use when sampling. The default value is 0.
    im_half_width_x : int
        The half width of the grid in the x direction. The full width will be
        2 * im_half_width_x + 1. The default value is 50.
    im_half_width_y : int
        The half width of the grid in the y direction. The full width will be
        2 * im_half_width_x + 1. The default value is 50.

    """
    sampled_values = ndimage.map_coordinates(
        image,
        transformed_sampling_grid.T, order=order
    )

    im_width_y = 2 * im_half_width_y + 1
    im_width_x = 2 * im_half_width_x + 1
    sampled_image = np.zeros((im_width_y, im_width_x))
    y_indices = original_sampling_grid[:, 1] + im_half_width_y
    x_indices = original_sampling_grid[:, 2] + im_half_width_x
    sampled_image[y_indices, x_indices] = sampled_values

    return sampled_image


def slice_image(
    im: np.ndarray,
    spline: BSpline,
    slice_increment: float = 0.01,
    im_half_width_x: int = 150,
    im_half_width_y: int = 150,
    order: int = 0
) -> np.ndarray:
    """

    Parameters
    ----------
    im : np.ndarray
        The 3D image to be sliced. The image is assumed to be ordered (z, y, x).
    spline : BSpline
        The spline object to slice along.
    slice_increment : float
        The increment for slicing the spline. This is in the parametric coordinate.
        For example, an increment of 0.01 yields 100 slices (1 slice every 0.01 of
        the total spline length). The default value is 0.01.
    im_half_width_x : int
        The half width of the grid in the x direction. The full width will be
        2 * im_half_width_x + 1. The default value is 150.
    im_half_width_y : int
        The half width of the grid in the y direction. The full width will be
        2 * im_half_width_x + 1. The default value is 150.
    order : int
        The order of interpolation to use when sampling. The default value is 0.

    Returns
    -------
    sliced_im : np.ndarray
        The sliced image. The image is ordered (s, y, x), where s is the slice.
    """

    # create the sample grid
    sampling_grid = create_sampling_grid(
        im_half_width_x=im_half_width_x,
        im_half_width_y=im_half_width_y,
        step_x=1,
        step_y=1
    )

    # sampling grid was created in the YX plane
    reference_plane_normal = [1, 0, 0]

    # loop over positions and get values
    parametric_coords = np.arange(0, 1, slice_increment)
    slices = []

    for u in parametric_coords:
        # for the position, find the tangent to the curve
        tangent_data = operations.tangent(spline, u)
        center_point = np.asarray(tangent_data[0])
        plane_normal = np.asarray(tangent_data[1])

        if np.allclose(plane_normal, reference_plane_normal):
            rotated_coords = sampling_grid.copy()
        else:
            rot = calculate_rotation_matrix(reference_plane_normal, plane_normal)
            rotated_coords = sampling_grid @ rot.T

        translated_coords = rotated_coords + center_point

        # sample the segmentation
        im_slice = sample_image(
            im,
            transformed_sampling_grid=translated_coords,
            original_sampling_grid=sampling_grid,
            im_half_width_x = im_half_width_x,
            im_half_width_y = im_half_width_y,
            order=order
        )
        slices.append(im_slice)

    sliced_im = np.stack(slices)

    return sliced_im


@magic_factory(
    auto_call=True,
    im_fpath={'label': 'image file path', 'widget_type': 'FileEdit', 'mode': 'r', 'filter': '*.h5'},
    spline_fpath={'label': 'spline file path', 'widget_type': 'FileEdit', 'mode': 'r', 'filter': '*.json'},
    output_fpath={'label': 'output file path', 'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.h5'},
    call_button='slice image'
)
def slice_image_from_file(
    im_fpath: str,
    im_key: str,
    spline_fpath: str,
    output_fpath: str,
    n_slices: int = 100,
    im_half_width_x: int = 150,
    im_half_width_y: int = 150,
    interpolation_order: int = 0,
):
    # load the spline
    spline = exchange.import_json(spline_fpath)[0]

    # load the images
    with h5py.File(im_fpath) as f:
        im = f[im_key][:]

    # ensure the image is 4d (assumed czyx)
    if (im.ndim < 3) or (im.ndim > 4):
        raise ValueError('Image must be 3D or 4D')
    elif im.ndim == 3:
        im = np.expand_dims(im, axis=0)

    # slice the images
    slice_increment = 1 / n_slices
    sliced_images = []
    for channel_im in im:
        sliced_im = slice_image(
            channel_im,
            spline=spline,
            slice_increment=slice_increment,
            im_half_width_x=im_half_width_x,
            im_half_width_y=im_half_width_y,
            order=interpolation_order
        )
        sliced_images.append(sliced_im)
    im_stack = np.stack(sliced_images)

    # write the file
    with h5py.File(output_fpath, 'w') as f_out:
        f_out.create_dataset(
            'sliced_stack',
            im_stack.shape,
            data=im_stack,
            compression='gzip'
        )
