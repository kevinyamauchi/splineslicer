from typing import Any, Dict, List, Optional, TYPE_CHECKING
import h5py
from magicgui import magicgui
import napari
from napari.layers import Layer, Image, Shapes
import numpy as np
from splineslicer.slice.slice_utils import calculate_rotation_matrix, create_sampling_grid
from splineslicer.slice.slice import sample_image
from geomdl import exchange, operations, BSpline


if TYPE_CHECKING:
    import napari

def slice_it(
    im_layer: Image,
    spline_layer: Shapes,
    n_slices: int = 100,
    im_half_width_x: int = 150,
    im_half_width_y: int = 150,
    interpolation_order: int = 0) -> "napari.types.LayerDataTuple":
    
    sliced_images = []
    im = im_layer.data[...]
    spline = spline_layer.metadata["spline"]
    slice_increment = 1 / (n_slices-1)
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
    return (im_stack, {'name': 'output_slicing', 'colormap': 'gray'}, 'image')
 
def slice_image(
    im: np.ndarray,
    spline: BSpline,
    slice_increment: float = 0.01,
    im_half_width_x: int = 150,
    im_half_width_y: int = 150,
    order: int = 0) -> np.ndarray:
    """
    This is a copy of kevin's slice_image function but replacing the parametric_coords variable to include the first and last points of the spline 
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
    parametric_coords = np.arange(0, 1+slice_increment/1000, slice_increment )#### /!\ here is the replacement to include 1 and not more
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
