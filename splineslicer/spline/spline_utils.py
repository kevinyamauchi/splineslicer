from typing import List

from geomdl import exchange, fitting, BSpline
from napari.layers import Labels
from napari.types import LayerDataTuple
import numpy as np


def fit_spline(
        coords: np.ndarray,
        order: int = 3,
        n_ctrl_pts: int = 25
) -> BSpline:
    spline = fitting.approximate_curve(coords.tolist(), order, ctrlpts_size=n_ctrl_pts)
    return spline


def get_spline_points(spline, increment: float = 0.01) -> np.ndarray:
    spline.delta = increment
    return np.asarray(spline.evalpts)


def fit_spline_to_skeleton_layer(
        skeleton_layer: Labels,
        output_path: str,
        order: int = 3,
        n_ctrl_pts: int = 25,
) -> List[LayerDataTuple]:

    if output_path == '':
        raise ValueError('invalid output path')

    skeleton_obj = skeleton_layer.metadata['skan_obj']

    # get the indices to flip
    flip_mask = skeleton_layer.properties['flip']
    segments_to_flip = skeleton_layer.properties['index'][flip_mask] - 1

    # get the indices to keep
    keep_mask = skeleton_layer.properties['keep']
    segments_to_keep = skeleton_layer.properties['index'][keep_mask] - 1

    clean_coords = []
    for i in segments_to_keep:
        path_coords = skeleton_obj.path_coordinates(i)
        if i in segments_to_flip:
            path_coords = np.flip(path_coords, axis=0)
        clean_coords.append(path_coords)
    clean_coords = np.vstack(clean_coords).astype(int)

    n_coords = len(clean_coords)

    order_image = np.zeros_like(skeleton_layer.data, dtype=float)

    increment = 1 / n_coords
    value = 0
    for i in range(n_coords):
        coord = clean_coords[i]
        order_image[coord[0], coord[1], coord[2]] = value
        value += increment

    # make the skeleton image
    curated_skeleton = order_image.copy()
    curated_skeleton[curated_skeleton > 0] = 1

    # fit the spline
    spline = fit_spline(clean_coords, order=order, n_ctrl_pts=n_ctrl_pts)
    spline_points = get_spline_points(spline, increment=0.01)

    # save the spline
    exchange.export_json(spline, output_path)

    # make the layer data
    order_layer_data = (
        order_image,
        {
            'interpolation': 'nearest',
            'colormap': 'turbo',
            'name': 'order image',
            'visible': False,
        },
        'image'
    )
    spline_layer_data = (
        spline_points,
        {
            'shape_type': 'path',
            'edge_width': 0.5,
            'edge_color': 'magenta',
            'metadata': {'spline': spline}
        },
        'shapes'
    )

    return [order_layer_data, spline_layer_data]
