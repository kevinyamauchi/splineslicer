from typing import List, Optional, Tuple

import numpy as np
from napari.layers import Layer, Image
from napari.types import LayerDataTuple, ImageData

from ..skeleton.binarize import binarize_image, keep_largest_region
from skimage.measure import regionprops
from skimage.transform import rotate


def binarize_per_slice(
        im_layer: Image,
        channel: int = 1,
        threshold: float = 0.5,
        closing_size: int = 1
) -> List[LayerDataTuple]:
    """Binaraize an image and keep only the largest structure in each slice.
     Small holes are filled with the scipy.ndimage.binary_fill_holes() function.

    Parameters
    ----------
    im_layer : napari.layers.Image
        The image layer to be binarized. The image is expected to be ordered (c, z, y, x).
    channel : int
        The channel to binarize in im. The image is expected to be ordered (c, z, y, x).
    threshold : float
        Threshold for binarization. Values less than or equal to this
        value will be set to False and values greater than this value
        will be set to True.
    closing_size: Optional[int]
        The size of morphological closing to apply to the binarized image.
        This is generally to fill in holes. If None, no closing will be applied.
        The default value is None.

    Returns
    -------
    binarized_clean : np.ndarray
        The binarized image containing only the largets segmented
        region.
    """
    im = im_layer.data
    binary_im = binarize_image(im=im, channel=channel, threshold=threshold, closing_size=closing_size)
    binarized_clean = np.asarray([keep_largest_region(s) for s in binary_im])

    layer_name = f"{im_layer.name} binarized"
    layer_metadata = {
        "metadata": {
            "binarize_settings": {
                "layer_name": im_layer.name,
                "binarize_channel": 1,
                "binarize_threshold": 0.5,
                "binarize_closing_size": closing_size
            }
        },
        "name": layer_name
    }

    return [(binarized_clean, layer_metadata, "image")]


def calculate_slice_rotations(im_stack: np.ndarray, max_rotation:float = 45) -> List[float]:
    """Calculate the rotation angle to align each slice so the
    objects long axis is aligned with the horizontal axis.

    Parameters
    ----------
    im_stack : np.ndarray
        A stack of images. The images should be binary
        or label iamges. The regions are found and processed
        with the scikit-image label and regionprops functions.
        The stack should have shape (z, y, x) for z images
        with shape (y, x).
    max_rotation : float
        The maximum allowed rotation between slices in degrees.
        If this value is exceeded, it is assumed that the
        opposite rotation was found and 180 is added to the rotation.
        The default value is 45.

    Returns
    -------
    rotations : List[float]
        The rotation for each slice in degrees.
    """
    # get the rotations of the images
    rotations = []
    rotations_raw = []
    prev_rot = 0
    previous_values = []
    for i, im in enumerate(im_stack):
        previous_values.append(prev_rot)
        rp = regionprops(im.astype(int))
        if len(rp) > 0:
            orientation = rp[0].orientation
            angle_in_degrees = orientation * (180 / np.pi) + 90
        else:
            angle_in_degrees = 0

        rotations_raw.append(angle_in_degrees)

        if i > 0:
            # check if we should flip the rotation
            if abs(prev_rot - angle_in_degrees) > max_rotation:
                angle_in_degrees = -1 * (180 - angle_in_degrees)

        prev_rot = angle_in_degrees

        rotations.append(angle_in_degrees)

    return rotations


def rotate_stack(
        im_stack: np.ndarray,
        rotations: List[float]
) -> np.ndarray:
    """Calculate the rotation angle to align each slice so the
    objects long axis is aligned with the horizontal axis.

    Parameters
    ----------
    im_stack : np.ndarray
        The stack of images to rotate. The stack should have
        shape (z, y, x) for z images with shape (y, x).
    rotations : List[float]
        The rotation for each slice in degrees.

    Returns
    -------
    rotated_stack : np.ndarray

    """
    rotated_stack = []
    for im, rot in zip(im_stack, rotations):
        rotated_stack.append(rotate(im, -rot, resize=False))

    return np.stack(rotated_stack)


def align_stack(
        im_stack: np.ndarray,
        flip_rotation_indices: Optional[List[int]] = None,
        start_slice: Optional[int] = None,
        end_slice: Optional[int] = None,
        max_rotation: float = 90,
        invert_rotation: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the rotation angle to align each slice so the
        objects long axis is aligned with the horizontal axis.

    Parameters
    ----------
    im_stack : np.ndarray
        A stack of images. The images should be binary
        or label iamges. The regions are found and processed
        with the scikit-image label and regionprops functions.
        The stack should have shape (z, y, x) for z images
        with shape (y, x).
    flip_rotation_indices : Optional[List[int]]
        The indices at which to flip rotation 180 degrees.
        If None, no flipping is performed. Default value
        is None.
    start_slice : Optional[int]
        The first slice to extract for alignment.
        If None, starts at the beginning of the stack.
        Default value is None.
    end_slice : Optional[int]
        The last slice to extract for alignment.
        If None, the last slice of the stack is used..
        Default value is None.
    max_rotation : float
        The maximum allowed rotation between slices in degrees.
        If this value is exceeded, it is assumed that the
        opposite rotation was found and 180 is added to the rotation.
        The default value is 45.

    Returns
    -------
    aligned_stack : np.ndarray
        The rotated and aligned stack.
    rotations : np.ndarray
        The rotations that were applied in degrees.
    """

    if start_slice is None:
        start_slice = 0
    if end_slice is None:
        end_slice = im_stack.shape[0]
    selected_indices = im_stack[start_slice:end_slice]

    # get the rotations of the images
    rotations = calculate_slice_rotations(
        selected_indices,
        max_rotation=max_rotation
    )

    if invert_rotation is True:
        rotations = np.asarray(rotations) + 180
    else:
        rotations = np.asarray(rotations)

    # correct flipped alignments
    if flip_rotation_indices is not None:
        for slice_index in flip_rotation_indices:
            rotations[slice_index::] = rotations[slice_index::] + 180

    # rotate the stack
    aligned_stack = rotate_stack(selected_indices, rotations)

    return aligned_stack, rotations


def _align_stack_mg(
        im_layer:  Image,
        start_slice: str,
        end_slice: str,
        flip_rotation_indices: str,
        max_rotation: float = 90,
        invert_rotation: bool = False
) -> LayerDataTuple:
    """magicgui wrapper for align_stack

    Parameters
    ----------
    im_layer : np.ndarray
        A stack of images. The images should be binary
        or label iamges. The regions are found and processed
        with the scikit-image label and regionprops functions.
        The stack should have shape (z, y, x) for z images
        with shape (y, x).
    flip_rotation_indices : Optional[List[int]]
        The indices at which to flip rotation 180 degrees.
        If None, no flipping is performed. Default value
        is None.
    start_slice : Optional[int]
        The first slice to extract for alignment.
        If None, starts at the beginning of the stack.
        Default value is None.
    end_slice : Optional[int]
        The last slice to extract for alignment.
        If None, the last slice of the stack is used..
        Default value is None.
    max_rotation : float
        The maximum allowed rotation between slices in degrees.
        If this value is exceeded, it is assumed that the
        opposite rotation was found and 180 is added to the rotation.
        The default value is 45.

    Returns
    -------
    aligned_stack : np.ndarray
        The rotated and aligned stack.
    """
    # convert inputs to kwarg values
    im = im_layer.data
    start_slice = int(start_slice)
    end_slice = int(end_slice)

    flip_rotation_indices.replace(" ", "")
    if flip_rotation_indices == '':
        flip_rotation_indices = None
    else:
        flip_rotation_indices = flip_rotation_indices.replace(" ", "").split(",")
        flip_rotation_indices = [int(index) for index in flip_rotation_indices]

    aligned_stack, rotations = align_stack(
        im_stack=im,
        flip_rotation_indices=flip_rotation_indices,
        start_slice=start_slice,
        end_slice=end_slice,
        max_rotation=max_rotation,
        invert_rotation=invert_rotation
    )
    rotation_settings = {
        'flip_rotation_indices': flip_rotation_indices,
        'start_slice': start_slice,
        'end_slice': end_slice,
        'max_rotation': max_rotation,
        'invert_rotation': invert_rotation,
    }

    metadata = {
        'rotation_settings': rotation_settings,
        'rotations': rotations
    }

    if "binarize_settings" in im_layer.metadata:
        binarize_settings = im_layer.metadata["binarize_settings"]
        metadata.update({"binarize_settings": binarize_settings})

    layer_name = f'{im_layer.name} aligned'
    layer_metadata = {
        'metadata': metadata,
        'name': layer_name,
        "opacity": 0.7,
        "colormap": "bop blue",
    }

    return (
        aligned_stack,
        layer_metadata,
        'image'
    )


def align_stack_from_layer(source_layer: Image, layer_to_align: Image) -> LayerDataTuple:
    rotations = source_layer.metadata['rotations']
    start_slice = source_layer.metadata['rotation_settings']['start_slice']
    end_slice = source_layer.metadata['rotation_settings']['end_slice']

    multi_channel_stack = layer_to_align.data[:, start_slice:end_slice, ...]
    aligned_stack = []
    for channel in multi_channel_stack:
        aligned_channel = rotate_stack(channel, rotations)
        aligned_stack.append(aligned_channel)
    aligned_stack = np.stack(aligned_stack)

    layer_name = f'{layer_to_align.name} aligned'
    layer_metadata = {
        'name': layer_name,
        'metadata': {
           'rotations': rotations,
        },
    }

    return aligned_stack, layer_metadata , 'image'


def align_mask_and_stain_from_layer(
        mask_layer: Image,
        stain_layer: Image,
        start_slice: str,
        end_slice: str,
        flip_rotation_indices: str,
        max_rotation: float = 90,
        invert_rotation: bool = False
) -> List[LayerDataTuple]:

    # align the mask
    aligned_mask_data = _align_stack_mg(
        im_layer=mask_layer,
        start_slice=start_slice,
        end_slice=end_slice,
        flip_rotation_indices=flip_rotation_indices,
        max_rotation=max_rotation,
        invert_rotation=invert_rotation
    )
    aligned_layer = Layer.create(*aligned_mask_data)

    # apply the same rotations to the stain layer
    aligned_stain_data = align_stack_from_layer(
        source_layer=aligned_layer,
        layer_to_align=stain_layer
    )


    return [aligned_stain_data, aligned_mask_data]
