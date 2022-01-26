from typing import Optional

from napari.layers import Image
from napari.types import LayerDataTuple, LabelsData
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops_table
from skimage.morphology import binary_closing, cube

from napari_tools_menu import register_function

@register_function(menu="Segmentation post-processing > Keep largest region (splisli)")
def keep_largest_region(binary_im: LabelsData) -> LabelsData:
    """Keep only the largest region in a binary image. The region
    is calculated using the scipy.ndimage.label function.

    Parameters
    ----------
    binary_im : np.ndarray
        The binary image to filter for the largest region.

    Returns
    -------
    binarized_clean : np.ndarray
        The binary_im with only the largest conencted region.
    """
    label_im = ndi.label(binary_im)[0]
    rp = regionprops_table(label_im, properties=["label", "area"])
    if len(rp["area"]) > 0:
        max_ind = np.argmax(rp["area"])
        max_label = rp["label"][max_ind]
        binarized_clean = np.zeros_like(label_im, dtype=bool)
        binarized_clean[label_im == max_label] = True
    else:
        binarized_clean = np.zeros_like(label_im, dtype=bool)

    return binarized_clean


def binarize_image(im: np.ndarray, channel: int = 1, threshold: float = 0.5, closing_size: Optional[int] = None) -> np.ndarray:
    """Binaraize an image and keep only the largest structure. Small holes are filled
    with the scipy.ndimage.binary_fill_holes() function.

    Parameters
    ----------
    im : np.ndarray
        The image to be binarized. The image is expected to be ordered (c, z, y, x).
    channel : int
        The channel to binarize in im. The image is expected to be ordered (c, z, y, x).
    threshold : float
        Threshold for binarization. Values less than or equal to this
        value will be set to False and values greater than this value
        will be set to True.
    close_size: Optional[int]
        The size of morphological closing to apply to the binarized image.
        This is generally to fill in holes. If None, no closing will be applied.
        The default value is None.

    Returns
    -------
    binarized_clean : np.ndarray
        The binarized image containing only the largets segmented
        region.
    """
    # make the mask and label image
    selected_channel = im[channel, ...]
    mask_im = selected_channel > threshold
    mask_im_filled = ndi.binary_fill_holes(mask_im)

    # keep only the largest structure
    binarized_clean = keep_largest_region(mask_im_filled)

    if closing_size is not None:
        return binary_closing(binarized_clean, selem=cube(closing_size))
    else:
        return binarized_clean


def _binarize_image_mg(im_layer: Image, channel: int = 0, threshold: float = 0.5, closing_size: int = 3) -> LayerDataTuple:
    """This is a magicgui wrapper for the binarize_image() function"""
    if closing_size == 0:
        closing_size = None

    im = im_layer.data

    if channel < 0 or channel > im.shape[0]:
        raise ValueError("channel index invalid")
    binarized_clean = binarize_image(im=im,channel=channel, threshold=threshold, closing_size=closing_size)

    # make the new layer name
    output_name = im_layer.name + ' binarized'


    return (binarized_clean, {'name': output_name}, 'image')
