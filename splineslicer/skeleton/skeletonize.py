from typing import Tuple

from napari.layers import Image
from napari.types import LabelsData, ImageData
import numpy as np
import pandas as pd
from skan import Skeleton, summarize
from skimage.morphology import binary_dilation, cube, skeletonize
from napari_tools_menu import register_function

def _remove_non_main(skeleton_obj: Skeleton) -> Skeleton:
    """Remove the non-main branches from the skeleton

    Parameters
    ----------
    skeleton_obj : skan.Skeleton
        The Skeleton object to remove the non-main branches from

    Returns
    -------
    main_skeleton : skan.Skeleton
        skeleton_obj with the non-main branches removed
    """

    # get the properties of all of the branches
    skeleton_df = summarize(skeleton_obj, find_main_branch=True)

    # get the non-main branchesn
    non_main_df = skeleton_df.loc[skeleton_df["main"] == False]

    # remove the non-main branches
    paths_to_prune = non_main_df.index.tolist()
    main_skeleton = skeleton_obj.prune_paths(paths_to_prune)

    return main_skeleton


def prune_skeleton(
    skeleton_obj: Skeleton,
    min_branch_length: float = 10,
) -> Skeleton:
    """Remove short end-point branches from a skeleton.

    Parameters
    ----------
    skeleton_obj : skan.Skeleton
        The Skeleton object to remove short end-point branches from
    min_branch_length : float
        The miniumum branch length for end point branches. The default value is 10.

    Returns
    -------
    pruned_skeleton_obj : skan.Skeleton
        skeleton_obj with the short end point branches removed
    """
    skeleton_df = summarize(skeleton_obj)
    bad_branches = skeleton_df.loc[
        (skeleton_df["branch-type"] != 2)
        & (skeleton_df["branch-distance"] < min_branch_length)
    ]
    bad_branch_indices = bad_branches.index.tolist()

    if len(bad_branch_indices) > 0:
        clean_skeleton = skeleton_obj.prune_paths(bad_branch_indices)
        clean_skeleton_im = np.asarray(clean_skeleton).astype(bool)
    else:
        clean_skeleton_im = np.asarray(skeleton_obj).astype(bool)

    dilated_skeleton = binary_dilation(clean_skeleton_im, selem=cube(3))
    pruned_skeleton = skeletonize(dilated_skeleton)
    pruned_skeleton_obj = Skeleton(pruned_skeleton)

    return pruned_skeleton_obj

def make_skeleton(
    im: ImageData,
    min_branch_length: float = 10,
) -> Tuple[LabelsData, Skeleton, pd.DataFrame]:
    """Skeletonize a binary image

    Parameters
    ----------
    im : np.ndarray
        The binary image to be skeletonized
    min_branch_length : float
        The miniumum branch length for end point branches. The default value is 10.
    """
    skeleton = skeletonize(im)
    skeleton_obj = Skeleton(skeleton)

    # remove the non-main branches
    main_skeleton = _remove_non_main(skeleton_obj)

    # remove the short end point branches
    pruned_skeleton = prune_skeleton(main_skeleton, min_branch_length)
    pruned_summary = summarize(pruned_skeleton)

    pruned_summary['index'] = np.arange(pruned_summary.shape[0]) + 1
    pruned_summary['keep'] = True
    pruned_summary['flip'] = False

    skeleton_labels = np.asarray(pruned_skeleton)

    return skeleton_labels, pruned_skeleton, pruned_summary

def _make_skeleton_mg(im_layer: Image, min_branch_length: float = 10):
    """Skeletonize a binary image. This is a wrapper for use with magicgui

    Parameters
    ----------
    im_layer : Image
        The image layer to make the skeleton from
    min_branch_length : float
        The miniumum branch length for end point branches. The default value is 10.
    """
    im = im_layer.data
    return make_skeleton(im=im, min_branch_length=min_branch_length)

@register_function(menu="Segmentation post-processing > Make skeleton from binary image (splisli)")
def make_skeleton_ntm(
    binary_image: LabelsData,
    min_branch_length: float = 10,
) -> LabelsData:
    """Skeletonize a binary image. This is a wrapper for use with magicgui + napari tools menu + assistant

    Parameters
    ----------
    binary_image
    min_branch_length

    Returns
    -------

    """

    skeleton_labels, pruned_skeleton, pruned_summary = make_skeleton(binary_image, min_branch_length)
    return skeleton_labels