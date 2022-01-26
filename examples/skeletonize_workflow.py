import h5py

from splineslicer.skeleton.binarize import binarize_image
from splineslicer.skeleton.skeletonize import make_skeleton
from splineslicer.spline.spline_utils import fit_spline_to_skeleton


def fit_spline(
        skeleton_output,
        segments_to_flip,
        segments_to_keep,
        order_image_shape,
        order: int = 3,
        n_ctrl_pts: int = 25,
):
    skeleton_obj = skeleton_output[1]

    spline, order_image = fit_spline_to_skeleton(
        skeleton_obj=skeleton_obj,
        segments_to_flip=segments_to_flip,
        segments_to_keep=segments_to_keep,
        order_image_shape=order_image_shape,
        order=order,
        n_ctrl_pts=n_ctrl_pts,
    )

    return spline, order_image


# load image
fpath = '/Users/kyamauch/Documents/neural_tube_analysis/full_data_20210615/converted_images_20210716/nt_25s-raw_rescaled_Probabilities.h5'

with h5py.File(fpath, 'r') as f:
    im = f['exported_data'][:]

binarized = binarize_image(
    im,
    channel=1,
    threshold=0.5,
    closing_size=3
)

skeleton = make_skeleton(binarized, min_branch_length=10)

# fit spline
spline = fit_spline(
    skeleton,
    segments_to_flip=[],
    segments_to_keep=[0],
    order_image_shape=(448, 505, 505),
    order=3,
    n_ctrl_pts=25
)

