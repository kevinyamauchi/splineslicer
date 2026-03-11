from typing import Any, Dict, List, Optional, TYPE_CHECKING
from napari.layers import Image
import numpy as np 
import napari
from scipy.signal import find_peaks
from skimage.measure import label, regionprops
import math 
from ..measure.align_slices import rotate_stack


def new_align_rotate(
        mask_layer: Image,
        stain_layer: Image,
        start_slice: Optional[int] = None,
        end_slice: Optional[int] = None,
        NT_segmentation_index: int=0,
        background_index: int=0,
        invert_rotation: bool = False
        ) -> "napari.types.LayerDataTuple":
    
    print(mask_layer.data.shape)
    if start_slice is None:
        start_slice = 0
    if end_slice is None:
        end_slice = mask_layer.data.shape[0]

    rotations, line_scans, orientations, pos, adapted_rotations = calculate_slice_rotations(mask_layer.data[NT_segmentation_index,start_slice:end_slice,...]
                                                                        ,mask_layer.data[background_index,start_slice:end_slice,...], p=0.6)
    
    print(invert_rotation)
    if invert_rotation is True:
        adapted_rotations = np.asarray(adapted_rotations) + 180
    else:
        adapted_rotations = np.asarray(adapted_rotations)

    # rotate the segmentation
    rotated_seg = rotate_stack(mask_layer.data[NT_segmentation_index,start_slice:end_slice,...], adapted_rotations)
    rotated_stack_adapted = []
    for i, chan in enumerate(stain_layer.data[:,start_slice:end_slice,...]):
        rotated_stack_adapted.append(rotate_stack(np.asarray(chan), adapted_rotations))
    
    # if no NT segmented or multiple segmented elements are found then the slice is not taken and axis 1 (nb of slices)
    # may become <100. needs to account for this 
    m = np.shape(rotated_stack_adapted[0:3])[1]
    rotated_stack_adapted.append(rotated_seg[0:m])

    # transform to np array for saving as h5
    rotated_stack_adapted = np.asarray(rotated_stack_adapted)
    return (rotated_stack_adapted, {'name': 'output_rotation', 'colormap': 'gray'}, 'image')


def calculate_slice_rotations(im_stack: np.ndarray, im_stack_chan: np.ndarray, max_rotation:float = 45, p=0.5) -> List[float]:
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
    line_scans = []
    orientations = []
    pos = []
    adapted_rotations = []
    for i, im in enumerate(im_stack):
        previous_values.append(prev_rot)
        rp = regionprops(im.astype(int))
        if len(rp) > 0:
            orientation = rp[0].orientation
            y0, x0 = rp[0].centroid

            # compute boundary points of the NT 
            x1 = x0 - math.sin(orientation) * p * rp[0].axis_major_length
            y1 = y0 - math.cos(orientation) * p * rp[0].axis_major_length
            x2 = x0 + math.sin(orientation) * p * rp[0].axis_major_length
            y2 = y0 + math.cos(orientation) * p * rp[0].axis_major_length

            if x1 <0:
                x1 = 0
            if y1 <0: 
                y1 = 0
            if x1 > 300:
                x1 = 300
            if y1 > 300: 
                y1 = 300

            if x2 <0:
                x2 = 0
            if y2 <0: 
                y2 = 0

            if x2 > 300:
                x2 = 300
            if y2 > 300: 
                y2 = 300
                
            length = int(np.hypot(x1-x2, y1-y2))
            x, y = np.linspace(x1, x2, length), np.linspace(y1, y2, length)
            pos.append([x1,y1,x2,y2])
            # Extract the values along the line
            linescan = im_stack_chan[i, y.astype(int), x.astype(int)]
            line_scans.append(linescan)
            orientations.append(orientation)
            angle = orientation * (180 / np.pi)

            if linescan[0]:
                if angle <0:
                    adapted_angle = angle+90
                else:
                    adapted_angle = angle+90
            else:
                if angle <0:
                    adapted_angle = angle-90
                else:
                    adapted_angle = angle-90
    
            angle_in_degrees = orientation * (180 / np.pi) + 90
            adapted_rotations.append(adapted_angle)

        else:
            angle_in_degrees = 0

        rotations_raw.append(angle_in_degrees)

        if i > 0:
            # check if we should flip the rotation
            if abs(prev_rot - angle_in_degrees) > max_rotation:
                angle_in_degrees = -1 * (180 - angle_in_degrees)

        prev_rot = angle_in_degrees

        rotations.append(angle_in_degrees+360)

    # method to check for switches
    check_angles = abs(np.diff(adapted_rotations)) # get the diff to get 
    a = np.where(check_angles > 90, 1, 0) # find the abrupt changes in the differential of angles
    switches = a[:-1]+a[1:] # sum consequent indices to find changes which are sudden and not maitained
    switching_ind = np.asarray(np.where(switches == 2)[0]+1) # these flipped would be equal to 2, these are the flipped indices
    if len(switching_ind)>0:
        np.array(adapted_rotations)[switching_ind] = np.array(adapted_rotations)[switching_ind]+180

    return rotations, line_scans, orientations, pos, np.asarray(adapted_rotations)
