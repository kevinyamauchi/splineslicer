from enum import Enum
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from geomdl import exchange, operations
from napari.layers import Image
from napari.types import LayerDataTuple
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.measure import label, regionprops


class BoundaryModes(Enum):
    DERIVATIVE = 'derivative'
    PERCENT_MAX = 'percent_max'
    TANH = 'tanh'


def find_edge_derivative(bg_sub_profile: np.ndarray) -> Tuple[int, int]:
    # get the rising edge
    first_deriv = np.diff(bg_sub_profile)
    peaks_rising, _ = find_peaks(first_deriv, height=0)
    rising_edge = peaks_rising[0]

    # get the falling edge
    peaks_falling, _ = find_peaks(-first_deriv, height=0)
    falling_edge = peaks_falling[0]

    return rising_edge, falling_edge


def _detect_threshold_crossing(
        bg_sub_profile: np.ndarray,
        threshold: float,
        iloc: int,
        increment: int):
    run = True
    max_index = len(bg_sub_profile) - 1

    crossing_index = iloc
    while run:
        value = bg_sub_profile[crossing_index]

        if value < threshold:
            run = False
        elif (crossing_index == 0) or (crossing_index == max_index):
            # stop if the end of the array is reached
            run = False
        else:
            crossing_index += increment

    return crossing_index


def find_edge_percent_max(bg_sub_profile: np.ndarray, threshold: float) -> Tuple[int, int]:
    # find the peak index
    peaks, _ = find_peaks(bg_sub_profile, distance=10)
    if len(peaks > 0):
        peak_index = np.min(peaks)

        # find the peak value
        peak_value = bg_sub_profile[peak_index]

        # calculate the threshold value
        threshold_value = threshold * peak_value

        # find rising edge
        rising_edge = _detect_threshold_crossing(
            bg_sub_profile,
            threshold=threshold_value,
            iloc=peak_index,
            increment=-1
        )

        # find trailing edge
        falling_edge = _detect_threshold_crossing(
            bg_sub_profile,
            threshold=threshold_value,
            iloc=peak_index,
            increment=1
        )
    else:
        rising_edge = 0
        falling_edge = 0

    return rising_edge, falling_edge


def _tanh(x: np.ndarray, a: float, w_0: float, phi_0: float, w_1: float, phi_1:float):
    """Sum of two opposing tanh functions that forms"""
    return a * 0.5 * (
        np.tanh((x - phi_0) / w_0) - np.tanh((x - phi_1) / w_1)
    )


def _estimate_tanh_parameters(
        peak_value: float,
        peak_index: Union[int, float],
        est_rising_edge: Union[int, float],
        est_falling_edge: Union[int, float]
) -> List[float]:
    # estimate domain parameters
    a_init = peak_value
    w_0_init = (peak_index - est_rising_edge) / 4
    phi_0_init = (peak_index + est_rising_edge) / 2
    w_1_init = (est_falling_edge - peak_index) / 4
    phi_1_init = (est_falling_edge + peak_index) / 2

    p_0 = [a_init, w_0_init, phi_0_init, w_1_init, phi_1_init]

    return p_0


def _estimate_tanh_bounds(
        peak_value: float,
        peak_index: Union[int, float],
        est_falling_edge: Union[int, float]
) -> List[List[float]]:
    # define the bounds
    a_bounds = [0, peak_value]
    w_0_bounds = [0, peak_index]
    phi_0_bounds = [0, peak_index]
    w_1_bounds = [0, est_falling_edge - peak_index]
    phi_1_bounds = [peak_index, est_falling_edge]

    bounds_lower = [
        a_bounds[0],
        w_0_bounds[0],
        phi_0_bounds[0],
        w_1_bounds[0],
        phi_1_bounds[0],
    ]
    bounds_upper = [
        a_bounds[1],
        w_0_bounds[1],
        phi_0_bounds[1],
        w_1_bounds[1],
        phi_1_bounds[1],
    ]
    bounds = [bounds_lower, bounds_upper]

    return bounds


def find_edge_fit(bg_sub_profile: np.ndarray, threshold: float, fit_func: Optional[Callable] = None):
    if fit_func is None:
        fit_func = _tanh
        param_func = _estimate_tanh_parameters
        bounds_func = _estimate_tanh_bounds

    # find the peak index
    peaks, _ = find_peaks(bg_sub_profile, distance=10)
    if len(peaks > 0):
        peak_index = np.min(peaks)

        # find the peak value
        peak_value = bg_sub_profile[peak_index]

        # estimate the falling edge location
        est_rising_edge, est_falling_edge = find_edge_percent_max(bg_sub_profile, threshold)

        # estimate the fit parameters
        p_0 = param_func(
            peak_value=peak_value,
            peak_index=peak_index,
            est_rising_edge=est_rising_edge,
            est_falling_edge=est_falling_edge
        )

        # define the parameter bounds
        bounds = bounds_func(
            peak_index=peak_index,
            peak_value=peak_value,
            est_falling_edge=est_falling_edge
        )

        far_index = np.clip(est_falling_edge + 10, 0, len(bg_sub_profile))
        fit_profile = bg_sub_profile[0:far_index]
        xdata = np.arange(far_index)
        popt, pcov = curve_fit(
            fit_func, xdata, fit_profile, bounds=bounds, p0=p_0
        )

        fit_values = fit_func(xdata, *popt)

        # set the threshold as a function of the amplitude
        threshold_value = popt[0] * threshold

        # find rising edge
        rising_edge = _detect_threshold_crossing(
            fit_values,
            threshold=threshold_value,
            iloc=peak_index,
            increment=-1
        )

        # find trailing edge
        falling_edge = _detect_threshold_crossing(
            fit_values,
            threshold=threshold_value,
            iloc=peak_index,
            increment=1
        )

    else:
        rising_edge = 0
        falling_edge = 0

    return rising_edge, falling_edge


def find_boundaries(
        seg_im: np.ndarray,
        stain_im: np.ndarray,
        half_width: int = 10,
        bg_sample_pos: float = 0.7,
        bg_half_width: int = 2,
        edge_method: BoundaryModes = BoundaryModes.DERIVATIVE,
        edge_value: float = 0.1
):
    ventral_boundary = []
    dorsal_boundary = []
    nt_length = []
    bg_sub_profiles = []
    cropped_ims = []
    for nt_seg_slice, raw_slice in zip(seg_im, stain_im):
        # find the centroids and end points of the nt
        nt_label_im = label(nt_seg_slice)
        rp = regionprops(nt_label_im)

        nt_centroid = rp[0].centroid
        nt_bbox = rp[0].bbox

        # get the crop region
        row_min = int(nt_centroid[0] - half_width)
        row_max = int(nt_centroid[0] + half_width + 1)
        col_min = int(nt_bbox[1])
        col_max = int(nt_bbox[3])

        raw_crop = raw_slice[nt_bbox[0]:nt_bbox[2], nt_bbox[1]:nt_bbox[3]]
        sample_region = raw_slice[row_min:row_max, col_min:col_max]
        summed_profile = sample_region.sum(axis=0)

        # correct for columsn with no nt labels
        n_sample_rows = nt_seg_slice[row_min:row_max, col_min:col_max].sum(axis=0)
        no_nt_indices = np.argwhere(n_sample_rows == 0)
        n_sample_rows[no_nt_indices] = 1
        summed_profile[no_nt_indices] = 0

        # normalize the intensity by the number of nt pixels in the column
        raw_profile = summed_profile / n_sample_rows

        bg_sample_center = int(bg_sample_pos * (col_max - col_min))
        bg_sample_min = bg_sample_center - bg_half_width
        bg_sample_max = bg_sample_center + bg_half_width
        bg_value = np.mean(raw_profile[bg_sample_min:bg_sample_max])
        bg_sub_profile = (raw_profile - bg_value).clip(min=0, max=1)

        if isinstance(edge_method, str):
            edge_method = BoundaryModes(edge_method)
        if edge_method == 'derivative':
            rising_edge, falling_edge = find_edge_derivative(bg_sub_profile)
        elif edge_method == 'percent_max':
            rising_edge, falling_edge = find_edge_percent_max(bg_sub_profile, threshold=edge_value)
        elif edge_method == 'tanh':
            rising_edge, falling_edge = find_edge_fit(bg_sub_profile, threshold=edge_value)

        # store the values
        nt_length.append(col_max - col_min)
        ventral_boundary.append(rising_edge)
        dorsal_boundary.append(falling_edge)
        cropped_ims.append(raw_crop)
        bg_sub_profiles.append(bg_sub_profile)

    return nt_length, ventral_boundary, dorsal_boundary, bg_sub_profiles, cropped_ims


def find_boundaries_from_layer(
        segmentation_layer: Image,
        stain_layer: Image,
        spline_file_path: str,
        table_output_path: str,
        channel_names: str,
        half_width: int = 10,
        bg_sample_pos: float = 0.7,
        bg_half_width: int = 2,
        edge_method: BoundaryModes = BoundaryModes.PERCENT_MAX,
        edge_value: float = 0.1,
        pixel_size_um: float = 5.79
) -> LayerDataTuple:
    channel_names = channel_names.replace(" ", "").split(",")

    seg_im = segmentation_layer.data
    stain_im = stain_layer.data

    # get data from the layers
    start_slice = segmentation_layer.metadata['rotation_settings']['start_slice']
    end_slice = segmentation_layer.metadata['rotation_settings']['end_slice']
    rotations = segmentation_layer.metadata['rotations']

    channel_data = []
    all_cropped_images = []
    for i, channel_im in enumerate(stain_im):
        nt_length, ventral_boundary, dorsal_boundary, bg_sub_profiles, cropped_ims = find_boundaries(
            seg_im=seg_im,
            stain_im=channel_im,
            half_width=half_width,
            bg_sample_pos=bg_sample_pos,
            bg_half_width=bg_half_width,
            edge_method=edge_method,
            edge_value=edge_value
        )
        all_cropped_images.append(cropped_ims)

        nt_length = np.asarray(nt_length)
        nt_length_um = nt_length * pixel_size_um
        ventral_boundary = np.asarray(ventral_boundary) * pixel_size_um
        dorsal_boundary = np.asarray(dorsal_boundary) * pixel_size_um

        ventral_boundary_rel = ventral_boundary / nt_length_um
        dorsal_boundary_rel = dorsal_boundary / nt_length_um

        # get the slice length and increment in microns
        spline = exchange.import_json(spline_file_path)[0]
        spline_length = operations.length_curve(spline) * pixel_size_um
        spline_increment = spline_length / channel_im.shape[0]

        n_slices = len(nt_length_um)
        target_name = channel_names[i]
        target = [target_name] * n_slices
        threshold_list = [edge_value] * n_slices
        slice_index = np.arange(start_slice, end_slice)
        slice_position_um = [i * spline_increment for i in slice_index]
        slice_position_rel_um = [i * spline_increment for i in range(n_slices)]

        df = pd.DataFrame(
            {
                'slice_index': slice_index,
                'slice_position_um': slice_position_um,
                'slice_position_rel_um': slice_position_rel_um,
                'nt_length_um': nt_length_um,
                'target': target,
                'ventral_boundary_um': ventral_boundary,
                'dorsal_boundary_um': dorsal_boundary,
                'ventral_boundary_rel': ventral_boundary_rel,
                'dorsal_boundary_rel': dorsal_boundary_rel,
                'domain_edge_threshold': threshold_list,
                'rotation': rotations
            }
        )
        channel_data.append(df)


    all_data = pd.concat(channel_data, ignore_index=True)
    all_data.to_csv(table_output_path)

    layer_kwargs = {
        'metadata': {'measurements': all_data},
        'name': 'measurement_crops'
    }

    return (np.stack(all_cropped_images), layer_kwargs, 'image')
