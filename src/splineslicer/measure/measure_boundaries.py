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


from ..io.hdf5 import save_aligned_slices


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
    nt_col_start = []
    nt_col_end = []
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

        # correct for columns with no nt labels
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
        if edge_method == BoundaryModes.DERIVATIVE:
            rising_edge, falling_edge = find_edge_derivative(bg_sub_profile)
        elif edge_method == BoundaryModes.PERCENT_MAX:
            rising_edge, falling_edge = find_edge_percent_max(bg_sub_profile, threshold=edge_value)
        elif edge_method == BoundaryModes.TANH:
            rising_edge, falling_edge = find_edge_fit(bg_sub_profile, threshold=edge_value)
        else:
            raise ValueError('unknown boundary mode')

        # store the values
        nt_length.append(col_max - col_min)
        ventral_boundary.append(rising_edge)
        dorsal_boundary.append(falling_edge)
        cropped_ims.append(raw_crop)
        bg_sub_profiles.append(bg_sub_profile)
        nt_col_start.append(col_min)
        nt_col_end.append(col_max - 1)

    return nt_length, ventral_boundary, dorsal_boundary, bg_sub_profiles, cropped_ims, nt_col_start, nt_col_end


def find_boundaries_from_layer(
        segmentation_layer: Image,
        stain_layer: Image,
        spline_file_path: str,
        table_output_path: str,
        aligned_slices_output_path: str,
        channel_names: str,
        half_width: int = 10,
        bg_sample_pos: float = 0.7,
        bg_half_width: int = 2,
        edge_method: BoundaryModes = BoundaryModes.PERCENT_MAX,
        edge_value: float = 0.1,
        pixel_size_um: float = 5.79
) -> List[List[np.ndarray]]:
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
        nt_length, ventral_boundary_px, dorsal_boundary_px, bg_sub_profiles, cropped_ims, nt_col_start, nt_col_end = find_boundaries(
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
        ventral_boundary_um = np.asarray(ventral_boundary_px) * pixel_size_um
        dorsal_boundary_um = np.asarray(dorsal_boundary_px) * pixel_size_um

        ventral_boundary_rel = ventral_boundary_um / nt_length_um
        dorsal_boundary_rel = dorsal_boundary_um / nt_length_um

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
                'ventral_boundary_um': ventral_boundary_um,
                'dorsal_boundary_um': dorsal_boundary_um,
                'ventral_boundary_rel': ventral_boundary_rel,
                'dorsal_boundary_rel': dorsal_boundary_rel,
                'domain_edge_threshold': threshold_list,
                'rotation': rotations,
                'nt_start_column_px': nt_col_start,
                'nt_end_column_px': nt_col_end,
                'ventral_boundary_px': ventral_boundary_px,
                'dorsal_boundary_px': dorsal_boundary_px
            }
        )
        channel_data.append(df)

    all_data = pd.concat(channel_data, ignore_index=True)
    all_data.to_csv(table_output_path)

    # save the aligned images
    save_aligned_slices(
        file_path=aligned_slices_output_path,
        segmentation_layer=segmentation_layer,
        stain_layer=stain_layer,
        stain_channel_names=channel_names,
        compression="gzip"
    )

    return all_cropped_images


### New functions for the updated version by Alexis Villars


def update_metadata(
    image_layer: Image,
    channel_0: str,
    channel_1: str,
    channel_2: str,
    channel_3: Optional[str] = None
):
    if channel_3 is None:
        stain_channel_names = [channel_0, channel_1, channel_2, 'segmentation']
    else: 
        stain_channel_names = [channel_0, channel_1, channel_2, channel_3, 'segmentation']

    image_metadata =  image_layer.metadata
    image_metadata.update({"channel_names": stain_channel_names})
    print(image_metadata)


def measure_boundaries(
        image_layer: Image,
        spline_file_path: str,
        table_output_path: str,
        half_width: float = 0.5,
        edge_method: BoundaryModes = BoundaryModes.PERCENT_MAX,
        edge_value: float = 0.5,
        pixel_size_um: float = 5.79,
        upper_range: float = 1,
        lower_range: float = 0,
        start_slice: int = 0,
        end_slice: int = 99):
    all_cropped_images = []
    channel_data = []
    bg_sub_profiles = []
    raw_profiles = []
    
    im_channels = image_layer.data
    # seg_im= image_layer.data[-1,...]

    targets =  image_layer.metadata.get(
            "channel_names", {}
    )

    # run through the 3 first channels: I should make this automatically adapt to the channel number. 
    # restricts the measurements to the area that are correctly rotated and correclty segmented
    seg_im = np.asarray(im_channels[-1, start_slice:end_slice, ...])
    
    channel_data = []
    bg_sub_profiles = []
    raw_profiles = []

    # run the measurement for each channels except the last one (the segmentation one)
    for k, chan in enumerate(im_channels[:-1]):
        
        # restrict measurement to current channels
        stain_im = np.asarray(im_channels[k, start_slice:end_slice, ...])

        #measure
        
        nt_length, ventral_boundary_px, dorsal_boundary_px, bg_sub_profile, raw_profile, cropped_ims, nt_col_start, nt_col_end = find_boundaries_method2(
                seg_im= seg_im,
                stain_im= stain_im,
                half_width = half_width,
                edge_method = edge_method,
                edge_value = edge_value, 
                upper_range = upper_range,
                lower_range = lower_range
        )
        
        # set the result array
        bg_sub_profiles.append(bg_sub_profile)
        raw_profiles.append(raw_profile)
        nt_length = np.asarray(nt_length)
        nt_length_um = nt_length * pixel_size_um
        ventral_boundary_um = np.asarray(ventral_boundary_px) * pixel_size_um
        dorsal_boundary_um = np.asarray(dorsal_boundary_px) * pixel_size_um
        ventral_boundary_rel = ventral_boundary_um / nt_length_um
        dorsal_boundary_rel = dorsal_boundary_um / nt_length_um

        # get spline for length measurement
        spline = exchange.import_json(spline_file_path)[0]
        spline_length = operations.length_curve(spline) * pixel_size_um
        spline_increment = spline_length / stain_im.shape[0]

        
        n_slices = len(nt_length_um)
        target_name = targets[k]
        target = [target_name] * n_slices
        threshold_list = [edge_value] * n_slices
        slice_index = np.arange(start_slice, end_slice)
        slice_position_um = [i * spline_increment for i in slice_index]
        slice_position_rel_um = [i * spline_increment for i in range(n_slices)]
        param_measure_half_width = [half_width] * n_slices
        param_measure_edge_method = [edge_method] * n_slices
        param_measure_edge_value = [edge_value] * n_slices

        # put into a dataframe to save
        df = pd.DataFrame(
                {
                    'slice_index': slice_index,
                    'slice_position_um': slice_position_um,
                    'slice_position_rel_um': slice_position_rel_um,
                    'nt_length_um': nt_length_um,
                    'target': target,
                    'ventral_boundary_um': ventral_boundary_um,
                    'dorsal_boundary_um': dorsal_boundary_um,
                    'ventral_boundary_rel': ventral_boundary_rel,
                    'dorsal_boundary_rel': dorsal_boundary_rel,
                    'domain_edge_threshold': threshold_list,
                    'nt_start_column_px': nt_col_start,
                    'nt_end_column_px': nt_col_end,
                    'ventral_boundary_px': ventral_boundary_px,
                    'dorsal_boundary_px': dorsal_boundary_px,
                    'param_measure_half_width': param_measure_half_width,
                    'param_measure_edge_method': param_measure_edge_method,
                    'param_measure_edge_value': param_measure_edge_value
                }
            )
        channel_data.append(df)

        # if target_name == 'Olig2':
        #     for j, crop_im in enumerate(cropped_ims):
        #         file_name = row['root_path']+'/profiles/NT_crops/'+row['File'].replace('.h5','_')+target_name+'_slice_'+str(j+start_slice)+'.jpg'
        #         plt.ioff()
        #         fig, axes = plt.subplots()
        #         axes.imshow(crop_im)
        #         ventral = ventral_boundary_px[j]
        #         dorsal = dorsal_boundary_px[j]
        #         axes.plot([ventral,ventral], [0,crop_im.shape[0]], 'orange')
        #         axes.plot([dorsal,dorsal], [0,crop_im.shape[0]], 'green')
        #         plt.savefig(file_name)
        #         # tifffile.imwrite(file_name, ((crop_im/crop_im.max())*255).astype('uint8'))

    # concatenate the data frames for the channels and save
    all_data = pd.concat(channel_data, ignore_index=True)
    all_data.to_csv(table_output_path)

    # here add the saving for the aligned slices

def find_boundaries_method2( 
        seg_im: np.ndarray,
        stain_im: np.ndarray,
        half_width: float = 0.5,
        edge_method: BoundaryModes = BoundaryModes.PERCENT_MAX,
        edge_value: float = 0.1,
        upper_range: float = 1,
        lower_range: float = 0,):
    ventral_boundary = []
    dorsal_boundary = []
    nt_length = []
    nt_col_start = []
    nt_col_end = []
    bg_sub_profiles = []
    cropped_ims = []
    raw_profiles = []

    filtered_im = seg_im*stain_im
    bg_value = np.median(filtered_im[np.nonzero(filtered_im)])

    for nt_seg_slice, raw_slice in zip(seg_im, stain_im):

        # normalize the intensity by the number of nt pixels in the column
        raw_profile, raw_crop, col_min, col_max = method_2(raw_slice, nt_seg_slice, half_width)
        bg_sub_profile = (raw_profile - bg_value).clip(min=0, max=1)

        lim_up = round(upper_range*len(bg_sub_profile))
        lim_down = round(lower_range*len(bg_sub_profile))
        search_profile = bg_sub_profile[lim_down:lim_up]
        arr_min = np.min(search_profile)
        arr_max = np.max(search_profile)
        search_profile = (search_profile - arr_min) / (arr_max - arr_min)

        if isinstance(edge_method, str):
            edge_method = BoundaryModes(edge_method)
        if edge_method == BoundaryModes.DERIVATIVE:
            rising_edge, falling_edge = find_edge_derivative(search_profile)
        elif edge_method == BoundaryModes.PERCENT_MAX:
            rising_edge, falling_edge = find_edge_percent_max(search_profile, threshold=edge_value)
        elif edge_method == BoundaryModes.TANH:
            rising_edge, falling_edge = find_edge_fit(search_profile, threshold=edge_value)
        else:
            raise ValueError('unknown boundary mode')

        # store the values
        nt_length.append(col_max - col_min)
        ventral_boundary.append(rising_edge)
        dorsal_boundary.append(falling_edge)
        cropped_ims.append(raw_crop)
        bg_sub_profiles.append(bg_sub_profile)
        raw_profiles.append(raw_profile)
        nt_col_start.append(col_min)
        nt_col_end.append(col_max - 1)

    return nt_length, ventral_boundary, dorsal_boundary, bg_sub_profiles, raw_profiles, cropped_ims, nt_col_start, nt_col_end

def method_2(raw_slice, nt_seg_slice, half_width):
    intermediate_im = nt_seg_slice*raw_slice
    nt_label_im = label(nt_seg_slice)
    rp = regionprops(nt_label_im)

    nt_centroid = rp[0].centroid
    nt_bbox = rp[0].bbox

    restrict_width = half_width*(rp[0].axis_minor_length/2)
    # get the crop region
    row_min = int(nt_centroid[0] - restrict_width)
    row_max = int(nt_centroid[0] + restrict_width + 1)
    col_min = int(nt_bbox[1])
    col_max = int(nt_bbox[3])

    raw_crop = intermediate_im[nt_bbox[0]:nt_bbox[2], nt_bbox[1]:nt_bbox[3]]
    sample_region = intermediate_im[row_min:row_max, col_min:col_max]

    summed_profile = sample_region.sum(axis=0)

    # correct for columns with no nt labels
    n_sample_rows = nt_seg_slice[row_min:row_max, col_min:col_max].sum(axis=0)
    no_nt_indices = np.argwhere(n_sample_rows == 0)
    n_sample_rows[no_nt_indices] = 1
    summed_profile[no_nt_indices] = 0

    # normalize the intensity by the number of nt pixels in the column
    raw_profile = summed_profile / n_sample_rows
    return raw_profile, raw_crop, col_min, col_max
