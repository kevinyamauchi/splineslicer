from typing import Any, Dict, List

import h5py
from napari.layers import Image
import numpy as np


def _write_dataset_from_array(
    file_handle: h5py.File,
    dataset_name: str,
    dataset_array: np.ndarray,
    compression: str = 'gzip',
):
    file_handle.create_dataset(
        dataset_name,
        dataset_array.shape,
        dtype=dataset_array.dtype,
        data=dataset_array,
        compression=compression,
    )


def _write_dataset_from_dict(
    file_handle: h5py.File,
    dataset_name: str,
    dataset: Dict[str, Any],
    compression: str = 'gzip',
):
    dataset_array = dataset['data']
    dset = file_handle.create_dataset(
        dataset_name,
        dataset_array.shape,
        dtype=dataset_array.dtype,
        data=dataset_array,
        compression=compression,
    )

    dataset_attrs = dataset.get('attrs', None)
    if dataset_attrs is not None:
        for k, v in dataset_attrs.items():
            dset.attrs[k] = v


def write_multi_dataset_hdf(
    file_path: str, compression: str = 'gzip', **kwargs
):
    if len(kwargs) == 0:
        raise ValueError('Must supply at least one dataset as a kwarg')
    with h5py.File(file_path, 'w') as f:
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                _write_dataset_from_array(
                    file_handle=f,
                    dataset_name=k,
                    dataset_array=v,
                    compression=compression,
                )
            else:
                _write_dataset_from_dict(
                    file_handle=f,
                    dataset_name=k,
                    dataset=v,
                    compression=compression,
                )


def save_aligned_slices(
    file_path: str,
    segmentation_layer: Image,
    stain_layer: Image,
    stain_channel_names: List[str],
    compression: str = "gzip"
) -> None:
    # make the segmentation dataset
    segmentation_metadata =  segmentation_layer.metadata.get(
            "binarize_settings", {}
    )

    segmentation_metadata.update(
        segmentation_layer.metadata.get(
                "rotation_settings", {}
            )
    )
    aligned_segmentation_dataset = {
        "data": segmentation_layer.data,
        "attrs": segmentation_metadata
    }

    # make the stain dataset
    aligned_stain_dataset = {
        "data": stain_layer.data,
        "attrs": {"channel_names": stain_channel_names}
    }

    write_multi_dataset_hdf(
        file_path=file_path,
        compression=compression,
        aligned_segmentation_image=aligned_segmentation_dataset,
        aligned_stain_image=aligned_stain_dataset
    )
