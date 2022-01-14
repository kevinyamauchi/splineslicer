import h5py
import numpy as np
from napari.types import ImageData


def load_ilastik_predictions(filepath: str) -> ImageData:
    with h5py.File(filepath, 'r') as f:
        im = f['exported_data'][:]

    return [(im, {}, 'image')]
