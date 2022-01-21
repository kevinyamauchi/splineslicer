from typing import Any, Dict, Union

import numpy as np
import pandas as pd


def make_points_data(df: Union[pd.DataFrame, Dict[str, Any]]):
    coords_start = np.column_stack(
        (
            df['image-coord-src-0'],
            df['image-coord-src-1'],
            df['image-coord-src-2'],

        )
    )
    coords_end = np.column_stack(
        (
            df['image-coord-dst-0'],
            df['image-coord-dst-1'],
            df['image-coord-dst-2'],

        )
    )
    coords = np.concatenate((coords_start, coords_end))

    first_position = []
    second_position = []
    for k_value, f_value in zip(df['keep'], df['flip']):
        if bool(k_value) is not True:
            first_position.append('hide')
            second_position.append('hide')
        elif bool(f_value) is True:
            first_position.append('end')
            second_position.append('start')
        else:
            first_position.append('start')
            second_position.append('end')
    position = np.concatenate([first_position, second_position])
    selected = np.concatenate([df['selected'], df['selected']])
    properties = {'position': position, 'selected': selected}

    return coords, properties
