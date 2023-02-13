from typing import List

from geomdl import exchange
from napari.types import ShapesData

from ..spline.spline_utils import get_spline_points


def load_spline_geomdl(filepath: str) -> List[ShapesData]:
    spline = exchange.import_json(filepath)[0]
    spline_points = get_spline_points(
        spline=spline,
        increment=0.01
    )

    return [(
        spline_points,
        {
            'shape_type': 'path',
            'edge_width': 0.5,
            'edge_color': 'magenta',
            'blending': 'translucent_no_depth',
            'metadata': {'spline': spline}
        },
        'shapes'
    )]