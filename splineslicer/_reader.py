"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the ``napari_get_reader`` hook specification, (to create
a reader plugin) but your plugin may choose to implement any of the hook
specifications offered by napari.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below accordingly.  For complete documentation see:
https://napari.org/docs/dev/plugins/for_plugin_developers.html
"""
from napari_plugin_engine import napari_hook_implementation

from .io.ilastik import load_ilastik_predictions, load_aligned
from .io.spline import load_spline_geomdl


@napari_hook_implementation
def napari_get_reader(path):
    """A basic implementation of the napari_get_reader hook specification.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    if not isinstance(path, str):
        path = path.as_posix()

    if "_Probabilities.h5" in path:
        # ilastik probability images end with _Probabilities.h5
        return load_ilastik_predictions
    elif "Probabilities Stage 2.h5" in path:
        # ilastik autocontext segs
        return load_ilastik_predictions
    elif ".h5" in path:
        return load_aligned
    elif ".json" in path:
        # geomdl spline
        return load_spline_geomdl
    else:
        return None
