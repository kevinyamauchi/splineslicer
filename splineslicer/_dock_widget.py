"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation

from .measure._qt_measure import QtMeasure
from .skeleton._qt_skeletonize import QtSkeletonize
from .slice.slice import slice_image_from_file
from .view.results_viewer import QtResultsViewer


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [
        (QtSkeletonize, {"name": "skeletonize"}),
        (slice_image_from_file, {"name": "slice image from file"}),
        (QtMeasure, {"name": "measure domains"}),
        (QtResultsViewer, {"name": "view results"})
    ]
