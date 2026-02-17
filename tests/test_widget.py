from splineslicer.measure._qt_measure import QtMeasure
from splineslicer.skeleton._qt_skeletonize import QtSkeletonize
from splineslicer.slice.slice import slice_image_from_file
from splineslicer.view.results_viewer import QtResultsViewer


def test_qt_measure(make_napari_viewer):
    """Test QtMeasure widget can be instantiated and added to viewer."""
    viewer = make_napari_viewer()
    widget = QtMeasure(viewer)

    viewer.window.add_dock_widget(widget, area='right')


def test_qt_skeletonize(make_napari_viewer):
    """Test QtSkeletonize widget can be instantiated and added to viewer."""
    viewer = make_napari_viewer()
    widget = QtSkeletonize(viewer)

    viewer.window.add_dock_widget(widget, area='right')


def test_slice_image_from_file(make_napari_viewer):
    """Test slice_image_from_file magicgui widget can be instantiated and added to viewer."""
    viewer = make_napari_viewer()
    # slice_image_from_file is a magic_factory, so we call it to get the widget
    widget = slice_image_from_file()

    viewer.window.add_dock_widget(widget, area='right')


def test_qt_results_viewer(make_napari_viewer):
    """Test QtResultsViewer widget can be instantiated and added to viewer."""
    viewer = make_napari_viewer()
    widget = QtResultsViewer(viewer)

    viewer.window.add_dock_widget(widget, area='right')

