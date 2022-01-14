from napari.layers import Labels
import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget
from superqt.collapsible import QCollapsible

from .skeleton_pruner import SkeletonPruner


class QtSkeletonSelector(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        self.model = SkeletonPruner(viewer=napari_viewer)

        self.layer_selection_widget = magicgui.magicgui(
            self._select_skeleton_layer,
            call_button='select skeleton layer'
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._binarize_section)

    def _select_skeleton_layer(self, skeleton_layer: Labels):
        self.model.skeleton_layer = skeleton_layer.name