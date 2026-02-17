from typing import List

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
            skeleton_layer={'choices': self._get_labels_layers},
            call_button='curate skeleton layer'
        )
        self._viewer.layers.events.inserted.connect(
            self.layer_selection_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self.layer_selection_widget.reset_choices
        )

        # stop curating button
        self._end_curation_button = QPushButton('stop curating')
        self._end_curation_button.setVisible(False)
        self._end_curation_button.clicked.connect(self._on_end_curation_pressed)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.layer_selection_widget.native)
        self.layout().addWidget(self._end_curation_button)

    @property
    def curating(self) -> bool:
        """State variable set to true when curating"""
        return self.model.curating

    @curating.setter
    def curating(self, curating: bool):
        if curating:
            if self.model.skeleton_layer == '':
                self.model.curating = False
                raise RuntimeError('A skeleton layer must be selected to start curating')
            self._setup_curation()
        else:
            self._teardown_curation()
        self.model.curating = curating

    def _setup_curation(self):
        self._end_curation_button.setVisible(True)
        self.layer_selection_widget.native.setVisible(False)
        if self._viewer.dims.ndisplay != 3:
            self._viewer.dims.ndisplay = 3

    def _teardown_curation(self):
        self._end_curation_button.setVisible(False)
        self.layer_selection_widget.native.setVisible(True)

    def _on_end_curation_pressed(self, event):
        self.curating = False

    def _select_skeleton_layer(self, skeleton_layer: Labels):
        self.model.skeleton_layer = skeleton_layer.name
        self.curating = True

    def _get_labels_layers(self, combo_widget) -> List[Labels]:
        """Get a list of Labels layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Labels)]
