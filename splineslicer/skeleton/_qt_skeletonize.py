from typing import List

import magicgui
from napari.layers import Image, Labels
from qtpy.QtWidgets import QVBoxLayout, QWidget
from superqt.collapsible import QCollapsible

from ._qt_skeleton_pruner import QtSkeletonSelector
from .binarize import _binarize_image_mg
from .skeletonize import _make_skeleton_mg
from ..spline.spline_utils import fit_spline_to_skeleton_layer


class QtSkeletonize(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        # store the viewer
        self._viewer = napari_viewer
        self.skeleton = {}
        self.summary = {}

        # make the binarize section
        self._binarize_section = QCollapsible(title='1. binarize', parent=self)
        self._binarize_widget = magicgui.magicgui(
            _binarize_image_mg,
            im_layer={'choices': self._get_image_layers},
            call_button='binarize image'
        )
        self._binarize_section.addWidget(self._binarize_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._binarize_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._binarize_widget.reset_choices
        )
        self._binarize_widget.reset_choices()

        # make the skeletonize section
        self._skeletonize_section = QCollapsible(title='2. skeletonize', parent=self)
        self._skeletonize_widget = magicgui.magicgui(
            _make_skeleton_mg,
            im_layer={'choices': self._get_image_layers},
            call_button='skeletonize image'
        )
        self._skeletonize_section.addWidget(self._skeletonize_widget.native)
        self._skeletonize_widget.called.connect(self._on_skeletonize)
        self._viewer.layers.events.inserted.connect(
            self._skeletonize_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._skeletonize_widget.reset_choices
        )

        # make the curate section
        self._curate_section = QCollapsible(title=' 3. curate skeleton', parent=self)
        self._curate_widget = QtSkeletonSelector(napari_viewer=napari_viewer)
        self._curate_section.addWidget(self._curate_widget)

        # make the fit section
        self._fit_section = QCollapsible(title='4. fit spline', parent=self)
        self._fit_widget = magicgui.magicgui(
            fit_spline_to_skeleton_layer,
            skeleton_layer={'choices': self._get_labels_layers},
            output_path={'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.json'},
            call_button='fit spline'
        )
        self._fit_section.addWidget(self._fit_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._fit_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._fit_widget.reset_choices
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._binarize_section)
        self.layout().addWidget(self._skeletonize_section)
        self.layout().addWidget(self._curate_section)
        self.layout().addWidget(self._fit_section)

    def _on_skeletonize(self, event):
        # get the results from the event object
        skeletononized_im, skeleton_obj, summary = event.value

        # store the skeleton data
        self.skeleton.update({'skeletonize': skeleton_obj})
        self.summary.update({'skeletonize': summary})

        print(skeletononized_im)

        # make the layer with the skeleton
        self._viewer.add_labels(
            skeletononized_im,
            name="skeletonize",
            properties=summary,
            metadata={'skan_obj': skeleton_obj}
        )

    def _get_image_layers(self, combo_widget) -> List[Image]:
        """Get a list of Image layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]

    def _get_labels_layers(self, combo_widget) -> List[Labels]:
        """Get a list of Labels layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Labels)]
