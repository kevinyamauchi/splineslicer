from typing import List

import magicgui
from napari.layers import Image, Labels
from qtpy.QtWidgets import QVBoxLayout, QWidget
from superqt.collapsible import QCollapsible

from .align_slices import binarize_per_slice, align_mask_and_stain_from_layer, align_stack_from_layer
from .measure_boundaries import find_boundaries_from_layer


class QtMeasure(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()

        # store the viewer
        self._viewer = napari_viewer

        # make the binarize section
        self._setup_binarize_widget()

        # make the align section
        self._setup_align_widget()

        # make the measure section
        self._setup_measure_widget()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._binarize_section)
        self.layout().addWidget(self._align_section)
        self.layout().addWidget(self._measure_section)

    def _setup_binarize_widget(self):
        self._binarize_section = QCollapsible(title='1. binarize', parent=self)
        self._binarize_widget = magicgui.magicgui(
            binarize_per_slice,
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

    def _setup_align_widget(self):
        self._align_section = QCollapsible(title='2. align images', parent=self)
        self._align_widget = magicgui.magicgui(
            align_mask_and_stain_from_layer,
            mask_layer={'choices': self._get_image_layers},
            stain_layer={'choices': self._get_image_layers},
            call_button='align layers'
        )
        self._align_section.addWidget(self._align_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._align_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._align_widget.reset_choices
        )
        self._align_widget.reset_choices()

    def _setup_measure_widget(self):
        self._measure_section = QCollapsible(title='3. measure boundaries', parent=self)
        self._measure_widget = magicgui.magicgui(
            find_boundaries_from_layer,
            segmentation_layer={'choices': self._get_aligned_layer},
            stain_layer={'choices': self._get_aligned_layer},
            spline_file_path={'widget_type': 'FileEdit', 'mode': 'r', 'filter': '*.json'},
            table_output_path={'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.csv'},
            aligned_slices_output_path={'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.h5'},
            call_button='measure image'
        )
        self._measure_section.addWidget(self._measure_widget.native)
        self._viewer.layers.events.inserted.connect(
            self._measure_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._measure_widget.reset_choices
        )
        self._measure_widget.reset_choices()

    def _get_image_layers(self, combo_widget) -> List[Image]:
        """Get a list of Image layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]

    def _get_aligned_layer(self, combo_widget) -> List[Image]:
        im_layers = [layer for layer in self._viewer.layers if isinstance(layer, Image)]

        return [layer for layer in im_layers if 'rotations' in layer.metadata]