import magicgui
from qtpy.QtWidgets import QLabel, QVBoxLayout, QPushButton, QWidget
from superqt.collapsible import QCollapsible

from .binarize import _binarize_image_mg
from .skeletonize import make_skeleton
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
            make_skeleton,
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
        self._fit_section = QCollapsible(title='3. fit spline', parent=self)
        self._fit_widget = magicgui.magicgui(
            fit_spline_to_skeleton_layer,
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

        # make the export section
        self._export_section = QCollapsible(title='4. save', parent=self)
        self._export_section.addWidget(QLabel('hi'))

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self._binarize_section)
        self.layout().addWidget(self._skeletonize_section)
        self.layout().addWidget(self._fit_section)
        self.layout().addWidget(self._export_section)

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

