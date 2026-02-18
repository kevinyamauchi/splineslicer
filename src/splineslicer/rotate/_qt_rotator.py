from typing import Any, Dict, List, Optional, TYPE_CHECKING
import h5py
from magicgui import magicgui
import napari
from napari.layers import Layer, Image, Shapes
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget, QPushButton
from superqt.sliders import QLabeledSlider
from skimage.transform import rotate
import splineslicer
from superqt.collapsible import QCollapsible
from .rotations_utils import new_align_rotate

class QtUpdatedRotation(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        self.load_data_widget = magicgui(
            self.load_data,
            sliced_image_path={
                'label': 'sliced image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            },
            segmented_image_path={
                'label': 'segmented image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            }
        )

        self.image_slices: Optional[np.ndarray] = None
        self.results_table: Optional[pd.DataFrame] = None
        self.pixel_size_um: float = 5.79
        self.stain_channeL_names: List[str] = []
        self.min_slice: int = 0
        self.max_slice: int = 0
        self.draw_domain_boundaries = True
        self.current_channel_index: int = 0

        # make the binarize section
        self._binarize_section = QCollapsible(title='1. binarize', parent=self)
        self._binarize_widget = magicgui(
            self._binarize_segmentation,
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

        # make rotation widget
        self._rotate_section = QCollapsible(title='2. align images', parent=self)
        self._rotate_widget = magicgui(
            new_align_rotate,
            mask_layer={'choices': self._get_image_layers},
            stain_layer={'choices': self._get_image_layers},
            NT_segmentation_index={"choices": [0, 1, 2]},
            background_index={"choices": [0, 1, 2]},
            call_button='rotate and align layers'
        )
        self._rotate_section.addWidget(self._rotate_widget.native)

        self._viewer.layers.events.inserted.connect(
            self._rotate_widget.reset_choices
        )
        self._viewer.layers.events.removed.connect(
            self._rotate_widget.reset_choices
        )
        self._rotate_widget.reset_choices()

        # create the saving button to save rotation
        self.save_rotation_widget = magicgui(
            self._save_rotation,
            output_path={
                'label': 'select saving path',
                'widget_type': 'FileEdit', 'mode': 'd',
                'filter': ''
            },
            call_button='save rotation'
        )

        # set the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self._binarize_section)
        self.layout().addWidget(self._rotate_section)
        self.layout().addWidget(self.save_rotation_widget.native)

    def _binarize_segmentation(self, threshold: float = 0.5, closing_size: int = 3) -> "napari.types.LayerDataTuple":
        im_seg = self._viewer.layers.selection.active.data
        binarized_im = np.zeros((im_seg.shape))
        for i, chan in enumerate(im_seg):
            binarized_im[i,...] = splineslicer.skeleton.binarize.binarize_image(im = im_seg, channel=i, threshold=threshold)
        layer_type = 'image'
        metadata = {
            'name': 'binarized_image',
            'colormap': 'blue'
        }
        # self._viewer._add_layer_from_data(binarized_im, metadata, binarized_im)
        return (binarized_im, metadata, layer_type)

    def load_data(
        self,
        sliced_image_path: str = "",
        segmented_image_path: str = ""
    ) -> List[napari.types.LayerDataTuple]: 
        # load the image containing the data
        hdf5_file = h5py.File(sliced_image_path, 'r+')
        image_slices = hdf5_file[list(hdf5_file.keys())[0]]
        layer_type ='image'
        metadata = {
            'name': 'sliced_image',
            'colormap': 'gray'
        }

        hdf5_file = h5py.File(segmented_image_path, 'r+')
        segmented_image_slices = hdf5_file[list(hdf5_file.keys())[0]]
        segmented_image_layer_type ='image'
        segmented_image_metadata = {
            'name': 'segmented_image',
            'colormap': 'gray'
        }

        return [(image_slices, metadata, layer_type), (segmented_image_slices, segmented_image_metadata, segmented_image_layer_type)]

    def _get_image_layers(self, combo_widget) -> List[Image]:
        """Get a list of Image layers in the viewer"""
        return [layer for layer in self._viewer.layers if isinstance(layer, Image)]
    
    def _save_rotation(self,
        output_path: str = ""):
        loaded_file = self.load_data_widget.sliced_image_path
        # print(re.match("([\w\d_\.]+\.[\w\d]+)[^\\]", str(loaded_file.value)))
        fname = str(loaded_file.value).split('\\')[-1]
        print(fname)
        output_path=str(output_path)+'/'+fname.replace('.h5','_after_rotation.h5')
        print(output_path)
        with h5py.File(output_path,'w') as f_out:
            f_out.create_dataset(
            'sliced_stack_rotated',
            self._viewer.layers.selection.active.data.shape,
            data=self._viewer.layers.selection.active.data,
            compression='gzip'
        )
        
class QtRotationWidget(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self._viewer = napari_viewer

        self.check_angle = 0
        self.make_copy = 0

        # create the rotation slider
        self.rotate_slider = QLabeledSlider(Qt.Orientation.Vertical)
        self.rotate_slider.setRange(-180, 180)
        self.rotate_slider.setSliderPosition(0)
        self.rotate_slider.setSingleStep(1)
        self.rotate_slider.setTickInterval(1)
        self.rotate_slider.valueChanged.connect(self._on_rotate_slider_moved)

        # reset the slider to 0 upon changing the slice to avoid unwanted rotations
        self.current_step = self._viewer.dims.current_step[1]
        self._viewer.dims.events.current_step.connect(self._update_slider)

        # create apply push button 
        self.apply_button = QPushButton(text='apply rotation')
        self.apply_button.clicked.connect(self._apply_button)

        # create save button
        self.save_rotation_widget = magicgui(
            self._save_rotation,
            output_path={'widget_type': 'FileEdit', 'mode': 'w', 'filter': '*.h5'},
            call_button='save rotation'
        )

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.rotate_slider)
        self.layout().addWidget(self.apply_button)
        self.layout().addWidget(self.save_rotation_widget.native)


    def _update_slider(self, event=None):
        self.rotate_slider.setSliderPosition(0)
        self.make_copy = 0
        self.check_angle = 0

    def _on_rotate_slider_moved(self, event=None):
        self.check_angle = int(self.rotate_slider.value())
        #is of form (2,50,150,150) where (channel, slice, x, y) and I want to change the rotation for all channel at one slice
        if self.make_copy == 0:
            self.ind = self._viewer.dims.current_step[1]
            self.current_slice = self._viewer.layers.selection.active.data[:,self.ind,...].copy()
            self.make_copy = 1
        
        for i, chan in enumerate(self._viewer.layers.selection.active.data[:,self.ind,...]):
            if i == self._viewer.layers.selection.active.data[:,self.ind,...].shape[0]-1:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self.current_slice[i,...], self.check_angle, order = 0, resize=False)
            else:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self.current_slice[i,...], self.check_angle, order = 3, resize=False)
        self._viewer.layers.selection.active.refresh()

    def _apply_button(self, event=None):
        self.current_slice = self._viewer.layers.selection.active.data[:,self.ind,...].copy()
        for i, chan in enumerate(self._viewer.layers.selection.active.data[:,self.ind,...]):
            if i == self._viewer.layers.selection.active.data[:,self.ind,...].shape[0]-1:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self._viewer.layers.selection.active.data[i,self.ind,...], self.check_angle, order = 0, resize=False)
            else:
                self._viewer.layers.selection.active.data[i,self.ind,...] = rotate(self._viewer.layers.selection.active.data[i,self.ind,...], self.check_angle, order = 3, resize=False)
        self.check_angle = 0

    def _save_rotation(self,
        output_path: str = ""):
        with h5py.File(output_path,'w') as f_out:
            f_out.create_dataset(
            'sliced_stack_rotated',
            self._viewer.layers.selection.active.data.shape,
            data=self._viewer.layers.selection.active.data,
            compression='gzip'
        )
        