from typing import Any, Dict, List, Optional, TYPE_CHECKING

import h5py
from magicgui import magicgui
import napari
from napari.layers import Layer
import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget
from skimage.transform import rotate
from superqt.sliders import QLabeledSlider

from ..measure.align_slices import rotate_stack
from .._reader import napari_get_reader
from .results_viewer_utils import get_plane_coords


class QtImageSliceWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.image_slices: Optional[np.ndarray] = None
        self.results_table: Optional[pd.DataFrame] = None
        self.pixel_size_um: float = 5.79
        self.stain_channeL_names: List[str] = []
        self.min_slice: int = 0
        self.max_slice: int = 0

        # create the slider
        self.slice_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 99)
        self.slice_slider.setSliderPosition(50)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.setTickInterval(1)
        self.slice_slider.valueChanged.connect(self._on_slider_moved)

        # create the stain channel selection box
        self.image_selector = QComboBox()
        self.image_selector.currentIndexChanged.connect(self._update_current_channel)

        # create the image
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.image_widget = pg.ImageView(parent=self)
        self.nt_ventral_position = pg.InfiniteLine(
            pen={'color': "#1b9e77", "width": 2},
            angle=90,
            movable=False
        )
        self.image_widget.addItem(self.nt_ventral_position)
        self.nt_dorsal_position = pg.InfiniteLine(
            pen={'color': "#1b9e77", "width": 2},
            angle=90,
            movable=False
        )
        self.image_widget.addItem(self.nt_dorsal_position)

        # create the plot
        self.plot_column_selector = QComboBox()
        self.plot_column_selector.currentIndexChanged.connect(self._update_plot)
        self.plot = pg.PlotWidget(parent=self)
        self.plot_slice_line = pg.InfiniteLine(
            angle=90,
            movable=False
        )
        self.plot.addItem(self.plot_slice_line)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.slice_slider)
        self.layout().addWidget(self.image_selector)
        self.layout().addWidget(self.image_widget)
        self.layout().addWidget(self.plot_column_selector)
        self.layout().addWidget(self.plot)

    def _on_slider_moved(self, event=None):
        self.draw_at_current_slice_index()

    def draw_at_current_slice_index(self):
        current_slice_index = int(self.slice_slider.value())
        self.draw_at_slice_index(current_slice_index)
        self._update_plot_slice_line(current_slice_index)

    def draw_at_slice_index(self, slice_index: int):
        self._update_image(slice_index)

    def _update_image(self, slice_index: int):
        # offset the slice index since we only have a subset of the slices
        if self.image_slices is None:
            # if images haven't been set yet, do nothing
            return

        offset_slice_index = slice_index - self.min_slice
        image_slice = self.image_slices[self.current_channel_index, offset_slice_index, ...]

        # update the image slice
        self.image_widget.setImage(image_slice)

        # update the vertical lines
        target_name = self.stain_channeL_names[self.current_channel_index]
        slice_row = self.results_table.loc[
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == slice_index)
        ]

        # set the neural tube boundaries
        neural_tube_ventral_boundary = slice_row["nt_start_column_px"].values[0]
        neural_tube_dorsal_boundary = slice_row["nt_end_column_px"].values[0]
        self.nt_ventral_position.setValue(neural_tube_ventral_boundary)
        self.nt_dorsal_position.setValue(neural_tube_dorsal_boundary)

    def _update_plot(self, event=None):
        self.plot.clear()

        # get the column
        current_target = self.stain_channeL_names[self.current_channel_index]
        target_measurements = self.results_table.loc[self.results_table["target"] == current_target]
        column_to_plot = self.plot_column_selector.currentText()
        y_data = target_measurements[column_to_plot].values
        x_data = target_measurements["slice_index"].values

        self.plot.plot(x_data, y_data)
        self.plot.addItem(self.plot_slice_line)
        current_slice_index = int(self.slice_slider.value())
        self.plot_slice_line.setValue(current_slice_index)

        # set the labels
        axis_parameters = {
            "bottom": pg.AxisItem(orientation="bottom", text="slice index"),
            "left": pg.AxisItem(orientation="left", text=column_to_plot)
        }
        self.plot.setAxisItems(axis_parameters)

    def _update_plot_slice_line(self, slice_index):
        self.plot_slice_line.setValue(slice_index)

    def set_data(
            self,
            stain_image: np.ndarray,
            results_table: pd.DataFrame,
            pixel_size_um: float,
            stain_channel_names: Optional[List[str]]=None,
    ):
        if stain_image.ndim == 3:
            # make sure the image is 4D
            # (channel, slice, y, x)
            stain_image = np.expand_dims(stain_image, axis=0)
        # set the range slider range
        self.min_slice = results_table["slice_index"].min()
        self.max_slice = results_table["slice_index"].max()
        self.slice_slider.setRange(self.min_slice, self.max_slice)

        self.pixel_size_um = pixel_size_um
        self.image_slices = stain_image
        self.results_table = results_table

        # add the image channels
        if stain_channel_names is not None:
            self.stain_channeL_names = stain_channel_names

        else:
            n_channels = stain_image.shape[0]
            self.stain_channeL_names = [
                f"channel {channel_index}" for channel_index in range(n_channels)
            ]
        self.image_selector.clear()
        self.image_selector.addItems(self.stain_channeL_names)

        # update the plot-able columns
        column_names = list(self.results_table.columns.values)
        self.plot_column_selector.clear()
        self.plot_column_selector.addItems(column_names)

        # refresh the selected channel index and redraw
        self._update_current_channel()

        self.setVisible(True)

    def _update_current_channel(self, event=None):
        if len(self.stain_channeL_names) == 0:
            # don't do anything if there aren't any channels
            return
        self.current_channel_index = self.image_selector.currentIndex()
        self.draw_at_current_slice_index()


class QtResultsViewer(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        # store the viewer
        self._viewer = napari_viewer

        # make the load data widget
        self.load_data_widget = magicgui(
            self.load_data,
            raw_image_path={
                'label': 'raw image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            },
            nt_segmentation_path={
                'label': 'segmentation image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            },
            spline_path={
                'label': 'spline path',
                'widget_type': 'FileEdit',
                'mode': 'r',
                'filter': '*.json'
            },
            sliced_image_path={
                'label': 'sliced image path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.h5'
            },
            results_table_path={
                'label': 'results table path',
                'widget_type': 'FileEdit', 'mode': 'r',
                'filter': '*.csv'
            }
        )

        self.image_slice_widget = QtImageSliceWidget()
        self.image_slice_widget.setVisible(False)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self.image_slice_widget)

    def load_data(
        self,
        raw_image_path: str,
        raw_image_key: str,
        nt_segmentation_path: str,
        spline_path: str,
        sliced_image_path: str,
        results_table_path: str,
        pixel_size_um: float = 5.79
    ):
        self._load_raw_image(image_path=raw_image_path, image_key=raw_image_key)
        self._load_neural_tube_segmentation(image_path=nt_segmentation_path)
        self._load_spline(spline_path=spline_path)
        self._load_slices(
            image_path=sliced_image_path,
            results_table_path=results_table_path,
            pixel_size_um=pixel_size_um
        )

    def _load_raw_image(self, image_path: str, image_key: str):
        # load the image
        with h5py.File(image_path) as f:
            image = f[image_key][:]

        # add the layer to the viewer
        self._viewer.add_image(
            image,
            name="raw image"
        )

    def _load_neural_tube_segmentation(self, image_path: str):
        # get the reader
        reader = napari_get_reader(image_path)
        if reader is None:
            raise ValueError(f"no reader found for {image_path}")

        # load the layer data
        data, metadata, layer_type = reader(image_path)[0]

        # add the layer name and set colormap to the metadata
        metadata.update(
            {
                "name": "nt segmentation",
                "colormap": "bop blue",
                "opacity": 0.7
            }
        )

        # take only prediction channel 1 (neural tube)
        data = data[1, ...]

        # add the layer to the viewer
        self._viewer.add_layer(Layer.create(data, metadata, layer_type))

    def _load_spline(self, spline_path: str):
        # get the reader
        reader = napari_get_reader(spline_path)
        if reader is None:
            raise ValueError(f"no reader found for {spline_path}")

        # load the layer data
        layer_data = reader(spline_path)[0]

        # add the layer to the viewer
        self._viewer.add_layer(Layer.create(*layer_data))

        # add the slicing plane
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, _, _ = get_plane_coords(
            spline_model, 0.5, 10
        )
        values = np.ones(4)
        self._viewer.add_surface(data=(plane_coords, faces, values), name="slice plane")
        self.image_slice_widget.slice_slider.valueChanged.connect(self._update_slice_plane)

        # add the slicing point
        self._viewer.add_points(data=[[0, 0, 0]], name="slice point", shading="spherical")

    def _load_slices(self, image_path: str, results_table_path: str, pixel_size_um: float):
        # load the data
        results_table = pd.read_csv(results_table_path)

        stain_image, stain_channels = self._prepare_slices(
            image_path=image_path,
            results_table=results_table
        )

        self.image_slice_widget.set_data(
            stain_image=stain_image,
            stain_channel_names=stain_channels,
            results_table=results_table,
            pixel_size_um=pixel_size_um
        )

    def _prepare_slices(self, image_path: str, results_table: pd.DataFrame):
        with h5py.File(image_path) as f:
            dataset_keys = list(f.keys())
            if "sliced_stack" in dataset_keys:
                stain_image = f["sliced_stack"][:]
                stain_metadata = {}
                stain_image_aligned = False
            else:
                segmentation_image = f["aligned_segmentation_image"][:]
                segmentation_metadata = dict(f["aligned_segmentation_image"].attrs.items())
                stain_image = f["aligned_stain_image"][:]
                stain_metadata = dict(f["aligned_stain_image"].attrs.items())

                # no need to align datasets
                stain_image_aligned = True

        if stain_image.ndim == 3:
            # make sure the image is 4D
            # (channel, slice, y, x)
            stain_image = np.expand_dims(stain_image, axis=0)

        if "channel_names" in stain_metadata:
            # get the
            stain_channel_names = stain_metadata["channel_names"]
        else:
            # if no channels are present, make names with the channel index
            n_channels = stain_image.shape[0]
            stain_channel_names = [
                f"channel {channel_index}" for channel_index in range(n_channels)
            ]

        if stain_image_aligned:
            return stain_image, stain_channel_names
        else:
            # if the images are not aligned, align them
            first_target = results_table["target"].unique()[0]
            rotations = results_table.loc[
                (results_table["target"] == first_target)
            ]["rotation"].values

            rotated_stain_image = []
            for stain_channel_image in stain_image:
                rotated_stain_image.append(
                    rotate_stack(
                        stain_channel_image,
                        rotations
                    )
                )
            return np.stack(rotated_stain_image), stain_channel_names

    def _update_slice_plane(self, slice_coordinate):
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, center_position, plane_normal = get_plane_coords(
            spline_model, slice_coordinate / 100, 10
        )
        values = np.ones(4)
        self._viewer.layers["slice plane"].data = (plane_coords, faces, values)
        self._viewer.layers["slice point"].data = center_position




