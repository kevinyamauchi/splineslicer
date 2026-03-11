from typing import Any, Dict, List, Optional, TYPE_CHECKING
import h5py
from magicgui import magicgui
from napari import Viewer
from napari.layers import Layer, Image, Shapes
import numpy as np
import pandas as pd
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QVBoxLayout, QWidget, QPushButton, QShortcut
from superqt.sliders import QLabeledSlider
import napari
from PyQt5.QtGui import QKeySequence

from ..view.results_viewer_utils import get_plane_coords
from splineslicer._reader import napari_get_reader

class QtBoundaryCorrection(QWidget):
    def __init__(self):
        super().__init__()

        self.image_slices: Optional[np.ndarray] = None
        self.results_table: Optional[pd.DataFrame] = None
        self.pixel_size_um: float = 5.79
        self.stain_channeL_names: List[str] = []
        self.min_slice: int = 0
        self.max_slice: int = 0
        self.draw_domain_boundaries = True
        self.current_channel_index: int = 0

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

        # create ventral slider
        self.ventral_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.ventral_slider.setRange(0, 99)
        self.ventral_slider.setSliderPosition(50)
        self.ventral_slider.setSingleStep(1)
        self.ventral_slider.setTickInterval(1)
        self.ventral_slider.valueChanged.connect(self._move_boundary_with_slider)

        # create dorsal slider
        self.dorsal_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.dorsal_slider.setRange(0, 99)
        self.dorsal_slider.setSliderPosition(50)
        self.dorsal_slider.setSingleStep(1)
        self.dorsal_slider.setTickInterval(1)
        self.dorsal_slider.valueChanged.connect(self._move_boundary_with_slider)

        # create None push button 
        self.Null_widget = QPushButton(text='No boundary')
        self.Null_widget.clicked.connect(self._remove_boundary_on_click)

        # create update push button
        self.update_widget = QPushButton(text='update')
        self.update_widget.clicked.connect(self._update_on_click)

        # create save push button
        self.save_widget = QPushButton(text='save')
        self.save_widget.clicked.connect(self._save_on_click)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.slice_slider)
        self.layout().addWidget(self.image_selector)
        self.layout().addWidget(self.image_widget)
        self.layout().addWidget(self.ventral_slider)
        self.layout().addWidget(self.dorsal_slider)
        self.layout().addWidget(self.update_widget)
        self.layout().addWidget(self.Null_widget)
        self.layout().addWidget(self.save_widget)

        self.shortcut_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_right.activated.connect(self.move_slider_right)
        self.shortcut_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_left.activated.connect(self.move_slider_left)

    def move_slider_right(self):
        current = self.slice_slider.value()
        max_value = self.slice_slider.maximum()
        if current < max_value:
            self.slice_slider.setValue(current + 1)

    def move_slider_left(self):
        current = self.slice_slider.value()
        min_value = self.slice_slider.minimum()
        if current > min_value:
            self.slice_slider.setValue(current - 1)

    def _remove_boundary_on_click(self, event=None):
        # avoid None type error if no data loaded
        if self.results_table is None:
            return

        current_slice_index = int(self.slice_slider.value())

        res, ind = np.unique(self.results_table["target"], return_index=True)
        target_names_result_table = res[np.argsort(ind)]
        target_name = target_names_result_table[self.current_channel_index]

        mask = (
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == current_slice_index)
        )
        if not np.any(mask):
            return

        nt_length_um = self.results_table.loc[mask, "nt_length_um"].iloc[0]
        self.results_table.loc[mask, "ventral_boundary_px"] = -1
        self.results_table.loc[mask, "dorsal_boundary_px"] = -1
        self.results_table.loc[mask, "ventral_boundary_um"] = -1
        self.results_table.loc[mask, "dorsal_boundary_um"] = -1
        self.results_table.loc[mask, "ventral_boundary_rel"] = -1 if nt_length_um == 0 else (-1 / nt_length_um)
        self.results_table.loc[mask, "dorsal_boundary_rel"] = -1 if nt_length_um == 0 else (-1 / nt_length_um)

    def _save_on_click(self, event=None):
        res_path = str(self.results_table_path)
        print('saving')
        self.results_table.to_csv(res_path[0:-4]+'_corrected.csv')
        print('saved')

    def _update_on_click(self, event=None):
        if self.results_table is None:
            return

        current_ventral_index = int(self.ventral_slider.value())
        current_dorsal_index = int(self.dorsal_slider.value())
        current_slice_index = int(self.slice_slider.value())

        res, ind = np.unique(self.results_table["target"], return_index=True)
        target_names_result_table = res[np.argsort(ind)]
        target_name = target_names_result_table[self.current_channel_index]

        mask = (
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == current_slice_index)
        )
        if not np.any(mask):
            return

        nt_length_um = self.results_table.loc[mask, "nt_length_um"].iloc[0]
        ventral_um = current_ventral_index * self.pixel_size_um
        dorsal_um = current_dorsal_index * self.pixel_size_um

        self.results_table.loc[mask, "ventral_boundary_px"] = current_ventral_index
        self.results_table.loc[mask, "dorsal_boundary_px"] = current_dorsal_index
        self.results_table.loc[mask, "ventral_boundary_um"] = ventral_um
        self.results_table.loc[mask, "dorsal_boundary_um"] = dorsal_um
        self.results_table.loc[mask, "ventral_boundary_rel"] = ventral_um / nt_length_um if nt_length_um != 0 else np.nan
        self.results_table.loc[mask, "dorsal_boundary_rel"] = dorsal_um / nt_length_um if nt_length_um != 0 else np.nan
        
    def _move_boundary_with_slider(self, event=None):
        current_ventral_index = int(self.ventral_slider.value()) #+ self.start_nt
        current_dorsal_index = int(self.dorsal_slider.value()) #+ self.start_nt
        self.nt_ventral_position.setValue(current_ventral_index)
        self.nt_dorsal_position.setValue(current_dorsal_index)   

    def _on_slider_moved(self, event=None):
        if self.results_table is None:
            return
        self.draw_at_current_slice_index()
        self.ventral_slider.setSliderPosition(int(self.neural_tube_ventral_boundary))
        self.dorsal_slider.setSliderPosition(int(self.neural_tube_dorsal_boundary))

    def draw_at_current_slice_index(self):
        current_slice_index = int(self.slice_slider.value())
        self.draw_at_slice_index(current_slice_index)
        # self._update_plot_slice_line(current_slice_index)

    def draw_at_slice_index(self, slice_index: int):
        self._get_boundaries(slice_index)
        self._update_image(slice_index)

    def _get_boundaries(self, slice_index: int):

        if self.results_table is None:
            return

        # update the vertical lines
        res, ind = np.unique(self.results_table["target"], return_index=True)

        # Sorting indices
        target_names_result_table = res[np.argsort(ind)]
        target_name = target_names_result_table[self.current_channel_index]

        # target_name = self.stain_channeL_names[self.current_channel_index]
        slice_row = self.results_table.loc[
            (self.results_table["target"] == target_name) &
            (self.results_table["slice_index"] == slice_index)
        ]
        if slice_row.empty:
            return
        
        # set the neural tube boundaries
        if self.draw_domain_boundaries:
            self.start_nt = slice_row["nt_start_column_px"].values[0]
            self.end_nt = slice_row["nt_end_column_px"].values[0]
            self.neural_tube_ventral_boundary =  slice_row["ventral_boundary_px"].values[0] #+ self.start_nt
            self.neural_tube_dorsal_boundary =  slice_row["dorsal_boundary_px"].values[0] #+ self.start_nt
            self.nt_ventral_position.setValue(self.neural_tube_ventral_boundary)
            self.nt_dorsal_position.setValue(self.neural_tube_dorsal_boundary)
        else:
            self.nt_ventral_position.setVisible(False)
            self.nt_dorsal_position.setVisible(False)

    def _update_image(self, slice_index: int):
        # offset the slice index since we only have a subset of the slices
        if self.image_slices is None:
            # if images haven't been set yet, do nothing
            return
        # self.min_slice = min(self.results_table["slice_index"].values)
        offset_slice_index = slice_index #- self.min_slice
        upper_bound = np.shape(self.image_slices)[2]//2 + 60
        lower_bound = np.shape(self.image_slices)[2]//2 - 60
        image_slice = self.image_slices[self.current_channel_index, offset_slice_index, lower_bound:upper_bound, self.start_nt:self.end_nt]

        # update the image slice
        self.image_widget.setImage(image_slice)

    def set_data(
            self,
            stain_image: np.ndarray,
            results_table: pd.DataFrame,
            pixel_size_um: float,
            results_table_path: str,
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
        self.results_table_path = results_table_path

        # update the plot-able columns
        # column_names = list(self.results_table.columns.values)
        # self.plot_column_selector.clear()
        # self.plot_column_selector.addItems(column_names)

        # add the image channels
        if stain_channel_names is not None:
            self.stain_channeL_names = stain_channel_names

        else:
            n_channels = stain_image.shape[0]
            self.stain_channeL_names = [
                f"channel {channel_index}" for channel_index in range(n_channels)
            ]

        # check if all stain channels are in the results table
        contains_channel = [
            np.any(results_table["target"] == channel)
            for channel in self.stain_channeL_names
        ]
        all_channels_in_results_table = np.all(contains_channel)

        self.draw_domain_boundaries = True    
        # if ("nt_start_column_px" in self.results_table.columns) and all_channels_in_results_table:
        #     self.draw_domain_boundaries = True
        # else:
        #     self.draw_domain_boundaries = False

        self.image_selector.clear()
        self.image_selector.addItems(self.stain_channeL_names)

        # refresh the selected channel index and redraw
        self._update_current_channel()

        self.setVisible(True)

    def _update_current_channel(self, event=None):
        if len(self.stain_channeL_names) == 0:
            # don't do anything if there aren't any channels
            return
        self.current_channel_index = self.image_selector.currentIndex()
        self.draw_at_current_slice_index()

class QtBoundaryReader(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()

        # store the viewer
        self._viewer = napari_viewer

        # make the load data widget
        self.load_data_widget = magicgui(
            self.load_data,
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
        self.image_slice_widget = QtBoundaryCorrection()
        self.image_slice_widget.setVisible(False)

        # make the layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.load_data_widget.native)
        self.layout().addWidget(self.image_slice_widget)

    def load_data(
        self,
        segmentation_channel: int = 3,
        sliced_image_path: str = "",
        results_table_path: str = "",
        pixel_size_um: float = 5.79
    ):
        self._load_slices(
            image_path=sliced_image_path,
            results_table_path=results_table_path,
            pixel_size_um=pixel_size_um
        )


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
            results_table_path=results_table_path,
            pixel_size_um=pixel_size_um
        )

    def _prepare_slices(self, image_path: str, results_table: pd.DataFrame):
        # hdf5_file = h5py.File(image_path)
        with h5py.File(image_path) as hdf5_file:
            stain_image = hdf5_file[list(hdf5_file.keys())[0]][:]

        if stain_image.ndim == 3:
            # make sure the image is 4D
            # (channel, slice, y, x)
            stain_image = np.expand_dims(stain_image, axis=0)

        # if no channels are present, make names with the channel index
        n_channels = stain_image.shape[0]
        

        stain_channel_names = [
            f"channel {channel_index}" for channel_index in range(n_channels)
        ]
        return stain_image, stain_channel_names

    def _load_spline(self, spline_path: str):
        """
        Load a spline from a file and set up the visualization with slicing plane and point.
        This method reads a spline file using napari's reader system, adds it as a layer
        to the viewer, and initializes the slicing interface with a plane and control point.
        Parameters
        ----------
        spline_path : str
            Path to the spline file to load.
        Raises
        ------
        ValueError
            If no suitable reader is found for the given spline file path.
        Notes
        -----
        This method performs the following operations:
        - Loads the spline data using napari's reader system
        - Adds the spline as a layer to the viewer
        - Extracts the spline model from the layer metadata
        - Creates and adds a slicing plane surface at the midpoint (0.5) with a size of 10
        - Connects the slice slider to update the plane position when changed
        - Adds a spherical point at the origin for slice point reference
        """
        
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
        plane_coords, faces, center_position, plane_normal = get_plane_coords(
            spline_model, 0.5, 10
        )
        values = np.ones(4)
        self._viewer.add_surface(data=(plane_coords, faces, values), name="slice plane")
        self.image_slice_widget.slice_slider.valueChanged.connect(self._update_slice_plane)

        # add the slicing point
        self._viewer.add_points(data=[[0, 0, 0]], name="slice point", shading="spherical")

    def _update_slice_plane(self, slice_coordinate):
        """
        Update the slice plane visualization based on the given slice coordinate.
        Parameters
        ----------
        slice_coordinate : float
            The coordinate position for the slice plane, scaled by 1/100.
        Returns
        -------
        None
        Notes
        -----
        This method retrieves the spline model from the viewer's metadata, calculates
        the plane coordinates and normal vector at the specified slice position, and
        updates the visualization layers ("slice plane" and "slice point") with the
        computed geometry and position data.
        """
        
        spline_model = self._viewer.layers["spline"].metadata["spline"]
        plane_coords, faces, center_position, plane_normal = get_plane_coords(
            spline_model, slice_coordinate / 100, 10
        )
        values = np.ones(4)
        self._viewer.layers["slice plane"].data = (plane_coords, faces, values)
        self._viewer.layers["slice point"].data = center_position

    def _load_raw_image(self, image_path: str):
        # load the image
        with h5py.File(image_path) as f:
            image = f[list(f.keys())[0]][:]

        # add the layer to the viewer
        self._viewer.add_image(
            image,
            name="raw image"
        )
