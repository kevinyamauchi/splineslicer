import pandas as pd
from napari.layers import Labels
import numpy as np

from .skeleton_utils import make_points_data


class SkeletonPruner():
    def __init__(self, viewer):
        self._viewer = viewer

        self._skeleton_layer = ''
        self._points_layer = None
        self._curating = False
        self._selected_branches = set()

    @property
    def skeleton_layer(self):
        return self._skeleton_layer

    @skeleton_layer.setter
    def skeleton_layer(self, skeleton_layer):
        self._initialize_skeleton_layer(
            new_skeleton_layer=skeleton_layer,
            prev_skeleton_layer=self._skeleton_layer
        )

        self._skeleton_layer = skeleton_layer

    @property
    def curating(self) -> bool:
        return self._curating

    @curating.setter
    def curating(self, curating: bool):
        if curating is True:
            self._initialize_points_layer()

            skeleton_layer = self._viewer.layers[self.skeleton_layer]
            skeleton_layer.events.properties.connect(self._on_properties_update)
            self._setup_curated_skeleton_layer()
            self._viewer.layers.selection = [skeleton_layer]

            # make only the skeleton layers visible
            layers_to_view = [
                skeleton_layer,
                self._curated_skeleton_layer,
                self._points_layer
            ]
            for layer in self._viewer.layers:
                if layer not in layers_to_view:
                    layer.visible = False

        else:
            self._cleanup_points_layer()
            self._cleanup_curated_skeleton_layer()
        self._curating = curating

    def _initialize_skeleton_layer(self, new_skeleton_layer: str, prev_skeleton_layer: str):
        """set up the skeleton layer"""
        if prev_skeleton_layer != '':
            # clean up the old skeleton layer
            self._cleanup_skeleton_layer(prev_skeleton_layer)
        layer = self._viewer.layers[new_skeleton_layer]
        self._connect_layer_events(layer)

        # add the required column to the layer properties
        if 'keep' not in layer.properties:
            n_branches = len(layer.properties['skeleton-id'])
            layer.properties['keep'] = np.zeros((n_branches,), dtype=bool)
        if 'flip' not in layer.properties:
            n_branches = len(layer.properties['skeleton-id'])
            layer.properties['flip'] = np.zeros((n_branches,), dtype=bool)
        if 'selected' not in layer.properties:
            n_branches = len(layer.properties['skeleton-id'])
            layer.properties['selected'] = np.zeros((n_branches,), dtype=bool)

    def _cleanup_skeleton_layer(self, skeleton_layer):
        """clean up a skeleton layer after deselecting it"""
        layer = self._viewer.layers[skeleton_layer]
        self._disconnect_layer_events(layer)

        layer.properties.update(
            {
                'selected': np.zeros_like(
                    layer.properties['skeleton-id'], dtype=bool
                )
            }
        )

    def _initialize_points_layer(self):
        """Create a points layer for curation viz"""
        skeleton_layer = self._viewer.layers[self.skeleton_layer]
        coords, properties = make_points_data(skeleton_layer.properties)

        face_color = {
            'colors': 'position',
            'categorical_colormap': {'colormap': {'start': 'green', 'end': 'magenta', 'hide': [0, 0, 0, 0]}}
        }

        # create the points layer
        self._points_layer = self._viewer.add_points(
            coords,
            properties=properties,
            face_color=face_color,
            edge_color='selected',
            edge_color_cycle=['black', 'white'],
            name='skeleton points'
        )

    def _cleanup_points_layer(self):
        if self._points_layer is not None:
            self._viewer.layers.remove(self._points_layer)
            self._points_layer = None

    def _setup_curated_skeleton_layer(self):
        if self.skeleton_layer == "":
            return
        curated_skeleton_paths, skeleton_properties = self._draw_curated_skeleton()

        self._curated_skeleton_layer = self._viewer.add_shapes(
            curated_skeleton_paths,
            properties=skeleton_properties,
            shape_type='path',
            opacity=0.7,
            name="curated skeleton"
        )
        self._curated_skeleton_layer.edge_color_cycle_map = {True: np.array([1, 1, 1, 1]), False: np.array([0, 0, 0, 0])}
        self._curated_skeleton_layer.edge_color = "keep"

    def _cleanup_curated_skeleton_layer(self):
        if self._curated_skeleton_layer is not None:
            self._viewer.layers.remove(self._curated_skeleton_layer)
            self._points_layer = None

    def _update_curated_skeleton_layer(self):
        if self.skeleton_layer == "":
            return

        skeleton_layer = self._viewer.layers[self.skeleton_layer]
        self._curated_skeleton_layer.properties = skeleton_layer.properties
        self._curated_skeleton_layer.refresh_colors()

    def _draw_curated_skeleton(self) -> np.ndarray:
        skeleton_layer = self._viewer.layers[self.skeleton_layer]
        skeleton_image = skeleton_layer.data
        curated_skeleton_image = np.zeros_like(skeleton_image, dtype=float)

        skeleton_properties = skeleton_layer.properties
        skeleton = skeleton_layer.metadata["skan_obj"]
        branches_to_keep = skeleton_properties["keep"]

        # get the branches to include
        # skeleton_branch_ids = skeleton_properties["index"]
        # labels_to_draw = skeleton_branch_ids[branches_to_keep]
        # skeleton_ids_to_draw = labels_to_draw - 1
        all_paths = [
            skeleton.path_coordinates(i)
            for i in range(skeleton.n_paths)
        ]

        return all_paths, skeleton_properties

    @property
    def selected_branches(self):
        return self._selected_branches

    @selected_branches.setter
    def selected_branches(self, selected_branches):
        selected_branches = set(selected_branches)
        # set the selected branches in the features table
        skeleton_layer = self._viewer.layers[self.skeleton_layer]
        new_layer_properties = skeleton_layer.properties.copy()
        new_selected = np.zeros_like(new_layer_properties['index'], dtype=bool)
        for label in selected_branches:
            feature_index = np.argwhere(
                new_layer_properties['index'] == label
            )
            new_selected[feature_index] = True
        new_layer_properties.update({'selected': new_selected})
        skeleton_layer.properties = new_layer_properties

        self._selected_branches = selected_branches

    def toggle_keep_branches(self, event=None):
        skeleton_layer = self._viewer.layers[self.skeleton_layer]
        new_layer_properties = skeleton_layer.properties.copy()
        for label in self.selected_branches:
            # flip keep bool of the selected labels
            feature_index = np.argwhere(
                new_layer_properties['index'] == label
            )
            skeleton_layer.properties['keep'][feature_index] = np.logical_not(
                new_layer_properties['keep'][feature_index]
            )
        skeleton_layer.properties = new_layer_properties
        self._update_curated_skeleton_layer()

    def toggle_flip_branches(self, event=None):
        skeleton_layer = self._viewer.layers[self.skeleton_layer]
        new_layer_properties = skeleton_layer.properties.copy()
        for label in self.selected_branches:
            # flip the flip bool of the selected labels
            feature_index = np.argwhere(
                new_layer_properties['index'] == label
            )
            skeleton_layer.properties['flip'][feature_index] = np.logical_not(
                new_layer_properties['flip'][feature_index]
            )
        skeleton_layer.properties = new_layer_properties

    def _connect_layer_events(self, layer: Labels):
        layer.mouse_drag_callbacks.append(self._on_click)
        layer.bind_key('d', self._on_keep_toggle)
        layer.bind_key('f', self._on_flip_toggle)

    def _disconnect_layer_events(self, layer: Labels):
        layer.mouse_drag_callbacks.remove(self._on_click)
        layer.bind_key('d', None)
        layer.bind_key('f', None)

    def _on_click(self, layer, event):
        selected_label = layer.get_value(
            position=event.position,
            view_direction=event.view_direction,
            dims_displayed=event.dims_displayed,
            world=True
        )

        if selected_label == 0:
            self.selected_branches = set()
        else:
            self.selected_branches = {selected_label}

    def _on_keep_toggle(self, event):
        if self.curating is True:
            self.toggle_keep_branches()

    def _on_flip_toggle(self, event):
        if self.curating is True:
            self.toggle_flip_branches()

    def _on_properties_update(self, event):
        if self._points_layer is not None:
            skeleton_layer = self._viewer.layers[self.skeleton_layer]
            _, new_points_properties = make_points_data(skeleton_layer.properties)
            self._points_layer.properties = new_points_properties

