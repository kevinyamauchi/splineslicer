from napari.layers import Labels
import numpy as np


class SkeletonPruner():
    def __init__(self, viewer):
        self._viewer = viewer

        self._skeleton_layer = ''
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

    def _initialize_skeleton_layer(self, new_skeleton_layer: str, prev_skeleton_layer: str):
        """set up the skeleton layer"""
        if prev_skeleton_layer == '':
            # clean up the old skeleton layer
            self._cleanup_skeleton_layer(prev_skeleton_layer)
        layer = self._viewer[new_skeleton_layer]
        self._connect_layer_events(layer)

        # add the required column to the layer properties
        if 'keep' not in layer.properties:
            n_branches = len(layer.properties['skeleton-id'])
            layer.properties['keep'] = np.zeros((n_branches,), dtype=bool)

    def _cleanup_skeleton_layer(self, skeleton_layer):
        """clean up a skeleton layer after deselecting it"""
        layer = self._viewer.layers[skeleton_layer]
        self._disconnect_layer_events(layer)

    @property
    def selected_branches(self):
        return self._selected_branches

    @selected_branches.setter
    def selected_branches(self, selected_branches):
        self._selected_branches = set(selected_branches)

    def toggle_selected_branches(self):
        pass

    def _connect_layer_events(self, layer: Labels):
        pass

    def _disconnect_layer_events(self, layer: Labels):
        pass
