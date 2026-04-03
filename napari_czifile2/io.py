from functools import cached_property
from pathlib import Path
from typing import List, Optional, Union
from xml.etree import ElementTree

import numpy as np
from czifile import CziDirectoryEntryDV, CziFile


class CZISceneFile(CziFile):
    @staticmethod
    def get_num_scenes(path: Union[str, Path], *args, **kwargs) -> int:
        with CziFile(path, *args, **kwargs) as czi_file:
            return len(czi_file.scenes)

    def __init__(self, path: Union[str, Path], scene_index: int, *args, **kwargs):
        super(CZISceneFile, self).__init__(str(path), *args, **kwargs)
        self.scene_index = scene_index
        self.axes = list(self.scenes[self.scene_index].sizes.keys())

    def _get_start(self, dim: str) -> int:
        try:
            return self.scenes[self.scene_index].start[self.axes.index(dim)]
        except ValueError:
            return 0

    @cached_property
    def pos_x_um(self) -> float:
        return self.scale_x_um * self._get_start("X")

    @cached_property
    def pos_y_um(self) -> float:
        return self.scale_y_um * self._get_start("Y")

    @cached_property
    def pos_z_um(self) -> float:
        return self.scale_z_um * self._get_start("Z")

    @cached_property
    def pos_t_seconds(self) -> float:
        return self.scale_t_seconds * self._get_start("T")

    @cached_property
    def scale_x_um(self) -> float:
        return self._get_scale("X", multiplier=10.0**6)

    @cached_property
    def scale_y_um(self) -> float:
        return self._get_scale("Y", multiplier=10.0**6)

    @cached_property
    def scale_z_um(self) -> float:
        return self._get_scale("Z", multiplier=10.0**6)

    @cached_property
    def scale_t_seconds(self) -> float:
        return self._get_scale("T")

    @cached_property
    def channel_names(self) -> Optional[List[str]]:
        if "C" in self.axes:
            channel_elements = self._metadata_xml.findall(
                ".//Metadata/Information/Image/Dimensions/Channels/Channel"
            )
            if len(channel_elements) == self.scenes[self.scene_index].sizes["C"]:
                return [c.attrib.get("Name", c.attrib["Id"]) for c in channel_elements]
        return None

    @cached_property
    def is_rgb(self) -> bool:
        return "0" in self.axes and self.scenes[self.scene_index].sizes["0"] > 1

    def as_tzcyx0_array(self, *args, **kwargs) -> np.ndarray:
        data = self.asarray(*args, **kwargs)
        tzcyx0_axis_indices = []
        if "T" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("T"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        if "Z" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("Z"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        if "C" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("C"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        tzcyx0_axis_indices.append(self.axes.index("Y"))
        tzcyx0_axis_indices.append(self.axes.index("X"))
        if "0" in self.axes:
            tzcyx0_axis_indices.append(self.axes.index("0"))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        for axis_index in range(len(self.axes)):
            if axis_index not in tzcyx0_axis_indices:
                tzcyx0_axis_indices.append(axis_index)
        data = data.transpose(tzcyx0_axis_indices)
        data.shape = data.shape[:6]
        return data

    def _get_scale(self, dimension: str, multiplier: float = 1.0):
        scale_element = self._metadata_xml.find(
            f'.//Metadata/Scaling/Items/Distance[@Id="{dimension}"]/Value'
        )
        if scale_element is not None:
            scale = float(scale_element.text)
            if scale > 0:
                return scale * multiplier
        return 1.0

    @cached_property
    def _metadata_xml(self) -> ElementTree.Element:
        return ElementTree.fromstring(self.metadata())

    @cached_property
    def filtered_subblock_directory(self) -> List[CziDirectoryEntryDV]:
        dir_entries = super(CZISceneFile, self).filtered_subblock_directory
        return list(
            filter(
                lambda dir_entry: self._get_scene_index(dir_entry) == self.scene_index,
                dir_entries,
            )
        )

    @staticmethod
    def _get_scene_index(dir_entry: CziDirectoryEntryDV) -> int:
        scene_index = dir_entry.scene_index
        return 0 if scene_index == -1 else scene_index
