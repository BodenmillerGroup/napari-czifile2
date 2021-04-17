import numpy as np

from czifile import CziFile, DimensionEntryDV1, DirectoryEntryDV
from pathlib import Path
from tifffile import lazyattr
from typing import Iterable, List, Optional, Union
from xml.etree import ElementTree


class CZISceneFile(CziFile):
    @staticmethod
    def get_num_scenes(path: Union[str, Path], *args, **kwargs) -> int:
        with CziFile(path, *args, **kwargs) as czi_file:
            if 'S' in czi_file.axes:
                return czi_file.shape[czi_file.axes.index('S')]
            return 1

    def __init__(self, path: Union[str, Path], scene_index: int, *args, **kwargs):
        super(CZISceneFile, self).__init__(str(path), *args, **kwargs)
        self.scene_index = scene_index

    @lazyattr
    def pos_x_um(self) -> float:
        return min((dim_entry.start for dim_entry in self._iter_dim_entries('X')), default=0.) * self.scale_x_um

    @lazyattr
    def pos_y_um(self) -> float:
        return min((dim_entry.start for dim_entry in self._iter_dim_entries('Y')), default=0.) * self.scale_y_um

    @lazyattr
    def pos_z_um(self) -> float:
        return min((dim_entry.start for dim_entry in self._iter_dim_entries('Z')), default=0.) * self.scale_z_um

    @lazyattr
    def pos_t_seconds(self) -> float:
        return min((dim_entry.start for dim_entry in self._iter_dim_entries('T')), default=0.) * self.scale_t_seconds

    @lazyattr
    def scale_x_um(self) -> float:
        return self._get_scale('X', multiplier=10. ** 6)

    @lazyattr
    def scale_y_um(self) -> float:
        return self._get_scale('Y', multiplier=10. ** 6)

    @lazyattr
    def scale_z_um(self) -> float:
        return self._get_scale('Z', multiplier=10. ** 6)

    @lazyattr
    def scale_t_seconds(self) -> float:
        return self._get_scale('T')

    @lazyattr
    def channel_names(self) -> Optional[List[str]]:
        if 'C' in self.axes:
            channel_elements = self._metadata_xml.findall('.//Metadata/Information/Image/Dimensions/Channels/Channel')
            if len(channel_elements) == self.shape[self.axes.index('C')]:
                return [c.attrib.get('Name', c.attrib['Id']) for c in channel_elements]
        return None

    @lazyattr
    def is_rgb(self) -> bool:
        return '0' in self.axes and self.shape[self.axes.index('0')] > 1

    def as_tzcyx0_array(self, *args, **kwargs) -> np.ndarray:
        data = self.asarray(*args, **kwargs)
        tzcyx0_axis_indices = []
        if 'T' in self.axes:
            tzcyx0_axis_indices.append(self.axes.index('T'))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        if 'Z' in self.axes:
            tzcyx0_axis_indices.append(self.axes.index('Z'))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        if 'C' in self.axes:
            tzcyx0_axis_indices.append(self.axes.index('C'))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        tzcyx0_axis_indices.append(self.axes.index('Y'))
        tzcyx0_axis_indices.append(self.axes.index('X'))
        if '0' in self.axes:
            tzcyx0_axis_indices.append(self.axes.index('0'))
        else:
            tzcyx0_axis_indices.append(data.ndim)
            data = np.expand_dims(data, -1)
        for axis_index in range(len(self.axes)):
            if axis_index not in tzcyx0_axis_indices:
                tzcyx0_axis_indices.append(axis_index)
        data = data.transpose(tzcyx0_axis_indices)
        data.shape = data.shape[:6]
        return data

    def _iter_dim_entries(self, dimension: str) -> Iterable[DimensionEntryDV1]:
        for dir_entry in self.filtered_subblock_directory:
            for dim_entry in dir_entry.dimension_entries:
                if dim_entry.dimension == dimension:
                    yield dim_entry

    def _get_scale(self, dimension: str, multiplier: float = 1.):
        scale_element = self._metadata_xml.find(f'.//Metadata/Scaling/Items/Distance[@Id="{dimension}"]/Value')
        if scale_element is not None:
            scale = float(scale_element.text)
            if scale > 0:
                return scale * multiplier
        return 1.

    @lazyattr
    def _metadata_xml(self) -> ElementTree.Element:
        return ElementTree.fromstring(self.metadata())

    @lazyattr
    def filtered_subblock_directory(self) -> List[DirectoryEntryDV]:
        dir_entries = super(CZISceneFile, self).filtered_subblock_directory
        return list(filter(lambda dir_entry: self._get_scene_index(dir_entry) == self.scene_index, dir_entries))

    @staticmethod
    def _get_scene_index(dir_entry: DirectoryEntryDV) -> int:
        scene_indices = {dim_entry.start for dim_entry in dir_entry.dimension_entries if dim_entry.dimension == 'S'}
        if len(scene_indices) == 0:
            return 0
        assert len(scene_indices) == 1
        return scene_indices.pop()
