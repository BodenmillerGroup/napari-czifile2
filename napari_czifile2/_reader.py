from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

from .io import CZISceneFile


def napari_get_reader(path):
    if isinstance(path, list):
        if any(Path(p).suffix.lower() != ".czi" for p in path):
            return None
    else:
        if Path(path).suffix.lower() != ".czi":
            return None
    return reader_function


def reader_function(paths):
    layer_data = []
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        num_scenes = CZISceneFile.get_num_scenes(path)
        for scene_index in range(num_scenes):
            with CZISceneFile(path, scene_index) as f:
                data = f.as_tzcyx0_array(max_workers=cpu_count())
                # https://github.com/BodenmillerGroup/napari-czifile2/issues/5
                contrast_limits = None
                if data.dtype == np.uint16:
                    contrast_limits = (0, 65535)
                # https://github.com/napari/napari/issues/2348
                if not f.is_rgb:
                    data = data[:, :, :, :, :, 0]
                metadata = {
                    "rgb": f.is_rgb,
                    "channel_axis": 2,
                    "translate": (f.pos_t_seconds, f.pos_z_um, f.pos_y_um, f.pos_x_um),
                    "scale": (
                        f.scale_t_seconds,
                        f.scale_z_um,
                        f.scale_y_um,
                        f.scale_x_um,
                    ),
                    "contrast_limits": contrast_limits,
                }
                if f.channel_names is not None:
                    if num_scenes == 1:
                        metadata["name"] = f.channel_names
                    elif num_scenes > 1:
                        metadata["name"] = [
                            f"S{scene_index:02d} {channel_name}"
                            for channel_name in f.channel_names
                        ]
            layer_data.append((data, metadata, "image"))
    return layer_data
