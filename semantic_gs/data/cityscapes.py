"""Cityscapes class table (single source of truth inside ``semantic_gs``).

Mirrors ``scripts/utils.py::CITYSCAPES_LABELS`` from the ``lift-semantic``
branch. Kept in sync **by hand** until that file is promoted to a proper
shared package — see ``semantic_gs/README.md`` for the open coordination
item.

Every Mask2Former / lift-script consumer in this module should look up
class names and palette colours through these constants instead of
hard-coding magic numbers.
"""

from __future__ import annotations

import numpy as np


# (label_id, name, is_dynamic_vehicle, is_sky, rgb_colour_uint8)
CITYSCAPES_LABELS: list[tuple[int, str, bool, bool, tuple[int, int, int]]] = [
    (0,  "road",          False, False, (128,  64, 128)),
    (1,  "sidewalk",      False, False, (244,  35, 232)),
    (2,  "building",      False, False, ( 70,  70,  70)),
    (3,  "wall",          False, False, (102, 102, 156)),
    (4,  "fence",         False, False, (190, 153, 153)),
    (5,  "pole",          False, False, (153, 153, 153)),
    (6,  "traffic light", False, False, (250, 170,  30)),
    (7,  "traffic sign",  False, False, (220, 220,   0)),
    (8,  "vegetation",    False, False, (107, 142,  35)),
    (9,  "terrain",       False, False, (152, 251, 152)),
    (10, "sky",           False, True,  ( 70, 130, 180)),
    (11, "person",        False, False, (220,  20,  60)),
    (12, "rider",         False, False, (255,   0,   0)),
    (13, "car",           True,  False, (  0,   0, 142)),
    (14, "truck",         True,  False, (  0,   0,  70)),
    (15, "bus",           True,  False, (  0,  60, 100)),
    (16, "train",         True,  False, (  0,  80, 100)),
    (17, "motorcycle",    True,  False, (  0,   0, 230)),
    (18, "bicycle",       True,  False, (119,  11,  32)),
]


CITYSCAPES_ID_TO_NAME: dict[int, str] = {
    lid: name for lid, name, *_ in CITYSCAPES_LABELS
}


CITYSCAPES_ID_TO_COLOR_U8: dict[int, tuple[int, int, int]] = {
    lid: col for lid, _, _, _, col in CITYSCAPES_LABELS
}


# Float-normalised palette ([0, 1] RGB) for 3-D point-cloud colouring.
CITYSCAPES_ID_TO_COLOR_F32: dict[int, np.ndarray] = {
    lid: np.array(col, dtype=np.float32) / 255.0
    for lid, _, _, _, col in CITYSCAPES_LABELS
}


__all__ = [
    "CITYSCAPES_LABELS",
    "CITYSCAPES_ID_TO_NAME",
    "CITYSCAPES_ID_TO_COLOR_U8",
    "CITYSCAPES_ID_TO_COLOR_F32",
]

