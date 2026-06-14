"""Build per-pixel static masks from Mask2Former panoptic outputs.

A *static* pixel is one that is safe to feed into the static-scene
Gaussian Splatting model: it must (a) belong to a confidently predicted
segment (not void), and (b) belong to a semantic class that is not part
of the dynamic-asset set (cars, pedestrians, ...).

The dynamic-class set is matched by **name** (lower-cased) rather than
by ID, so swapping the underlying Mask2Former checkpoint (different
``id2label`` maps) does not silently break this filter.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import AbstractSet

import numpy as np


# Cityscapes "thing" classes that move between frames — must NOT be baked
# into the static Gaussian model. Names match the default Mask2Former
# Cityscapes-panoptic id2label values.
#
# NOTE: ``person`` and ``rider`` are intentionally NOT in this set, matching
# the teammate's ``scripts/lift_to_semantic_pointcloud.py`` policy. In
# forward-driving sequences these are mostly static over the 1-3 s aggregation
# window. Flip them back into the dynamic set via the ``dynamic_classes``
# argument if Phase 3 visuals show ghosting around pedestrians.
DEFAULT_DYNAMIC_CLASSES: frozenset[str] = frozenset({
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
})


# Cityscapes "stuff" classes that are static in the scene but whose
# per-pixel depth from MobileStereoNet (or any matcher) is unreliable
# enough that we don't want them seeding the static GS model.
DEFAULT_UNRELIABLE_DEPTH_CLASSES: frozenset[str] = frozenset({
    "sky",
})


def semantic_boundary_mask(label_map: np.ndarray, margin: int) -> np.ndarray:
    """Return a mask of pixels within ``margin`` px of any semantic class boundary.

    Stereo aggregation kernels straddling two classes produce depth-bleeding
    "floater" artefacts in back-projected clouds. Dilating semantic
    boundaries by a few pixels and discarding those pixels is a cheap fix.

    Adopted from ``scripts/utils.py::semantic_boundary_mask`` (lift-semantic
    branch) for behavioural parity with the teammate's lift script.
    """
    if margin <= 0:
        return np.zeros(label_map.shape, dtype=bool)

    from scipy.ndimage import binary_dilation  # local import: scipy is heavy

    edge = np.zeros(label_map.shape, dtype=bool)
    edge[:-1, :] |= label_map[:-1, :] != label_map[1:,  :]
    edge[1:,  :] |= label_map[:-1, :] != label_map[1:,  :]
    edge[:, :-1] |= label_map[:, :-1] != label_map[:,  1:]
    edge[:,  1:] |= label_map[:, :-1] != label_map[:,  1:]

    struct = np.ones((2 * margin + 1, 2 * margin + 1), dtype=bool)
    return binary_dilation(edge, structure=struct)


def build_static_mask(
    panoptic_seg: np.ndarray,
    segment_ids: np.ndarray,
    label_ids: np.ndarray,
    id2label: Mapping[int, str],
    dynamic_classes: AbstractSet[str] | None = None,
    unreliable_depth_classes: AbstractSet[str] | None = None,
    void_is_static: bool = False,
    boundary_margin: int = 0,
) -> np.ndarray:
    """Return a boolean mask of static pixels suitable for GS initialisation.

    A pixel is "static" iff (a) it is not void, (b) its semantic class is
    not in ``dynamic_classes`` (things that move), (c) its semantic class
    is not in ``unreliable_depth_classes`` (e.g. sky — static but garbage
    depth), and (d) it is not within ``boundary_margin`` pixels of any
    semantic class boundary (stereo depth-bleeding zone).

    Parameters
    ----------
    panoptic_seg
        ``int`` ``(H, W)`` — per-pixel segment ID (``0`` = void).
    segment_ids
        ``int`` ``(N,)`` — unique segment IDs in this frame.
    label_ids
        ``int`` ``(N,)`` — semantic class ID for each segment, indexed
        into ``id2label``.
    id2label
        Mapping from class ID to class name (e.g. ``{13: "car"}``).
    dynamic_classes
        Class names considered dynamic. Defaults to
        :data:`DEFAULT_DYNAMIC_CLASSES`. Pass ``frozenset()`` to disable.
    unreliable_depth_classes
        Class names that are static but should still be removed because
        their depth is not trustworthy. Defaults to ``None`` (no extra
        exclusion); the KITTI loader sets this to
        :data:`DEFAULT_UNRELIABLE_DEPTH_CLASSES`. Pass ``frozenset()``
        to disable.
    void_is_static
        If ``True``, pixels with ``panoptic_seg == 0`` are kept as static.
        Default is ``False`` because Mask2Former emits void where it is
        uncertain.
    boundary_margin
        If > 0, also remove pixels within this many pixels of any semantic
        class boundary. ``5`` matches the teammate's default.

    Returns
    -------
    static_mask : ``bool`` ``(H, W)``
    """
    if panoptic_seg.ndim != 2:
        raise ValueError(f"panoptic_seg must be 2-D, got shape {panoptic_seg.shape}")
    if not (segment_ids.shape == label_ids.shape and segment_ids.ndim == 1):
        raise ValueError(
            "segment_ids and label_ids must be 1-D arrays of equal length; "
            f"got {segment_ids.shape} and {label_ids.shape}"
        )

    if dynamic_classes is None:
        dynamic_classes = DEFAULT_DYNAMIC_CLASSES
    excluded: Set[str] = {c.lower() for c in dynamic_classes}
    if unreliable_depth_classes is not None:
        excluded |= {c.lower() for c in unreliable_depth_classes}

    # Resolve which segment IDs should be excluded from the static mask.
    excluded_seg_ids: list[int] = []
    for sid, lid in zip(segment_ids.tolist(), label_ids.tolist()):
        name = str(id2label.get(int(lid), "")).lower()
        if name in excluded:
            excluded_seg_ids.append(int(sid))

    # Start with all non-void pixels (or all pixels if void_is_static).
    if void_is_static:
        static = np.ones(panoptic_seg.shape, dtype=bool)
    else:
        static = panoptic_seg != 0

    if excluded_seg_ids:
        excluded_pixels = np.isin(panoptic_seg, np.asarray(excluded_seg_ids))
        static = static & ~excluded_pixels

    # Optional semantic-boundary erosion (matches teammate's lift script).
    if boundary_margin > 0:
        # Build a per-pixel class-label map so we can detect class boundaries.
        label_map = np.zeros(panoptic_seg.shape, dtype=np.int32)
        for sid, lid in zip(segment_ids.tolist(), label_ids.tolist()):
            label_map[panoptic_seg == sid] = int(lid)
        boundary = semantic_boundary_mask(label_map, boundary_margin)
        static = static & ~boundary

    return static.astype(bool, copy=False)



