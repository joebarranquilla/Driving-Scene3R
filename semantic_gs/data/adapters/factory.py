"""Single place that turns ``--seg-source`` + CLI paths into a KITTI loader.

Both CLI entry points (``train_gs``, ``smoke_test_adapters``) share this
factory so the seg-source branch, its required-flag validation, and the
loader keyword wiring cannot drift apart between scripts.
"""

from __future__ import annotations

from semantic_gs.data.adapters.kitti_odom import KITTIOdomSequenceLoader
from semantic_gs.data.adapters.kitti_odom_sam3 import KITTISam3SequenceLoader
from semantic_gs.data.dataset import SequenceLoader

SEG_SOURCES = ("mask2former", "sam3")


def build_kitti_loader(
    seg_source: str,
    *,
    sequence_dir,
    depth_dir,
    pose_path,
    pano_dir=None,
    sam3_dir=None,
    id2label_path=None,
    concepts_path=None,
    camera_index: int = 2,
    boundary_margin: int = 0,
) -> SequenceLoader:
    """Build the KITTI loader for ``seg_source``.

    Raises ``ValueError`` naming the missing CLI flags (callers surface it as
    a ``SystemExit``), so both entry points report identical errors.
    """
    if seg_source not in SEG_SOURCES:
        raise ValueError(f"--seg-source must be one of {SEG_SOURCES}, "
                         f"got {seg_source!r}")

    if seg_source == "sam3":
        missing = [name for name, val in
                   (("--depth-dir", depth_dir),
                    ("--sam3-dir",  sam3_dir),
                    ("--pose-path", pose_path))
                   if val is None]
        if missing:
            raise ValueError(
                f"--seg-source sam3 requires {missing}; pass them on the CLI."
            )
        return KITTISam3SequenceLoader(
            sequence_dir    = sequence_dir,
            depth_dir       = depth_dir,
            sam3_dir        = sam3_dir,
            pose_path       = pose_path,
            concepts_path   = concepts_path,
            camera_index    = camera_index,
            boundary_margin = boundary_margin,
        )

    missing = [name for name, val in
               (("--depth-dir", depth_dir),
                ("--pano-dir",  pano_dir),
                ("--pose-path", pose_path))
               if val is None]
    if missing:
        raise ValueError(
            f"--kitti-odom-seq requires {missing}; pass them on the CLI."
        )
    return KITTIOdomSequenceLoader(
        sequence_dir  = sequence_dir,
        depth_dir     = depth_dir,
        panoptic_dir  = pano_dir,
        pose_path     = pose_path,
        id2label_path = id2label_path,
        camera_index  = camera_index,
    )
