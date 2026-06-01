#!/usr/bin/env python3
"""
Instance tracking over Mask2Former panoptic predictions.
=========================================================

Reads the per-frame NPZ files produced by run_mask2former_inference.py and
assigns globally consistent instance IDs ("track IDs") across the time axis
using the Hungarian (Kuhn–Munkres) assignment algorithm.

The output mirrors the input NPZ schema exactly so it can be used as a
drop-in replacement for downstream scripts such as lift_to_semantic_pointcloud.py.

Algorithm
---------
"Things" — instance-level classes (person 11, rider 12, car 13, truck 14,
            bus 15, train 16, motorcycle 17, bicycle 18):

  For each consecutive frame pair (t, t+1):
    1.  Build IoU matrix  I[i,j] = IoU(track_i_mask, det_j_mask).
    2.  Construct cost matrix C[i,j] = 1 − I[i,j].
        Set C[i,j] = 1 (invalid) when:
          • track_i and det_j have different label_ids, or
          • I[i,j] < iou_threshold.
    3.  Solve C with scipy.optimize.linear_sum_assignment (Hungarian).
    4.  Pairs with C < 1 are valid matches → detection inherits the
        track's globally unique ID.
    5.  Unmatched detections → new globally unique track ID.
    6.  Tracks unmatched for more than ``max_age`` frames are deleted.

"Stuff" — semantic-level classes (road 0 … sky 10):
  Assigned deterministic IDs of the form
      STUFF_OFFSET + label_id × 1000 + per-frame-class-instance-counter
  so they never collide with thing track IDs (which start at 1).
  No temporal association is performed for stuff.

Output per frame
----------------
{output_dir}/{sequence}/{frame_stem}.npz
    "panoptic_seg"  →  int32  (H, W)   globally consistent ID per pixel  (0 = void)
    "segment_ids"   →  int32  (N,)     globally consistent IDs (tracks / stuff)
    "label_ids"     →  int32  (N,)     semantic class — unchanged
    "scores"        →  float32 (N,)    detection confidence — unchanged

Usage
-----
python scripts/track_instances.py \\
    --panoptic_dir  /usr/prakt/<user>/panoptic_predictions \\
    --sequences     00 01 \\
    --output_dir    /usr/prakt/<user>/tracked_predictions

python scripts/track_instances.py \\
    --panoptic_dir  /usr/prakt/<user>/panoptic_predictions \\
    --sequences     00 \\
    --iou_threshold 0.3 \\
    --max_age       3 \\
    --start_frame   0 \\
    --n_frames      100 \\
    --output_dir    /usr/prakt/<user>/tracked_predictions
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import CITYSCAPES_LABELS, load_panoptic, load_id2label  # noqa: E402


# ---------------------------------------------------------------------------
# Class taxonomy
# ---------------------------------------------------------------------------

# Cityscapes "things": instance-level classes worth tracking individually.
# person (11), rider (12), car (13), truck (14), bus (15),
# train (16), motorcycle (17), bicycle (18)
_THING_IDS: frozenset[int] = frozenset(range(11, 19))

# Stuff IDs are encoded as STUFF_OFFSET + label_id * 1000 + n so they
# cannot collide with thing track IDs (which start at 1).
_STUFF_OFFSET: int = 1_000_000

# Label-id to human-readable name (for diagnostics)
_ID_TO_NAME: dict[int, str] = {lid: name for lid, name, *_ in CITYSCAPES_LABELS}


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _iou_matrix(
    masks_a: np.ndarray,  # (N, K) bool or uint8
    masks_b: np.ndarray,  # (M, K) bool or uint8
) -> np.ndarray:          # (N, M) float32
    """
    Compute pairwise IoU between two sets of flattened binary masks.

    Uses integer dot products for the intersection count, which is faster
    than boolean logic on large (H×W)-dimensional vectors.
    """
    a = masks_a.astype(np.uint8)
    b = masks_b.astype(np.uint8)
    intersection = (a @ b.T).astype(np.float32)        # (N, M)
    area_a = a.sum(axis=1, keepdims=True).astype(np.float32)   # (N, 1)
    area_b = b.sum(axis=1, keepdims=True).astype(np.float32)   # (M, 1)
    union = area_a + area_b.T - intersection            # (N, M)
    return np.where(union > 0, intersection / union, 0.0)


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------

@dataclass
class Track:
    """One tracked instance."""
    track_id: int
    label_id: int
    mask: np.ndarray   # (H*W,) bool — most recent matched mask (flat)
    age: int = 0       # frames since last successful match (0 = matched this frame)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class InstanceTracker:
    """
    Frame-by-frame instance tracker using Hungarian matching on mask IoU.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to allow a match.  Pairs below this value cannot be
        matched even if they are the optimal assignment.
    max_age : int
        Maximum number of consecutive unmatched frames before a track is
        deleted.  Setting this to 0 deletes tracks after a single miss.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self._tracks: dict[int, Track] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_track_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        panoptic_seg: np.ndarray,  # (H, W) int32
        segment_ids: np.ndarray,   # (N,) int32
        label_ids: np.ndarray,     # (N,) int32
        scores: np.ndarray,        # (N,) float32
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Assign globally consistent IDs to all segments in one frame.

        Returns
        -------
        new_panoptic_seg : (H, W) int32   — remapped IDs
        new_segment_ids  : (N,) int32     — globally consistent IDs
        label_ids        : (N,) int32     — unchanged
        scores           : (N,) float32   — unchanged
        """
        H, W = panoptic_seg.shape
        K = H * W

        # ---- Increment ages of all existing tracks at frame start --------
        # Matched tracks will have their age reset to 0 below.
        for t in self._tracks.values():
            t.age += 1

        # ---- Partition detections into things and stuff ------------------
        is_thing = np.array([int(lid) in _THING_IDS for lid in label_ids], dtype=bool)

        t_segs   = segment_ids[is_thing]
        t_labels = label_ids[is_thing]
        t_scores = scores[is_thing]

        s_segs   = segment_ids[~is_thing]
        s_labels = label_ids[~is_thing]

        # ---- Flat binary masks for every things detection ----------------
        if len(t_segs) > 0:
            det_flat = np.stack(
                [(panoptic_seg == sid).reshape(K) for sid in t_segs]
            )  # (M_det, K)
        else:
            det_flat = np.zeros((0, K), dtype=bool)

        # ---- Hungarian matching ------------------------------------------
        seg_to_new: dict[int, int] = {}   # local_seg_id → global_id
        matched_det_indices: set[int] = set()

        active_ids = [
            tid for tid, t in self._tracks.items()
            if t.label_id in _THING_IDS   # safety guard
        ]

        if len(active_ids) > 0 and len(t_segs) > 0:
            trk_flat   = np.stack([self._tracks[tid].mask for tid in active_ids])
            trk_labels = np.array(
                [self._tracks[tid].label_id for tid in active_ids], dtype=np.int32
            )

            iou = _iou_matrix(trk_flat, det_flat)  # (N_trk, M_det)

            # Class gate: different-class pairs are forbidden (cost=1)
            class_ok = trk_labels[:, np.newaxis] == t_labels[np.newaxis, :]
            cost = np.where(class_ok, 1.0 - iou, 1.0).astype(np.float64)

            # IoU threshold gate: low-overlap pairs are also forbidden
            cost[iou < self.iou_threshold] = 1.0

            row_ind, col_ind = linear_sum_assignment(cost)

            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 1.0:          # valid match
                    tid    = active_ids[r]
                    seg_id = int(t_segs[c])
                    seg_to_new[seg_id] = tid
                    # Refresh matched track
                    self._tracks[tid].mask = det_flat[c]
                    self._tracks[tid].age  = 0
                    matched_det_indices.add(c)

        # ---- Unmatched things → new tracks -------------------------------
        for c in range(len(t_segs)):
            if c not in matched_det_indices:
                seg_id   = int(t_segs[c])
                label_id = int(t_labels[c])
                new_tid  = self._new_track_id()
                seg_to_new[seg_id] = new_tid
                self._tracks[new_tid] = Track(
                    track_id=new_tid,
                    label_id=label_id,
                    mask=det_flat[c],
                    age=0,
                )

        # ---- Remove dead tracks ------------------------------------------
        self._tracks = {
            tid: t for tid, t in self._tracks.items()
            if t.age <= self.max_age
        }

        # ---- Deterministic IDs for stuff ---------------------------------
        per_class_ctr: dict[int, int] = {}
        for seg_id, label_id in zip(s_segs, s_labels):
            lid = int(label_id)
            n   = per_class_ctr.get(lid, 0)
            per_class_ctr[lid] = n + 1
            seg_to_new[int(seg_id)] = _STUFF_OFFSET + lid * 1000 + n

        # ---- Remap panoptic_seg map --------------------------------------
        new_panoptic = np.zeros(K, dtype=np.int32)
        flat_seg = panoptic_seg.reshape(K)
        for old_id, new_id in seg_to_new.items():
            new_panoptic[flat_seg == old_id] = new_id
        new_panoptic = new_panoptic.reshape(H, W)

        # ---- Build output segment_ids array (same ordering as input) -----
        new_segment_ids = np.array(
            [seg_to_new.get(int(sid), 0) for sid in segment_ids],
            dtype=np.int32,
        )

        return new_panoptic, new_segment_ids, label_ids.copy(), scores.copy()

    @property
    def n_active_tracks(self) -> int:
        """Number of tracks currently alive (age ≤ max_age)."""
        return len(self._tracks)


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq_id: str,
    panoptic_dir: str,
    output_dir: str,
    iou_threshold: float,
    max_age: int,
    start_frame: int,
    n_frames: int,
) -> None:
    seq_id = seq_id.zfill(2)
    in_seq_dir  = Path(panoptic_dir) / seq_id
    out_seq_dir = Path(output_dir) / seq_id
    out_seq_dir.mkdir(parents=True, exist_ok=True)

    # Discover and slice frames
    all_npz = sorted(in_seq_dir.glob("*.npz"))
    if not all_npz:
        print(f"  [warn] No NPZ files found in {in_seq_dir}")
        return

    end_frame = (
        min(start_frame + n_frames, len(all_npz))
        if n_frames > 0
        else len(all_npz)
    )
    npz_files = all_npz[start_frame:end_frame]
    print(f"  Frames     : {len(all_npz)} total → processing [{start_frame}, {end_frame})")

    # Resume: skip already-saved frames
    pending = [p for p in npz_files if not (out_seq_dir / p.name).exists()]
    n_done  = len(npz_files) - len(pending)
    if n_done:
        print(f"  Skipping   : {n_done} already-saved frames")

    if not pending:
        print("  Nothing to do — all frames already saved.")
        return
    print(f"  To process : {len(pending)} frames")

    # When resuming mid-sequence, we must replay from the beginning so the
    # tracker state is correct for the first pending frame.
    replay_from = all_npz.index(pending[0])
    replay_files = all_npz[start_frame:replay_from]
    save_files   = pending

    tracker = InstanceTracker(iou_threshold=iou_threshold, max_age=max_age)

    # --- Warm-up (replay without saving) ----------------------------------
    if replay_files:
        print(f"  Replaying  : {len(replay_files)} frames to restore tracker state…")
        for npz_path in tqdm(replay_files, desc="  warm-up", unit="frame", leave=False):
            pred = load_panoptic(str(npz_path))
            tracker.update(
                pred["panoptic_seg"].astype(np.int32),
                pred["segment_ids"].astype(np.int32),
                pred["label_ids"].astype(np.int32),
                pred["scores"].astype(np.float32),
            )

    # --- Main processing loop ---------------------------------------------
    for npz_path in tqdm(save_files, desc=f"  seq {seq_id}", unit="frame"):
        pred = load_panoptic(str(npz_path))

        new_panoptic, new_seg_ids, label_ids, scores = tracker.update(
            pred["panoptic_seg"].astype(np.int32),
            pred["segment_ids"].astype(np.int32),
            pred["label_ids"].astype(np.int32),
            pred["scores"].astype(np.float32),
        )

        np.savez_compressed(
            str(out_seq_dir / npz_path.name),
            panoptic_seg=new_panoptic,
            segment_ids=new_seg_ids,
            label_ids=label_ids,
            scores=scores,
        )

    print(f"  Active tracks at end: {tracker.n_active_tracks}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Assign globally consistent instance track IDs to Mask2Former "
            "panoptic predictions using the Hungarian algorithm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- required ---
    p.add_argument(
        "--panoptic_dir", required=True,
        help=(
            "Root directory of Mask2Former panoptic predictions produced by "
            "run_mask2former_inference.py.  "
            "Frames are expected at {panoptic_dir}/{sequence}/{frame_stem}.npz."
        ),
    )
    p.add_argument(
        "--sequences", nargs="+", required=True, metavar="SEQ",
        help="One or more two-digit sequence IDs, e.g.: --sequences 00 01 02",
    )

    # --- optional ---
    p.add_argument(
        "--output_dir",
        default="/usr/prakt/<user>/tracked_predictions",
        help="Root directory for tracked NPZ output files.",
    )
    p.add_argument(
        "--iou_threshold", type=float, default=0.3,
        help=(
            "Minimum IoU between a track and a detection for a valid match.  "
            "Pairs below this threshold are forbidden from matching, causing the "
            "detection to start a fresh track.  "
            "Increase for noisier sequences; decrease if tracks fragment too much."
        ),
    )
    p.add_argument(
        "--max_age", type=int, default=3,
        help=(
            "Maximum number of consecutive frames a track can go unmatched "
            "before being deleted.  Allows short occlusions to be bridged."
        ),
    )
    p.add_argument(
        "--start_frame", type=int, default=0,
        help="Index of the first frame to process (0-based, matches NPZ filename).",
    )
    p.add_argument(
        "--n_frames", type=int, default=0,
        help=(
            "Number of frames to process starting from --start_frame.  "
            "0 means process all available frames."
        ),
    )
    p.add_argument(
        "--copy_id2label", action="store_true", default=True,
        help=(
            "Copy id2label.json from --panoptic_dir to --output_dir so that "
            "downstream scripts can resolve class names."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    panoptic_dir = Path(args.panoptic_dir)
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Propagate id2label.json alongside the tracked predictions
    if args.copy_id2label:
        src = panoptic_dir / "id2label.json"
        dst = output_dir   / "id2label.json"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[i] Copied id2label.json → {dst}")

    for seq_id in args.sequences:
        print(f"\n{'=' * 60}")
        print(f"  Sequence: {seq_id}")
        print(f"  iou_threshold={args.iou_threshold}  max_age={args.max_age}")
        print(f"{'=' * 60}")
        process_sequence(
            seq_id        = seq_id,
            panoptic_dir  = str(panoptic_dir),
            output_dir    = str(output_dir),
            iou_threshold = args.iou_threshold,
            max_age       = args.max_age,
            start_frame   = args.start_frame,
            n_frames      = args.n_frames,
        )

    print(f"\nAll sequences done.  Tracked maps written to: {output_dir}")

    # --- Diagnostics: decode a few example track IDs ---------------------
    print("\n[i] ID encoding reference:")
    print(f"    Thing track IDs  : 1, 2, 3, …  (globally unique, grow over time)")
    print(f"    Stuff segment IDs: {_STUFF_OFFSET} + label_id × 1000 + n")
    for lid, name, _, _, _ in CITYSCAPES_LABELS:
        if lid <= 10:
            print(f"      {name:16s} (label {lid:2d}) → base ID {_STUFF_OFFSET + lid * 1000}")


if __name__ == "__main__":
    main()
