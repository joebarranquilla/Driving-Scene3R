"""Export a trained :class:`GaussianModel` as an Inria-format 3DGS ``.ply``.

This is the de-facto interchange format for 3D Gaussian Splatting and is
read directly by all common viewers:

* **SuperSplat**   (browser, easiest)  https://playcanvas.com/supersplat/editor
* **PlayCanvas** viewer
* **Polycam** Gaussian viewer
* **Inria SIBR** real-time viewer  https://github.com/graphdeco-inria/gaussian-splatting

Per-vertex layout (all ``float32``, binary little-endian; 17 fields = 68 B):

==============  ============================================================
Field           Meaning
==============  ============================================================
``x y z``       position (metres, world coords)
``nx ny nz``    normals (zeros — required-but-ignored by some viewers)
``f_dc_*``      DC spherical-harmonic coefficient = colour in SH basis,
                computed as ``(rgb - 0.5) / C0``, ``C0 = 1/(2*sqrt(pi))``
``opacity``     logit-space opacity (viewer applies sigmoid)
``scale_*``     log-space scales (viewer applies exp)
``rot_*``       quaternion ``wxyz`` (viewer normalises)
==============  ============================================================

Storing logit/log/raw-quat (rather than the activated values) is what
the Inria reference implementation does. Doing otherwise breaks
compatibility with downstream viewers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from semantic_gs.model.gaussians import GaussianModel


# SH degree-0 basis value: 1 / (2 * sqrt(pi))
_SH_C0 = 0.28209479177387814


def _rgb_to_sh0_dc(rgb: np.ndarray) -> np.ndarray:
    """Convert ``rgb`` in ``[0, 1]`` to the DC SH coefficient used by 3DGS PLY."""
    return (rgb - 0.5) / _SH_C0


@torch.no_grad()
def save_gaussians_ply(model: GaussianModel, path: str | Path) -> None:
    """Write the model to a binary-little-endian PLY in Inria 3DGS format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    means       = model.means.detach().cpu().numpy().astype(np.float32)
    quats_raw   = model.quats_raw.detach().cpu().numpy().astype(np.float32)
    log_scales  = model.log_scales.detach().cpu().numpy().astype(np.float32)
    opac_logits = model.opacity_logits.detach().cpu().numpy().astype(np.float32)
    colors      = model.colors.detach().cpu().numpy().astype(np.float32)

    n = means.shape[0]
    f_dc    = _rgb_to_sh0_dc(np.clip(colors, 0.0, 1.0)).astype(np.float32)
    normals = np.zeros((n, 3), dtype=np.float32)

    # 17 float32 columns per vertex, in the order viewers expect.
    rows = np.concatenate(
        [
            means,                          # 3
            normals,                        # 3
            f_dc,                           # 3
            opac_logits[:, None],           # 1
            log_scales,                     # 3
            quats_raw,                      # 4
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if rows.shape != (n, 17):
        raise AssertionError(
            f"save_gaussians_ply: packed row shape {rows.shape} != (n, 17)"
        )

    fields = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {f}\n" for f in fields)
        + "end_header\n"
    ).encode("ascii")

    with path.open("wb") as fh:
        fh.write(header)
        fh.write(rows.tobytes())

