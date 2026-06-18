"""Vanilla 3DGS trainer for Phase 3 (RGB-only photometric supervision).

Workflow per iteration:

1. Sample a random training :class:`Frame` from the loader.
2. Render it with the current :class:`GaussianModel`.
3. Compute photometric loss against the GT RGB, masked by
   ``frame.static_mask`` so dynamic vehicles + sky do not contribute.
4. Adam step on all per-Gaussian parameters.

Periodically, render every held-out eval frame, log PSNR/SSIM, and dump
side-by-side comparison PNGs so the user can visually verify the model
is learning the actual scene (this is the smoking-gun output of Phase
3).

At the end, export the trained Gaussians as an Inria-format ``.ply``
that can be opened in SuperSplat / PlayCanvas / the Inria SIBR viewer.

This trainer does NOT use semantics yet (Phase 4 will add a logits
head). It also does not (yet) densify / prune — the initial cloud from
``lift_to_semantic_pointcloud.py`` is dense enough for a POC. See
``--strategy`` in :mod:`semantic_gs.scripts.train_gs` for the planned
hook-up of ``gsplat.DefaultStrategy`` / ``MCMCStrategy``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from semantic_gs.data.dataset import SequenceLoader
from semantic_gs.data.frame import Frame
from semantic_gs.data.pointcloud import SemanticPointCloud
from semantic_gs.model.gaussians import GaussianModel
from semantic_gs.train.losses import photometric_loss, psnr, ssim as ssim_fn
from semantic_gs.train.render import render_frame


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All Phase-3 hyper-parameters. ``train_gs.py`` exposes these as flags."""

    max_iters:    int   = 10_000
    eval_every:   int   = 1_000
    ckpt_every:   int   = 2_500
    eval_stride:  int   = 10      # every Nth frame held out for eval
    lambda_ssim:  float = 0.2
    near_plane:   float = 0.1
    far_plane:    float = 200.0
    log_every:    int   = 50
    seed:         int   = 0
    save_renders: bool  = True
    # Render background colour (RGB in [0, 1]) for pixels no Gaussian covers.
    # A sky-blue value complements the sky dome by filling inter-splat cracks.
    bg_color:     tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ---- synthetic sky dome (free-view backdrop) --------------------------
    sky_dome:               bool  = False
    sky_dome_points:        int   = 40_000
    sky_dome_ground:        bool  = False
    sky_dome_radius_scale:  float = 3.0
    # Per-tensor learning rates (Inria 3DGS defaults).
    lr_means:    float = 1.6e-4
    lr_colors:   float = 2.5e-3
    lr_opacity:  float = 5.0e-2
    lr_quats:    float = 1.0e-3
    lr_scales:   float = 5.0e-3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _split_train_eval(
    n_frames: int, stride: int,
) -> tuple[list[int], list[int]]:
    """Take every ``stride``-th frame as eval; rest are train."""
    if stride < 1:
        raise ValueError(f"eval_stride must be >= 1, got {stride}")
    eval_idx_set = set(range(0, n_frames, stride))
    eval_idx     = sorted(eval_idx_set)
    train_idx    = [i for i in range(n_frames) if i not in eval_idx_set]
    if not train_idx:
        # Sequence too short — fall back to using everything for both.
        train_idx = list(range(n_frames))
    return train_idx, eval_idx


def _to_gpu_image(rgb_uint8: np.ndarray, device: torch.device) -> torch.Tensor:
    """``uint8 (H, W, 3)`` -> ``float32 (H, W, 3)`` in ``[0, 1]`` on ``device``."""
    return torch.from_numpy(rgb_uint8.astype(np.float32) / 255.0).to(device)


def _to_gpu_mask(mask_bool: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask_bool).to(device)


def _save_render_png(
    out_path: Path,
    pred:     torch.Tensor,    # (H, W, 3)
    gt:       torch.Tensor,    # (H, W, 3)
    mask:     torch.Tensor,    # (H, W)   bool
    title:    str,
) -> None:
    """Write a 4-panel diagnostic image (pred ∥ gt ∥ error ∥ mask)."""
    import matplotlib.pyplot as plt  # local import: heavy, only used here

    out_path.parent.mkdir(parents=True, exist_ok=True)
    p = pred.detach().clamp(0.0, 1.0).cpu().numpy()
    g = gt.detach().cpu().numpy()
    m = mask.detach().cpu().numpy()
    err = np.abs(p - g).mean(axis=-1)
    err_static = np.where(m, err, np.nan)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(p);  ax[0].set_title("Predicted (3DGS)"); ax[0].axis("off")
    ax[1].imshow(g);  ax[1].set_title("Ground truth");     ax[1].axis("off")
    im = ax[2].imshow(err_static, vmin=0.0, vmax=0.3, cmap="hot")
    ax[2].set_title("|pred - gt|  (static only)"); ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].imshow(m, cmap="gray"); ax[3].set_title("Static mask"); ax[3].axis("off")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:       GaussianModel,
    eval_frames: list[Frame],
    out_dir:     Path,
    it:          int,
    cfg:         TrainConfig,
    *,
    csv_writer:  Optional[csv.writer] = None,   # type: ignore[type-arg]
    csv_file=None,
    save_pngs:   bool                 = True,
    t_start:     float                = 0.0,
    label:       str                  = "eval",
) -> dict:
    """Render every eval frame; return mean PSNR / SSIM and optionally save PNGs."""
    if not eval_frames:
        return {"psnr_mean": 0.0, "ssim_mean": 0.0,
                "psnr_per_frame": [], "ssim_per_frame": []}

    device = model.means.device
    psnrs:  list[float] = []
    ssims:  list[float] = []
    for frame in eval_frames:
        gt_rgb  = _to_gpu_image(frame.rgb, device)
        gt_mask = _to_gpu_mask(frame.static_mask, device)
        out     = render_frame(
            model, frame,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            bg_color=cfg.bg_color,
        )
        pred = out["rgb"].clamp(0.0, 1.0)
        p = psnr(pred, gt_rgb)
        s = float(ssim_fn(pred, gt_rgb).item())
        psnrs.append(p)
        ssims.append(s)
        if save_pngs:
            png = out_dir / "renders" / f"iter_{it:06d}_eval_{frame.frame_id}.png"
            _save_render_png(
                png, pred, gt_rgb, gt_mask,
                f"iter {it} | eval frame {frame.frame_id} | "
                f"PSNR {p:.2f} dB | SSIM {s:.3f}",
            )

    psnr_mean = float(np.mean(psnrs))
    ssim_mean = float(np.mean(ssims))
    elapsed   = time.time() - t_start
    print(
        f"[{label}] iter {it:>6d} | eval PSNR={psnr_mean:.2f} dB  "
        f"SSIM={ssim_mean:.3f}  N={model.num_gaussians:,}  ({elapsed:.0f}s)"
    )
    if csv_writer is not None:
        csv_writer.writerow([
            it, "eval", "", "", "",
            f"{psnr_mean:.3f}", model.num_gaussians, f"{elapsed:.1f}",
        ])
        if csv_file is not None:
            csv_file.flush()
    return {
        "psnr_mean": psnr_mean, "ssim_mean": ssim_mean,
        "psnr_per_frame": psnrs, "ssim_per_frame": ssims,
    }


# ---------------------------------------------------------------------------
# main training entry point
# ---------------------------------------------------------------------------

def train(
    loader:  SequenceLoader,
    init_pc: SemanticPointCloud,
    out_dir: Path,
    cfg:     TrainConfig,
    *,
    device:  torch.device | None = None,
) -> dict:
    """Run vanilla 3DGS optimisation and return a summary dict.

    The summary contains final PSNR/SSIM and the path to the exported
    Inria-format PLY file (openable in SuperSplat / SIBR viewer).
    """
    # ---- device guard ------------------------------------------------
    # The whole project targets Linux + NVIDIA CUDA. gsplat has no CPU
    # fallback, so we hard-fail with a clear message instead of
    # producing a cryptic torch.cuda error 200 lines later.
    if device is None:
        device = torch.device("cuda")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 3 training requires a CUDA GPU (gsplat is CUDA-only). "
            "Run this on the i9 Linux workstation / cluster GPU."
        )

    out_dir = Path(out_dir)
    (out_dir / "renders").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "exports").mkdir(parents=True, exist_ok=True)

    # ---- reproducibility --------------------------------------------
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- model + optimizer ------------------------------------------
    model = GaussianModel.from_semantic_pointcloud(init_pc).to(device)
    print(f"[train] initialised model with {model.num_gaussians:,} Gaussians")
    print(f"[train] device = {device}  ({torch.cuda.get_device_name(0)})")

    # ---- optional synthetic sky dome (free-view backdrop) -----------
    # Appended BEFORE the optimizer so its parameters join the param groups.
    # Sky pixels are masked out of the photometric loss, so dome Gaussians
    # receive ~no gradient and stay put as a clean, parallax-free backdrop.
    if cfg.sky_dome:
        from semantic_gs.model.sky_dome import (
            SkyDomeConfig,
            make_sky_dome_gaussians,
            scene_center_radius,
        )
        mn, mx = init_pc.bbox
        center, scene_radius = scene_center_radius(mn, mx)
        dome_cfg = SkyDomeConfig(
            enabled        = True,
            n_points       = cfg.sky_dome_points,
            radius_scale   = cfg.sky_dome_radius_scale,
            include_ground = cfg.sky_dome_ground,
        )
        dome = make_sky_dome_gaussians(
            center, scene_radius, dome_cfg, far_plane=cfg.far_plane,
        )
        model.append_gaussians(
            means=dome["means"], colors=dome["colors"],
            scales=dome["scales"], opacities=dome["opacities"],
        )
        print(
            f"[train] added sky dome: +{dome['means'].shape[0]:,} Gaussians "
            f"(radius {float(dome['radius'][0]):.1f} m, "
            f"ground={'on' if cfg.sky_dome_ground else 'off'}) -> "
            f"{model.num_gaussians:,} total"
        )

    lrs = {
        "means":   cfg.lr_means,   "colors":  cfg.lr_colors,
        "opacity": cfg.lr_opacity, "quats":   cfg.lr_quats,
        "scales":  cfg.lr_scales,
    }
    optimizer = torch.optim.Adam(model.param_groups(lrs), eps=1e-15)

    # ---- frame split ------------------------------------------------
    n_frames = len(loader)
    train_idx, eval_idx = _split_train_eval(n_frames, cfg.eval_stride)
    print(
        f"[train] {n_frames} frames total -> "
        f"{len(train_idx)} train / {len(eval_idx)} eval"
    )

    # Pre-cache eval frames; train frames are cached lazily on first hit
    # so we never re-decode a PNG more than once.
    eval_frames: list[Frame] = [loader[i] for i in eval_idx]
    train_cache: dict[int, Frame] = {}

    def _get_train_frame(idx: int) -> Frame:
        f = train_cache.get(idx)
        if f is None:
            f = loader[idx]
            train_cache[idx] = f
        return f

    # ---- metrics CSV ------------------------------------------------
    metrics_path = out_dir / "metrics.csv"
    csv_f        = metrics_path.open("w", newline="")
    csv_w        = csv.writer(csv_f)
    csv_w.writerow(
        ["iter", "phase", "loss", "l1", "ssim", "psnr",
         "n_gaussians", "seconds"]
    )

    # ---- training loop ----------------------------------------------
    t_start = time.time()
    for it in range(1, cfg.max_iters + 1):
        idx   = int(rng.choice(train_idx))
        frame = _get_train_frame(idx)
        gt_rgb  = _to_gpu_image(frame.rgb, device)
        gt_mask = _to_gpu_mask(frame.static_mask, device)

        out      = render_frame(
            model, frame,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            bg_color=cfg.bg_color,
        )
        pred_rgb = out["rgb"]            # NOTE: do not clamp before loss
        loss, comp = photometric_loss(
            pred_rgb, gt_rgb,
            mask=gt_mask, lambda_ssim=cfg.lambda_ssim,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if it == 1 or it % cfg.log_every == 0:
            elapsed = time.time() - t_start
            print(
                f"[train] iter {it:>5d}/{cfg.max_iters}  "
                f"frame={frame.frame_id}  loss={comp['loss']:.4f}  "
                f"l1={comp['l1']:.4f}  ssim={comp['ssim']:.3f}  "
                f"N={model.num_gaussians:,}  ({elapsed:.0f}s)"
            )
            csv_w.writerow([
                it, "train", comp["loss"], comp["l1"], comp["ssim"],
                "", model.num_gaussians, f"{elapsed:.1f}",
            ])
            csv_f.flush()

        # ---- periodic eval & checkpoint ----
        if it % cfg.eval_every == 0 or it == cfg.max_iters:
            evaluate(
                model, eval_frames, out_dir, it, cfg,
                csv_writer=csv_w, csv_file=csv_f,
                save_pngs=cfg.save_renders, t_start=t_start,
            )
        if it % cfg.ckpt_every == 0 or it == cfg.max_iters:
            ckpt_path = out_dir / "checkpoints" / f"iter_{it:06d}.pt"
            torch.save(
                {"model": model.state_dict(), "iter": it, "cfg": asdict(cfg)},
                ckpt_path,
            )
            print(f"[train] wrote checkpoint {ckpt_path}")

    csv_f.close()

    # ---- export PLY (Inria 3DGS, openable in SuperSplat etc.) --------
    from semantic_gs.export.ply import save_gaussians_ply
    ply_path = out_dir / "exports" / "gaussians_final.ply"
    save_gaussians_ply(model, ply_path)
    print(
        f"[train] wrote 3DGS PLY {ply_path}  "
        f"({ply_path.stat().st_size / 1e6:.1f} MB)"
    )

    # ---- final evaluate (no PNGs, just numbers) ----------------------
    final = evaluate(
        model, eval_frames, out_dir, cfg.max_iters, cfg,
        csv_writer=None, csv_file=None,
        save_pngs=False, t_start=t_start, label="final",
    )

    summary = {
        "n_gaussians":           model.num_gaussians,
        "n_train_frames":        len(train_idx),
        "n_eval_frames":         len(eval_idx),
        "final_eval_psnr_mean":  final["psnr_mean"],
        "final_eval_ssim_mean":  final["ssim_mean"],
        "final_eval_psnr_per_frame": final["psnr_per_frame"],
        "final_eval_ssim_per_frame": final["ssim_per_frame"],
        "ply_path":              str(ply_path),
        "metrics_csv":           str(metrics_path),
        "config":                asdict(cfg),
        "loader":                loader.name,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] wrote summary {out_dir / 'summary.json'}")
    return summary



