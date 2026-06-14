"""Hermetic CPU tests for the Inria-format 3DGS PLY exporter."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from semantic_gs.data.pointcloud import SemanticPointCloud
from semantic_gs.export.ply import save_gaussians_ply, _rgb_to_sh0_dc, _SH_C0
from semantic_gs.model.gaussians import GaussianModel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_ply_header(path: Path) -> tuple[int, list[str], int]:
    """Return (n_vertices, list_of_float_field_names, header_byte_length)."""
    fields: list[str] = []
    n_vertices = 0
    with path.open("rb") as f:
        magic = f.readline().strip()
        assert magic == b"ply"
        fmt = f.readline().strip()
        assert fmt == b"format binary_little_endian 1.0"
        for line in f:
            stripped = line.strip()
            if stripped.startswith(b"element vertex "):
                n_vertices = int(stripped.split()[-1])
            elif stripped.startswith(b"property float"):
                fields.append(stripped.split()[-1].decode())
            elif stripped == b"end_header":
                return n_vertices, fields, f.tell()
    raise AssertionError("end_header not found")


def _tiny_model(n: int = 50, seed: int = 0) -> GaussianModel:
    rng = np.random.default_rng(seed)
    pc = SemanticPointCloud(
        xyz    = rng.normal(size=(n, 3)).astype(np.float32),
        rgb    = rng.uniform(size=(n, 3)).astype(np.float32),
        labels = np.zeros(n, dtype=np.int32),
    )
    return GaussianModel.from_semantic_pointcloud(pc)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_header_has_expected_field_order(tmp_path: Path) -> None:
    out = tmp_path / "g.ply"
    save_gaussians_ply(_tiny_model(n=80), out)
    n_read, fields, _ = _read_ply_header(out)
    assert n_read == 80
    assert fields == [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]


def test_body_byte_size_matches_expected(tmp_path: Path) -> None:
    n   = 64
    out = tmp_path / "g.ply"
    save_gaussians_ply(_tiny_model(n=n), out)
    _, _, header_end = _read_ply_header(out)
    body_bytes = out.stat().st_size - header_end
    assert body_bytes == n * 17 * 4   # 17 float32 columns


def test_first_vertex_position_roundtrips(tmp_path: Path) -> None:
    n   = 16
    model = _tiny_model(n=n)
    out = tmp_path / "g.ply"
    save_gaussians_ply(model, out)

    _, _, header_end = _read_ply_header(out)
    with out.open("rb") as f:
        f.seek(header_end)
        first = f.read(17 * 4)
    floats = struct.unpack("<17f", first)
    expected = model.means.detach().cpu().numpy()[0]
    np.testing.assert_allclose(floats[:3], expected, rtol=0, atol=1e-6)


def test_first_vertex_color_is_sh0_encoded(tmp_path: Path) -> None:
    n     = 8
    model = _tiny_model(n=n)
    out   = tmp_path / "g.ply"
    save_gaussians_ply(model, out)

    _, _, header_end = _read_ply_header(out)
    with out.open("rb") as f:
        f.seek(header_end)
        first = struct.unpack("<17f", f.read(17 * 4))
    f_dc = np.array(first[6:9], dtype=np.float32)
    rgb_clipped = np.clip(model.colors.detach().cpu().numpy()[0], 0.0, 1.0)
    expected_f_dc = (rgb_clipped - 0.5) / _SH_C0
    np.testing.assert_allclose(f_dc, expected_f_dc, rtol=1e-5, atol=1e-5)


def test_normals_are_zero(tmp_path: Path) -> None:
    out = tmp_path / "g.ply"
    save_gaussians_ply(_tiny_model(n=4), out)
    _, _, header_end = _read_ply_header(out)
    with out.open("rb") as f:
        f.seek(header_end)
        first = struct.unpack("<17f", f.read(17 * 4))
    assert first[3:6] == (0.0, 0.0, 0.0)


def test_rgb_to_sh0_round_trip() -> None:
    rgb = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    f_dc = _rgb_to_sh0_dc(rgb)
    # Recover rgb from SH0: rgb = f_dc * C0 + 0.5
    recovered = f_dc * _SH_C0 + 0.5
    np.testing.assert_allclose(recovered, rgb, atol=1e-6)

