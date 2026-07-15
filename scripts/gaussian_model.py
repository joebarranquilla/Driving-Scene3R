# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from .general_utils import inverse_sigmoid, strip_symmetric, build_scaling_rotation
import math


class Gaussian:
    def __init__(
        self,
        aabb: list,
        sh_degree: int = 0,
        mininum_kernel_size: float = 0.0,
        scaling_bias: float = 0.01,
        opacity_bias: float = 0.1,
        scaling_activation: str = "exp",
        device="cuda",
    ):
        self.init_params = {
            "aabb": aabb,
            "sh_degree": sh_degree,
            "mininum_kernel_size": mininum_kernel_size,
            "scaling_bias": scaling_bias,
            "opacity_bias": opacity_bias,
            "scaling_activation": scaling_activation,
        }

        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.mininum_kernel_size = mininum_kernel_size
        self.scaling_bias = scaling_bias
        self.opacity_bias = opacity_bias
        self.scaling_activation_type = scaling_activation
        self.device = device
        self.aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.setup_functions()

        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._scaling = None
        self._rotation = None
        self._opacity = None

    def setup_functions(self):
        if self.scaling_activation_type == "exp":
            self.scaling_activation = torch.exp
            self.inverse_scaling_activation = torch.log
        elif self.scaling_activation_type == "softplus":
            self.scaling_activation = torch.nn.functional.softplus
            self.inverse_scaling_activation = softplus_inverse_scaling_activation

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.scale_bias = self.inverse_scaling_activation(
            torch.tensor(self.scaling_bias)
        ).cuda()
        self.rots_bias = torch.zeros((4)).cuda()
        self.rots_bias[0] = 1
        self.opacity_bias = self.inverse_opacity_activation(
            torch.tensor(self.opacity_bias)
        ).cuda()

    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        scales = self.scaling_activation(self._scaling + self.scale_bias)
        scales = torch.square(scales) + self.mininum_kernel_size**2
        scales = torch.sqrt(scales)
        return scales

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation + self.rots_bias[None, :])

    @property
    def get_xyz(self):
        return self._xyz * self.aabb[None, 3:] + self.aabb[None, :3]

    @property
    def get_features(self):
        return (
            torch.cat((self._features_dc, self._features_rest), dim=2)
            if self._features_rest is not None
            else self._features_dc
        )

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity + self.opacity_bias)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation + self.rots_bias[None, :]
        )

    def from_scaling(self, scales):
        scales = torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
        self._scaling = self.inverse_scaling_activation(scales) - self.scale_bias

    def from_rotation(self, rots):
        self._rotation = rots - self.rots_bias[None, :]

    def from_xyz(self, xyz):
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]

    def from_features(self, features):
        self._features_dc = features

    def from_opacity(self, opacities):
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l
    
    def scale_to_target_size(self, target_max_dim=4.5, center=True):
        """
        Uniformly scales the splat scene so its longest bounding-box
        side equals target_max_dim (in the same units as your xyz, e.g. meters).
        """
        with torch.no_grad():
            xyz = self.get_xyz.detach()
            bbox_min = xyz.min(dim=0).values
            bbox_max = xyz.max(dim=0).values
            extent = bbox_max - bbox_min
            current_max_dim = extent.max().item()

            if current_max_dim == 0:
                raise ValueError("Bounding box has zero extent, cannot scale.")

            factor = target_max_dim / current_max_dim

            if center:
                centroid = (bbox_min + bbox_max) / 2.0
                # move to origin, scale, keep at origin (or shift back if you prefer)
                self._xyz.sub_(centroid)
                self._xyz.mul_(factor)
            else:
                self._xyz.mul_(factor)

            # Scale the Gaussian extents too — critical step!
            self._scaling.add_(math.log(factor))

        return factor

    def save_ply(self, path):
        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = inverse_sigmoid(self.get_opacity).detach().cpu().numpy()
        scale = torch.log(self.get_scaling).detach().cpu().numpy()
        rotation = (self._rotation + self.rots_bias[None, :]).detach().cpu().numpy()

        # 1. Base attributes from the original 3DGS code
        names = self.construct_list_of_attributes()
        dtype_full = [(attribute, "f4") for attribute in names]
        
        # 2. FIX: Append explicit RGB fields to the PLY data structure
        dtype_full.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")]) # "u1" means uint8

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, opacities, scale, rotation), axis=1
        )
        attributes = attributes.astype("f4")

        # Fill base attributes
        for i, name in enumerate(names):
            elements[name] = attributes[:, i]

        # 3. Calculate RGB values safely
        try:
            if f_dc.shape[1] >= 3:
                rgb = f_dc[:, :3]
            else:
                rgb = np.repeat(f_dc[:, :1], 3, axis=1)
        except Exception:
            N = xyz.shape[0]
            rgb = np.zeros((N, 3), dtype=np.float32)

        # 4. Convert SH base to 0-255 RGB bytes. 
        # Note: 3DGS usually scales SH by a constant (0.28209). 
        # If the colors look deeply oversaturated or weird, you may need to use:
        # SH_C0 = 0.28209479177387814
        # rgb = 0.5 + SH_C0 * rgb
        rgb_bytes = np.clip((rgb + 0.5) * 255.0, 0, 255).astype(np.uint8)

        # 5. FIX: These will now work because "red", "green", "blue" are in dtype_full
        elements["red"] = rgb_bytes[:, 0]
        elements["green"] = rgb_bytes[:, 1]
        elements["blue"] = rgb_bytes[:, 2]

        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.sh_degree > 0:
            extra_f_names = [
                p.name
                for p in plydata.elements[0].properties
                if p.name.startswith("f_rest_")
            ]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            assert len(extra_f_names) == 3 * (self.sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # convert to actual gaussian attributes
        xyz = torch.tensor(xyz, dtype=torch.float, device=self.device)
        features_dc = (
            torch.tensor(features_dc, dtype=torch.float, device=self.device)
            .transpose(1, 2)
            .contiguous()
        )
        if self.sh_degree > 0:
            features_extra = (
                torch.tensor(features_extra, dtype=torch.float, device=self.device)
                .transpose(1, 2)
                .contiguous()
            )
        opacities = torch.sigmoid(
            torch.tensor(opacities, dtype=torch.float, device=self.device)
        )
        scales = torch.exp(torch.tensor(scales, dtype=torch.float, device=self.device))
        rots = torch.tensor(rots, dtype=torch.float, device=self.device)

        # convert to _hidden attributes
        self._xyz = (xyz - self.aabb[None, :3]) / self.aabb[None, 3:]
        self._features_dc = features_dc
        if self.sh_degree > 0:
            self._features_rest = features_extra
        else:
            self._features_rest = None
        self._opacity = self.inverse_opacity_activation(opacities) - self.opacity_bias
        self._scaling = (
            self.inverse_scaling_activation(
                torch.sqrt(torch.square(scales) - self.mininum_kernel_size**2)
            )
            - self.scale_bias
        )
        self._rotation = rots - self.rots_bias[None, :]

def softplus_inverse_scaling_activation(x):
    return x + torch.log(-torch.expm1(-x))