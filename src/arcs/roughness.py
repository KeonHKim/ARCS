"""Gaussian conductivity-gradient roughness represented by BEM interfaces."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr
from .materials import Metal


@dataclass(frozen=True)
class Roughness:
    """Homogenized wall-depth model, not longitudinal surface corrugations.

    rms_height is in metres. The supplied aperture is the truncated air-side
    reference plane; the first mean metal surface is extent_sigma*rms_height
    outward from it. The profile is sigma_metal*Phi(x/rms_height) for a bare
    wall. Slices per 2*extent_sigma RMS transition control profile convergence.
    For offset coatings, each interface is smoothed with the same RMS height,
    or with interface_heights (one per vacuum/material and material/material
    transition). Permeability, permittivity and relaxation time must be common
    to the stack; this model smooths only conductivity.
    """
    rms_height: float
    slices: int = 32
    extent_sigma: float = 5.0
    interface_heights: tuple = ()

    def __post_init__(self):
        if isinstance(self.rms_height, (bool, np.bool_)) or not np.isfinite(self.rms_height) or self.rms_height <= 0:
            raise ValueError("Roughness RMS height must be positive and finite in metres; omit roughness for a smooth wall.")
        if isinstance(self.slices, (bool, np.bool_)) or not isinstance(self.slices, (int, np.integer)) or self.slices < 4:
            raise ValueError("Use at least 4 integer roughness slices; check convergence by doubling.")
        if not np.isfinite(self.extent_sigma) or self.extent_sigma < 4:
            raise ValueError("Use extent_sigma >= 4 for a truncated Gaussian profile.")
        heights = tuple(self.interface_heights)
        if any(not np.isfinite(x) or x <= 0 for x in heights):
            raise ValueError("Every interface RMS height must be positive in metres.")
        object.__setattr__(self, "interface_heights", heights)


def gradient_layers(wall, roughness):
    layers = [layer for layer in wall.layers if layer.thickness > 0]
    media = [layer.metal for layer in layers]+[wall.substrate]
    if any((m.mu_r, m.epsilon_r, m.relaxation_time) !=
           (media[0].mu_r, media[0].epsilon_r, media[0].relaxation_time) for m in media):
        raise NotImplementedError("Conductivity-gradient roughness requires common mu_r, epsilon_r and relaxation_time across the stack.")
    heights = np.asarray(roughness.interface_heights or (roughness.rms_height,)*len(media))
    if len(heights) != len(media):
        raise ValueError("Supply one interface_heights entry per transition, including vacuum to metal.")
    extent = roughness.extent_sigma
    means = extent*heights[0]+np.r_[0., np.cumsum([x.thickness for x in layers])]
    outer = means[-1]+extent*heights[-1]
    depth = np.unique(np.r_[0., outer, *[np.clip(
        mean+np.linspace(-extent*height, extent*height, roughness.slices+1), 0., outer)
        for mean,height in zip(means,heights)]])
    midpoint = (depth[:-1]+depth[1:])/2
    jumps = np.diff(np.r_[0., [m.conductivity for m in media]])
    sigma = ndtr((midpoint[:, None]-means)/heights) @ jumps
    if np.any(sigma <= 0):
        raise ValueError("The selected rough interface profiles produce nonpositive conductivity.")
    base = media[0]
    materials = [Metal(float(s), name=f"gradient_{i}", mu_r=base.mu_r,
                       epsilon_r=base.epsilon_r, relaxation_time=base.relaxation_time)
                 for i,s in enumerate(sigma)] + [wall.substrate]
    return depth, materials
