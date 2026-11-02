"""
Ribbon Array & Waveguide Flange Coupling Geometry.
Models tightly packed line arrays of ribbon drivers (e.g. 4x 10x50mm ribbons)
integrated into a common horn/waveguide flange to study comb-filtering,
acoustic impedance matching, and high-frequency vertical directivity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from acoustics_platform.models import MeshStructure


@dataclass
class RibbonArrayWaveguideGeometry:
    """
    Parametric geometry for a vertical array of ribbon planar transducers
    coupled to a common waveguide horn flange.
    """
    num_ribbons: int = 4            # 4x ribbons in vertical line
    ribbon_width: float = 0.010     # 10 mm width
    ribbon_height: float = 0.050    # 50 mm height
    vertical_gap: float = 0.003     # 3 mm tight inter-element spacing
    flange_width: float = 0.120     # 120 mm mouth width
    flange_depth: float = 0.035     # 35 mm waveguide depth
    baffle_z: float = 0.0           # Reference Z plane

    @property
    def total_active_height(self) -> float:
        """Total vertical array span (m)."""
        return float(self.num_ribbons * self.ribbon_height + (self.num_ribbons - 1) * self.vertical_gap)

    def generate_mesh(self, target_elem_size: float = 0.008) -> MeshStructure:
        """
        Generates 3D surface mesh containing:
        - Individual ribbon diaphragm surfaces with physical tags ("ribbon_0", "ribbon_1", ...)
        - Waveguide sidewalls and top/bottom flares with physical tag "waveguide_flange"
        """
        nodes_list: List[List[float]] = []
        elements_list: List[List[int]] = []
        physical_tags: Dict[str, List[int]] = {
            "ribbons_all": [],
            "waveguide_flange": [],
        }
        for i in range(self.num_ribbons):
            physical_tags[f"ribbon_{i}"] = []

        rw_half = self.ribbon_width / 2.0
        rh = self.ribbon_height
        gap = self.vertical_gap
        total_h = self.total_active_height
        y_start = -total_h / 2.0

        fw_half = self.flange_width / 2.0
        fd = self.flange_depth

        def add_quad(p0, p1, p2, p3, tag_keys: List[str]):
            base = len(nodes_list)
            nodes_list.extend([p0, p1, p2, p3])
            t1 = [base, base + 1, base + 2]
            t2 = [base, base + 2, base + 3]
            e_start = len(elements_list)
            elements_list.extend([t1, t2])
            for k in tag_keys:
                physical_tags[k].extend([e_start, e_start + 1])

        # 1. Generate ribbon rectangular diaphragms at throat (Z = baffle_z)
        for i in range(self.num_ribbons):
            y_bot = y_start + i * (rh + gap)
            y_top = y_bot + rh

            # Subdivide ribbon face
            nx = max(2, int(np.ceil(self.ribbon_width / target_elem_size)))
            ny = max(4, int(np.ceil(rh / target_elem_size)))

            xs = np.linspace(-rw_half, rw_half, nx)
            ys = np.linspace(y_bot, y_top, ny)

            for ix in range(nx - 1):
                for iy in range(ny - 1):
                    p0 = [xs[ix], ys[iy], self.baffle_z]
                    p1 = [xs[ix + 1], ys[iy], self.baffle_z]
                    p2 = [xs[ix + 1], ys[iy + 1], self.baffle_z]
                    p3 = [xs[ix], ys[iy + 1], self.baffle_z]
                    add_quad(p0, p1, p2, p3, ["ribbons_all", f"ribbon_{i}"])

        # 2. Waveguide flange walls flaring out from throat (Z = baffle_z) to mouth (Z = baffle_z + fd)
        mouth_y_half = (total_h / 2.0) + 0.015 # slight vertical expansion
        z_mouth = self.baffle_z + fd

        # Left flange flare
        add_quad(
            [-rw_half, -total_h / 2.0, self.baffle_z],
            [-fw_half, -mouth_y_half, z_mouth],
            [-fw_half, mouth_y_half, z_mouth],
            [-rw_half, total_h / 2.0, self.baffle_z],
            ["waveguide_flange"],
        )

        # Right flange flare
        add_quad(
            [rw_half, -total_h / 2.0, self.baffle_z],
            [rw_half, total_h / 2.0, self.baffle_z],
            [fw_half, mouth_y_half, z_mouth],
            [fw_half, -mouth_y_half, z_mouth],
            ["waveguide_flange"],
        )

        # Top flare
        add_quad(
            [-rw_half, total_h / 2.0, self.baffle_z],
            [-fw_half, mouth_y_half, z_mouth],
            [fw_half, mouth_y_half, z_mouth],
            [rw_half, total_h / 2.0, self.baffle_z],
            ["waveguide_flange"],
        )

        # Bottom flare
        add_quad(
            [-rw_half, -total_h / 2.0, self.baffle_z],
            [rw_half, -total_h / 2.0, self.baffle_z],
            [fw_half, -mouth_y_half, z_mouth],
            [-fw_half, -mouth_y_half, z_mouth],
            ["waveguide_flange"],
        )

        nodes_np = np.array(nodes_list, dtype=np.float64)
        elems_np = np.array(elements_list, dtype=np.int64)
        phys_np = {k: np.array(v, dtype=np.int64) for k, v in physical_tags.items() if len(v) > 0}

        mesh = MeshStructure(nodes=nodes_np, elements=elems_np, element_type="tri3", physical_tags=phys_np)
        mesh.compute_geometry_properties()
        return mesh
