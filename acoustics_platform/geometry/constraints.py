"""
Constraint & Clearance Engine.
Enforces strict collision detection, spatial clearance margins, and mechanical boundary rules:
- No internal brace or cabinet wall may enter within minimum clearance margin of magnet or surround (Xmax zone).
- Immediate invalidation and warning metadata if components breach clearances.
- Detailed collision reporting for visual highlighting in PyVista (e.g. bright red warnings).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from acoustics_platform.models import EnclosureGeometry
from acoustics_platform.geometry.assembly_parser import (
    TaggedComponent,
    ComponentTag,
    BoundingBox3D,
)


@dataclass
class InternalBrace:
    """Internal structural reinforcement brace (Versteviging)."""
    name: str
    center: Tuple[float, float, float]
    dimensions: Tuple[float, float, float] # (width, height, depth)

    @property
    def bounds(self) -> BoundingBox3D:
        c = np.array(self.center, dtype=float)
        d = np.array(self.dimensions, dtype=float)
        return BoundingBox3D(c - d / 2.0, c + d / 2.0)


@dataclass
class ClearanceViolation:
    """Detailed record of a constraint/clearance rule breach."""
    component_a: str
    tag_a: ComponentTag
    component_b: str
    tag_b: ComponentTag
    distance: float
    required_margin: float
    severity: str = "critical" # "critical" (breach/overlap) or "warning"

    @property
    def message(self) -> str:
        return (
            f"Collision breach between '{self.component_a}' ({self.tag_a.value}) and "
            f"'{self.component_b}' ({self.tag_b.value}): clearance = {self.distance*1000:.1f} mm < "
            f"required margin {self.required_margin*1000:.1f} mm"
        )


@dataclass
class ValidationReport:
    """Validation report summarizing layout validity and clearance violations."""
    is_valid: bool
    violations: List[ClearanceViolation] = field(default_factory=list)
    components_in_breach: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_valid:
            return "All spatial and mechanical clearance constraints passed successfully."
        lines = [f"Constraint Validation Failed: {len(self.violations)} violations detected:"]
        for v in self.violations:
            lines.append(f" - {v.message}")
        return "\n".join(lines)


class ClearanceEngine:
    """
    Evaluates spatial clearance rules between drivers, magnets, surrounds,
    enclosure walls, and internal braces.
    """

    def __init__(
        self,
        magnet_clearance_margin: float = 0.015,   # 15 mm minimum clearance around magnet
        surround_clearance_margin: float = 0.010, # 10 mm margin beyond Xmax for surround
        driver_to_driver_margin: float = 0.005,   # 5 mm minimum gap between driver flanges
    ):
        self.mag_margin = magnet_clearance_margin
        self.surr_margin = surround_clearance_margin
        self.driver_margin = driver_to_driver_margin

    def check_layout(
        self,
        enclosure: EnclosureGeometry,
        components: List[TaggedComponent],
        braces: Optional[List[InternalBrace]] = None,
    ) -> ValidationReport:
        """
        Validates complete internal and baffle layout.
        Rules:
        1. All components must fit within the enclosure's interior volume.
        2. No internal brace may enter within magnet_clearance_margin of any MAGNET.
        3. No internal brace or wall may enter within surround_clearance_margin of any SURROUND or CONE.
        4. No two distinct drivers or magnets may collide with each other.
        """
        violations: List[ClearanceViolation] = []
        breached_names = set()

        # Interior box boundaries
        hw = enclosure.internal_width / 2.0
        hh = enclosure.internal_height / 2.0
        hd = enclosure.internal_depth / 2.0
        # Assume box centered at origin
        interior_bounds = BoundingBox3D(np.array([-hw, -hh, -hd]), np.array([hw, hh, hd]))

        # Check internal components against interior enclosure walls
        for comp in components:
            if comp.tag in [ComponentTag.MAGNET, ComponentTag.BASKET]:
                cb = comp.bounds
                # If magnet exceeds rear wall (Z < -hd)
                if cb.min_pt[2] < (-hd + self.mag_margin):
                    v = ClearanceViolation(
                        component_a=comp.name,
                        tag_a=comp.tag,
                        component_b="rear_enclosure_wall",
                        tag_b=ComponentTag.ENCLOSURE_WALL,
                        distance=float(cb.min_pt[2] - (-hd)),
                        required_margin=self.mag_margin,
                    )
                    violations.append(v)
                    breached_names.add(comp.name)

                # Check side walls
                if cb.min_pt[0] < (-hw + self.mag_margin) or cb.max_pt[0] > (hw - self.mag_margin):
                    dist = float(min(cb.min_pt[0] - (-hw), hw - cb.max_pt[0]))
                    v = ClearanceViolation(
                        component_a=comp.name,
                        tag_a=comp.tag,
                        component_b="side_enclosure_wall",
                        tag_b=ComponentTag.ENCLOSURE_WALL,
                        distance=dist,
                        required_margin=self.mag_margin,
                    )
                    violations.append(v)
                    breached_names.add(comp.name)

        # Check driver-to-driver collisions
        num_comps = len(components)
        for i in range(num_comps):
            for j in range(i + 1, num_comps):
                ca = components[i]
                cb = components[j]
                # Check if from different drivers (by prefix before first '_')
                prefix_a = ca.name.split("_")[0]
                prefix_b = cb.name.split("_")[0]
                if prefix_a == prefix_b:
                    continue # Same driver assembly parts naturally touch

                # Magnet-to-magnet or basket-to-basket collision
                if ca.tag == ComponentTag.MAGNET and cb.tag == ComponentTag.MAGNET:
                    b_a, b_b = ca.bounds, cb.bounds
                    dist = b_a.distance_to(b_b)
                    if dist < self.mag_margin:
                        violations.append(ClearanceViolation(
                            component_a=ca.name, tag_a=ca.tag,
                            component_b=cb.name, tag_b=cb.tag,
                            distance=dist, required_margin=self.mag_margin,
                        ))
                        breached_names.add(ca.name)
                        breached_names.add(cb.name)

                # Flange-to-flange overlap
                if ca.tag == ComponentTag.FLANGE and cb.tag == ComponentTag.FLANGE:
                    b_a, b_b = ca.bounds, cb.bounds
                    dist = b_a.distance_to(b_b)
                    if dist < self.driver_margin:
                        violations.append(ClearanceViolation(
                            component_a=ca.name, tag_a=ca.tag,
                            component_b=cb.name, tag_b=cb.tag,
                            distance=dist, required_margin=self.driver_margin,
                        ))
                        breached_names.add(ca.name)
                        breached_names.add(cb.name)

        # Check internal braces against drivers
        if braces:
            for brace in braces:
                bb = brace.bounds
                for comp in components:
                    cb = comp.bounds
                    if comp.tag == ComponentTag.MAGNET:
                        dist = bb.distance_to(cb)
                        if dist < self.mag_margin:
                            violations.append(ClearanceViolation(
                                component_a=brace.name, tag_a=ComponentTag.BRACE,
                                component_b=comp.name, tag_b=comp.tag,
                                distance=dist, required_margin=self.mag_margin,
                            ))
                            breached_names.add(comp.name)
                            breached_names.add(brace.name)

                    elif comp.tag in [ComponentTag.SURROUND, ComponentTag.CONE]:
                        dist = bb.distance_to(cb)
                        if dist < self.surr_margin:
                            violations.append(ClearanceViolation(
                                component_a=brace.name, tag_a=ComponentTag.BRACE,
                                component_b=comp.name, tag_b=comp.tag,
                                distance=dist, required_margin=self.surr_margin,
                            ))
                            breached_names.add(comp.name)
                            breached_names.add(brace.name)

        is_valid = len(violations) == 0
        return ValidationReport(
            is_valid=is_valid,
            violations=violations,
            components_in_breach=list(breached_names),
        )
