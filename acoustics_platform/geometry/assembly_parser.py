"""
Multi-Component CAD & Metadata Tagging.
Parses multi-component objects (.obj, STEP/IGES, or structured geometries)
with labeled parts:
- flange: mounting flange and screw holes
- cone: vibrating diaphragm piston
- surround: flexible suspension and linear peak excursion envelope (X_max)
- basket: chassis frame structure
- magnet: motor assembly / rear clearance obstacle

Automatically associates component tags with driver Thiele/Small parameters and LEM models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from acoustics_platform.models import TSParameters, MeshStructure


class ComponentTag(str, Enum):
    FLANGE = "flange"
    CONE = "cone"
    SURROUND = "surround"
    BASKET = "basket"
    MAGNET = "magnet"
    ENCLOSURE_WALL = "enclosure_wall"
    BRACE = "brace"


@dataclass
class BoundingBox3D:
    """3D Axis-Aligned Bounding Box (AABB)."""
    min_pt: np.ndarray # (3,)
    max_pt: np.ndarray # (3,)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.min_pt + self.max_pt)

    @property
    def dimensions(self) -> np.ndarray:
        return np.maximum(0.0, self.max_pt - self.min_pt)

    def intersects(self, other: BoundingBox3D, margin: float = 0.0) -> bool:
        """Check if this AABB intersects with another given an extra safety margin."""
        return (
            (self.min_pt[0] - margin <= other.max_pt[0]) and (self.max_pt[0] + margin >= other.min_pt[0]) and
            (self.min_pt[1] - margin <= other.max_pt[1]) and (self.max_pt[1] + margin >= other.min_pt[1]) and
            (self.min_pt[2] - margin <= other.max_pt[2]) and (self.max_pt[2] + margin >= other.min_pt[2])
        )

    def distance_to(self, other: BoundingBox3D) -> float:
        """Calculate minimum Euclidean separation between two AABBs (0 if overlapping)."""
        delta = np.maximum(0.0, np.maximum(self.min_pt - other.max_pt, other.min_pt - self.max_pt))
        return float(np.linalg.norm(delta))


@dataclass
class TaggedComponent:
    """Individual tagged structural or acoustic part of a loudspeaker assembly."""
    name: str
    tag: ComponentTag
    nodes: np.ndarray            # (N, 3)
    elements: np.ndarray         # (M, 3)
    driver_params: Optional[TSParameters] = None
    excursion_envelope_z: float = 0.0 # Additional Xmax clearance needed in front/rear

    @property
    def bounds(self) -> BoundingBox3D:
        if len(self.nodes) == 0:
            return BoundingBox3D(np.zeros(3), np.zeros(3))
        min_p = np.min(self.nodes, axis=0).copy()
        max_p = np.max(self.nodes, axis=0).copy()
        if self.tag in [ComponentTag.CONE, ComponentTag.SURROUND] and self.excursion_envelope_z > 0:
            # Expand bounding box along Z for Xmax forward motion
            max_p[2] += self.excursion_envelope_z
        return BoundingBox3D(min_p, max_p)


class MultiComponentAssemblyParser:
    """
    Parses and builds multi-component loudspeaker assemblies with metadata tagging.
    """

    @classmethod
    def create_synthetic_driver_assembly(
        cls,
        name: str,
        driver_params: TSParameters,
        center_position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        orientation_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> List[TaggedComponent]:
        """
        Synthesizes a realistic geometric assembly of a loudspeaker driver with:
        - Flange (annular ring at baffle plane)
        - Cone (inner diaphragm)
        - Surround (outer roll suspension with Xmax envelope)
        - Basket (rear chassis frame)
        - Magnet (heavy cylindrical motor obstacle at back)
        """
        cx, cy, cz = center_position
        r_cone = np.sqrt(max(driver_params.sd, 1e-4) / np.pi)
        r_surround = r_cone * 1.15
        r_flange = r_cone * 1.25
        depth_cone = r_cone * 0.35
        depth_basket = r_cone * 0.70
        r_magnet = r_cone * 0.65
        depth_magnet = r_cone * 0.40
        xmax = driver_params.xmax

        n_pts = 24
        theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
        components: List[TaggedComponent] = []

        # 1. Cone (diaphragm)
        cone_nodes = [[cx, cy, cz - depth_cone]]
        for th in theta:
            cone_nodes.append([cx + r_cone * np.cos(th), cy + r_cone * np.sin(th), cz])
        cone_elems = []
        for i in range(n_pts):
            cone_elems.append([0, 1 + i, 1 + ((i + 1) % n_pts)])
        components.append(TaggedComponent(
            name=f"{name}_cone",
            tag=ComponentTag.CONE,
            nodes=np.array(cone_nodes, dtype=float),
            elements=np.array(cone_elems, dtype=int),
            driver_params=driver_params,
            excursion_envelope_z=xmax,
        ))

        # 2. Surround (suspension with Xmax clearance envelope)
        surr_nodes = []
        surr_elems = []
        for th in theta:
            surr_nodes.append([cx + r_cone * np.cos(th), cy + r_cone * np.sin(th), cz])
            surr_nodes.append([cx + r_surround * np.cos(th), cy + r_surround * np.sin(th), cz])
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            p0 = 2 * i
            p1 = 2 * i + 1
            p2 = 2 * next_i + 1
            p3 = 2 * next_i
            surr_elems.append([p0, p1, p2])
            surr_elems.append([p0, p2, p3])
        components.append(TaggedComponent(
            name=f"{name}_surround",
            tag=ComponentTag.SURROUND,
            nodes=np.array(surr_nodes, dtype=float),
            elements=np.array(surr_elems, dtype=int),
            driver_params=driver_params,
            excursion_envelope_z=xmax,
        ))

        # 3. Flange (mounting rim)
        flange_nodes = []
        flange_elems = []
        for th in theta:
            flange_nodes.append([cx + r_surround * np.cos(th), cy + r_surround * np.sin(th), cz])
            flange_nodes.append([cx + r_flange * np.cos(th), cy + r_flange * np.sin(th), cz])
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            p0 = 2 * i
            p1 = 2 * i + 1
            p2 = 2 * next_i + 1
            p3 = 2 * next_i
            flange_elems.append([p0, p1, p2])
            flange_elems.append([p0, p2, p3])
        components.append(TaggedComponent(
            name=f"{name}_flange",
            tag=ComponentTag.FLANGE,
            nodes=np.array(flange_nodes, dtype=float),
            elements=np.array(flange_elems, dtype=int),
            driver_params=driver_params,
        ))

        # 4. Basket (chassis struts extending backwards)
        basket_nodes = []
        basket_elems = []
        z_basket_rear = cz - depth_basket
        for th in theta:
            basket_nodes.append([cx + r_cone * np.cos(th), cy + r_cone * np.sin(th), cz])
            basket_nodes.append([cx + r_magnet * np.cos(th), cy + r_magnet * np.sin(th), z_basket_rear])
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            p0 = 2 * i
            p1 = 2 * i + 1
            p2 = 2 * next_i + 1
            p3 = 2 * next_i
            basket_elems.append([p0, p1, p2])
            basket_elems.append([p0, p2, p3])
        components.append(TaggedComponent(
            name=f"{name}_basket",
            tag=ComponentTag.BASKET,
            nodes=np.array(basket_nodes, dtype=float),
            elements=np.array(basket_elems, dtype=int),
            driver_params=driver_params,
        ))

        # 5. Magnet (rear cylindrical motor obstacle)
        mag_nodes = []
        mag_elems = []
        z_mag_front = z_basket_rear
        z_mag_back = z_mag_front - depth_magnet
        # Front circle, rear circle, center front, center back
        cf_idx = 2 * n_pts
        cb_idx = 2 * n_pts + 1
        for th in theta:
            mag_nodes.append([cx + r_magnet * np.cos(th), cy + r_magnet * np.sin(th), z_mag_front])
            mag_nodes.append([cx + r_magnet * np.cos(th), cy + r_magnet * np.sin(th), z_mag_back])
        mag_nodes.append([cx, cy, z_mag_front])
        mag_nodes.append([cx, cy, z_mag_back])

        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            # Side cylinder
            p0 = 2 * i
            p1 = 2 * i + 1
            p2 = 2 * next_i + 1
            p3 = 2 * next_i
            mag_elems.append([p0, p1, p2])
            mag_elems.append([p0, p2, p3])
            # Back cap
            mag_elems.append([cb_idx, 2 * i + 1, 2 * next_i + 1])

        components.append(TaggedComponent(
            name=f"{name}_magnet",
            tag=ComponentTag.MAGNET,
            nodes=np.array(mag_nodes, dtype=float),
            elements=np.array(mag_elems, dtype=int),
            driver_params=driver_params,
        ))

        return components

    @classmethod
    def parse_tagged_obj(cls, filepath: str, driver_params: Optional[TSParameters] = None) -> List[TaggedComponent]:
        """
        Parses Wavefront .obj files with group/object tags (e.g. 'g flange', 'o magnet').
        """
        components: List[TaggedComponent] = []
        current_name = "default"
        current_tag = ComponentTag.ENCLOSURE_WALL
        all_verts: List[List[float]] = []
        group_faces: Dict[str, Tuple[ComponentTag, List[List[int]]]] = {}

        tag_map = {
            "flange": ComponentTag.FLANGE,
            "cone": ComponentTag.CONE,
            "surround": ComponentTag.SURROUND,
            "basket": ComponentTag.BASKET,
            "magnet": ComponentTag.MAGNET,
            "brace": ComponentTag.BRACE,
            "wall": ComponentTag.ENCLOSURE_WALL,
        }

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()[1:4]
                    all_verts.append([float(p) for p in parts])
                elif line.startswith("g ") or line.startswith("o "):
                    tag_str = line.split()[1].lower()
                    current_name = tag_str
                    current_tag = tag_map.get(tag_str, ComponentTag.ENCLOSURE_WALL)
                    if current_name not in group_faces:
                        group_faces[current_name] = (current_tag, [])
                elif line.startswith("f "):
                    parts = line.split()[1:4]
                    # Convert 1-indexed to 0-indexed
                    face = [int(p.split("/")[0]) - 1 for p in parts]
                    if current_name not in group_faces:
                        group_faces[current_name] = (current_tag, [])
                    group_faces[current_name][1].append(face)

        verts_np = np.array(all_verts, dtype=float)
        for name, (tag, faces) in group_faces.items():
            if len(faces) == 0:
                continue
            used_vert_indices = np.unique(np.array(faces).flatten())
            vert_map = {old_i: new_i for new_i, old_i in enumerate(used_vert_indices)}
            c_nodes = verts_np[used_vert_indices]
            c_elems = np.array([[vert_map[v] for v in f] for f in faces], dtype=int)
            components.append(TaggedComponent(
                name=name,
                tag=tag,
                nodes=c_nodes,
                elements=c_elems,
                driver_params=driver_params,
                excursion_envelope_z=driver_params.xmax if driver_params and tag in [ComponentTag.CONE, ComponentTag.SURROUND] else 0.0,
            ))

        return components
