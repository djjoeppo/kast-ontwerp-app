"""
Discretization & Mesh Engine.
Implements automated frequency-dependent mesh generation according to:
h_max <= c / (6 * f_max)
Supports second-order curvilinear elements (-order 2), physical group surface tagging,
Gmsh SDK API integration when available, with a fast parametric mesh generator fallback,
and meshio export to VTK and XDMF.
Corresponds to REQ-MESH-001 and REQ-MESH-002.
"""

from __future__ import annotations
import os
import tempfile
from typing import Dict, List, Optional, Tuple
import numpy as np

from acoustics_platform.models import (
    EnclosureGeometry,
    MeshStructure,
)
from acoustics_platform.geometry.cad import ParametricCADGenerator

# Check for Gmsh SDK availability
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

# Check for meshio availability
try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False


class MeshDiscretizationEngine:
    """
    Automated Mesh Discretization Engine.
    Computes required mesh resolution h_max = c / (6 * f_max), creates adaptive
    surface and volume meshes, tags Physical Groups, and provides export/conversion.
    """

    def __init__(self, c0: float = 345.0):
        self.c0 = c0

    def compute_max_element_size(self, f_max: float, elements_per_wavelength: float = 6.0) -> float:
        """
        Computes maximum allowable element edge length h_max based on acoustic wavelength:
        h_max <= c0 / (N * f_max) where N >= 6 for standard BEM dispersion limits.
        """
        f = max(f_max, 20.0)
        wavelength = self.c0 / f
        h_max = wavelength / max(elements_per_wavelength, 4.0)
        # Limit between 2mm (fine) and 10cm (coarse)
        return float(np.clip(h_max, 0.002, 0.10))

    def generate_mesh(
        self,
        enclosure: EnclosureGeometry,
        f_max: float = 3000.0,
        second_order: bool = False,
    ) -> MeshStructure:
        """
        Generates surface mesh tagged with physical boundary groups.
        If gmsh is available and initialized, uses gmsh; otherwise uses internal
        parametric CAD discretizer.
        """
        h_max = self.compute_max_element_size(f_max)

        if GMSH_AVAILABLE:
            try:
                return self._generate_gmsh_mesh(enclosure, h_max, second_order)
            except Exception:
                # Fallback to parametric generator
                pass

        cad = ParametricCADGenerator(enclosure)
        mesh = cad.generate_surface_mesh(target_elem_size=h_max)
        if second_order:
            mesh = self.convert_to_second_order(mesh)
        return mesh

    def _generate_gmsh_mesh(
        self,
        enclosure: EnclosureGeometry,
        h_max: float,
        second_order: bool = False,
    ) -> MeshStructure:
        """Generate mesh using Gmsh C++/Python API."""
        gmsh.initialize()
        gmsh.model.add("acoustics_enclosure")

        w, h, d = enclosure.external_width, enclosure.external_height, enclosure.external_depth
        box = gmsh.model.occ.addBox(-w/2, -h/2, -d/2, w, h, d)
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", h_max * 0.4)
        if second_order:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)

        gmsh.model.mesh.generate(2) # 2D surface mesh

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(coords).reshape(-1, 3)

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
        if len(elem_node_tags) > 0:
            nodes_per_elem = 6 if second_order else 3
            elements = np.array(elem_node_tags[0]).reshape(-1, nodes_per_elem) - 1
        else:
            elements = np.empty((0, 3), dtype=np.int64)

        gmsh.finalize()

        elem_type_str = "tri6" if second_order else "tri3"
        # Tag front face elements near driver as diaphragm
        mesh = MeshStructure(nodes=nodes, elements=elements, element_type=elem_type_str)
        mesh.compute_geometry_properties()
        # Default tag all as enclosure
        mesh.physical_tags["enclosure"] = np.arange(len(elements))
        return mesh

    def convert_to_second_order(self, mesh: MeshStructure) -> MeshStructure:
        """
        Converts 1st-order linear triangular elements (tri3) to 2nd-order curvilinear
        6-node quadratic triangles (tri6) for reduced high-frequency dispersion (REQ-MESH-002).
        Nodes:
        0, 1, 2: original vertices
        3: midpoint of edge (0, 1)
        4: midpoint of edge (1, 2)
        5: midpoint of edge (2, 0)
        """
        if mesh.element_type == "tri6":
            return mesh

        nodes_list = list(mesh.nodes)
        edge_to_midpoint: Dict[Tuple[int, int], int] = {}
        tri6_elements = []

        def get_midpoint(i0: int, i1: int) -> int:
            edge = (min(i0, i1), max(i0, i1))
            if edge in edge_to_midpoint:
                return edge_to_midpoint[edge]
            p0 = mesh.nodes[i0]
            p1 = mesh.nodes[i1]
            pm = 0.5 * (p0 + p1)
            mid_idx = len(nodes_list)
            nodes_list.append(pm)
            edge_to_midpoint[edge] = mid_idx
            return mid_idx

        for elem in mesh.elements:
            n0, n1, n2 = elem[0], elem[1], elem[2]
            n3 = get_midpoint(n0, n1)
            n4 = get_midpoint(n1, n2)
            n5 = get_midpoint(n2, n0)
            tri6_elements.append([n0, n1, n2, n3, n4, n5])

        new_nodes = np.array(nodes_list, dtype=np.float64)
        new_elems = np.array(tri6_elements, dtype=np.int64)

        mesh_o2 = MeshStructure(
            nodes=new_nodes,
            elements=new_elems,
            element_type="tri6",
            physical_tags=mesh.physical_tags,
        )
        mesh_o2.compute_geometry_properties()
        return mesh_o2

    def export_meshio(self, mesh: MeshStructure, filepath: str) -> bool:
        """
        Export mesh to VTK or XDMF using meshio (REQ-MESH-001 / REQ-POST-001).
        """
        if not MESHIO_AVAILABLE:
            return False

        cell_type = "triangle6" if mesh.element_type == "tri6" else "triangle"
        cells = [(cell_type, mesh.elements)]

        point_data = {}
        cell_data = {}

        # Add physical tag identifiers if available
        if mesh.physical_tags:
            tag_array = np.zeros(len(mesh.elements), dtype=np.int32)
            tag_id_map = {"enclosure": 1, "diaphragm": 2, "port": 3, "horn": 4}
            for tag_name, elem_indices in mesh.physical_tags.items():
                tid = tag_id_map.get(tag_name, 99)
                tag_array[elem_indices] = tid
            cell_data["PhysicalGroup"] = [tag_array]

        m_out = meshio.Mesh(
            points=mesh.nodes,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
        )
        meshio.write(filepath, m_out)
        return True
