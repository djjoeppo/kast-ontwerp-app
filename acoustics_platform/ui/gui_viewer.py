"""
Visual 3D Color Coding in PyVista Viewer.
Renders real-time visual feedback according to specification rules:
- Blue / Green: Safe zones, correct clearance, active vibrating membranes.
- Gray / Metallic: Solid enclosure walls and internal structures.
- Bright Red: Warning / clearance violation breach (collisions with magnet, surround, basket, or wall).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

from acoustics_platform.models import EnclosureGeometry, MeshStructure
from acoustics_platform.geometry.assembly_parser import TaggedComponent, ComponentTag
from acoustics_platform.geometry.constraints import InternalBrace, ValidationReport


class Layout3DColorMapper:
    """
    Assigns color codes to layout geometry parts based on component tags and clearance checks:
    - Safe membrane (cone/surround): Green (#2e7d32) or Blue (#1976d2)
    - Cabinet walls & braces: Metallic gray (#90a4ae / #546e7a)
    - Collision / breach: Bright Red (#d32f2f)
    """

    COLOR_MEMBRANE_SAFE = "#1976d2"   # Blue / Green safe zone
    COLOR_SURROUND_SAFE = "#388e3c"   # Green
    COLOR_STRUCTURE_GRAY = "#78909c"  # Gray / Metallic
    COLOR_MAGNET_SAFE = "#455a64"     # Dark metallic
    COLOR_BREACH_ALERT = "#d32f2f"    # Bright Red

    @classmethod
    def get_component_color(cls, comp: TaggedComponent, is_breached: bool) -> str:
        if is_breached:
            return cls.COLOR_BREACH_ALERT
        if comp.tag == ComponentTag.CONE:
            return cls.COLOR_MEMBRANE_SAFE
        if comp.tag == ComponentTag.SURROUND:
            return cls.COLOR_SURROUND_SAFE
        if comp.tag in [ComponentTag.MAGNET, ComponentTag.BASKET]:
            return cls.COLOR_MAGNET_SAFE
        return cls.COLOR_STRUCTURE_GRAY

    @classmethod
    def build_colored_actors(
        cls,
        plotter: pv.Plotter,
        enclosure: EnclosureGeometry,
        components: List[TaggedComponent],
        braces: Optional[List[InternalBrace]] = None,
        validation_report: Optional[ValidationReport] = None,
    ) -> None:
        """
        Populates a PyVista plotter with colored 3D meshes for enclosure, components, and braces.
        """
        if not PYVISTA_AVAILABLE:
            return

        breached_names = set(validation_report.components_in_breach) if validation_report else set()

        # 1. Add translucent cabinet shell
        w = enclosure.external_width
        h = enclosure.external_height
        d = enclosure.external_depth
        box_mesh = pv.Box(bounds=(-w/2, w/2, -h/2, h/2, -d/2, d/2))
        plotter.add_mesh(box_mesh, color=cls.COLOR_STRUCTURE_GRAY, opacity=0.25, style="wireframe", line_width=2)

        # 2. Add tagged components
        for comp in components:
            if len(comp.nodes) == 0 or len(comp.elements) == 0:
                continue
            cells = np.hstack([[3, elem[0], elem[1], elem[2]] for elem in comp.elements])
            pv_mesh = pv.PolyData(comp.nodes, cells)
            is_breached = comp.name in breached_names
            color = cls.get_component_color(comp, is_breached)
            opacity = 0.9 if not is_breached else 1.0
            plotter.add_mesh(pv_mesh, color=color, opacity=opacity, show_edges=True)

        # 3. Add internal braces
        if braces:
            for brace in braces:
                c = brace.center
                dim = brace.dimensions
                b_mesh = pv.Box(bounds=(
                    c[0] - dim[0]/2, c[0] + dim[0]/2,
                    c[1] - dim[1]/2, c[1] + dim[1]/2,
                    c[2] - dim[2]/2, c[2] + dim[2]/2,
                ))
                is_b_breached = brace.name in breached_names
                b_color = cls.COLOR_BREACH_ALERT if is_b_breached else cls.COLOR_STRUCTURE_GRAY
                plotter.add_mesh(b_mesh, color=b_color, opacity=0.8, show_edges=True)
