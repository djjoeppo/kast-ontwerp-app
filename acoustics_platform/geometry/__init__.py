"""
Parametric Geometry and CAD module.
"""

from acoustics_platform.geometry.cad import (
    ParametricCADGenerator,
    HornProfileGenerator,
    OCC_AVAILABLE,
)
from acoustics_platform.geometry.ribbon_array import RibbonArrayWaveguideGeometry
from acoustics_platform.geometry.assembly_parser import (
    ComponentTag,
    BoundingBox3D,
    TaggedComponent,
    MultiComponentAssemblyParser,
)
from acoustics_platform.geometry.constraints import (
    InternalBrace,
    ClearanceViolation,
    ValidationReport,
    ClearanceEngine,
)

__all__ = [
    "ParametricCADGenerator",
    "HornProfileGenerator",
    "OCC_AVAILABLE",
    "RibbonArrayWaveguideGeometry",
    "ComponentTag",
    "BoundingBox3D",
    "TaggedComponent",
    "MultiComponentAssemblyParser",
    "InternalBrace",
    "ClearanceViolation",
    "ValidationReport",
    "ClearanceEngine",
]
