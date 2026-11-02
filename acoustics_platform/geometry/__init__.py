"""
Parametric Geometry and CAD module.
"""

from acoustics_platform.geometry.cad import (
    ParametricCADGenerator,
    HornProfileGenerator,
    OCC_AVAILABLE,
)
from acoustics_platform.geometry.ribbon_array import RibbonArrayWaveguideGeometry

__all__ = [
    "ParametricCADGenerator",
    "HornProfileGenerator",
    "OCC_AVAILABLE",
    "RibbonArrayWaveguideGeometry",
]
