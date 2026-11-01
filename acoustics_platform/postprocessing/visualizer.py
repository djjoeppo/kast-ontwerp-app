"""
Post-Processing, Polar Directivity & Visualization Module.
Calculates 1m on-axis SPL, phase, electrical impedance, 2D/3D directivity maps,
and provides visualization exports via PyVista and Matplotlib.
Corresponds to REQ-POST-001.
"""

from __future__ import annotations
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

from acoustics_platform.models import (
    SimulationResult,
    MeshStructure,
)

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


class DirectivityProcessor:
    """
    Computes acoustic radiation patterns, polar directivity balloons,
    and Directivity Index (DI) across azimuth and elevation angles.
    """

    @staticmethod
    def generate_polar_observation_arc(
        distance: float = 1.0,
        num_angles: int = 37, # 0 to 180 degrees in 5-degree increments
        plane: str = "horizontal", # "horizontal" (XZ) or "vertical" (YZ)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate polar observation coordinates and angle array in degrees.
        Forward axis is +Z (0 degrees).
        """
        angles_deg = np.linspace(0.0, 180.0, num_angles)
        angles_rad = np.deg2rad(angles_deg)

        points = np.zeros((num_angles, 3))
        if plane == "horizontal":
            # X = r * sin(theta), Z = r * cos(theta), Y = 0
            points[:, 0] = distance * np.sin(angles_rad)
            points[:, 2] = distance * np.cos(angles_rad)
        else:
            # Y = r * sin(theta), Z = r * cos(theta), X = 0
            points[:, 1] = distance * np.sin(angles_rad)
            points[:, 2] = distance * np.cos(angles_rad)

        return points, angles_deg

    @staticmethod
    def compute_directivity_map(
        p_surface: np.ndarray,
        v_normal: np.ndarray,
        mesh: MeshStructure,
        k: float,
        angles_deg: np.ndarray,
        distance: float = 1.0,
        c0: float = 345.0,
        rho0: float = 1.18,
    ) -> np.ndarray:
        """
        Compute SPL polar response at a single frequency wavenumber k across angles.
        """
        from acoustics_platform.bem.solver import HelmholtzGreenFunction

        obs_points, _ = DirectivityProcessor.generate_polar_observation_arc(distance, len(angles_deg))

        # Collocation centroids
        p0 = mesh.nodes[mesh.elements[:, 0]]
        p1 = mesh.nodes[mesh.elements[:, 1]]
        p2 = mesh.nodes[mesh.elements[:, 2]]
        centroids = (p0 + p1 + p2) / 3.0

        omega = k * c0
        q = -1j * omega * rho0 * v_normal

        g_ext, dg_ext = HelmholtzGreenFunction.evaluate(obs_points, centroids, k, mesh.normals)
        p_far = np.sum((g_ext * q[None, :] - dg_ext * p_surface[None, :]) * mesh.areas[None, :], axis=1)

        p_ref = 20e-6
        spl = 20.0 * np.log10(np.maximum(np.abs(p_far) / p_ref, 1e-6))
        return spl


class AcousticVisualizer:
    """
    3D Field & Directivity Visualization using PyVista/VTK.
    """

    @staticmethod
    def render_surface_pressure_to_file(
        mesh: MeshStructure,
        p_surface: np.ndarray,
        filepath: str,
        colormap: str = "viridis",
    ) -> bool:
        """
        Renders surface acoustic pressure distribution onto the 3D mesh and saves screenshot.
        """
        if not PYVISTA_AVAILABLE:
            return False

        try:
            # PyVista polydata representation
            # Create cell connectivity
            n_cells = len(mesh.elements)
            cells = np.hstack([[3, elem[0], elem[1], elem[2]] for elem in mesh.elements])
            pv_mesh = pv.PolyData(mesh.nodes, cells)

            # Assign scalar pressure magnitude
            p_mag = np.abs(p_surface)
            pv_mesh.cell_data["Pressure_Magnitude"] = p_mag

            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(pv_mesh, scalars="Pressure_Magnitude", cmap=colormap, show_edges=True)
            plotter.add_text("3D Surface Acoustic Pressure (Pa)", font_size=12)
            plotter.view_isometric()
            plotter.screenshot(filepath)
            plotter.close()
            return True
        except Exception:
            return False
