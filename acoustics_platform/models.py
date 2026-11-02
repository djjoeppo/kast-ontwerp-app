"""
Core data models for acoustics simulation and enclosure design.
Defines strongly typed structures for driver parameters, enclosure geometries,
meshes, and simulation outputs.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np


class EnclosureType(str, Enum):
    """Enclosure topology type (Kastvorm / Behuizingstopologie)."""
    CLOSED = "closed"          # Gesloten kast
    VENTED = "vented"          # Basreflex
    BANDPASS_4TH = "bandpass_4th"  # 4e orde bandpass
    HORN = "horn"              # Hoornconstructie


class FlowRegime(str, Enum):
    """Aerodynamic flow regime in reflex port (Stromingsregimes in de poort)."""
    LAMINAR = "laminar"              # Re < 2000
    TRANSITIONAL = "transitional"    # 2000 <= Re < 10000
    TURBULENT = "turbulent"          # Re >= 10000 (chuffing)


class SolverMode(str, Enum):
    """Simulation solver routing mode."""
    RIGOROUS_BEM = "bem"             # Rigoureuze BEM solver
    AI_SURROGATE = "ai_surrogate"     # U-FNO / LU-NO neuraal surrogaat
    HYBRID_AUTO = "hybrid_auto"       # AI met automatische fallback naar BEM


@dataclass
class TSParameters:
    """
    Thiele/Small lumped element driver parameters (T/S parameters).
    All units are SI base units unless specified otherwise.
    """
    fs: float                # Free air resonance frequency (Hz)
    qts: Optional[float] = None  # Total Q factor
    qes: Optional[float] = None  # Electrical Q factor
    qms: Optional[float] = None  # Mechanical Q factor
    vas: float = 0.0         # Equivalent acoustic compliance volume (m^3 or converted from L)
    sd: float = 0.0          # Diaphragm piston area (m^2)
    cms: Optional[float] = None  # Mechanical suspension compliance (m/N)
    mms: Optional[float] = None  # Total moving mass including air load (kg)
    xmax: float = 0.005      # Peak linear excursion limit (m)
    re: float = 6.0          # DC voice coil resistance (Ohms)
    bl: Optional[float] = None   # Force factor (T*m or N/A)
    le: float = 0.001        # Voice coil inductance (H)
    pe_max: float = 100.0    # Continuous thermal power rating (W)

    # Physical ambient constants
    rho0: float = 1.18       # Air density (kg/m^3)
    c0: float = 345.0        # Speed of sound in air (m/s)

    def __post_init__(self):
        """Synthesize missing interdependent T/S parameters."""
        self._synthesize_parameters()

    def _synthesize_parameters(self) -> None:
        """Derive consistent mechanical & electrical parameters."""
        omega_s = 2.0 * np.pi * self.fs

        # If Cms or Mms are missing, derive them from Vas and Fs
        if self.vas > 0.0 and self.sd > 0.0:
            derived_cms = self.vas / (self.rho0 * (self.c0 ** 2) * (self.sd ** 2))
            if self.cms is None or self.cms <= 0:
                self.cms = float(derived_cms)

        if self.cms is not None and self.cms > 0.0:
            if self.mms is None or self.mms <= 0:
                self.mms = float(1.0 / ((omega_s ** 2) * self.cms))
            elif self.fs <= 0:
                self.fs = float(1.0 / (2.0 * np.pi * np.sqrt(self.cms * self.mms)))

        # Q factor relations
        if self.qts is None:
            if self.qes is not None and self.qms is not None:
                self.qts = float((self.qes * self.qms) / (self.qes + self.qms))
        elif self.qes is None and self.qms is not None:
            if self.qms != self.qts:
                self.qes = float((self.qts * self.qms) / (self.qms - self.qts))
        elif self.qms is None and self.qes is not None:
            if self.qes != self.qts:
                self.qms = float((self.qts * self.qes) / (self.qes - self.qts))

        # Re-derive Vas if not set
        if self.vas <= 0.0 and self.cms is not None and self.sd > 0.0:
            self.vas = float(self.rho0 * (self.c0 ** 2) * (self.sd ** 2) * self.cms)

        # Force factor BL estimate if missing
        if (self.bl is None or self.bl <= 0.0) and self.qes is not None and self.mms is not None:
            # Qes = (2*pi*Fs*Mms*Re) / (BL^2) => BL = sqrt((2*pi*Fs*Mms*Re)/Qes)
            self.bl = float(np.sqrt((omega_s * self.mms * self.re) / max(self.qes, 1e-6)))

    @property
    def eta0(self) -> float:
        """Reference power efficiency eta_0."""
        if self.qes is not None and self.qes > 0:
            return float(9.7822e-10 * self.vas * (self.fs ** 3) / self.qes)
        return 0.0

    @property
    def spl_1w_1m(self) -> float:
        """Theoretic Sound Pressure Level at 1 Watt / 1 meter (dB SPL)."""
        eff = max(self.eta0, 1e-12)
        return float(112.02 + 10.0 * np.log10(eff))

    @property
    def ebp(self) -> float:
        """Efficiency Bandwidth Product (EBP = Fs / Qes)."""
        if self.qes is not None and self.qes > 0:
            return float(self.fs / self.qes)
        return 0.0

    @property
    def vd(self) -> float:
        """Peak displacement volume V_d = Sd * Xmax (m^3)."""
        return float(self.sd * self.xmax)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fs": self.fs,
            "qts": self.qts,
            "qes": self.qes,
            "qms": self.qms,
            "vas": self.vas,
            "sd": self.sd,
            "cms": self.cms,
            "mms": self.mms,
            "xmax": self.xmax,
            "re": self.re,
            "bl": self.bl,
            "le": self.le,
            "pe_max": self.pe_max,
            "eta0": self.eta0,
            "spl_1w_1m": self.spl_1w_1m,
            "ebp": self.ebp,
            "vd": self.vd,
        }


@dataclass
class PortGeometry:
    """Parametric reflex port geometry (Poortgeometrie)."""
    diameter: float = 0.07          # Inner diameter (m)
    length: float = 0.20            # Physical port duct length (m)
    flared: bool = True             # Aerodynamic bellmouth flare
    flare_radius: float = 0.015     # Radius of flare curvature (m)
    position: Tuple[float, float, float] = (0.0, 0.15, 0.0) # Relative coordinate on baffle

    @property
    def area(self) -> float:
        """Cross-sectional duct area S_p (m^2)."""
        return float(np.pi * (self.diameter / 2.0) ** 2)

    @property
    def hydraulic_diameter(self) -> float:
        """Hydraulic diameter D_port (m)."""
        return float(self.diameter)

    @property
    def end_correction(self) -> float:
        """Acoustic end correction length (delta L). Flared port ~ 0.732 * r, unflared ~ 0.85 * r."""
        r = self.diameter / 2.0
        return float(0.732 * r if self.flared else 0.850 * r)

    @property
    def effective_length(self) -> float:
        """Total effective acoustic length L_eff = L_p + 2 * delta_L."""
        return float(self.length + 2.0 * self.end_correction)


@dataclass
class HornGeometry:
    """Acoustic horn / waveguide parametric curve."""
    throat_diameter: float = 0.0254 # 1-inch throat (m)
    mouth_width: float = 0.25       # Mouth width (m)
    mouth_height: float = 0.15      # Mouth height (m)
    length: float = 0.20            # Axial length (m)
    profile: str = "exponential"    # "conical", "exponential", "tractrix"
    cutoff_freq: float = 500.0      # Cutoff frequency (Hz)


@dataclass
class EnclosureGeometry:
    """3D CAD Parametric Enclosure Geometry definition."""
    enclosure_type: EnclosureType = EnclosureType.VENTED
    external_width: float = 0.28     # m
    external_height: float = 0.45    # m
    external_depth: float = 0.32     # m
    wall_thickness: float = 0.019    # 19mm MDF / Baltic Birch
    driver_radius: float = 0.10      # Cutout radius for driver (m)
    driver_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    chamfer_radius: float = 0.012    # Edge chamfer / roundover (m)
    port: Optional[PortGeometry] = field(default_factory=PortGeometry)
    horn: Optional[HornGeometry] = None

    @property
    def internal_width(self) -> float:
        return max(0.01, self.external_width - 2.0 * self.wall_thickness)

    @property
    def internal_height(self) -> float:
        return max(0.01, self.external_height - 2.0 * self.wall_thickness)

    @property
    def internal_depth(self) -> float:
        return max(0.01, self.external_depth - 2.0 * self.wall_thickness)

    @property
    def internal_volume(self) -> float:
        """Net internal air volume V_b (m^3)."""
        gross = self.internal_width * self.internal_height * self.internal_depth
        port_vol = (self.port.area * self.port.length) if (self.port and self.enclosure_type == EnclosureType.VENTED) else 0.0
        driver_displacement = 0.0015 # approx 1.5 L driver basket/magnet displacement
        return max(0.001, gross - port_vol - driver_displacement)


@dataclass
class MeshStructure:
    """
    Surface or Volume mesh representation with Physical Group tagging.
    """
    nodes: np.ndarray             # (N, 3) float64 array of vertex coordinates
    elements: np.ndarray          # (M, 3) or (M, 4) or (M, 6) int64 element connectivity
    element_type: str = "tri3"    # "tri3" (1st order), "tri6" (curvilinear 2nd order), "tet4"
    physical_tags: Dict[str, np.ndarray] = field(default_factory=dict) # Name -> array of element indices
    normals: Optional[np.ndarray] = None # (M, 3) element surface normals
    areas: Optional[np.ndarray] = None   # (M,) element surface areas

    def compute_geometry_properties(self) -> None:
        """Compute surface element normals and areas."""
        if self.element_type.startswith("tri") and len(self.elements) > 0:
            p0 = self.nodes[self.elements[:, 0]]
            p1 = self.nodes[self.elements[:, 1]]
            p2 = self.nodes[self.elements[:, 2]]
            cross = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(cross, axis=1, keepdims=True)
            norm_safe = np.where(norm < 1e-12, 1.0, norm)
            self.normals = cross / norm_safe
            self.areas = 0.5 * norm.squeeze(axis=-1)


@dataclass
class SimulationConfig:
    """Simulation run configuration."""
    frequencies: np.ndarray = field(default_factory=lambda: np.logspace(np.log10(20), np.log10(20000), 100))
    solver_mode: SolverMode = SolverMode.HYBRID_AUTO
    input_voltage: float = 2.83          # 2.83V RMS (nominal 1W @ 8 Ohm)
    f_max_mesh: float = 3000.0           # Maximum frequency for BEM mesh refinement
    observation_distance: float = 1.0    # Distance for SPL response (m)
    ai_confidence_threshold: float = 0.85 # Below this, fallback from U-FNO to BEM
    num_polar_angles: int = 37           # Number of polar angles for directivity (0 to 180 deg)


@dataclass
class SimulationResult:
    """Complete simulation outputs across LEM, BEM/AI, and aerodynamics."""
    frequencies: np.ndarray
    spl_total: np.ndarray                # Total Sound Pressure Level at observation distance (dB)
    spl_driver: np.ndarray               # Contribution from driver cone (dB)
    spl_port: Optional[np.ndarray] = None # Contribution from port (dB)
    phase_response: np.ndarray = field(default_factory=lambda: np.array([]))
    electrical_impedance: np.ndarray = field(default_factory=lambda: np.array([]))
    diaphragm_displacement: np.ndarray = field(default_factory=lambda: np.array([])) # Peak cone excursion (m)
    port_velocity: Optional[np.ndarray] = None   # Air particle velocity in port (m/s)
    reynolds_numbers: Optional[np.ndarray] = None# Dynamic Reynolds number in port
    flow_regime: Optional[FlowRegime] = None     # Maximum flow regime reached
    directivity_angles: Optional[np.ndarray] = None # Angles in radians/degrees
    directivity_spl_map: Optional[np.ndarray] = None # 2D matrix (num_freqs, num_angles)
    solver_used: SolverMode = SolverMode.RIGOROUS_BEM
    ai_confidence: Optional[float] = None
    pressure_field_grid: Optional[Dict[str, np.ndarray]] = None # {"x": ..., "y": ..., "z": ..., "pressure": ...}

    def to_json(self) -> str:
        data = {
            "frequencies": self.frequencies.tolist(),
            "spl_total": self.spl_total.tolist(),
            "solver_used": str(self.solver_used),
            "flow_regime": str(self.flow_regime) if self.flow_regime else None,
            "ai_confidence": self.ai_confidence,
        }
        return json.dumps(data, indent=2)
