"""
Aerodynamics and Port Turbulence Module.
Calculates air particle velocity in the reflex duct, evaluates dynamic Reynolds number:
Re = (|v_port| * D_port) / nu_air
and classifies flow regimes:
- Laminar: Re < 2000
- Transitional: 2000 <= Re < 10000
- Fully Turbulent: Re >= 10000 (acoustic compression, chuffing noise, loss of QP)
Corresponds to REQ-AERO-001.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from acoustics_platform.models import FlowRegime, PortGeometry


class PortAerodynamicsAnalyzer:
    """
    Evaluates nonlinear duct flow dynamics, acoustic velocity, and turbulence regimes.
    """

    # Kinematic viscosity of air at 20 deg C (m^2 / s)
    NU_AIR: float = 1.5e-5

    @classmethod
    def calculate_reynolds_number(
        cls,
        v_port: np.ndarray,
        port: PortGeometry,
    ) -> np.ndarray:
        """
        Computes dynamic Reynolds number array across frequencies:
        Re = (|v_port| * D_port) / nu_air
        """
        v_mag = np.abs(v_port)
        d_port = port.hydraulic_diameter
        re = (v_mag * d_port) / cls.NU_AIR
        return re

    @classmethod
    def classify_regime(cls, reynolds_number: float) -> FlowRegime:
        """
        Classifies instantaneous or peak Reynolds number into physical flow regime:
        - Laminar: Re < 2000
        - Transitional: 2000 <= Re < 10000
        - Turbulent: Re >= 10000
        """
        if reynolds_number < 2000.0:
            return FlowRegime.LAMINAR
        elif reynolds_number < 10000.0:
            return FlowRegime.TRANSITIONAL
        else:
            return FlowRegime.TURBULENT

    @classmethod
    def evaluate_port_performance(
        cls,
        v_port: np.ndarray,
        port: PortGeometry,
    ) -> Dict[str, object]:
        """
        Comprehensive port turbulence and chuffing diagnosis.
        """
        re_arr = cls.calculate_reynolds_number(v_port, port)
        max_re = float(np.max(re_arr)) if len(re_arr) > 0 else 0.0
        regime = cls.classify_regime(max_re)

        # Estimate effective QP reduction due to turbulence:
        # QP starts near 50-80 laminar, drops to < 5 when turbulent
        if regime == FlowRegime.LAMINAR:
            qp_effective = 60.0
            chuffing_warning = False
        elif regime == FlowRegime.TRANSITIONAL:
            # Interpolate between 60 and 15
            frac = (max_re - 2000.0) / 8000.0
            qp_effective = float(60.0 - frac * 45.0)
            chuffing_warning = False
        else:
            qp_effective = float(max(2.0, 15.0 * (10000.0 / max(max_re, 10000.0))))
            chuffing_warning = True

        return {
            "reynolds_numbers": re_arr,
            "max_reynolds": max_re,
            "flow_regime": regime,
            "qp_effective": qp_effective,
            "chuffing_warning": chuffing_warning,
        }
