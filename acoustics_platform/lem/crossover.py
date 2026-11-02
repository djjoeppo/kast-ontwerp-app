"""
Crossover Filter Network & Multi-Driver Synthesis.
Implements digital and analog crossover filters:
- Linkwitz-Riley (LR2, LR4) with -6dB crossover summation and flat magnitude response
- Butterworth (BW2, BW3, BW4) filters
- Multi-driver array complex velocity and sound pressure phase summation
- Configurable driver sections (e.g. Woofers, Midranges, Tweeters/Ribbons)
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from acoustics_platform.models import TSParameters, EnclosureGeometry
from acoustics_platform.lem.solver import LEMSolver, EnclosureLosses


class FilterType(str, Enum):
    LINKWITZ_RILEY_2ND = "LR2"
    LINKWITZ_RILEY_4TH = "LR4"
    BUTTERWORTH_2ND = "BW2"
    BUTTERWORTH_3RD = "BW3"
    BUTTERWORTH_4TH = "BW4"


class CrossoverFilter:
    """
    Computes complex s-domain transfer functions for crossover filter sections.
    """

    @staticmethod
    def lowpass(frequencies: np.ndarray, f_cross: float, order: FilterType = FilterType.LINKWITZ_RILEY_4TH) -> np.ndarray:
        s = 1j * (frequencies / max(f_cross, 1e-3))
        if order == FilterType.LINKWITZ_RILEY_4TH:
            # LR4 = (BW2)^2 -> 1 / (s^2 + sqrt(2)*s + 1)^2
            bw2 = 1.0 / (s**2 + np.sqrt(2.0) * s + 1.0)
            return bw2 ** 2
        elif order == FilterType.LINKWITZ_RILEY_2ND:
            # LR2 = (BW1)^2 -> 1 / (s + 1)^2
            return 1.0 / ((s + 1.0) ** 2)
        elif order == FilterType.BUTTERWORTH_2ND:
            return 1.0 / (s**2 + np.sqrt(2.0) * s + 1.0)
        elif order == FilterType.BUTTERWORTH_3RD:
            return 1.0 / ((s + 1.0) * (s**2 + s + 1.0))
        elif order == FilterType.BUTTERWORTH_4TH:
            return 1.0 / ((s**2 + 0.765367 * s + 1.0) * (s**2 + 1.847759 * s + 1.0))
        return np.ones_like(frequencies, dtype=complex)

    @staticmethod
    def highpass(frequencies: np.ndarray, f_cross: float, order: FilterType = FilterType.LINKWITZ_RILEY_4TH) -> np.ndarray:
        s = 1j * (frequencies / max(f_cross, 1e-3))
        if order == FilterType.LINKWITZ_RILEY_4TH:
            # LR4 highpass = (s^2 / (s^2 + sqrt(2)*s + 1))^2
            bw2_hp = (s**2) / (s**2 + np.sqrt(2.0) * s + 1.0)
            return bw2_hp ** 2
        elif order == FilterType.LINKWITZ_RILEY_2ND:
            return (s**2) / ((s + 1.0) ** 2)
        elif order == FilterType.BUTTERWORTH_2ND:
            return (s**2) / (s**2 + np.sqrt(2.0) * s + 1.0)
        elif order == FilterType.BUTTERWORTH_3RD:
            return (s**3) / ((s + 1.0) * (s**2 + s + 1.0))
        elif order == FilterType.BUTTERWORTH_4TH:
            return (s**4) / ((s**2 + 0.765367 * s + 1.0) * (s**2 + 1.847759 * s + 1.0))
        return np.ones_like(frequencies, dtype=complex)

    @staticmethod
    def bandpass(frequencies: np.ndarray, f_low: float, f_high: float, order: FilterType = FilterType.LINKWITZ_RILEY_4TH) -> np.ndarray:
        hp = CrossoverFilter.highpass(frequencies, f_low, order)
        lp = CrossoverFilter.lowpass(frequencies, f_high, order)
        return hp * lp


class DriverBranch:
    """
    Represents a multi-driver branch in a loudspeaker system (e.g. 2x 10" woofers, 4x ribbons).
    """
    def __init__(
        self,
        name: str,
        driver: TSParameters,
        count: int = 1,
        parallel: bool = True,
        enclosure: Optional[EnclosureGeometry] = None,
        filter_type: str = "bandpass", # "lowpass", "bandpass", "highpass"
        f_low: Optional[float] = None,
        f_high: Optional[float] = None,
        crossover_order: FilterType = FilterType.LINKWITZ_RILEY_4TH,
        polarity_invert: bool = False,
        delay_seconds: float = 0.0,
    ):
        self.name = name
        self.driver = driver
        self.count = count
        self.parallel = parallel
        self.enclosure = enclosure
        self.filter_type = filter_type
        self.f_low = f_low
        self.f_high = f_high
        self.crossover_order = crossover_order
        self.polarity_invert = polarity_invert
        self.delay_seconds = delay_seconds


class MultiDriverLEMSolver:
    """
    Multi-Driver Lumped Element Model & Crossover Solver.
    Computes branch transfer functions, individual driver complex velocities v_n,
    acoustic pressure contributions, and coherent acoustic phase summation at observation point.
    """

    def __init__(self, branches: List[DriverBranch]):
        self.branches = branches

    def solve(self, frequencies: np.ndarray, eg: float = 2.83) -> Dict[str, object]:
        total_p = np.zeros_like(frequencies, dtype=complex)
        branch_results: Dict[str, Dict[str, np.ndarray]] = {}

        w = 2.0 * np.pi * frequencies

        for branch in self.branches:
            # Multi-driver equivalence:
            # If 2 identical drivers in parallel: Sd_eff = count * Sd, Vas_eff = count * Vas, Re_eff = Re / count
            drv = branch.driver
            count = max(1, branch.count)
            sd_eff = drv.sd * count
            vas_eff = drv.vas * count
            re_eff = (drv.re / count) if branch.parallel else (drv.re * count)

            eff_driver = TSParameters(
                fs=drv.fs,
                qes=drv.qes,
                qms=drv.qms,
                vas=vas_eff,
                sd=sd_eff,
                xmax=drv.xmax,
                re=re_eff,
            )

            enc = branch.enclosure if branch.enclosure else EnclosureGeometry()
            solver = LEMSolver(eff_driver, enc)
            raw_res = solver.solve(frequencies, eg=eg)

            # Evaluate branch crossover filter transfer function H_xover
            if branch.filter_type == "lowpass" and branch.f_high:
                h_x = CrossoverFilter.lowpass(frequencies, branch.f_high, branch.crossover_order)
            elif branch.filter_type == "highpass" and branch.f_low:
                h_x = CrossoverFilter.highpass(frequencies, branch.f_low, branch.crossover_order)
            elif branch.filter_type == "bandpass" and branch.f_low and branch.f_high:
                h_x = CrossoverFilter.bandpass(frequencies, branch.f_low, branch.f_high, branch.crossover_order)
            else:
                h_x = np.ones_like(frequencies, dtype=complex)

            # Polarity inversion & time alignment delay: exp(-i * omega * delta_t)
            polarity_sign = -1.0 if branch.polarity_invert else 1.0
            time_delay_phase = np.exp(-1j * w * branch.delay_seconds)
            h_branch = h_x * polarity_sign * time_delay_phase

            # Filtered responses
            p_branch = raw_res["p_total"] * h_branch
            v_n_branch = raw_res["v_n"] * h_branch

            p_ref = 20e-6
            spl_branch = 20.0 * np.log10(np.maximum(np.abs(p_branch) / p_ref, 1e-6))

            branch_results[branch.name] = {
                "p_complex": p_branch,
                "v_n": v_n_branch,
                "spl": spl_branch,
                "h_crossover": h_branch,
                "raw_solver_res": raw_res,
            }

            total_p += p_branch

        p_ref = 20e-6
        spl_total = 20.0 * np.log10(np.maximum(np.abs(total_p) / p_ref, 1e-6))
        phase_total = np.angle(total_p, deg=True)

        return {
            "frequencies": frequencies,
            "p_total": total_p,
            "spl_total": spl_total,
            "phase_total": phase_total,
            "branch_results": branch_results,
        }
