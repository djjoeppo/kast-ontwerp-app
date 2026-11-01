"""
Analytical Thiele/Small Lumped Element Modeling (LEM) Engine.
Implements electro-mechanical-acoustical network theory, Small's enclosure loss models,
and classic filter alignments (B4, QB3, Butterworth/Chebyshev).
Corresponds to REQ-LEM-001 and REQ-LEM-002.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np

from acoustics_platform.models import (
    TSParameters,
    EnclosureGeometry,
    EnclosureType,
    PortGeometry,
)


class EnclosureLosses:
    """
    Small's Enclosure Loss Model (Akoestisch analoog schema van Small).
    Total enclosure Q-factor QB at fb: 1/QB = 1/QL + 1/QA + 1/QP
    """
    def __init__(self, ql: float = 10.0, qa: float = 30.0, qp: float = 50.0):
        self.ql = float(ql)  # Leakage losses (typisch 5 <= QL <= 20)
        self.qa = float(qa)  # Absorption damping losses (QA >= 100 in unlined, 15-30 lined)
        self.qp = float(qp)  # Port viscous losses (QP ~ 20-100 laminar, < 5 turbulent)

    @property
    def qb(self) -> float:
        """Total box quality factor Q_B."""
        inv_qb = (1.0 / max(self.ql, 1e-4)) + (1.0 / max(self.qa, 1e-4)) + (1.0 / max(self.qp, 1e-4))
        return float(1.0 / inv_qb)


class LEMAlignmentSynthesizer:
    """
    Synthesizes optimal enclosure volumes and tuning frequencies based on classic alignments:
    - B4: 4th-order Butterworth flat response alignment
    - QB3: Quasi-Butterworth 3rd-order alignment (ideal for low Qts drivers, Qts < 0.38)
    - SC4: Sub-Chebyshev 4th-order alignment
    - Sealed: Butterworth B2 maximally flat alignment (Qtc = 0.707)
    """

    @staticmethod
    def calculate_b4_alignment(driver: TSParameters) -> Tuple[float, float]:
        """
        Calculates optimal box volume V_b (m^3) and tuning frequency f_b (Hz) for B4 alignment.
        For ideal B4, driver Qts = 0.383:
        Vb = Vas / alpha, with alpha approx 1.0 (or Vas * (Qts / 0.383)^2)
        fb = Fs
        """
        qts = driver.qts if driver.qts else 0.383
        # Small's polynomial fit for B4/vented
        # alpha = 11.0 * (Qts ** 3.3) approx for B4 target
        alpha = (0.383 / max(qts, 1e-3)) ** 2.87
        vb = driver.vas / max(alpha, 1e-3)
        fb = driver.fs * (driver.vas / max(vb, 1e-6)) ** 0.32
        return float(vb), float(fb)

    @staticmethod
    def calculate_qb3_alignment(driver: TSParameters) -> Tuple[float, float]:
        """
        Calculates optimal box volume V_b (m^3) and tuning frequency f_b (Hz) for QB3 alignment.
        QB3 alignment is suited for drivers with Qts < 0.38.
        alpha = Vas / Vb = (0.383 / Qts)^2.87 * (adjustment for QB3)
        """
        qts = driver.qts if driver.qts else 0.3
        alpha = 3.0 * (qts ** (-1.8)) * 0.3
        vb = driver.vas / max(alpha, 0.1)
        fb = driver.fs * (qts ** (-0.35)) * 0.95
        return float(vb), float(fb)

    @staticmethod
    def calculate_sealed_alignment(driver: TSParameters, qtc_target: float = 0.707) -> Tuple[float, float]:
        """
        Calculates optimal sealed box volume V_b (m^3) and system resonance f_c (Hz)
        for target system Q factor Qtc (default: Butterworth B2 Qtc = 0.707).
        Qtc / Qts = sqrt(1 + Vas / Vb) = fc / fs
        """
        qts = driver.qts if driver.qts else 0.4
        if qtc_target <= qts:
            qtc_target = qts * 1.05
        ratio_sq = (qtc_target / qts) ** 2
        alpha = ratio_sq - 1.0
        vb = driver.vas / max(alpha, 1e-3)
        fc = driver.fs * np.sqrt(1.0 + alpha)
        return float(vb), float(fc)


class LEMSolver:
    """
    Thiele/Small Lumped Element Model (LEM) Solver.
    Solves continuous electro-mechanical-acoustical network transfer functions.
    Computes:
    - Diaphragm normal particle velocity v_n(omega) (Neumann condition for BEM/AI)
    - Port particle velocity v_port(omega)
    - Electrical input impedance Z_in(omega)
    - Radiated sound pressure p(omega) and SPL at r = 1 m
    - Cone excursion X(omega)
    """

    def __init__(
        self,
        driver: TSParameters,
        enclosure: EnclosureGeometry,
        losses: Optional[EnclosureLosses] = None,
        tuning_freq: Optional[float] = None,
    ):
        self.driver = driver
        self.enclosure = enclosure
        self.losses = losses or EnclosureLosses()
        self.vb = enclosure.internal_volume

        # Determine tuning frequency fb
        if tuning_freq is not None and tuning_freq > 0:
            self.fb = float(tuning_freq)
        elif enclosure.port is not None and enclosure.enclosure_type == EnclosureType.VENTED:
            # Calculate Helmhotz resonance: fb = (c / 2*pi) * sqrt(Sp / (Vb * Leff))
            c = driver.c0
            sp = enclosure.port.area
            l_eff = enclosure.port.effective_length
            self.fb = float((c / (2.0 * np.pi)) * np.sqrt(sp / max(self.vb * l_eff, 1e-12)))
        else:
            self.fb = float(driver.fs)

    def solve(self, frequencies: np.ndarray, eg: float = 2.83) -> Dict[str, np.ndarray]:
        """
        Solve frequency response for given frequencies (Hz) and input voltage Eg (V RMS).
        Returns a dict of complex response arrays and magnitudes.
        """
        w = 2.0 * np.pi * frequencies
        s = 1j * w

        fs = self.driver.fs
        ws = 2.0 * np.pi * fs
        qms = self.driver.qms if self.driver.qms else 4.0
        qes = self.driver.qes if self.driver.qes else 0.4
        re = self.driver.re
        bl = self.driver.bl if self.driver.bl else 10.0
        mms = self.driver.mms if self.driver.mms else 0.05
        cms = self.driver.cms if self.driver.cms else (1.0 / (ws**2 * mms))
        sd = self.driver.sd if self.driver.sd else 0.035
        rho0 = self.driver.rho0
        c0 = self.driver.c0

        # Acoustic compliances
        # Driver suspension acoustic compliance
        cas = self.driver.vas / (rho0 * c0**2)
        # Enclosure acoustic compliance
        cab = self.vb / (rho0 * c0**2)

        if self.enclosure.enclosure_type == EnclosureType.CLOSED:
            # Closed box system parameters
            alpha = self.driver.vas / max(self.vb, 1e-6)
            fc = fs * np.sqrt(1.0 + alpha)
            wc = 2.0 * np.pi * fc
            qmc = qms * np.sqrt(1.0 + alpha)
            qec = qes * np.sqrt(1.0 + alpha)
            qtc = (qmc * qec) / (qmc + qec)

            # Normalized transfer function: H(s) = (s^2 / wc^2) / (s^2/wc^2 + s/(wc*Qtc) + 1)
            sn = s / wc
            h_s = (sn ** 2) / (sn ** 2 + sn / qtc + 1.0)

            # Diaphragm velocity: v_n = Eg * BL / (Re * Mms_total * ...)
            # Mechanical impedance of driver in box
            # Zm(s) = s*Mms + Rms + 1/(s*Cms_tot) + (BL^2)/(Re + s*Le)
            cms_tot = cms / (1.0 + alpha)
            rms = ws * mms / qms
            ze = re + s * self.driver.le
            zm = s * mms + rms + 1.0 / (s * max(cms_tot, 1e-12))
            # Total input electrical impedance
            z_in = ze + (bl ** 2) / zm

            # Voice coil current
            i_g = eg / z_in
            # Voice coil / cone velocity
            v_n = (bl * i_g) / zm
            # Cone displacement: x = v_n / s
            x_cone = np.abs(v_n / s)

            # Radiated far-field pressure at r = 1m (half-space / 2pi steradian ground plane):
            # p(r=1m) = (rho0 * s * Sd * v_n) / (2*pi*r)
            r = 1.0
            p_driver = (rho0 * s * sd * v_n) / (2.0 * np.pi * r)
            p_port = np.zeros_like(p_driver)
            p_total = p_driver

            v_port = np.zeros_like(frequencies, dtype=complex)

        else:
            # Basreflex (Vented Enclosure) 4th-order Small network
            wb = 2.0 * np.pi * self.fb
            alpha = self.driver.vas / max(self.vb, 1e-6)
            h = self.fb / fs
            qb = self.losses.qb

            # Small's high-pass 4th order transfer function polynomial
            # T(s) = s^4 / (s^4 + a1*s^3 + a2*s^2 + a3*s + a4)
            # Normalization w.r.t ws:
            sn = s / ws

            # Coefficients based on Small & Margolis
            a1 = (1.0 / (h * qb)) + (1.0 / self.driver.qts)
            a2 = 1.0 + (1.0 + alpha) * (h ** 2) + (h / (qb * self.driver.qts))
            a3 = (h / self.driver.qts) + ((1.0 + alpha) * (h ** 2) / (h * qb))
            a4 = (h ** 2)

            denom = (sn ** 4) + a1 * (sn ** 3) + a2 * (sn ** 2) + a3 * sn + a4
            h_vent = (sn ** 4) / denom

            # Enclosure acoustic impedance seen by the back of the cone:
            # Zab(s) = 1 / [ s*Cab + 1/(Rab) + 1/(s*Map + Rap) ]
            # Using acoustic parameters:
            map_acoustic = rho0 * self.enclosure.port.effective_length / max(self.enclosure.port.area, 1e-6) if self.enclosure.port else 100.0
            rap = (wb * map_acoustic) / max(self.losses.qp, 1.0)
            rab = 1.0 / ((wb * cab) / max(self.losses.qa, 1.0) + 1e-12)

            z_port_acoustic = s * map_acoustic + rap
            y_box = s * cab + (1.0 / max(rab, 1e-6)) + (1.0 / z_port_acoustic)
            z_box_acoustic = 1.0 / y_box

            # Mechanical impedance reflected to driver cone: Z_mb = Sd^2 * Zab
            z_mb = (sd ** 2) * z_box_acoustic
            rms = ws * mms / qms
            zm = s * mms + rms + 1.0 / (s * max(cms, 1e-12)) + z_mb
            ze = re + s * self.driver.le
            z_in = ze + (bl ** 2) / zm

            i_g = eg / z_in
            v_n = (bl * i_g) / zm
            x_cone = np.abs(v_n / s)

            # Back-box pressure: p_b = (v_n * Sd - v_port * Sp) * ...
            # Volume velocity of diaphragm: U_d = Sd * v_n
            u_d = sd * v_n
            # Volume velocity of port: U_p = p_b / Z_port_acoustic
            # In parallel branch: U_p = U_d * (1 / (Z_port * Y_box))
            u_port = u_d / (z_port_acoustic * y_box)

            # Port air particle velocity: v_port = U_port / Sp
            sp = self.enclosure.port.area if self.enclosure.port else 0.01
            v_port = u_port / sp

            # Radiated acoustic pressure at r=1m
            # Diaphragm radiation:
            r = 1.0
            p_driver = (rho0 * s * u_d) / (2.0 * np.pi * r)
            # Port radiation (in phase with cone back wave, so radiation forward is + or - depending on sign convention):
            # Port volume velocity flows OUT of box when pressure inside is positive, so it radiates in phase with cone above fb
            p_port = (rho0 * s * u_port) / (2.0 * np.pi * r)
            p_total = p_driver + p_port

        # Sound Pressure Level in dB SPL (ref: 20 uPa)
        p_ref = 20e-6
        spl_total = 20.0 * np.log10(np.maximum(np.abs(p_total) / p_ref, 1e-6))
        spl_driver = 20.0 * np.log10(np.maximum(np.abs(p_driver) / p_ref, 1e-6))
        spl_port = 20.0 * np.log10(np.maximum(np.abs(p_port) / p_ref, 1e-6))

        phase_resp = np.angle(p_total, deg=True)

        return {
            "frequencies": frequencies,
            "v_n": v_n,                     # Complex normal diaphragm velocity (m/s)
            "v_port": v_port,               # Complex port velocity (m/s)
            "p_total": p_total,             # Complex acoustic pressure at 1m (Pa)
            "p_driver": p_driver,
            "p_port": p_port,
            "spl_total": spl_total,
            "spl_driver": spl_driver,
            "spl_port": spl_port,
            "phase": phase_resp,
            "z_in": z_in,                   # Complex electrical input impedance (Ohm)
            "impedance_magnitude": np.abs(z_in),
            "displacement": x_cone,         # Peak cone displacement (m)
        }
