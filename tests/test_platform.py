"""
Unit and physical benchmark validation tests for Acoustics Platform.
Tests:
- LEM engine against classic Thiele/Small B4 and QB3 alignments (REQ-LEM-001/002)
- Multi-driver LEM & Crossover network matrix (Linkwitz-Riley / Butterworth)
- Helmholtz BEM solver against analytical pulsating sphere benchmark (REQ-BEM-001/002)
- Aerodynamic Reynolds number calculation and flow regime classification (REQ-AERO-001)
- Parametric CAD generation, physical group tagging, and mesh export (REQ-CAD-001/002, REQ-MESH-001/002)
- Ribbon array waveguide coupling geometry (4x 10x50mm ribbons)
- AI Fourier Neural Operator (U-FNO & LU-NO) forward inference and BEM fallback logic (REQ-AI-001/002)
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from acoustics_platform.models import (
    TSParameters,
    EnclosureGeometry,
    EnclosureType,
    PortGeometry,
    HornGeometry,
    FlowRegime,
    SolverMode,
)
from acoustics_platform.lem import (
    LEMSolver,
    LEMAlignmentSynthesizer,
    EnclosureLosses,
    DriverBranch,
    MultiDriverLEMSolver,
    FilterType,
    CrossoverFilter,
)
from acoustics_platform.geometry import (
    ParametricCADGenerator,
    HornProfileGenerator,
    RibbonArrayWaveguideGeometry,
)
from acoustics_platform.mesh import MeshDiscretizationEngine
from acoustics_platform.bem import HelmholtzBEMSolver, HelmholtzGreenFunction
from acoustics_platform.aerodynamics import PortAerodynamicsAnalyzer
from acoustics_platform.ai_surrogate import UFNO3d, LUNO3d, AISurrogateEngine
from acoustics_platform.postprocessing import DirectivityProcessor


class TestLEMValidation:
    """Validation of Thiele/Small analytical Lumped Element Model."""

    def test_ts_parameter_synthesis(self):
        driver = TSParameters(fs=30.0, qes=0.35, qms=3.5, vas=0.08, sd=0.035)
        expected_qts = (0.35 * 3.5) / (0.35 + 3.5)
        assert np.isclose(driver.qts, expected_qts, rtol=1e-3)
        assert np.isclose(driver.ebp, 30.0 / 0.35, rtol=1e-3)
        assert np.isclose(driver.fs, 1.0 / (2.0 * np.pi * np.sqrt(driver.cms * driver.mms)), rtol=1e-3)

    def test_b4_and_qb3_alignments(self):
        driver = TSParameters(fs=35.0, qes=0.38, qms=4.0, vas=0.06, sd=0.035)
        vb_b4, fb_b4 = LEMAlignmentSynthesizer.calculate_b4_alignment(driver)
        assert vb_b4 > 0.010 and vb_b4 < 0.200
        assert fb_b4 > 20.0 and fb_b4 < 80.0

        vb_qb3, fb_qb3 = LEMAlignmentSynthesizer.calculate_qb3_alignment(driver)
        assert vb_qb3 > 0.005 and vb_qb3 < vb_b4
        assert fb_qb3 >= driver.fs

    def test_lem_vented_and_sealed_frequency_response(self):
        driver = TSParameters(fs=40.0, qes=0.4, qms=4.0, vas=0.05, sd=0.035)
        enc_vented = EnclosureGeometry(enclosure_type=EnclosureType.VENTED, port=PortGeometry(diameter=0.07, length=0.15))
        solver_v = LEMSolver(driver, enc_vented, tuning_freq=45.0)

        freqs = np.array([20.0, 45.0, 100.0, 500.0])
        res_v = solver_v.solve(freqs)

        v_cone_fb = np.abs(res_v["v_n"][1])
        v_port_fb = np.abs(res_v["v_port"][1])
        assert v_port_fb > v_cone_fb
        assert np.all(res_v["spl_total"] > 60.0)
        assert np.all(res_v["spl_total"] < 120.0)

    def test_multi_driver_and_crossover_matrix(self):
        """Verify 3-way multi-driver configuration (2x 10\" woofers, 2x 5\" mids, 4x ribbons)."""
        w10 = TSParameters(fs=32.0, qes=0.35, qms=4.5, vas=0.065, sd=0.035)
        branch_bass = DriverBranch("Woofers_10inch", w10, count=2, filter_type="lowpass", f_high=400.0)

        m5 = TSParameters(fs=75.0, qes=0.45, qms=3.8, vas=0.008, sd=0.008)
        branch_mid = DriverBranch("Mids_5inch", m5, count=2, filter_type="bandpass", f_low=400.0, f_high=3000.0)

        r_ribbon = TSParameters(fs=1200.0, qes=1.2, qms=2.0, vas=0.0001, sd=0.0005, re=6.0)
        branch_ribbons = DriverBranch("Ribbon_Array_4x", r_ribbon, count=4, filter_type="highpass", f_low=3000.0)

        solver = MultiDriverLEMSolver([branch_bass, branch_mid, branch_ribbons])
        freqs = np.array([50.0, 400.0, 1000.0, 3000.0, 10000.0])
        res = solver.solve(freqs)

        assert len(res["branch_results"]) == 3
        # Bass branch dominates at 50 Hz
        assert res["branch_results"]["Woofers_10inch"]["spl"][0] > res["branch_results"]["Ribbon_Array_4x"]["spl"][0]
        # Ribbon branch dominates at 10 kHz
        assert res["branch_results"]["Ribbon_Array_4x"]["spl"][-1] > res["branch_results"]["Woofers_10inch"]["spl"][-1]


class TestBEMValidation:
    """Validation of 3D Helmholtz Boundary Element Method Solver."""

    def test_pulsating_sphere_benchmark(self):
        a = 0.12
        v0 = 0.02
        f = 250.0
        c0 = 345.0
        rho0 = 1.18
        k = 2.0 * np.pi * f / c0

        phi = (1.0 + np.sqrt(5.0)) / 2.0
        v = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        v = v / np.linalg.norm(v, axis=1, keepdims=True) * a

        f_idx = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ], dtype=np.int64)

        from acoustics_platform.models import MeshStructure
        mesh = MeshStructure(nodes=v, elements=f_idx)
        solver = HelmholtzBEMSolver(mesh, rho0=rho0, c0=c0)

        v_normal = np.full(len(f_idx), v0, dtype=complex)
        p_surf = solver.solve_surface_pressure(k, v_normal)

        r_obs = 1.0
        obs_points = np.array([[0.0, 0.0, r_obs]])
        p_num = solver.evaluate_field(obs_points, k, p_surf, v_normal)[0]

        ka = k * a
        p_ana = 1j * rho0 * c0 * v0 * (a / r_obs) * (ka / (1.0 + 1j * ka)) * np.exp(1j * k * (r_obs - a))

        rel_diff = np.abs(np.abs(p_num) - np.abs(p_ana)) / np.abs(p_ana)
        assert rel_diff < 0.25


class TestAerodynamicsValidation:
    """Validation of port air flow velocity and Reynolds regimes."""

    def test_reynolds_regime_classification(self):
        port = PortGeometry(diameter=0.08, length=0.20)
        v_lam = np.array([0.2])
        res_lam = PortAerodynamicsAnalyzer.evaluate_port_performance(v_lam, port)
        assert res_lam["flow_regime"] == FlowRegime.LAMINAR
        assert not res_lam["chuffing_warning"]

        v_trans = np.array([1.0])
        res_trans = PortAerodynamicsAnalyzer.evaluate_port_performance(v_trans, port)
        assert res_trans["flow_regime"] == FlowRegime.TRANSITIONAL
        assert not res_trans["chuffing_warning"]

        v_turb = np.array([10.0])
        res_turb = PortAerodynamicsAnalyzer.evaluate_port_performance(v_turb, port)
        assert res_turb["flow_regime"] == FlowRegime.TURBULENT
        assert res_turb["chuffing_warning"]
        assert res_turb["qp_effective"] < 15.0


class TestCADAndMesh:
    """Validation of CAD parametric generation and ribbon waveguide coupling."""

    def test_cad_and_mesh_generation(self):
        enc = EnclosureGeometry(
            enclosure_type=EnclosureType.VENTED,
            external_width=0.25,
            external_height=0.40,
            external_depth=0.30,
            driver_radius=0.08,
            port=PortGeometry(diameter=0.06, length=0.15),
            horn=HornGeometry(throat_diameter=0.025, mouth_width=0.18, mouth_height=0.10, length=0.12),
        )
        engine = MeshDiscretizationEngine()
        h_max = engine.compute_max_element_size(2000.0)
        assert h_max <= (345.0 / (6.0 * 2000.0)) + 1e-6

        mesh = engine.generate_mesh(enc, f_max=1500.0, second_order=False)
        assert len(mesh.nodes) > 100
        assert len(mesh.elements) > 100
        assert "diaphragm" in mesh.physical_tags
        assert "port" in mesh.physical_tags
        assert "enclosure" in mesh.physical_tags
        assert "horn" in mesh.physical_tags

        mesh2 = engine.convert_to_second_order(mesh)
        assert mesh2.element_type == "tri6"
        assert mesh2.elements.shape[1] == 6

        with tempfile.TemporaryDirectory() as tmpdir:
            cad = ParametricCADGenerator(enc)
            obj_f = os.path.join(tmpdir, "test.obj")
            stl_f = os.path.join(tmpdir, "test.stl")
            cad.export_obj(obj_f, mesh)
            cad.export_stl(stl_f, mesh)
            assert os.path.exists(obj_f) and os.path.getsize(obj_f) > 0
            assert os.path.exists(stl_f) and os.path.getsize(stl_f) > 0

    def test_ribbon_array_waveguide_geometry(self):
        """Verify 4x 10x50mm ribbon array and waveguide flange tagging."""
        geom = RibbonArrayWaveguideGeometry(
            num_ribbons=4,
            ribbon_width=0.010,
            ribbon_height=0.050,
            vertical_gap=0.003,
            flange_width=0.120,
            flange_depth=0.040,
        )
        mesh = geom.generate_mesh(target_elem_size=0.006)
        assert len(mesh.nodes) > 50
        assert "ribbons_all" in mesh.physical_tags
        assert "waveguide_flange" in mesh.physical_tags
        for i in range(4):
            assert f"ribbon_{i}" in mesh.physical_tags


class TestAISurrogateEngine:
    """Validation of AI Neural Operators and fallback mechanisms."""

    def test_ufno_and_luno_architectures(self):
        ufno = UFNO3d(in_channels=4, out_channels=2, hidden_channels=16, modes=4, num_layers=2)
        x = torch.randn(2, 4, 12, 12, 12)
        out = ufno(x)
        assert out.shape == (2, 2, 12, 12, 12)

        luno = LUNO3d(in_channels=10, out_channels=2, hidden_channels=16, modes=4)
        x_luno = torch.randn(1, 10, 12, 12, 12)
        out_luno = luno(x_luno)
        assert out_luno.shape == (1, 2, 12, 12, 12)

    def test_ai_fallback_routing(self):
        engine = AISurrogateEngine(grid_resolution=12)
        enc = EnclosureGeometry()
        cad = ParametricCADGenerator(enc)
        mesh = cad.generate_surface_mesh(target_elem_size=0.12)

        p_ai, mode_ai, conf_ai = engine.predict_field(enc, k=1.0, v_diaphragm=0.01)
        assert mode_ai == SolverMode.AI_SURROGATE
        assert conf_ai >= 0.80

        obs = np.array([[0.0, 0.0, 1.0]])
        p_bem, mode_bem, conf_bem = engine.predict_field(
            enc, k=100.0, v_diaphragm=0.01, confidence_threshold=0.85,
            mesh_for_bem_fallback=mesh, observation_points=obs
        )
        assert mode_bem == SolverMode.RIGOROUS_BEM
        assert conf_bem < 0.85
