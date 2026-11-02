"""
Validation tests for constraints engine, assembly parser, and multi-stage layout optimizer.
"""

import tempfile
import os
import numpy as np
import pytest

from acoustics_platform.models import TSParameters, EnclosureGeometry
from acoustics_platform.geometry.assembly_parser import (
    MultiComponentAssemblyParser,
    ComponentTag,
    BoundingBox3D,
)
from acoustics_platform.geometry.constraints import (
    ClearanceEngine,
    InternalBrace,
    ClearanceViolation,
    ValidationReport,
)
from acoustics_platform.optimization import (
    DriverPlacementSpec,
    EnclosureLayoutOptimizer,
    LayoutCandidate,
)
from acoustics_platform.ui.gui_viewer import Layout3DColorMapper


class TestAssemblyAndConstraints:

    def test_multi_component_synthetic_driver(self):
        driver = TSParameters(fs=40.0, qes=0.36, qms=4.2, vas=0.05, sd=0.035, xmax=0.005)
        comps = MultiComponentAssemblyParser.create_synthetic_driver_assembly("W10", driver, (0.0, 0.0, 0.15))
        assert len(comps) == 5
        tags = {c.tag for c in comps}
        assert ComponentTag.CONE in tags
        assert ComponentTag.SURROUND in tags
        assert ComponentTag.FLANGE in tags
        assert ComponentTag.BASKET in tags
        assert ComponentTag.MAGNET in tags

        # Verify surround has Xmax envelope
        surround_comp = [c for c in comps if c.tag == ComponentTag.SURROUND][0]
        assert surround_comp.excursion_envelope_z == driver.xmax

    def test_clearance_engine_collision_detection(self):
        enc = EnclosureGeometry(external_width=0.35, external_height=0.70, external_depth=0.35)
        driver1 = TSParameters(fs=35.0, qes=0.38, qms=4.0, vas=0.06, sd=0.035)
        driver2 = TSParameters(fs=35.0, qes=0.38, qms=4.0, vas=0.06, sd=0.035)

        # 1. Collision between 2 drivers (overlapping flanges and magnets)
        comps_d1 = MultiComponentAssemblyParser.create_synthetic_driver_assembly("D1", driver1, (0.0, 0.10, 0.15))
        comps_d2_overlap = MultiComponentAssemblyParser.create_synthetic_driver_assembly("D2", driver2, (0.0, 0.12, 0.15))

        engine = ClearanceEngine()
        report_overlap = engine.check_layout(enc, comps_d1 + comps_d2_overlap)
        assert not report_overlap.is_valid
        assert len(report_overlap.violations) > 0
        assert "D1_flange" in report_overlap.components_in_breach or "D1_magnet" in report_overlap.components_in_breach

        # 2. Collision with internal brace penetrating magnet clearance
        comps_d2_valid = MultiComponentAssemblyParser.create_synthetic_driver_assembly("D2", driver2, (0.0, -0.18, 0.15))
        # Place brace right inside D1 magnet
        mag_b = [c for c in comps_d1 if c.tag == ComponentTag.MAGNET][0].bounds
        bad_brace = InternalBrace("ClashingBrace", tuple(mag_b.center), (0.2, 0.02, 0.05))
        report_brace = engine.check_layout(enc, comps_d1 + comps_d2_valid, [bad_brace])
        assert not report_brace.is_valid
        assert any(v.component_a == "ClashingBrace" or v.component_b == "ClashingBrace" for v in report_brace.violations)

        # 3. Clean layout
        safe_brace = InternalBrace("SafeBrace", (0.0, -0.04, 0.0), (0.25, 0.02, 0.08))
        report_clean = engine.check_layout(enc, comps_d1 + comps_d2_valid, [safe_brace])
        assert report_clean.is_valid
        assert len(report_clean.violations) == 0

    def test_multi_component_obj_parser(self):
        # Create a simple mock .obj with tagged groups
        with tempfile.TemporaryDirectory() as tmpdir:
            obj_path = os.path.join(tmpdir, "tagged_driver.obj")
            with open(obj_path, "w") as f:
                f.write("""v 0 0 0
v 1 0 0
v 0 1 0
g cone
f 1 2 3
v 2 0 0
v 3 0 0
v 2 1 0
g magnet
f 4 5 6
""")
            comps = MultiComponentAssemblyParser.parse_tagged_obj(obj_path)
            assert len(comps) == 2
            tags = {c.tag for c in comps}
            assert ComponentTag.CONE in tags
            assert ComponentTag.MAGNET in tags


class TestMultiStageOptimizer:

    def test_four_stage_optimization_pipeline(self):
        enc = EnclosureGeometry(external_width=0.35, external_height=0.95, external_depth=0.38)
        specs = [
            DriverPlacementSpec("Tweeter", TSParameters(fs=1500.0, sd=0.001, xmax=0.001), count=1, vertical_span_limits=(0.15, 0.30)),
            DriverPlacementSpec("Midrange", TSParameters(fs=120.0, sd=0.008, xmax=0.003), count=1, vertical_span_limits=(-0.05, 0.10)),
            DriverPlacementSpec("Woofer", TSParameters(fs=35.0, sd=0.035, xmax=0.006), count=1, vertical_span_limits=(-0.35, -0.12)),
        ]

        opt = EnclosureLayoutOptimizer(enc, specs, random_seed=42)

        # Step 1: 50 coarse candidates
        coarse_50 = opt.step1_generate_coarse_candidates(num_candidates=50)
        assert len(coarse_50) == 50

        # Step 2: Selection of top 5
        top_5 = opt.step2_evaluate_and_select_top(coarse_50, top_k=5)
        assert len(top_5) == 5
        # Top 1 should have higher or equal fitness compared to top 5
        assert top_5[0].fitness >= top_5[-1].fitness

        # Step 3: Exactly 50 mutations from top 5
        mutations_50 = opt.step3_mutate_and_refine(top_5, total_mutations=50)
        assert len(mutations_50) == 50

        # Step 4: Validation and winner selection
        winner = opt.step4_validate_and_select_winner(top_5 + mutations_50)
        assert isinstance(winner, LayoutCandidate)
        assert winner.is_valid
        assert winner.fitness > 0


class TestVisualColorMapper:

    def test_color_rules(self):
        driver = TSParameters(fs=40.0, sd=0.035)
        comps = MultiComponentAssemblyParser.create_synthetic_driver_assembly("TestDr", driver)
        cone = [c for c in comps if c.tag == ComponentTag.CONE][0]
        surround = [c for c in comps if c.tag == ComponentTag.SURROUND][0]
        magnet = [c for c in comps if c.tag == ComponentTag.MAGNET][0]

        # Valid states: Blue/Green
        assert Layout3DColorMapper.get_component_color(cone, is_breached=False) == Layout3DColorMapper.COLOR_MEMBRANE_SAFE
        assert Layout3DColorMapper.get_component_color(surround, is_breached=False) == Layout3DColorMapper.COLOR_SURROUND_SAFE

        # Breach state: Bright Red
        assert Layout3DColorMapper.get_component_color(magnet, is_breached=True) == Layout3DColorMapper.COLOR_BREACH_ALERT
        assert Layout3DColorMapper.get_component_color(cone, is_breached=True) == Layout3DColorMapper.COLOR_BREACH_ALERT
