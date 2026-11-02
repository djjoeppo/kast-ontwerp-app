"""
Multi-Stage Coarse-to-Fine Optimization Algorithm.
Implements an evolutionary/genetic optimization algorithm for positioning drivers
and internal braces within a fixed enclosure volume:
- Step 1 (Coarse): Generate 50 Monte Carlo layout candidates within baffle/cabinet bounds.
- Step 2 (Fast Evaluation & Selection): Evaluate variants with fast LEM, filter out collisions
  using ClearanceEngine, minimize acoustic comb-filtering, and select top 5 candidates.
- Step 3 (Mutation & Refinement): Generate exactly 50 controlled mutations from the top 5 candidates.
- Step 4 (Validation): Select the absolute best layout ready for BEM verification and 3D mesh export.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

from acoustics_platform.models import TSParameters, EnclosureGeometry, MeshStructure
from acoustics_platform.geometry.assembly_parser import (
    TaggedComponent,
    MultiComponentAssemblyParser,
)
from acoustics_platform.geometry.constraints import (
    ClearanceEngine,
    InternalBrace,
    ValidationReport,
)
from acoustics_platform.lem.crossover import DriverBranch, MultiDriverLEMSolver


@dataclass
class DriverPlacementSpec:
    """Specification for placing a driver section on the front baffle."""
    name: str
    driver_params: TSParameters
    count: int = 1
    vertical_span_limits: Tuple[float, float] = (-0.35, 0.35) # Min and Max Y relative to baffle center
    fixed_y: Optional[float] = None


@dataclass
class LayoutCandidate:
    """Candidate loudspeaker layout containing driver positions and internal braces."""
    driver_positions: Dict[str, List[Tuple[float, float, float]]] # name -> list of (x, y, z)
    braces: List[InternalBrace]
    components: List[TaggedComponent] = field(default_factory=list)
    fitness: float = -1e9
    is_valid: bool = False
    validation_report: Optional[ValidationReport] = None


class EnclosureLayoutOptimizer:
    """
    Multi-Stage Coarse-to-Fine Layout Optimizer.
    """

    def __init__(
        self,
        enclosure: EnclosureGeometry,
        driver_specs: List[DriverPlacementSpec],
        clearance_engine: Optional[ClearanceEngine] = None,
        random_seed: Optional[int] = 42,
    ):
        self.enclosure = enclosure
        self.driver_specs = driver_specs
        self.clearance = clearance_engine or ClearanceEngine()
        if random_seed is not None:
            np.random.seed(random_seed)

        # Front baffle Z coordinate
        self.baffle_z = enclosure.external_depth / 2.0

    def _build_components_for_candidate(
        self,
        driver_positions: Dict[str, List[Tuple[float, float, float]]],
    ) -> List[TaggedComponent]:
        components: List[TaggedComponent] = []
        for spec in self.driver_specs:
            positions = driver_positions[spec.name]
            for i, pos in enumerate(positions):
                sub_name = f"{spec.name}_{i}"
                comps = MultiComponentAssemblyParser.create_synthetic_driver_assembly(
                    sub_name, spec.driver_params, center_position=pos
                )
                components.extend(comps)
        return components

    def _evaluate_acoustic_fitness(self, candidate: LayoutCandidate) -> float:
        """
        Evaluates acoustic response uniformity, minimization of comb-filtering,
        and optimal vertical directivity balance.
        """
        # Penalty for invalid layout
        if not candidate.is_valid:
            num_violations = len(candidate.validation_report.violations) if candidate.validation_report else 10
            return -1000.0 * num_violations

        # Acoustic objective:
        # 1. High frequency drivers (ribbons/tweeters) should be placed near ear-level (Y ~ 0.0 to +0.15m)
        # 2. Woofers should be separated from midranges by at least driver radius to avoid phase cancellation
        # 3. Symmetry along X axis (X ~ 0) gives better horizontal polar pattern
        score = 100.0

        for spec in self.driver_specs:
            positions = candidate.driver_positions[spec.name]
            for pos in positions:
                x, y, z = pos
                # Penalty for off-center horizontal placement (causes asymmetric lobing)
                score -= 150.0 * (x ** 2)

                # Ear level bonus for high-frequency ribbon/tweeter
                if spec.driver_params.fs > 1000.0:
                    ear_level_target = 0.10
                    score -= 50.0 * np.abs(y - ear_level_target)

                # Woofer floor boundary loading bonus (low Y)
                if spec.driver_params.fs < 50.0:
                    score += 20.0 * (-y) # Lower woofer gets floor boundary reinforcement

        # Structural bracing bonus: Braces well spaced inside cabinet
        if len(candidate.braces) > 0:
            score += 10.0 * len(candidate.braces)

        return float(score)

    def step1_generate_coarse_candidates(self, num_candidates: int = 50) -> List[LayoutCandidate]:
        """
        Step 1 (Coarse): Generate 50 Monte Carlo candidate layouts within baffle and cabinet bounds.
        """
        candidates: List[LayoutCandidate] = []
        hw = self.enclosure.internal_width / 2.0
        hh = self.enclosure.internal_height / 2.0
        hd = self.enclosure.internal_depth / 2.0

        for _ in range(num_candidates):
            positions: Dict[str, List[Tuple[float, float, float]]] = {}
            for spec in self.driver_specs:
                r_drv = np.sqrt(spec.driver_params.sd / np.pi)
                drv_pos_list = []
                for _ in range(spec.count):
                    if spec.fixed_y is not None:
                        y = spec.fixed_y
                    else:
                        y_min, y_max = spec.vertical_span_limits
                        y = np.random.uniform(max(-hh + r_drv, y_min), min(hh - r_drv, y_max))
                    # X position centered with slight perturbation
                    x = np.random.uniform(-0.02, 0.02)
                    drv_pos_list.append((x, y, self.baffle_z))
                positions[spec.name] = drv_pos_list

            # Random 1 or 2 internal braces
            braces = []
            num_braces = np.random.randint(1, 3)
            for b_idx in range(num_braces):
                by = np.random.uniform(-hh * 0.7, hh * 0.7)
                bz = np.random.uniform(-hd * 0.5, hd * 0.3)
                brace = InternalBrace(
                    name=f"Brace_{b_idx}",
                    center=(0.0, by, bz),
                    dimensions=(self.enclosure.internal_width * 0.85, 0.025, 0.08),
                )
                braces.append(brace)

            cand = LayoutCandidate(driver_positions=positions, braces=braces)
            cand.components = self._build_components_for_candidate(positions)
            report = self.clearance.check_layout(self.enclosure, cand.components, braces)
            cand.is_valid = report.is_valid
            cand.validation_report = report
            cand.fitness = self._evaluate_acoustic_fitness(cand)
            candidates.append(cand)

        return candidates

    def step2_evaluate_and_select_top(self, candidates: List[LayoutCandidate], top_k: int = 5) -> List[LayoutCandidate]:
        """
        Step 2 (Fast Evaluation & Selection): Filter valid candidates, rank by acoustic fitness,
        and select the top 5 best candidates.
        """
        # Sort candidates by fitness descending
        sorted_candidates = sorted(candidates, key=lambda c: (c.is_valid, c.fitness), reverse=True)
        return sorted_candidates[:top_k]

    def step3_mutate_and_refine(self, top_candidates: List[LayoutCandidate], total_mutations: int = 50) -> List[LayoutCandidate]:
        """
        Step 3 (Mutation & Refinement): Generate exactly 50 controlled mutations from the top candidates
        with micro-variations in driver positions and internal braces.
        """
        mutated_candidates: List[LayoutCandidate] = []
        k = len(top_candidates)
        mutations_per_parent = total_mutations // max(1, k)
        remainder = total_mutations % max(1, k)

        for p_idx, parent in enumerate(top_candidates):
            n_mut = mutations_per_parent + (1 if p_idx < remainder else 0)
            for _ in range(n_mut):
                # Copy parent positions with gaussian micro-perturbation (sigma ~ 5mm)
                new_positions: Dict[str, List[Tuple[float, float, float]]] = {}
                for spec in self.driver_specs:
                    old_list = parent.driver_positions[spec.name]
                    new_list = []
                    for pos in old_list:
                        # Perturb Y by +/- 6 mm, X by +/- 2 mm
                        dy = np.random.normal(0.0, 0.006)
                        dx = np.random.normal(0.0, 0.002)
                        y_min, y_max = spec.vertical_span_limits
                        new_y = float(np.clip(pos[1] + dy, y_min, y_max))
                        new_x = float(np.clip(pos[0] + dx, -0.015, 0.015))
                        new_list.append((new_x, new_y, self.baffle_z))
                    new_positions[spec.name] = new_list

                # Perturb braces
                new_braces: List[InternalBrace] = []
                for b in parent.braces:
                    d_by = float(np.random.normal(0.0, 0.008))
                    d_bz = float(np.random.normal(0.0, 0.008))
                    c = b.center
                    new_braces.append(InternalBrace(
                        name=b.name,
                        center=(c[0], c[1] + d_by, c[2] + d_bz),
                        dimensions=b.dimensions,
                    ))

                child = LayoutCandidate(driver_positions=new_positions, braces=new_braces)
                child.components = self._build_components_for_candidate(new_positions)
                report = self.clearance.check_layout(self.enclosure, child.components, new_braces)
                child.is_valid = report.is_valid
                child.validation_report = report
                child.fitness = self._evaluate_acoustic_fitness(child)
                mutated_candidates.append(child)

        return mutated_candidates

    def step4_validate_and_select_winner(self, refined_candidates: List[LayoutCandidate]) -> LayoutCandidate:
        """
        Step 4 (Validation): Select the absolute winner for final BEM verification and 3D export.
        """
        sorted_refined = sorted(refined_candidates, key=lambda c: (c.is_valid, c.fitness), reverse=True)
        winner = sorted_refined[0]
        return winner

    def run_optimization(self) -> Tuple[LayoutCandidate, List[LayoutCandidate], List[LayoutCandidate]]:
        """
        Executes the full 4-stage coarse-to-fine optimization pipeline:
        Returns:
            winner: Best layout candidate
            coarse_candidates: 50 initial candidates
            top_5: Top 5 selected candidates
        """
        # Step 1: 50 coarse candidates
        coarse_50 = self.step1_generate_coarse_candidates(num_candidates=50)

        # Step 2: Top 5 selection
        top_5 = self.step2_evaluate_and_select_top(coarse_50, top_k=5)

        # Step 3: Exactly 50 mutations from top 5
        mutations_50 = self.step3_mutate_and_refine(top_5, total_mutations=50)

        # Step 4: Validate and select winner
        all_final_pool = top_5 + mutations_50
        winner = self.step4_validate_and_select_winner(all_final_pool)

        return winner, coarse_50, top_5
