"""
Command Line Interface (CLI) runner: acoustics-sim
Allows parameter input, LEM synthesis, mesh generation, BEM/AI simulation,
aerodynamics analysis, and report/mesh export.
"""

from __future__ import annotations
import argparse
import sys
import json
import numpy as np

from acoustics_platform.models import (
    TSParameters,
    EnclosureGeometry,
    EnclosureType,
    PortGeometry,
    SolverMode,
    SimulationConfig,
)
from acoustics_platform.lem import LEMSolver, LEMAlignmentSynthesizer
from acoustics_platform.geometry import ParametricCADGenerator
from acoustics_platform.mesh import MeshDiscretizationEngine
from acoustics_platform.bem import HelmholtzBEMSolver
from acoustics_platform.ai_surrogate import AISurrogateEngine
from acoustics_platform.aerodynamics import PortAerodynamicsAnalyzer


def main():
    parser = argparse.ArgumentParser(
        prog="acoustics-sim",
        description="Acoustics Simulation and Loudspeaker Design Platform CLI",
    )
    parser.add_argument("--fs", type=float, default=35.0, help="Free-air resonance frequency Fs (Hz)")
    parser.add_argument("--qes", type=float, default=0.38, help="Electrical Qes")
    parser.add_argument("--qms", type=float, default=4.0, help="Mechanical Qms")
    parser.add_argument("--vas", type=float, default=60.0, help="Equivalent volume Vas (Liters)")
    parser.add_argument("--sd", type=float, default=350.0, help="Piston area Sd (cm^2)")
    parser.add_argument("--xmax", type=float, default=5.0, help="Linear peak excursion Xmax (mm)")
    parser.add_argument("--type", choices=["vented", "closed"], default="vented", help="Enclosure topology")
    parser.add_argument("--alignment", choices=["b4", "qb3", "sealed"], default="b4", help="Synthesis alignment")
    parser.add_argument("--fmin", type=float, default=20.0, help="Min frequency (Hz)")
    parser.add_argument("--fmax", type=float, default=1000.0, help="Max frequency (Hz)")
    parser.add_argument("--solver", choices=["bem", "ai", "hybrid"], default="hybrid", help="Solver mode")
    parser.add_argument("--export-mesh", type=str, default=None, help="Export mesh to OBJ / STL / VTK file")
    parser.add_argument("--out-json", type=str, default=None, help="Save simulation results to JSON")

    args = parser.parse_args()

    # Convert units to SI
    vas_m3 = args.vas * 1e-3
    sd_m2 = args.sd * 1e-4
    xmax_m = args.xmax * 1e-3

    driver = TSParameters(
        fs=args.fs,
        qes=args.qes,
        qms=args.qms,
        vas=vas_m3,
        sd=sd_m2,
        xmax=xmax_m,
    )

    print("=================================================================")
    print("      ACOUSTICS SIMULATION & LOUDSPEAKER DESIGN PLATFORM         ")
    print("=================================================================")
    print(f"Driver T/S: Fs={driver.fs:.1f}Hz, Qts={driver.qts:.3f}, Vas={driver.vas*1000:.1f}L, EBP={driver.ebp:.1f}")
    print(f"Reference Efficiency: eta0={driver.eta0*100:.3f}%, SPL 1W/1m={driver.spl_1w_1m:.1f} dB")

    # Enclosure alignment synthesis
    if args.type == "vented":
        enc_type = EnclosureType.VENTED
        if args.alignment == "qb3":
            vb, fb = LEMAlignmentSynthesizer.calculate_qb3_alignment(driver)
            align_name = "QB3"
        else:
            vb, fb = LEMAlignmentSynthesizer.calculate_b4_alignment(driver)
            align_name = "B4"
        print(f"Synthesized {align_name} Alignment: Vb={vb*1000:.1f} L, fb={fb:.1f} Hz")
        port = PortGeometry(diameter=0.07, length=0.18)
    else:
        enc_type = EnclosureType.CLOSED
        vb, fc = LEMAlignmentSynthesizer.calculate_sealed_alignment(driver, qtc_target=0.707)
        fb = fc
        port = None
        print(f"Synthesized Sealed Alignment: Vb={vb*1000:.1f} L, fc={fc:.1f} Hz")

    # Derive rectangular box dimensions with golden ratio ~ 1 : 1.25 : 1.6
    depth = (vb / (1.25 * 1.6)) ** (1.0 / 3.0)
    width = 1.25 * depth
    height = 1.6 * depth

    enclosure = EnclosureGeometry(
        enclosure_type=enc_type,
        external_width=width + 0.038,
        external_height=height + 0.038,
        external_depth=depth + 0.038,
        driver_radius=np.sqrt(sd_m2 / np.pi),
        port=port,
    )

    # Solve analytical LEM
    lem_solver = LEMSolver(driver, enclosure, tuning_freq=fb)
    freqs = np.logspace(np.log10(args.fmin), np.log10(args.fmax), 50)
    lem_results = lem_solver.solve(freqs)

    print(f"LEM Simulation complete: 50 frequency points between {args.fmin} and {args.fmax} Hz.")
    print(f"Peak on-axis SPL: {np.max(lem_results['spl_total']):.1f} dB at r=1m")

    # Aerodynamics analysis for vented
    if enc_type == EnclosureType.VENTED and port:
        aero = PortAerodynamicsAnalyzer.evaluate_port_performance(lem_results["v_port"], port)
        print(f"Port Aerodynamics: Max Re = {aero['max_reynolds']:.0f}, Regime = {aero['flow_regime']}")
        if aero["chuffing_warning"]:
            print("WARNING: High port velocity detected! Risk of audible turbulence chuffing.")
        else:
            print("Port flow operates within linear aerodynamic boundary limits.")

    # 3D Mesh Generation
    mesh_engine = MeshDiscretizationEngine()
    mesh = mesh_engine.generate_mesh(enclosure, f_max=min(args.fmax, 2000.0))
    print(f"Generated 3D Enclosure Surface Mesh: {len(mesh.nodes)} nodes, {len(mesh.elements)} elements.")

    # Export mesh if requested
    if args.export_mesh:
        cad = ParametricCADGenerator(enclosure)
        if args.export_mesh.endswith(".obj"):
            cad.export_obj(args.export_mesh, mesh)
        elif args.export_mesh.endswith(".stl"):
            cad.export_stl(args.export_mesh, mesh)
        elif args.export_mesh.endswith(".vtk"):
            mesh_engine.export_meshio(mesh, args.export_mesh)
        print(f"Exported mesh to {args.export_mesh}")

    # JSON output
    if args.out_json:
        data = {
            "driver": driver.to_dict(),
            "enclosure_volume_liters": vb * 1000,
            "tuning_freq": fb,
            "frequencies": freqs.tolist(),
            "spl_total": lem_results["spl_total"].tolist(),
            "impedance": lem_results["impedance_magnitude"].tolist(),
            "v_cone_peak": float(np.max(np.abs(lem_results["v_n"]))),
        }
        with open(args.out_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON results to {args.out_json}")

    print("=================================================================")
    print("Simulation completed successfully.")


if __name__ == "__main__":
    main()
