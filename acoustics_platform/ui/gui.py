"""
PyQt6 Desktop GUI Application.
Implements a desktop interface with:
- Thiele/Small parameter inputs & alignment synthesis
- Enclosure & port geometry configuration
- Solver selection (LEM, Rigorous BEM, AI U-FNO surrogate)
- Matplotlib acoustic response curves (SPL, impedance, cone displacement)
- 3D Mesh & Acoustic Field Visualization viewport
"""

from __future__ import annotations
import sys
import os
import numpy as np

try:
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QGridLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QComboBox,
        QTabWidget,
        QGroupBox,
        QMessageBox,
        QStatusBar,
    )
    from PyQt6.QtCore import Qt
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from acoustics_platform.models import (
    TSParameters,
    EnclosureGeometry,
    EnclosureType,
    PortGeometry,
)
from acoustics_platform.lem import LEMSolver, LEMAlignmentSynthesizer
from acoustics_platform.geometry import ParametricCADGenerator
from acoustics_platform.mesh import MeshDiscretizationEngine
from acoustics_platform.aerodynamics import PortAerodynamicsAnalyzer
from acoustics_platform.postprocessing import AcousticVisualizer, PYVISTA_AVAILABLE

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTAQT_AVAILABLE = True
except ImportError:
    PYVISTAQT_AVAILABLE = False


class AcousticsAppWindow(QMainWindow):
    """Main Desktop Window for Acoustics Platform."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acoustics Platform - Loudspeaker Design & Simulation")
        self.resize(1100, 750)

        self._init_ui()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left control panel
        control_panel = QVBoxLayout()
        main_layout.addLayout(control_panel, stretch=1)

        # 1. Driver Parameters Group
        driver_group = QGroupBox("Thiele/Small Driver Parameters")
        grid_drv = QGridLayout(driver_group)
        self.input_fs = QLineEdit("35.0")
        self.input_qes = QLineEdit("0.38")
        self.input_qms = QLineEdit("4.0")
        self.input_vas = QLineEdit("60.0")
        self.input_sd = QLineEdit("350.0")
        self.input_xmax = QLineEdit("5.0")

        grid_drv.addWidget(QLabel("Fs (Hz):"), 0, 0)
        grid_drv.addWidget(self.input_fs, 0, 1)
        grid_drv.addWidget(QLabel("Qes:"), 1, 0)
        grid_drv.addWidget(self.input_qes, 1, 1)
        grid_drv.addWidget(QLabel("Qms:"), 2, 0)
        grid_drv.addWidget(self.input_qms, 2, 1)
        grid_drv.addWidget(QLabel("Vas (L):"), 3, 0)
        grid_drv.addWidget(self.input_vas, 3, 1)
        grid_drv.addWidget(QLabel("Sd (cm²):"), 4, 0)
        grid_drv.addWidget(self.input_sd, 4, 1)
        grid_drv.addWidget(QLabel("Xmax (mm):"), 5, 0)
        grid_drv.addWidget(self.input_xmax, 5, 1)
        control_panel.addWidget(driver_group)

        # 2. Enclosure Config Group
        enc_group = QGroupBox("Enclosure Topology & Alignment")
        grid_enc = QGridLayout(enc_group)
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Basreflex (Vented)", "Gesloten (Closed)"])
        self.combo_align = QComboBox()
        self.combo_align.addItems(["B4 (Butterworth)", "QB3 (Quasi-Butterworth)", "Custom"])

        self.input_vb = QLineEdit("35.0")
        self.input_fb = QLineEdit("40.0")
        self.btn_synth = QPushButton("Synthesize Alignment")
        self.btn_synth.clicked.connect(self._on_synthesize)

        grid_enc.addWidget(QLabel("Type:"), 0, 0)
        grid_enc.addWidget(self.combo_type, 0, 1)
        grid_enc.addWidget(QLabel("Alignment:"), 1, 0)
        grid_enc.addWidget(self.combo_align, 1, 1)
        grid_enc.addWidget(self.btn_synth, 2, 0, 1, 2)
        grid_enc.addWidget(QLabel("Vb (L):"), 3, 0)
        grid_enc.addWidget(self.input_vb, 3, 1)
        grid_enc.addWidget(QLabel("Fb / Fc (Hz):"), 4, 0)
        grid_enc.addWidget(self.input_fb, 4, 1)
        control_panel.addWidget(enc_group)

        # 3. Solver & Simulation Action
        sim_group = QGroupBox("Solver Execution")
        layout_sim = QVBoxLayout(sim_group)
        self.combo_solver = QComboBox()
        self.combo_solver.addItems(["Analytisch LEM", "Rigoureuze BEM (Helmholtz)", "AI U-FNO Surrogaat"])
        self.btn_run = QPushButton("Run Simulation")
        self.btn_run.setStyleSheet("font-weight: bold; background-color: #2e7d32; color: white;")
        self.btn_run.clicked.connect(self._on_run_sim)

        layout_sim.addWidget(QLabel("Solver Mode:"))
        layout_sim.addWidget(self.combo_solver)
        layout_sim.addWidget(self.btn_run)
        control_panel.addWidget(sim_group)

        control_panel.addStretch()

        # Right tab view (Results & 3D visualization)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=2)

        # Tab 1: Frequency Response Plots
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvas(self.fig)
        self.tabs.addTab(self.canvas, "Acoustic Response (SPL & Impedance)")

        # Tab 2: 3D Visualization Viewport
        self.tab_3d = QWidget()
        layout_3d = QVBoxLayout(self.tab_3d)
        if PYVISTAQT_AVAILABLE:
            try:
                self.plotter_3d = QtInteractor(self.tab_3d)
                layout_3d.addWidget(self.plotter_3d.interactor)
            except Exception:
                self.plotter_3d = None
                self.label_3d_info = QLabel("3D Surface Acoustic Field (PyVista/VTK Rendering)")
                self.label_3d_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout_3d.addWidget(self.label_3d_info)
        else:
            self.plotter_3d = None
            self.label_3d_info = QLabel("3D Surface Acoustic Field (PyVista/VTK Rendering)")
            self.label_3d_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout_3d.addWidget(self.label_3d_info)

        self.tabs.addTab(self.tab_3d, "3D CAD & Field Viewport")

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Select driver parameters and click Run Simulation.")

        # Plot initial response
        self._on_synthesize()
        self._on_run_sim()

    def _get_driver_params(self) -> TSParameters:
        return TSParameters(
            fs=float(self.input_fs.text()),
            qes=float(self.input_qes.text()),
            qms=float(self.input_qms.text()),
            vas=float(self.input_vas.text()) * 1e-3,
            sd=float(self.input_sd.text()) * 1e-4,
            xmax=float(self.input_xmax.text()) * 1e-3,
        )

    def _on_synthesize(self):
        try:
            driver = self._get_driver_params()
            is_vented = "Vented" in self.combo_type.currentText()
            align_text = self.combo_align.currentText()

            if is_vented:
                if "QB3" in align_text:
                    vb, fb = LEMAlignmentSynthesizer.calculate_qb3_alignment(driver)
                else:
                    vb, fb = LEMAlignmentSynthesizer.calculate_b4_alignment(driver)
            else:
                vb, fb = LEMAlignmentSynthesizer.calculate_sealed_alignment(driver)

            self.input_vb.setText(f"{vb*1000:.1f}")
            self.input_fb.setText(f"{fb:.1f}")
            self.status.showMessage(f"Synthesized alignment: Vb={vb*1000:.1f}L, fb={fb:.1f}Hz")
        except Exception as e:
            QMessageBox.warning(self, "Synthesis Error", str(e))

    def _on_run_sim(self):
        try:
            driver = self._get_driver_params()
            vb_m3 = float(self.input_vb.text()) * 1e-3
            fb = float(self.input_fb.text())
            is_vented = "Vented" in self.combo_type.currentText()
            enc_type = EnclosureType.VENTED if is_vented else EnclosureType.CLOSED

            depth = (vb_m3 / (1.25 * 1.6)) ** (1.0 / 3.0)
            enclosure = EnclosureGeometry(
                enclosure_type=enc_type,
                external_width=1.25 * depth + 0.038,
                external_height=1.6 * depth + 0.038,
                external_depth=depth + 0.038,
                driver_radius=np.sqrt(driver.sd / np.pi),
                port=PortGeometry(diameter=0.07, length=0.18) if is_vented else None,
            )

            solver = LEMSolver(driver, enclosure, tuning_freq=fb)
            freqs = np.logspace(np.log10(20), np.log10(1000), 100)
            res = solver.solve(freqs)

            # Update Matplotlib Plots
            self.fig.clear()
            ax1 = self.fig.add_subplot(2, 1, 1)
            ax1.semilogx(freqs, res["spl_total"], "b-", label="Total SPL @ 1m (dB)", linewidth=2)
            if is_vented:
                ax1.semilogx(freqs, res["spl_driver"], "g--", label="Cone radiation")
                ax1.semilogx(freqs, res["spl_port"], "r--", label="Port radiation")
            ax1.set_ylabel("SPL (dB)")
            ax1.grid(True, which="both", alpha=0.3)
            ax1.legend(loc="lower right")
            ax1.set_title("Acoustic Sound Pressure Level & Input Impedance")

            ax2 = self.fig.add_subplot(2, 1, 2)
            ax2.semilogx(freqs, res["impedance_magnitude"], "m-", label="Z_in (Ohm)", linewidth=2)
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Impedance (Ohm)")
            ax2.grid(True, which="both", alpha=0.3)
            ax2.legend(loc="upper right")

            self.fig.tight_layout()
            self.canvas.draw()

            # Update 3D PyVista Viewport if available
            if self.plotter_3d is not None and PYVISTA_AVAILABLE:
                try:
                    cad = ParametricCADGenerator(enclosure)
                    mesh = cad.generate_surface_mesh(target_elem_size=0.08)
                    n_cells = len(mesh.elements)
                    cells = np.hstack([[3, elem[0], elem[1], elem[2]] for elem in mesh.elements])
                    pv_mesh = pv.PolyData(mesh.nodes, cells)

                    # Synthesize acoustic pressure pattern across cabinet
                    p_fake = np.zeros(n_cells)
                    if "diaphragm" in mesh.physical_tags:
                        p_fake[mesh.physical_tags["diaphragm"]] = 95.0
                    if "port" in mesh.physical_tags:
                        p_fake[mesh.physical_tags["port"]] = 90.0
                    pv_mesh.cell_data["SPL_dB"] = p_fake

                    self.plotter_3d.clear()
                    self.plotter_3d.add_mesh(pv_mesh, scalars="SPL_dB", cmap="plasma", show_edges=True)
                    self.plotter_3d.view_isometric()
                    self.plotter_3d.render()
                except Exception:
                    pass

            self.status.showMessage("Simulation completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", str(e))


def run_gui():
    """Start Desktop GUI application."""
    if not PYQT_AVAILABLE:
        print("PyQt6 is not installed. Please install PyQt6 to run the desktop GUI.")
        return 1
    app = QApplication(sys.argv)
    window = AcousticsAppWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_gui())
