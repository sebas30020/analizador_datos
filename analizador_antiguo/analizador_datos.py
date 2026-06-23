import sys
import os

import h5py
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

# Forzar backend Qt5 para Matplotlib antes de cualquier import de plt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# Módulos locales
import config
import styles
import data_handler as dh
import plotter

# Aplicar tema oscuro de Matplotlib
plt.style.use('dark_background')
for k, v in config.MPLRC.items():
    matplotlib.rcParams[k] = v


class AnalizadorGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.WINDOW_TITLE)
        self.resize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

        # Estado interno
        self.filepath  = ""
        self.hdf_file  = None
        self.grupos: list[str] = []

        self.init_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # Construcción de la UI
    # ─────────────────────────────────────────────────────────────────────────

    def init_ui(self):
        widget_central = QtWidgets.QWidget()
        widget_central.setObjectName("centralWidget")
        self.setCentralWidget(widget_central)

        layout_principal = QtWidgets.QHBoxLayout(widget_central)
        layout_principal.setContentsMargins(15, 15, 15, 15)
        layout_principal.setSpacing(15)

        panel_lateral  = self._build_panel_lateral()
        panel_central  = self._build_panel_central()

        layout_principal.addWidget(panel_lateral)
        layout_principal.addWidget(panel_central, stretch=1)

        # Inicializar canvas vacío
        self.ax_time = None
        self.limpiar_plots()

        # Aplicar hoja de estilos
        self.setStyleSheet(styles.APP_QSS)

    # ── Panel lateral ─────────────────────────────────────────────────────────

    def _build_panel_lateral(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("PanelLateral")
        panel.setFixedWidth(config.SIDE_PANEL_WIDTH)

        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        # ── Sección: Archivo ─────────────────────────────────────────────────
        lay.addWidget(self._make_section_label("📁 SELECCIÓN DE ARCHIVO", "title_store"))

        self.btn_open = QtWidgets.QPushButton("📂 Abrir Archivo HDF5")
        self.btn_open.setObjectName("BtnOpen")
        self.btn_open.clicked.connect(self.abrir_archivo)
        lay.addWidget(self.btn_open)

        self.lbl_info_file = QtWidgets.QLabel(
            "Ruta: Sin archivo cargado\nFecha: --\nTotal de Grupos: --")
        self.lbl_info_file.setWordWrap(True)
        self.lbl_info_file.setStyleSheet("color: #A0A0AB; font-size: 11px;")
        lay.addWidget(self.lbl_info_file)

        lay.addWidget(self._make_separator())

        # ── Sección: Modo de Visualización ────────────────────────────────────
        lay.addWidget(self._make_section_label("⚙️ MODO DE VISUALIZACIÓN", "title_horiz"))

        self.combo_modo_vis = QtWidgets.QComboBox()
        self.combo_modo_vis.addItems([
            "Superponer Grupo (t=0)", 
            "Señal Individual (Paso a Paso)", 
            "Serie Temporal Completa"
        ])
        self.combo_modo_vis.currentIndexChanged.connect(self.on_cambio_modo_visualizacion)
        lay.addWidget(self.combo_modo_vis)

        lbl_grupo = QtWidgets.QLabel("Grupo HDF5 activo:")
        lbl_grupo.setStyleSheet("color: #5E6070;")
        lay.addWidget(lbl_grupo)

        self.combo_grupo = QtWidgets.QComboBox()
        self.combo_grupo.currentIndexChanged.connect(self.cambiar_grupo)
        lay.addWidget(self.combo_grupo)

        self.lbl_total_capturas = QtWidgets.QLabel("Capturas en grupo: --")
        self.lbl_total_capturas.setStyleSheet("color: #5E6070; font-weight: bold;")
        lay.addWidget(self.lbl_total_capturas)

        lay.addWidget(self._make_separator())

        # ── Sección: Canales ─────────────────────────────────────────────────
        lay.addWidget(self._make_section_label("📺 SELECCIÓN DE CANALES", "title_rango_y"))

        self.chk_ch3 = QtWidgets.QCheckBox("Visualizar Canal 3 (CH3)")
        self.chk_ch3.setChecked(True)
        self.chk_ch3.stateChanged.connect(self.on_channel_state_changed)
        lay.addWidget(self.chk_ch3)

        self.chk_ch4 = QtWidgets.QCheckBox("Visualizar Canal 4 (CH4)")
        self.chk_ch4.setChecked(True)
        self.chk_ch4.stateChanged.connect(self.on_channel_state_changed)
        lay.addWidget(self.chk_ch4)

        lay.addWidget(self._make_separator())

        # ── Acciones ─────────────────────────────────────────────────────────
        self.btn_plot = QtWidgets.QPushButton("📊 Graficar Datos")
        self.btn_plot.setObjectName("BtnPlot")
        self.btn_plot.clicked.connect(self.graficar_datos)
        lay.addWidget(self.btn_plot)

        self.btn_save_img = QtWidgets.QPushButton("💾 Guardar Gráfico (Imagen)")
        self.btn_save_img.setObjectName("BtnSave")
        self.btn_save_img.clicked.connect(self.guardar_grafico)
        lay.addWidget(self.btn_save_img)

        lay.addStretch()
        return panel

    # ── Panel central ─────────────────────────────────────────────────────────

    def _build_panel_central(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("PlotContainer")

        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)

        self.fig    = plt.figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(styles.TOOLBAR_QSS)

        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, stretch=1)

        # Panel de modo individual (oculto por defecto)
        self.panel_individual = QtWidgets.QFrame()
        self.panel_individual.setObjectName("PanelIndividual")
        self.panel_individual.setStyleSheet(styles.PANEL_INDIVIDUAL_QSS)
        self.panel_individual.hide()

        layout_indiv = QtWidgets.QHBoxLayout(self.panel_individual)
        layout_indiv.setContentsMargins(5, 5, 5, 5)
        layout_indiv.setSpacing(15)

        layout_indiv.addLayout(self._build_nav_subpanel(),    stretch=1)
        layout_indiv.addLayout(self._build_meta_subpanel(),   stretch=1)
        layout_indiv.addLayout(self._build_comment_subpanel(), stretch=2)

        lay.addWidget(self.panel_individual)
        return panel

    def _build_nav_subpanel(self) -> QtWidgets.QVBoxLayout:
        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(8)

        lbl = QtWidgets.QLabel("🎮 NAVEGACIÓN INDIVIDUAL")
        lbl.setStyleSheet("color: #8BA7C7; font-weight: bold; font-size: 10px;")
        vbox.addWidget(lbl)

        hbox = QtWidgets.QHBoxLayout()

        self.btn_prev = QtWidgets.QPushButton("◀ Ant.")
        self.btn_prev.setObjectName("BtnPrev")
        self.btn_prev.setStyleSheet(styles.BTN_NAV_QSS_TEMPLATE.format(obj_name="BtnPrev"))
        self.btn_prev.clicked.connect(self.prev_signal)

        self.spin_individual = QtWidgets.QSpinBox()
        self.spin_individual.setMinimum(1)
        self.spin_individual.setMaximum(1)
        self.spin_individual.setValue(1)
        self.spin_individual.valueChanged.connect(self.on_spin_individual_changed)
        self.spin_individual.setFixedWidth(60)

        self.lbl_total_indiv = QtWidgets.QLabel("/ 1")
        self.lbl_total_indiv.setStyleSheet("font-weight: bold; font-size: 11px;")

        self.btn_next = QtWidgets.QPushButton("Sig. ▶")
        self.btn_next.setObjectName("BtnNext")
        self.btn_next.setStyleSheet(styles.BTN_NAV_QSS_TEMPLATE.format(obj_name="BtnNext"))
        self.btn_next.clicked.connect(self.next_signal)

        hbox.addWidget(self.btn_prev)
        hbox.addWidget(self.spin_individual)
        hbox.addWidget(self.lbl_total_indiv)
        hbox.addWidget(self.btn_next)
        vbox.addLayout(hbox)

        self.slider_individual = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_individual.setMinimum(1)
        self.slider_individual.setMaximum(1)
        self.slider_individual.setValue(1)
        self.slider_individual.valueChanged.connect(self.on_slider_individual_changed)
        self.slider_individual.setStyleSheet(styles.SLIDER_QSS)
        vbox.addWidget(self.slider_individual)

        return vbox

    def _build_meta_subpanel(self) -> QtWidgets.QVBoxLayout:
        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(5)

        lbl = QtWidgets.QLabel("📌 METADATOS")
        lbl.setStyleSheet("color: #8BA7C7; font-weight: bold; font-size: 10px;")
        vbox.addWidget(lbl)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        grid.addWidget(QtWidgets.QLabel("Fecha/Hora:"), 0, 0)
        self.lbl_val_timestamp = QtWidgets.QLabel("--")
        self.lbl_val_timestamp.setObjectName("ValMeta")
        grid.addWidget(self.lbl_val_timestamp, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Disparo (Trig):"), 1, 0)
        self.lbl_val_trigger = QtWidgets.QLabel("--")
        self.lbl_val_trigger.setObjectName("ValMeta")
        grid.addWidget(self.lbl_val_trigger, 1, 1)

        vbox.addLayout(grid)
        vbox.addStretch()
        return vbox

    def _build_comment_subpanel(self) -> QtWidgets.QVBoxLayout:
        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(5)

        lbl = QtWidgets.QLabel("💬 COMENTARIO DEL EXPERIMENTO")
        lbl.setStyleSheet("color: #8BA7C7; font-weight: bold; font-size: 10px;")
        vbox.addWidget(lbl)

        self.txt_comentario = QtWidgets.QTextEdit()
        self.txt_comentario.setObjectName("TxtComentario")
        self.txt_comentario.setReadOnly(True)
        self.txt_comentario.setPlaceholderText("Sin comentarios grabados para este experimento.")
        self.txt_comentario.setMaximumHeight(70)
        vbox.addWidget(self.txt_comentario)

        return vbox

    # ── Helpers de construcción de UI ─────────────────────────────────────────

    @staticmethod
    def _make_section_label(text: str, obj_name: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setObjectName(obj_name)
        return lbl

    @staticmethod
    def _make_separator() -> QtWidgets.QFrame:
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setStyleSheet("background-color: #22222A;")
        return line

    # ─────────────────────────────────────────────────────────────────────────
    # Gestión del canvas de Matplotlib
    # ─────────────────────────────────────────────────────────────────────────

    def setup_axes(self):
        """Recrea un único eje temporal en la figura."""
        self.fig.clear()
        self.ax_time = self.fig.add_subplot(1, 1, 1)
        self.fig.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.96)

    def limpiar_plots(self):
        self.setup_axes()
        self.ax_time.set_title("Señales Adquiridas en el Dominio del Tiempo",
                               fontsize=11, color='#FFFFFF', pad=10)
        self.ax_time.set_xlabel("Tiempo relativo (s)", fontsize=9, labelpad=5)
        self.ax_time.set_ylabel("Voltaje (V)", fontsize=9, labelpad=5)
        self.ax_time.grid(True, linestyle=':', alpha=0.3)
        self.canvas.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks de controles del panel lateral
    # ─────────────────────────────────────────────────────────────────────────

    def on_cambio_modo_visualizacion(self, index):
        # index: 0 -> Superponer, 1 -> Individual, 2 -> Serie temporal completa
        self.combo_grupo.setEnabled(index != 2)
        
        if index == 1:
            self.panel_individual.show()
        else:
            self.panel_individual.hide()

        if self.hdf_file:
            self.graficar_datos()

    def on_channel_state_changed(self, state):
        if self.hdf_file:
            self.graficar_datos()

    def cambiar_grupo(self, index):
        if index < 0 or not self.hdf_file or self.combo_modo_vis.currentIndex() == 2:
            return
            
        group_name = self.combo_grupo.itemText(index)
        try:
            group = self.hdf_file[group_name]
            total_capturas = dh.contar_capturas(group)

            self.lbl_total_capturas.setText(f"Capturas en grupo: {total_capturas}")

            if total_capturas > 0:
                self.spin_individual.blockSignals(True)
                self.slider_individual.blockSignals(True)
                
                self.spin_individual.setMaximum(total_capturas)
                self.slider_individual.setMaximum(total_capturas)
                self.lbl_total_indiv.setText(f"/ {total_capturas}")
                self.spin_individual.setValue(1)
                self.slider_individual.setValue(1)
                
                self.spin_individual.blockSignals(False)
                self.slider_individual.blockSignals(False)

            if self.combo_modo_vis.currentIndex() != 2:
                self.graficar_datos()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error al leer grupo",
                f"No se pudo analizar el grupo seleccionado:\n{e}")

    # ── Navegación individual ────────────────────────────────────────────────

    def prev_signal(self):
        val = self.spin_individual.value()
        if val > 1:
            self.spin_individual.setValue(val - 1)

    def next_signal(self):
        val = self.spin_individual.value()
        if val < self.spin_individual.maximum():
            self.spin_individual.setValue(val + 1)

    def on_spin_individual_changed(self, value):
        self.slider_individual.blockSignals(True)
        self.slider_individual.setValue(value)
        self.slider_individual.blockSignals(False)
        if self.hdf_file:
            self.graficar_datos()

    def on_slider_individual_changed(self, value):
        self.spin_individual.blockSignals(True)
        self.spin_individual.setValue(value)
        self.spin_individual.blockSignals(False)
        if self.hdf_file:
            self.graficar_datos()

    # ─────────────────────────────────────────────────────────────────────────
    # Operaciones sobre archivos HDF5
    # ─────────────────────────────────────────────────────────────────────────

    def abrir_archivo(self):
        default_dir = config.DEFAULT_DATA_DIR
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()

        options = QtWidgets.QFileDialog.Options()
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Seleccionar archivo HDF5", default_dir,
            "Archivos HDF5 (*.h5 *.hdf5)", options=options)

        if not filepath:
            return

        self.cerrar_archivo_activo()
        try:
            self.filepath = filepath
            self.hdf_file, self.grupos, fecha, comentario = dh.abrir_hdf5(filepath)

            filename = os.path.basename(self.filepath)
            self.lbl_info_file.setText(
                f"<b>Archivo:</b> {filename}<br>"
                f"<b>Fecha:</b> {fecha}<br>"
                f"<b>Grupos:</b> {len(self.grupos)}<br>"
                f"<b>Comentario:</b> {comentario}")
            self.lbl_info_file.setTextFormat(QtCore.Qt.RichText)

            self.combo_grupo.blockSignals(True)
            self.combo_grupo.clear()
            self.combo_grupo.addItems(self.grupos)
            self.combo_grupo.blockSignals(False)

            if len(self.grupos) > 0:
                self.combo_grupo.setCurrentIndex(0)
                self.cambiar_grupo(0)
            else:
                self.lbl_total_capturas.setText("Capturas en grupo: --")
                self.spin_individual.setMaximum(1)
                self.slider_individual.setMaximum(1)
                self.lbl_total_indiv.setText("/ 1")

            self.graficar_datos()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error al abrir HDF5", f"No se pudo leer el archivo:\n{e}")
            self.cerrar_archivo_activo()

    def cerrar_archivo_activo(self):
        if self.hdf_file:
            try:
                self.hdf_file.close()
            except Exception:
                pass
            self.hdf_file = None

        self.filepath = ""
        self.grupos   = []
        self.combo_grupo.blockSignals(True)
        self.combo_grupo.clear()
        self.combo_grupo.blockSignals(False)
        self.lbl_info_file.setText("Ruta: Sin archivo cargado\nFecha: --\nTotal de Grupos: --")
        self.lbl_total_capturas.setText("Capturas en grupo: --")
        self.spin_individual.setMaximum(1)
        self.slider_individual.setMaximum(1)
        self.spin_individual.setValue(1)
        self.slider_individual.setValue(1)
        self.lbl_total_indiv.setText("/ 1")
        self.lbl_val_timestamp.setText("--")
        self.lbl_val_trigger.setText("--")
        self.txt_comentario.clear()
        self.limpiar_plots()

    # ─────────────────────────────────────────────────────────────────────────
    # Graficación
    # ─────────────────────────────────────────────────────────────────────────

    def graficar_datos(self):
        if not self.hdf_file:
            QtWidgets.QMessageBox.warning(
                self, "Sin Archivo", "Por favor, abre un archivo HDF5 primero.")
            return

        modo_idx = self.combo_modo_vis.currentIndex()
        if modo_idx == 1:
            self._graficar_individual()
        else:
            self._graficar_superposicion()

    def _graficar_individual(self):
        self.setup_axes()
        try:
            group_name = self.combo_grupo.currentText()
            if not group_name or group_name not in self.hdf_file:
                return
            group = self.hdf_file[group_name]

            if 'tiempos' not in group:
                return
            tiempos = group['tiempos'][:]

            total_capturas = dh.contar_capturas(group)
            idx_sel = self.spin_individual.value() - 1
            if idx_sel < 0 or idx_sel >= total_capturas:
                return

            # Metadatos
            self.lbl_val_timestamp.setText(dh.leer_timestamp_str(group, idx_sel))
            self.lbl_val_trigger.setText(dh.leer_trigger_str(group, idx_sel))

            comentario = dh.leer_comentario(self.hdf_file)
            self.txt_comentario.setPlainText(comentario)

            # Graficar
            plotter.graficar_senal_individual(
                self.ax_time, group, tiempos, idx_sel,
                mostrar_ch3=self.chk_ch3.isChecked(),
                mostrar_ch4=self.chk_ch4.isChecked())

            plotter.configurar_eje_tiempo(
                self.ax_time,
                titulo=f"Señal Individual (Índice {idx_sel+1} de {total_capturas})",
                xlabel="Tiempo relativo (s)")
            plotter.aplicar_leyenda(self.ax_time)
            self.canvas.draw()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error al Graficar",
                f"Ocurrió un error al procesar los datos:\n{e}")
            self.limpiar_plots()

    def _graficar_superposicion(self):
        modo_idx = self.combo_modo_vis.currentIndex()
        todos = (modo_idx == 2)

        if todos:
            grupos_a_procesar = self.grupos
            if not grupos_a_procesar:
                return
        else:
            group_name = self.combo_grupo.currentText()
            if not group_name or group_name not in self.hdf_file:
                return
            grupos_a_procesar = [group_name]

        self.setup_axes()
        try:
            alinear = todos
            t_first = dh.obtener_primer_timestamp(self.hdf_file, grupos_a_procesar)

            plotter.graficar_superposicion(
                self.ax_time,
                hdf_file=self.hdf_file,
                grupos_a_procesar=grupos_a_procesar,
                mostrar_ch3=self.chk_ch3.isChecked(),
                mostrar_ch4=self.chk_ch4.isChecked(),
                alinear=alinear,
                t_first=t_first)

            if todos:
                titulo = "Serie Temporal del Experimento (Línea de Tiempo Completa)"
                xlabel = "Tiempo absoluto relativo (s)"
            else:
                titulo = f"Superposición de Señales (Grupo: {grupos_a_procesar[0]} en t=0)"
                xlabel = "Tiempo relativo de adquisición (s)"

            plotter.configurar_eje_tiempo(self.ax_time, titulo=titulo, xlabel=xlabel)
            plotter.aplicar_leyenda(self.ax_time)
            self.canvas.draw()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error al Graficar",
                f"Ocurrió un error al procesar los datos:\n{e}")
            self.limpiar_plots()

    def guardar_grafico(self):
        if not self.hdf_file:
            QtWidgets.QMessageBox.warning(
                self, "Sin Datos", "No hay datos graficados para guardar.")
            return

        sufijo = "todos_grupos" if self.combo_modo_vis.currentIndex() == 2 else self.combo_grupo.currentText()
        default_name = f"analisis_senal_{sufijo}.png"
        options = QtWidgets.QFileDialog.Options()
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar Gráfico como Imagen", default_name,
            "Imágenes PNG (*.png);;Imágenes JPEG (*.jpg);;Documento PDF (*.pdf)",
            options=options)

        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                QtWidgets.QMessageBox.information(
                    self, "Guardado Exitoso",
                    f"El gráfico se ha guardado en:\n{filepath}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self, "Error al Guardar",
                    f"No se pudo guardar la imagen:\n{e}")

    def closeEvent(self, event):
        self.cerrar_archivo_activo()
        event.accept()


# =============================================================================
# Punto de entrada
# =============================================================================

if __name__ == '__main__':
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 9))

    ventana = AnalizadorGUI()
    ventana.show()
    sys.exit(app.exec_())
