import sys
import os
import h5py
import numpy as np

from scipy.signal import welch

from PyQt5 import QtWidgets, QtCore, QtGui

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# Local modules
import config
import styles
import data_handler as dhn

# Apply dark theme configuration to matplotlib
plt.style.use('dark_background')
for k, v in config.MPLRC.items():
    matplotlib.rcParams[k] = v

COLOR_VOLTAGE = "#00E5FF"   # Neon Cyan
COLOR_TEMP = "#FF7043"      # Soft Orange/Red for Temperature
COLOR_HUMID = "#42A5F5"     # Bright Blue for Humidity
COLOR_FFT = "#E040FB"       # Neon Purple/Violet for FFT


class AnalizadorNuevoGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analizador Nativo - Nuevo Formato HDF5")
        self.resize(config.WINDOW_WIDTH + 50, config.WINDOW_HEIGHT + 50)

        # Internal state
        self.filepath = ""
        self.hdf_file = None
        self.test_group_name = ""
        self.test_group = None
        self.metadata = {}
        self.chunks = []
        
        self.init_ui()

    def init_ui(self):
        widget_central = QtWidgets.QWidget()
        widget_central.setObjectName("centralWidget")
        self.setCentralWidget(widget_central)

        layout_principal = QtWidgets.QHBoxLayout(widget_central)
        layout_principal.setContentsMargins(15, 15, 15, 15)
        layout_principal.setSpacing(15)

        panel_lateral = self._build_panel_lateral()
        panel_central = self._build_panel_central()

        layout_principal.addWidget(panel_lateral)
        layout_principal.addWidget(panel_central, stretch=1)

        self.ax_signal = None
        self.ax_fft = None
        self.ax_env = None
        self.limpiar_plots()

        # Apply dark QSS stylesheets
        self.setStyleSheet(styles.APP_QSS)

    def _build_panel_lateral(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("PanelLateral")
        panel.setFixedWidth(config.SIDE_PANEL_WIDTH + 10)

        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(12)

        # --- Section: File Selection ---
        lay.addWidget(self._make_section_label("📁 SELECCIÓN DE EXPERIMENTO", "title_store"))

        self.btn_open = QtWidgets.QPushButton("📂 Abrir Experimento HDF5")
        self.btn_open.setObjectName("BtnOpen")
        self.btn_open.clicked.connect(self.abrir_archivo)
        lay.addWidget(self.btn_open)

        self.lbl_info_file = QtWidgets.QLabel(
            "Archivo: Sin cargar\nFecha: --\nChunks Totales: --"
        )
        self.lbl_info_file.setWordWrap(True)
        self.lbl_info_file.setStyleSheet("color: #A0A0AB; font-size: 11px;")
        lay.addWidget(self.lbl_info_file)

        lay.addWidget(self._make_separator())

        # --- Section: Visualization Mode ---
        lay.addWidget(self._make_section_label("⚙️ MODO DE VISUALIZACIÓN", "title_horiz"))

        self.combo_modo_vis = QtWidgets.QComboBox()
        self.combo_modo_vis.addItems([
            "Superponer Señales del Chunk", 
            "Señal Individual (Paso a Paso)", 
            "Serie Temporal Completa"
        ])
        self.combo_modo_vis.currentIndexChanged.connect(self.on_cambio_modo_visualizacion)
        lay.addWidget(self.combo_modo_vis)

        lbl_grupo = QtWidgets.QLabel("Chunk Activo:")
        lbl_grupo.setStyleSheet("color: #5E6070;")
        lay.addWidget(lbl_grupo)

        hbox_chunk = QtWidgets.QHBoxLayout()

        self.btn_prev_chunk = QtWidgets.QPushButton("◀")
        self.btn_prev_chunk.setObjectName("BtnPrevChunk")
        self.btn_prev_chunk.setFixedWidth(35)
        self.btn_prev_chunk.setStyleSheet(styles.BTN_NAV_QSS_TEMPLATE.format(obj_name="BtnPrevChunk"))
        self.btn_prev_chunk.clicked.connect(self.prev_chunk)

        self.combo_grupo = QtWidgets.QComboBox()
        self.combo_grupo.currentIndexChanged.connect(self.cambiar_chunk)

        self.btn_next_chunk = QtWidgets.QPushButton("▶")
        self.btn_next_chunk.setObjectName("BtnNextChunk")
        self.btn_next_chunk.setFixedWidth(35)
        self.btn_next_chunk.setStyleSheet(styles.BTN_NAV_QSS_TEMPLATE.format(obj_name="BtnNextChunk"))
        self.btn_next_chunk.clicked.connect(self.next_chunk)

        hbox_chunk.addWidget(self.btn_prev_chunk)
        hbox_chunk.addWidget(self.combo_grupo, stretch=1)
        hbox_chunk.addWidget(self.btn_next_chunk)
        lay.addLayout(hbox_chunk)

        self.lbl_info_chunk = QtWidgets.QLabel(
            "Señales en chunk: --\nTipo: --\nHumedad/Temp: --"
        )
        self.lbl_info_chunk.setStyleSheet("color: #8A8A9E; font-size: 11px;")
        lay.addWidget(self.lbl_info_chunk)

        lay.addWidget(self._make_separator())

        # --- Section: Multichunk Accumulation ---
        lay.addWidget(self._make_section_label("📦 ACUMULACIÓN MULTICHUNK", "title_store"))

        self.chk_multichunk = QtWidgets.QCheckBox("Activar Multichunk (Anidar)")
        self.chk_multichunk.setChecked(False)
        self.chk_multichunk.stateChanged.connect(self.on_multichunk_state_changed)
        lay.addWidget(self.chk_multichunk)

        # Rango de chunks
        hbox_range = QtWidgets.QHBoxLayout()
        lbl_from = QtWidgets.QLabel("Desde:")
        lbl_from.setStyleSheet("color: #5E6070;")
        hbox_range.addWidget(lbl_from)
        self.spin_from_chunk = QtWidgets.QSpinBox()
        self.spin_from_chunk.setMinimum(1)
        self.spin_from_chunk.setMaximum(1)
        self.spin_from_chunk.setEnabled(False)
        hbox_range.addWidget(self.spin_from_chunk)

        lbl_to = QtWidgets.QLabel("Hasta:")
        lbl_to.setStyleSheet("color: #5E6070;")
        hbox_range.addWidget(lbl_to)
        self.spin_to_chunk = QtWidgets.QSpinBox()
        self.spin_to_chunk.setMinimum(1)
        self.spin_to_chunk.setMaximum(1)
        self.spin_to_chunk.setEnabled(False)
        hbox_range.addWidget(self.spin_to_chunk)
        lay.addLayout(hbox_range)

        self.btn_load_range = QtWidgets.QPushButton("➕ Cargar Rango")
        self.btn_load_range.setObjectName("btn_load_range")
        self.btn_load_range.setStyleSheet(styles.BTN_NAV_QSS_TEMPLATE.format(obj_name="btn_load_range"))
        self.btn_load_range.clicked.connect(self.load_range_to_stack)
        self.btn_load_range.setEnabled(False)
        lay.addWidget(self.btn_load_range)

        self.chk_compare_groups = QtWidgets.QCheckBox("Comparar con Grupo B")
        self.chk_compare_groups.setChecked(False)
        self.chk_compare_groups.setEnabled(False)
        self.chk_compare_groups.stateChanged.connect(self.on_compare_groups_state_changed)
        lay.addWidget(self.chk_compare_groups)

        # Radio buttons for editing target
        hbox_target = QtWidgets.QHBoxLayout()
        self.rbt_target_a = QtWidgets.QRadioButton("Destino: A")
        self.rbt_target_a.setObjectName("RbtTargetA")
        self.rbt_target_b = QtWidgets.QRadioButton("Destino: B")
        self.rbt_target_b.setObjectName("RbtTargetB")
        self.rbt_target_a.setChecked(True)
        self.rbt_target_a.setEnabled(False)
        self.rbt_target_b.setEnabled(False)
        self.rbt_target_a.toggled.connect(self.actualizar_enfoque_listas)
        self.rbt_target_b.toggled.connect(self.actualizar_enfoque_listas)
        hbox_target.addWidget(self.rbt_target_a)
        hbox_target.addWidget(self.rbt_target_b)
        lay.addLayout(hbox_target)

        # Group A title and list
        self.lbl_group_a = QtWidgets.QLabel("Grupo A (Cian):")
        self.lbl_group_a.setStyleSheet("color: #00E5FF; font-weight: bold; font-size: 11px;")
        lay.addWidget(self.lbl_group_a)

        self.list_stacked_chunks = QtWidgets.QListWidget()
        self.list_stacked_chunks.setMaximumHeight(65)
        self.list_stacked_chunks.setEnabled(False)
        self.list_stacked_chunks.setStyleSheet("background-color: #181820; color: #FFFFFF; border: 1px solid #2D2D38; border-radius: 5px;")
        lay.addWidget(self.list_stacked_chunks)

        # Group B title and list
        self.lbl_group_b = QtWidgets.QLabel("Grupo B (Magenta):")
        self.lbl_group_b.setStyleSheet("color: #FF1744; font-weight: bold; font-size: 11px;")
        lay.addWidget(self.lbl_group_b)

        self.list_stacked_chunks_b = QtWidgets.QListWidget()
        self.list_stacked_chunks_b.setMaximumHeight(65)
        self.list_stacked_chunks_b.setEnabled(False)
        self.list_stacked_chunks_b.setStyleSheet("background-color: #181820; color: #FFFFFF; border: 1px solid #2D2D38; border-radius: 5px;")
        lay.addWidget(self.list_stacked_chunks_b)

        self.chk_show_ratio = QtWidgets.QCheckBox("Mostrar Ratio de FFT (A/B)")
        self.chk_show_ratio.setChecked(True)
        self.chk_show_ratio.setEnabled(False)
        self.chk_show_ratio.stateChanged.connect(self.on_channel_state_changed)
        lay.addWidget(self.chk_show_ratio)

        lay.addWidget(self._make_separator())

        # --- Section: Channels and Environmental Variables ---
        lay.addWidget(self._make_section_label("📺 CANALES Y AMBIENTE", "title_rango_y"))

        self.chk_signals = QtWidgets.QCheckBox("Visualizar Señales (Voltaje)")
        self.chk_signals.setChecked(True)
        self.chk_signals.stateChanged.connect(self.on_channel_state_changed)
        lay.addWidget(self.chk_signals)

        self.chk_fft = QtWidgets.QCheckBox("Calcular FFT (Welch)")
        self.chk_fft.setChecked(True)
        self.chk_fft.stateChanged.connect(self.on_channel_state_changed)
        lay.addWidget(self.chk_fft)

        self.chk_env = QtWidgets.QCheckBox("Mostrar Humedad y Temperatura")
        self.chk_env.setChecked(True)
        self.chk_env.stateChanged.connect(self.on_channel_state_changed)
        lay.addWidget(self.chk_env)

        lay.addWidget(self._make_separator())

        # --- Actions ---
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

    def _build_panel_central(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("PlotContainer")

        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(10, 10, 10, 10)

        self.fig = plt.figure(figsize=(8, 8))
        self.canvas = FigureCanvas(self.fig)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(styles.TOOLBAR_QSS)

        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, stretch=1)

        # Individual signal metadata panel
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

        lbl = QtWidgets.QLabel("🎮 NAVEGACIÓN DE SEÑALES")
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

        lbl = QtWidgets.QLabel("📌 METADATOS DE ADQUISICIÓN")
        lbl.setStyleSheet("color: #8BA7C7; font-weight: bold; font-size: 10px;")
        vbox.addWidget(lbl)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        grid.addWidget(QtWidgets.QLabel("Fecha/Hora:"), 0, 0)
        self.lbl_val_timestamp = QtWidgets.QLabel("--")
        self.lbl_val_timestamp.setObjectName("ValMeta")
        grid.addWidget(self.lbl_val_timestamp, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Nivel Disparo (V):"), 1, 0)
        self.lbl_val_trigger = QtWidgets.QLabel("--")
        self.lbl_val_trigger.setObjectName("ValMeta")
        grid.addWidget(self.lbl_val_trigger, 1, 1)

        vbox.addLayout(grid)
        vbox.addStretch()
        return vbox

    def _build_comment_subpanel(self) -> QtWidgets.QVBoxLayout:
        vbox = QtWidgets.QVBoxLayout()
        vbox.setSpacing(5)

        lbl = QtWidgets.QLabel("💬 DETALLES DEL EXPERIMENTO")
        lbl.setStyleSheet("color: #8BA7C7; font-weight: bold; font-size: 10px;")
        vbox.addWidget(lbl)

        self.txt_comentario = QtWidgets.QTextEdit()
        self.txt_comentario.setObjectName("TxtComentario")
        self.txt_comentario.setReadOnly(True)
        self.txt_comentario.setPlaceholderText("Sin comentarios en el experimento.")
        self.txt_comentario.setMaximumHeight(70)
        vbox.addWidget(self.txt_comentario)

        return vbox

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

    # --- Plot Layout management ---

    def setup_axes(self):
        """Recreates active plots dynamically based on options."""
        self.fig.clear()
        
        mostrar_signal = self.chk_signals.isChecked()
        mostrar_fft = self.chk_fft.isChecked()
        mostrar_env = self.chk_env.isChecked() and self.has_env_data()
        
        compare = self.chk_compare_groups.isChecked() and self.chk_multichunk.isChecked()
        mostrar_ratio = compare and mostrar_fft and self.chk_show_ratio.isChecked()
        
        active_plots = []
        if mostrar_signal:
            active_plots.append('signal')
        if mostrar_fft:
            active_plots.append('fft')
        if mostrar_ratio:
            active_plots.append('ratio')
        if mostrar_env:
            active_plots.append('env')
            
        n_rows = len(active_plots)
        n_cols = 2 if compare else 1
        
        self.ax_signal = None
        self.ax_fft = None
        self.ax_ratio = None
        self.ax_env = None
        
        self.ax_signal_b = None
        self.ax_fft_b = None
        self.ax_env_b = None
        
        if n_rows == 0:
            return
            
        for r, plot_type in enumerate(active_plots):
            if n_cols == 2:
                if plot_type == 'ratio':
                    ax_ratio = self.fig.add_subplot(n_rows, 1, r + 1)
                    self.ax_ratio = ax_ratio
                else:
                    ax_a = self.fig.add_subplot(n_rows, n_cols, r * n_cols + 1)
                    ax_b = self.fig.add_subplot(n_rows, n_cols, r * n_cols + 2, sharey=ax_a if plot_type != 'env' else None)
                    
                    if plot_type == 'signal':
                        self.ax_signal = ax_a
                        self.ax_signal_b = ax_b
                    elif plot_type == 'fft':
                        self.ax_fft = ax_a
                        self.ax_fft_b = ax_b
                    elif plot_type == 'env':
                        self.ax_env = ax_a
                        self.ax_env_b = ax_b
            else:
                ax = self.fig.add_subplot(n_rows, n_cols, r + 1)
                if plot_type == 'signal':
                    self.ax_signal = ax
                elif plot_type == 'fft':
                    self.ax_fft = ax
                elif plot_type == 'env':
                    self.ax_env = ax
                    
        if n_cols == 2:
            wsp = 0.35 if mostrar_env else 0.25
            self.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.45, wspace=wsp)
        else:
            if n_rows == 3:
                self.fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.92, hspace=0.45)
            elif n_rows == 2:
                self.fig.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.92, hspace=0.35)
            else:
                self.fig.subplots_adjust(top=0.95, bottom=0.10, left=0.08, right=0.95)

    def has_env_data(self) -> bool:
        if not self.hdf_file or not self.test_group or not self.chunks:
            return False
        first_chunk = self.chunks[0]
        return 'humidity' in self.test_group[first_chunk]

    def limpiar_plots(self):
        self.setup_axes()
        
        # Configure Signal Plot A
        if self.ax_signal:
            title_a = "Grupo A - Voltaje" if self.ax_signal_b else "Señales Adquiridas (Voltaje)"
            self.ax_signal.set_title(title_a, fontsize=10, color='#00E5FF', pad=5)
            self.ax_signal.set_xlabel("Tiempo (µs)", fontsize=9, labelpad=3)
            self.ax_signal.set_ylabel("Voltaje (V)", fontsize=9, labelpad=3)
            self.ax_signal.grid(True, linestyle=':', alpha=0.3)
            
        # Configure Signal Plot B
        if self.ax_signal_b:
            self.ax_signal_b.set_title("Grupo B - Voltaje", fontsize=10, color='#FF1744', pad=5)
            self.ax_signal_b.set_xlabel("Tiempo (µs)", fontsize=9, labelpad=3)
            self.ax_signal_b.set_ylabel("Voltaje (V)", fontsize=9, labelpad=3)
            self.ax_signal_b.grid(True, linestyle=':', alpha=0.3)
        
        # Configure FFT Plot A
        if self.ax_fft:
            title_fft_a = "Grupo A - Welch PSD" if self.ax_fft_b else "Espectro de Potencia (Método de Welch)"
            self.ax_fft.set_title(title_fft_a, fontsize=10, color='#E040FB', pad=5)
            self.ax_fft.set_xlabel("Frecuencia (MHz)", fontsize=9, labelpad=3)
            self.ax_fft.set_ylabel("PSD (V²/Hz)", fontsize=9, labelpad=3)
            self.ax_fft.set_yscale('log')
            self.ax_fft.grid(True, linestyle=':', alpha=0.3)
            
        # Configure FFT Plot B
        if self.ax_fft_b:
            self.ax_fft_b.set_title("Grupo B - Welch PSD", fontsize=10, color='#FFD600', pad=5)
            self.ax_fft_b.set_xlabel("Frecuencia (MHz)", fontsize=9, labelpad=3)
            self.ax_fft_b.set_ylabel("PSD (V²/Hz)", fontsize=9, labelpad=3)
            self.ax_fft_b.set_yscale('log')
            self.ax_fft_b.grid(True, linestyle=':', alpha=0.3)

        # Configure Ratio FFT Plot
        if self.ax_ratio:
            self.ax_ratio.set_title("Ratio de FFT (Grupo A / Grupo B)", fontsize=10, color='#FF4081', pad=5)
            self.ax_ratio.set_xlabel("Frecuencia (MHz)", fontsize=9, labelpad=3)
            self.ax_ratio.set_ylabel("Ratio (A/B)", fontsize=9, labelpad=3)
            self.ax_ratio.set_yscale('log')
            self.ax_ratio.grid(True, linestyle=':', alpha=0.3)
        
        # Configure Environmental Plot A
        if self.ax_env:
            title_env_a = "Grupo A - Ambiente" if self.ax_env_b else "Humedad y Temperatura Ambiental"
            self.ax_env.set_title(title_env_a, fontsize=10, color='#FFFFFF', pad=5)
            self.ax_env.set_xlabel("Tiempo absoluto del experimento (s)", fontsize=9)
            self.ax_env.set_ylabel("Temperatura (°C)", fontsize=9, color=COLOR_TEMP)
            self.ax_env.grid(True, linestyle=':', alpha=0.2)
            
        # Configure Environmental Plot B
        if self.ax_env_b:
            self.ax_env_b.set_title("Grupo B - Ambiente", fontsize=10, color='#FFFFFF', pad=5)
            self.ax_env_b.set_xlabel("Tiempo absoluto del experimento (s)", fontsize=9)
            self.ax_env_b.set_ylabel("Temperatura (°C)", fontsize=9, color=COLOR_TEMP)
            self.ax_env_b.grid(True, linestyle=':', alpha=0.2)
            
        self.canvas.draw()

    # --- Controller Callbacks ---

    def on_cambio_modo_visualizacion(self, index):
        enable_chunk = (index != 2)
        self.combo_grupo.setEnabled(enable_chunk)
        self.btn_prev_chunk.setEnabled(enable_chunk)
        self.btn_next_chunk.setEnabled(enable_chunk)
        
        if index == 1:
            self.panel_individual.show()
        else:
            self.panel_individual.hide()

        if self.hdf_file:
            self.graficar_datos()

    def prev_chunk(self):
        idx = self.combo_grupo.currentIndex()
        if idx > 0:
            self.combo_grupo.setCurrentIndex(idx - 1)

    def next_chunk(self):
        idx = self.combo_grupo.currentIndex()
        if idx < self.combo_grupo.count() - 1:
            self.combo_grupo.setCurrentIndex(idx + 1)

    def obtener_chunks_seleccionados(self) -> list[str]:
        if not self.chk_multichunk.isChecked():
            c = self.combo_grupo.currentText()
            return [c] if c else []
        return self.obtener_chunks_grupo_a()

    def obtener_chunks_grupo_a(self) -> list[str]:
        chunks = []
        for i in range(self.list_stacked_chunks.count()):
            chunks.append(self.list_stacked_chunks.item(i).text())
        return chunks

    def obtener_chunks_grupo_b(self) -> list[str]:
        if not self.chk_compare_groups.isChecked():
            return []
        chunks = []
        for i in range(self.list_stacked_chunks_b.count()):
            chunks.append(self.list_stacked_chunks_b.item(i).text())
        return chunks

    def obtener_lista_activa(self) -> QtWidgets.QListWidget:
        if self.chk_compare_groups.isChecked() and self.rbt_target_b.isChecked():
            return self.list_stacked_chunks_b
        return self.list_stacked_chunks

    def actualizar_enfoque_listas(self):
        if self.rbt_target_b.isChecked() and self.chk_compare_groups.isChecked():
            self.list_stacked_chunks_b.setStyleSheet("background-color: #181820; color: #FFFFFF; border: 1px solid #FF1744; border-radius: 5px;")
            self.list_stacked_chunks.setStyleSheet("background-color: #181820; color: #FFFFFF; border: 1px solid #2D2D38; border-radius: 5px;")
        else:
            self.list_stacked_chunks.setStyleSheet("background-color: #181820; color: #FFFFFF; border: 1px solid #00E5FF; border-radius: 5px;")
            self.list_stacked_chunks_b.setStyleSheet("background-color: #181820; color: #FFFFFF; border: 1px solid #2D2D38; border-radius: 5px;")

    def obtener_senal_grupo(self, chunks: list[str], index_1based: int) -> tuple[str, int, dict]:
        target_idx = index_1based - 1
        current_offset = 0
        for cname in chunks:
            if cname in self.test_group:
                info = dhn.obtener_info_chunk(self.test_group, cname)
                n_signals = info['n_signals']
                if current_offset <= target_idx < current_offset + n_signals:
                    local_idx = target_idx - current_offset
                    return cname, local_idx, info
                current_offset += n_signals
        raise ValueError("Índice fuera de rango")

    def obtener_senal_multichunk(self, index_1based: int) -> tuple[str, int, dict]:
        return self.obtener_senal_grupo(self.obtener_chunks_grupo_a(), index_1based)

    def actualizar_limites_navegacion(self):
        if not self.hdf_file:
            return
            
        modo_idx = self.combo_modo_vis.currentIndex()
        if modo_idx == 2:
            return
            
        total_capturas = 0
        if self.chk_compare_groups.isChecked():
            total_a = 0
            for cname in self.obtener_chunks_grupo_a():
                if cname in self.test_group:
                    info = dhn.obtener_info_chunk(self.test_group, cname)
                    total_a += info['n_signals']
            total_b = 0
            for cname in self.obtener_chunks_grupo_b():
                if cname in self.test_group:
                    info = dhn.obtener_info_chunk(self.test_group, cname)
                    total_b += info['n_signals']
            total_capturas = max(total_a, total_b)
        else:
            chunks = self.obtener_chunks_seleccionados()
            for cname in chunks:
                if cname in self.test_group:
                    info = dhn.obtener_info_chunk(self.test_group, cname)
                    total_capturas += info['n_signals']
                
        self.lbl_info_chunk.setText(f"Capturas activas: {total_capturas}")
        
        if total_capturas > 0:
            self.spin_individual.blockSignals(True)
            self.slider_individual.blockSignals(True)
            
            self.spin_individual.setMaximum(total_capturas)
            self.slider_individual.setMaximum(total_capturas)
            self.lbl_total_indiv.setText(f"/ {total_capturas}")
            
            current_val = self.spin_individual.value()
            if current_val > total_capturas or current_val < 1:
                self.spin_individual.setValue(1)
                self.slider_individual.setValue(1)
                
            self.spin_individual.blockSignals(False)
            self.slider_individual.blockSignals(False)
        else:
            self.spin_individual.setMaximum(1)
            self.slider_individual.setMaximum(1)
            self.lbl_total_indiv.setText("/ 1")

    def on_multichunk_state_changed(self, state):
        enabled = self.chk_multichunk.isChecked()
        self.list_stacked_chunks.setEnabled(enabled)
        
        self.chk_compare_groups.setEnabled(enabled)
        compare = self.chk_compare_groups.isChecked() and enabled
        
        self.rbt_target_a.setEnabled(compare)
        self.rbt_target_b.setEnabled(compare)
        self.list_stacked_chunks_b.setEnabled(compare)
        self.chk_show_ratio.setEnabled(compare and self.chk_fft.isChecked())
        
        self.spin_from_chunk.setEnabled(enabled)
        self.spin_to_chunk.setEnabled(enabled)
        self.btn_load_range.setEnabled(enabled)
        
        self.actualizar_enfoque_listas()
        self.actualizar_limites_navegacion()
        if self.hdf_file:
            self.graficar_datos()

    def on_compare_groups_state_changed(self, state):
        enabled = self.chk_multichunk.isChecked()
        compare = self.chk_compare_groups.isChecked() and enabled
        
        self.rbt_target_a.setEnabled(compare)
        self.rbt_target_b.setEnabled(compare)
        self.list_stacked_chunks_b.setEnabled(compare)
        self.chk_show_ratio.setEnabled(compare and self.chk_fft.isChecked())
        
        if not compare:
            self.rbt_target_a.setChecked(True)
            
        self.actualizar_enfoque_listas()
        self.actualizar_limites_navegacion()
        if self.hdf_file:
            self.graficar_datos()


    def load_range_to_stack(self):
        if not self.hdf_file or not self.chunks:
            return
            
        from_idx = self.spin_from_chunk.value() - 1
        to_idx = self.spin_to_chunk.value() - 1
        
        if from_idx > to_idx:
            QtWidgets.QMessageBox.warning(
                self, "Rango Inválido",
                "El chunk de inicio ('Desde') debe ser menor o igual al chunk de fin ('Hasta')."
            )
            return
            
        active_list = self.obtener_lista_activa()
        active_list.clear()
        
        chunks_to_add = self.chunks[from_idx : to_idx + 1]
        
        for cname in chunks_to_add:
            active_list.addItem(cname)
            
        self.actualizar_limites_navegacion()
        self.graficar_datos()

    def on_channel_state_changed(self, state):
        compare = self.chk_compare_groups.isChecked() and self.chk_multichunk.isChecked()
        self.chk_show_ratio.setEnabled(compare and self.chk_fft.isChecked())
        if self.hdf_file:
            self.graficar_datos()

    def cambiar_chunk(self, index):
        if index < 0 or not self.hdf_file or self.combo_modo_vis.currentIndex() == 2:
            return
            
        chunk_name = self.combo_grupo.itemText(index)
        try:
            info = dhn.obtener_info_chunk(self.test_group, chunk_name)
            
            tipo = "Línea Base" if info['is_baseline'] else "Experimental"
            hum_temp_available = "Disponible" if self.has_env_data() else "No Disponible"
            
            self.lbl_info_chunk.setText(
                f"<b>Señales en chunk:</b> {info['n_signals']}<br>"
                f"<b>Tipo de chunk:</b> {tipo}<br>"
                f"<b>Datos de ambiente:</b> {hum_temp_available}"
            )
            self.lbl_info_chunk.setTextFormat(QtCore.Qt.RichText)

            self.actualizar_limites_navegacion()
            if self.combo_modo_vis.currentIndex() != 2:
                self.graficar_datos()

        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error al leer chunk",
                f"No se pudo analizar el chunk seleccionado:\n{e}")

    # --- Signal Step-by-Step Navigation ---

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

    # --- File Operations ---

    def abrir_archivo(self):
        default_dir = config.DEFAULT_DATA_DIR
        if not os.path.exists(default_dir):
            default_dir = os.getcwd()

        options = QtWidgets.QFileDialog.Options()
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Seleccionar archivo HDF5 de Experimento", default_dir,
            "Archivos HDF5 (*.hdf5 *.h5)", options=options)

        if not filepath:
            return

        self.cerrar_archivo_activo()
        try:
            self.filepath = filepath
            self.hdf_file, self.test_group_name, self.metadata, self.chunks = dhn.abrir_experimento(filepath)
            self.test_group = self.hdf_file[self.test_group_name]

            filename = os.path.basename(self.filepath)
            self.lbl_info_file.setText(
                f"<b>Archivo:</b> {filename}<br>"
                f"<b>Fecha:</b> {self.metadata['date']}<br>"
                f"<b>Chunks:</b> {len(self.chunks)}<br>"
                f"<b>Duración Chunk:</b> {self.metadata['chunk_duration_s']} s<br>"
                f"<b>Versión:</b> {self.metadata['version']}"
            )
            self.lbl_info_file.setTextFormat(QtCore.Qt.RichText)
            self.txt_comentario.setPlainText(self.metadata['description'])

            self.combo_grupo.blockSignals(True)
            self.combo_grupo.clear()
            self.combo_grupo.addItems(self.chunks)
            self.combo_grupo.blockSignals(False)

            if len(self.chunks) > 0:
                self.combo_grupo.setCurrentIndex(0)
                self.cambiar_chunk(0)
                
                # Configurar límites de rango multichunk
                n_chunks = len(self.chunks)
                self.spin_from_chunk.setMaximum(n_chunks)
                self.spin_from_chunk.setValue(1)
                
                self.spin_to_chunk.setMaximum(n_chunks)
                self.spin_to_chunk.setValue(min(10, n_chunks))
            else:
                self.lbl_info_chunk.setText("Señales en chunk: --\nTipo: --")
                self.spin_individual.setMaximum(1)
                self.slider_individual.setMaximum(1)
                self.lbl_total_indiv.setText("/ 1")

            self.graficar_datos()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error al abrir Experimento", f"No se pudo cargar el archivo:\n{e}")
            self.cerrar_archivo_activo()

    def cerrar_archivo_activo(self):
        if self.hdf_file:
            try:
                self.hdf_file.close()
            except Exception:
                pass
            self.hdf_file = None
            self.test_group = None

        self.filepath = ""
        self.chunks = []
        self.combo_grupo.blockSignals(True)
        self.combo_grupo.clear()
        self.combo_grupo.blockSignals(False)
        
        self.chk_multichunk.blockSignals(True)
        self.chk_multichunk.setChecked(False)
        self.chk_multichunk.blockSignals(False)
        self.list_stacked_chunks.clear()
        
        self.chk_compare_groups.blockSignals(True)
        self.chk_compare_groups.setChecked(False)
        self.chk_compare_groups.blockSignals(False)
        self.list_stacked_chunks_b.clear()
        
        self.chk_show_ratio.blockSignals(True)
        self.chk_show_ratio.setChecked(True)
        self.chk_show_ratio.setEnabled(False)
        self.chk_show_ratio.blockSignals(False)
        
        self.rbt_target_a.blockSignals(True)
        self.rbt_target_a.setChecked(True)
        self.rbt_target_a.blockSignals(False)
        
        self.spin_from_chunk.setMaximum(1)
        self.spin_from_chunk.setValue(1)
        self.spin_to_chunk.setMaximum(1)
        self.spin_to_chunk.setValue(1)
        
        self.lbl_info_file.setText("Archivo: Sin cargar\nFecha: --\nChunks Totales: --")
        self.lbl_info_chunk.setText("Señales en chunk: --\nTipo: --\nHumedad/Temp: --")
        self.spin_individual.setMaximum(1)
        self.slider_individual.setMaximum(1)
        self.spin_individual.setValue(1)
        self.slider_individual.setValue(1)
        self.lbl_total_indiv.setText("/ 1")
        self.lbl_val_timestamp.setText("--")
        self.lbl_val_trigger.setText("--")
        self.txt_comentario.clear()
        self.limpiar_plots()

    # --- Plotting logic ---

    def graficar_datos(self):
        if not self.hdf_file:
            return

        self.setup_axes()

        modo_idx = self.combo_modo_vis.currentIndex()
        chunk_name = self.combo_grupo.currentText()
        if modo_idx != 2:
            if self.chk_multichunk.isChecked():
                if self.list_stacked_chunks.count() == 0:
                    return
            else:
                if not chunk_name:
                    return

        try:
            if modo_idx == 0:
                self._plot_superpose_chunk(chunk_name)
            elif modo_idx == 1:
                self._plot_individual_signal(chunk_name)
            elif modo_idx == 2:
                self._plot_full_time_series()
            
            self.canvas.draw()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error al graficar", f"Ocurrió un error al procesar el gráfico:\n{e}"
            )
            self.limpiar_plots()

    def _plot_superpose_chunk(self, chunk_name: str):
        if self.chk_compare_groups.isChecked():
            chunks_a = self.obtener_chunks_grupo_a()
            chunks_b = self.obtener_chunks_grupo_b()
        else:
            chunks_a = self.obtener_chunks_grupo_a() if self.chk_multichunk.isChecked() else ([chunk_name] if chunk_name else [])
            chunks_b = []
            
        if not chunks_a and not chunks_b:
            return

        psds_a = []
        psds_b = []
        f_axis = None

        # 1. Plot Group A signals in left column subplots
        total_signals_a = 0
        chunk_data_a = []
        for cname in chunks_a:
            if cname in self.test_group:
                info = dhn.obtener_info_chunk(self.test_group, cname)
                if info['n_signals'] > 0:
                    chunk_data_a.append((cname, info, total_signals_a))
                    total_signals_a += info['n_signals']

        if total_signals_a > 0:
            max_trazos = 150
            paso = max(1, total_signals_a // max_trazos)
            global_indices_a = set(range(0, total_signals_a, paso))

            for cname, info, start_idx in chunk_data_a:
                local_indices = [
                    gi - start_idx for gi in range(start_idx, start_idx + info['n_signals']) if gi in global_indices_a
                ]
                if local_indices:
                    data, timestamps, triggers = dhn.obtener_todos_datos_senales(self.test_group, cname)
                    n_signals, n_points = data.shape
                    dt = 1.0 / 3e9
                    tiempos_us = np.arange(n_points, dtype=np.float64) * (dt * 1e6)
                    
                    for l_idx in local_indices:
                        v_raw = data[l_idx, :]
                        if self.ax_signal:
                            self.ax_signal.plot(
                                tiempos_us, v_raw,
                                color=COLOR_VOLTAGE, alpha=0.35, linewidth=0.7
                            )
                        
                        f, psd = welch(v_raw, fs=3e9, nperseg=1024)
                        psds_a.append(psd)
                        if f_axis is None:
                            f_axis = f
                            
                        if self.ax_fft:
                            self.ax_fft.plot(
                                f / 1e6, psd,
                                color=COLOR_FFT, alpha=0.15, linewidth=0.5
                            )

        # 2. Plot Group B signals in right column subplots
        total_signals_b = 0
        chunk_data_b = []
        for cname in chunks_b:
            if cname in self.test_group:
                info = dhn.obtener_info_chunk(self.test_group, cname)
                if info['n_signals'] > 0:
                    chunk_data_b.append((cname, info, total_signals_b))
                    total_signals_b += info['n_signals']

        COLOR_VOLTAGE_B = "#FF1744"  # Bright Red/Pink for Group B Voltage
        COLOR_FFT_B = "#FFD600"      # Bright Yellow/Gold for Group B FFT

        if total_signals_b > 0:
            max_trazos = 150
            paso = max(1, total_signals_b // max_trazos)
            global_indices_b = set(range(0, total_signals_b, paso))

            for cname, info, start_idx in chunk_data_b:
                local_indices = [
                    gi - start_idx for gi in range(start_idx, start_idx + info['n_signals']) if gi in global_indices_b
                ]
                if local_indices:
                    data, timestamps, triggers = dhn.obtener_todos_datos_senales(self.test_group, cname)
                    n_signals, n_points = data.shape
                    dt = 1.0 / 3e9
                    tiempos_us = np.arange(n_points, dtype=np.float64) * (dt * 1e6)
                    
                    for l_idx in local_indices:
                        v_raw = data[l_idx, :]
                        if self.ax_signal_b:
                            self.ax_signal_b.plot(
                                tiempos_us, v_raw,
                                color=COLOR_VOLTAGE_B, alpha=0.35, linewidth=0.7
                            )
                        
                        f, psd = welch(v_raw, fs=3e9, nperseg=1024)
                        psds_b.append(psd)
                        if f_axis is None:
                            f_axis = f
                            
                        if self.ax_fft_b:
                            self.ax_fft_b.plot(
                                f / 1e6, psd,
                                color=COLOR_FFT_B, alpha=0.15, linewidth=0.5
                            )

        # Plot ghostly average FFT curves
        if total_signals_a > 0 and psds_a:
            mean_psd_a = np.mean(psds_a, axis=0)
            mean_val_a = np.mean(mean_psd_a)
            if self.ax_fft:
                self.ax_fft.plot(
                    f_axis / 1e6, mean_psd_a,
                    color=COLOR_FFT, linewidth=1.2, alpha=0.5, linestyle="--",
                    label=f"Promedio A (Media: {mean_val_a:.2e} V²/Hz)"
                )

        if total_signals_b > 0 and psds_b:
            mean_psd_b = np.mean(psds_b, axis=0)
            mean_val_b = np.mean(mean_psd_b)
            if self.ax_fft_b:
                self.ax_fft_b.plot(
                    f_axis / 1e6, mean_psd_b,
                    color=COLOR_FFT_B, linewidth=1.2, alpha=0.5, linestyle="--",
                    label=f"Promedio B (Media: {mean_val_b:.2e} V²/Hz)"
                )

        # 3. Plot ratio if requested and both groups have PSD data
        if self.ax_ratio and psds_a and psds_b:
            mean_psd_a = np.mean(psds_a, axis=0)
            mean_psd_b = np.mean(psds_b, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = mean_psd_a / mean_psd_b
            valid_ratio = ratio[np.isfinite(ratio)]
            mean_ratio = np.mean(valid_ratio) if len(valid_ratio) > 0 else 0.0
            self.ax_ratio.axhline(1.0, color='#8A8A9E', linestyle=':', linewidth=1.0, alpha=0.7, label="Igualdad (1:1)")
            self.ax_ratio.plot(
                f_axis / 1e6, ratio,
                color='#FF4081', linewidth=1.5,
                label=f"Ratio Promedio A/B (Media: {mean_ratio:.2f})"
            )
            self.ax_ratio.set_title("Ratio de FFT Promedio (Grupo A / Grupo B)", fontsize=10, color='#FF4081', pad=5)
            self.ax_ratio.set_xlabel("Frecuencia (MHz)", fontsize=9)
            self.ax_ratio.set_ylabel("Ratio A/B", fontsize=9)
            if self.ax_ratio.get_legend_handles_labels()[0]:
                self.ax_ratio.legend(loc='upper right', framealpha=0.6)

        if self.ax_signal:
            title = "Comparación: Grupo A" if self.ax_signal_b else ("Superposición Multichunk" if self.chk_multichunk.isChecked() else f"Superposición de Señales (Chunk: {chunk_name})")
            self.ax_signal.set_title(
                f"{title} ({total_signals_a} señales)" if self.ax_signal_b else f"{title} ({len(chunks_a)} chunks, {total_signals_a} señales)",
                fontsize=10, color='#00E5FF', pad=5
            )
            self.ax_signal.set_xlabel("Tiempo relativo de adquisición (µs)", fontsize=9)
            self.ax_signal.set_ylabel("Voltaje (V)", fontsize=9)
            self.ax_signal.grid(True, linestyle=':', alpha=0.25)

        if self.ax_signal_b:
            self.ax_signal_b.set_title(f"Comparación: Grupo B ({total_signals_b} señales)", fontsize=10, color='#FF1744', pad=5)
            self.ax_signal_b.set_xlabel("Tiempo relativo de adquisición (µs)", fontsize=9)
            self.ax_signal_b.set_ylabel("Voltaje (V)", fontsize=9)
            self.ax_signal_b.grid(True, linestyle=':', alpha=0.25)

        if self.ax_fft:
            title_fft = "Grupo A - Espectro (Welch PSD)" if self.ax_fft_b else "Espectro de Potencia (Welch PSD) - Superpuesto"
            self.ax_fft.set_title(title_fft, fontsize=10, color='#E040FB', pad=5)
            self.ax_fft.set_xlabel("Frecuencia (MHz)", fontsize=9)
            self.ax_fft.set_ylabel("PSD (V²/Hz)", fontsize=9)
            self.ax_fft.grid(True, linestyle=':', alpha=0.25)
            if self.ax_fft.get_legend_handles_labels()[0]:
                self.ax_fft.legend(loc='upper right', framealpha=0.6)

        if self.ax_fft_b:
            self.ax_fft_b.set_title("Grupo B - Espectro (Welch PSD)", fontsize=10, color='#FFD600', pad=5)
            self.ax_fft_b.set_xlabel("Frecuencia (MHz)", fontsize=9)
            self.ax_fft_b.set_ylabel("PSD (V²/Hz)", fontsize=9)
            self.ax_fft_b.grid(True, linestyle=':', alpha=0.25)
            if self.ax_fft_b.get_legend_handles_labels()[0]:
                self.ax_fft_b.legend(loc='upper right', framealpha=0.6)
            self.ax_fft_b.set_ylabel("PSD (V²/Hz)", fontsize=9)
            self.ax_fft_b.grid(True, linestyle=':', alpha=0.25)

        # 4. Plot environment in bottom subplot
        if self.ax_env:
            self._plot_environmental_sequential()

    def _plot_individual_signal(self, chunk_name: str):
        idx_sel = self.spin_individual.value() - 1
        
        if self.chk_compare_groups.isChecked():
            # Compare two groups
            chunks_a = self.obtener_chunks_grupo_a()
            chunks_b = self.obtener_chunks_grupo_b()
            
            has_a = False
            has_b = False
            
            try:
                chunk_name_a, local_idx_a, info_a = self.obtener_senal_grupo(chunks_a, idx_sel + 1)
                v_raw_a, timestamp_a, trigger_a = dhn.obtener_datos_senal(self.test_group, chunk_name_a, local_idx_a)
                has_a = True
            except ValueError:
                pass
                
            try:
                chunk_name_b, local_idx_b, info_b = self.obtener_senal_grupo(chunks_b, idx_sel + 1)
                v_raw_b, timestamp_b, trigger_b = dhn.obtener_datos_senal(self.test_group, chunk_name_b, local_idx_b)
                has_b = True
            except ValueError:
                pass
                
            if not has_a and not has_b:
                return
                
            # Metadata update
            meta_str_ts = ""
            meta_str_trig = ""
            if has_a:
                meta_str_ts += f"A: {dhn.formatear_timestamp(timestamp_a)}"
                meta_str_trig += f"A: {trigger_a:.3f} V"
            if has_b:
                if has_a:
                    meta_str_ts += "\n"
                    meta_str_trig += " | "
                meta_str_ts += f"B: {dhn.formatear_timestamp(timestamp_b)}"
                meta_str_trig += f"B: {trigger_b:.3f} V"
            self.lbl_val_timestamp.setText(meta_str_ts)
            self.lbl_val_trigger.setText(meta_str_trig)
            
            dt = 1.0 / 3e9
            
            # Plot Voltages on ax_signal and ax_signal_b
            if self.ax_signal and has_a:
                n_points = len(v_raw_a)
                tiempos_us = np.arange(n_points, dtype=np.float64) * (dt * 1e6)
                self.ax_signal.plot(tiempos_us, v_raw_a, color=COLOR_VOLTAGE, linewidth=1.2, label=f"Señal A #{idx_sel+1}")
                self.ax_signal.set_title(f"Grupo A - Señal #{idx_sel+1}", fontsize=10, color='#00E5FF', pad=5)
                self.ax_signal.set_xlabel("Tiempo relativo (µs)", fontsize=9)
                self.ax_signal.set_ylabel("Voltaje (V)", fontsize=9)
                self.ax_signal.grid(True, linestyle=':', alpha=0.25)
                if self.ax_signal.get_legend_handles_labels()[0]:
                    self.ax_signal.legend(loc='upper right', framealpha=0.6)
                
            if self.ax_signal_b and has_b:
                n_points = len(v_raw_b)
                tiempos_us = np.arange(n_points, dtype=np.float64) * (dt * 1e6)
                self.ax_signal_b.plot(tiempos_us, v_raw_b, color="#FF1744", linewidth=1.2, label=f"Señal B #{idx_sel+1}")
                self.ax_signal_b.set_title(f"Grupo B - Señal #{idx_sel+1}", fontsize=10, color='#FF1744', pad=5)
                self.ax_signal_b.set_xlabel("Tiempo relativo (µs)", fontsize=9)
                self.ax_signal_b.set_ylabel("Voltaje (V)", fontsize=9)
                self.ax_signal_b.grid(True, linestyle=':', alpha=0.25)
                if self.ax_signal_b.get_legend_handles_labels()[0]:
                    self.ax_signal_b.legend(loc='upper right', framealpha=0.6)
                
            # Plot Welch PSD on ax_fft and ax_fft_b
            psd_a = None
            psd_b = None
            f_axis = None

            if has_a:
                f, psd_a = welch(v_raw_a, fs=3e9, nperseg=1024)
                f_axis = f
                mean_val_a = np.mean(psd_a)
                if self.ax_fft:
                    self.ax_fft.plot(f / 1e6, psd_a, color=COLOR_FFT, linewidth=1.5, label=f"PSD A #{idx_sel+1} (Media: {mean_val_a:.2e} V²/Hz)")
                    self.ax_fft.set_title("Grupo A - Welch PSD", fontsize=10, color='#E040FB', pad=5)
                    self.ax_fft.set_xlabel("Frecuencia (MHz)", fontsize=9)
                    self.ax_fft.set_ylabel("PSD (V²/Hz)", fontsize=9)
                    self.ax_fft.grid(True, linestyle=':', alpha=0.25)
                    if self.ax_fft.get_legend_handles_labels()[0]:
                        self.ax_fft.legend(loc='upper right', framealpha=0.6)
                
            if has_b:
                f, psd_b = welch(v_raw_b, fs=3e9, nperseg=1024)
                f_axis = f
                mean_val_b = np.mean(psd_b)
                if self.ax_fft_b:
                    self.ax_fft_b.plot(f / 1e6, psd_b, color="#FFD600", linewidth=1.5, label=f"PSD B #{idx_sel+1} (Media: {mean_val_b:.2e} V²/Hz)")
                    self.ax_fft_b.set_title("Grupo B - Welch PSD", fontsize=10, color='#FFD600', pad=5)
                    self.ax_fft_b.set_xlabel("Frecuencia (MHz)", fontsize=9)
                    self.ax_fft_b.set_ylabel("PSD (V²/Hz)", fontsize=9)
                    self.ax_fft_b.grid(True, linestyle=':', alpha=0.25)
                    if self.ax_fft_b.get_legend_handles_labels()[0]:
                        self.ax_fft_b.legend(loc='upper right', framealpha=0.6)

            # Plot Ratio on ax_ratio if available
            if self.ax_ratio and psd_a is not None and psd_b is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = psd_a / psd_b
                valid_ratio = ratio[np.isfinite(ratio)]
                mean_ratio = np.mean(valid_ratio) if len(valid_ratio) > 0 else 0.0
                self.ax_ratio.axhline(1.0, color='#8A8A9E', linestyle=':', linewidth=1.0, alpha=0.7, label="Igualdad (1:1)")
                self.ax_ratio.plot(
                    f_axis / 1e6, ratio,
                    color='#FF4081', linewidth=1.5, label=f"Ratio Señal #{idx_sel+1} (Media: {mean_ratio:.2f})"
                )
                self.ax_ratio.set_title(f"Ratio de FFT (Grupo A / Grupo B) - Señal #{idx_sel+1}", fontsize=10, color='#FF4081', pad=5)
                self.ax_ratio.set_xlabel("Frecuencia (MHz)", fontsize=9)
                self.ax_ratio.set_ylabel("Ratio A/B", fontsize=9)
                if self.ax_ratio.get_legend_handles_labels()[0]:
                    self.ax_ratio.legend(loc='upper right', framealpha=0.6)
                
            # Plot environment on ax_env
            if self.ax_env:
                active_chunk_highlight = chunk_name_a if has_a else chunk_name_b
                ts_highlight = timestamp_a if has_a else timestamp_b
                self._plot_environmental_sequential(t_highlight=ts_highlight, active_chunk=active_chunk_highlight)
                
        else:
            # Single group mode
            if self.chk_multichunk.isChecked():
                try:
                    chunk_name_active, local_idx_sel, info = self.obtener_senal_multichunk(idx_sel + 1)
                except ValueError:
                    return
            else:
                chunk_name_active = chunk_name
                local_idx_sel = idx_sel
                info = dhn.obtener_info_chunk(self.test_group, chunk_name_active)

            v_raw, timestamp, trigger = dhn.obtener_datos_senal(self.test_group, chunk_name_active, local_idx_sel)
            n_points = len(v_raw)
            
            # Metadata labels update
            self.lbl_val_timestamp.setText(dhn.formatear_timestamp(timestamp))
            self.lbl_val_trigger.setText(f"{trigger:.3f} V")

            dt = 1.0 / 3e9
            tiempos_us = np.arange(n_points, dtype=np.float64) * (dt * 1e6)

            if self.ax_signal:
                self.ax_signal.plot(tiempos_us, v_raw, color=COLOR_VOLTAGE, linewidth=1.2, label=f"Señal #{idx_sel+1}")
                title = f"Señal Individual (Global: {idx_sel+1}, Chunk: {chunk_name_active}, Local: {local_idx_sel+1})"
                self.ax_signal.set_title(title, fontsize=10, color='#FFFFFF', pad=5)
                self.ax_signal.set_xlabel("Tiempo relativo (µs)", fontsize=9)
                self.ax_signal.set_ylabel("Voltaje (V)", fontsize=9)
                self.ax_signal.grid(True, linestyle=':', alpha=0.25)
                if self.ax_signal.get_legend_handles_labels()[0]:
                    self.ax_signal.legend(loc='upper right', framealpha=0.6)

            # Plot Welch PSD on ax_fft
            if self.ax_fft:
                f, psd = welch(v_raw, fs=3e9, nperseg=1024)
                mean_val = np.mean(psd)
                self.ax_fft.plot(f / 1e6, psd, color=COLOR_FFT, linewidth=1.5, label=f"PSD Señal #{idx_sel+1} (Media: {mean_val:.2e} V²/Hz)")
                self.ax_fft.set_title("Espectro de Potencia de la Señal Activa (Welch)", fontsize=10, color='#FFFFFF', pad=5)
                self.ax_fft.set_xlabel("Frecuencia (MHz)", fontsize=9)
                self.ax_fft.set_ylabel("PSD (V²/Hz)", fontsize=9)
                self.ax_fft.grid(True, linestyle=':', alpha=0.25)
                if self.ax_fft.get_legend_handles_labels()[0]:
                    self.ax_fft.legend(loc='upper right', framealpha=0.6)

            # Environmental variables
            if self.ax_env:
                if self.chk_multichunk.isChecked():
                    self._plot_environmental_sequential(t_highlight=timestamp, active_chunk=chunk_name_active)
                else:
                    hum, temp, env_ts = dhn.obtener_datos_ambientales(self.test_group, chunk_name_active)
                    if len(env_ts) > 0:
                        info_c0 = dhn.obtener_info_chunk(self.test_group, self.chunks[0])
                        t_first = info_c0['start_time']
                        t_rel_env = env_ts - t_first
                        t_signal = timestamp - t_first
                        
                        self.ax_env.plot(t_rel_env, temp, color=COLOR_TEMP, linewidth=1.5, label="Temp (°C)")
                        self.ax_env.set_title("Variables del Chunk con Señal Activa", fontsize=10, color='#FFFFFF')
                        self.ax_env.set_xlabel("Tiempo absoluto del experimento (s)", fontsize=9)
                        self.ax_env.set_ylabel("Temperatura (°C)", fontsize=9, color=COLOR_TEMP)
                        self.ax_env.tick_params(axis='y', labelcolor=COLOR_TEMP)
                        self.ax_env.grid(True, linestyle=':', alpha=0.2)
                        
                        ax_hum = self.ax_env.twinx()
                        ax_hum.plot(t_rel_env, hum, color=COLOR_HUMID, linewidth=1.5, label="Humedad (%)")
                        ax_hum.set_ylabel("Humedad (%)", fontsize=9, color=COLOR_HUMID)
                        ax_hum.tick_params(axis='y', labelcolor=COLOR_HUMID)
                        
                        self.ax_env.axvline(t_signal, color="#FFD700", linestyle="--", linewidth=1.2, label="Instante Señal")
                        
                        lines1, labels1 = self.ax_env.get_legend_handles_labels()
                        lines2, labels2 = ax_hum.get_legend_handles_labels()
                        if (lines1 + lines2):
                            self.ax_env.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.6)

    def _plot_environmental_sequential(self, t_highlight=None, active_chunk=None):
        if self.chk_compare_groups.isChecked():
            chunks_a = self.obtener_chunks_grupo_a()
            chunks_b = self.obtener_chunks_grupo_b()
        else:
            chunks_a = self.obtener_chunks_grupo_a() if self.chk_multichunk.isChecked() else ([self.combo_grupo.currentText()] if self.combo_grupo.currentText() else [])
            chunks_b = []
            
        if not chunks_a and not chunks_b:
            return
        if not self.ax_env:
            return
            
        info_c0 = dhn.obtener_info_chunk(self.test_group, self.chunks[0])
        t_first = info_c0['start_time']
        
        # Color constants for B
        COLOR_TEMP_B = "#FF5252"   # Bright Red
        COLOR_HUMID_B = "#69F0AE"  # Light Green
        
        # Plot Group A
        t_env_a = []
        temp_a = []
        hum_a = []
        t_signal_seq_a = None
        
        for cname in chunks_a:
            if cname in self.test_group:
                hum, temp, env_ts = dhn.obtener_datos_ambientales(self.test_group, cname)
                if len(env_ts) > 0:
                    t_env_a.extend(env_ts - t_first)
                    temp_a.extend(temp)
                    hum_a.extend(hum)
                if t_highlight is not None and cname == active_chunk:
                    t_signal_seq_a = t_highlight - t_first
                    
        # Plot Group B
        t_env_b = []
        temp_b = []
        hum_b = []
        t_signal_seq_b = None
        
        for cname in chunks_b:
            if cname in self.test_group:
                hum, temp, env_ts = dhn.obtener_datos_ambientales(self.test_group, cname)
                if len(env_ts) > 0:
                    t_env_b.extend(env_ts - t_first)
                    temp_b.extend(temp)
                    hum_b.extend(hum)
                if t_highlight is not None and cname == active_chunk:
                    t_signal_seq_b = t_highlight - t_first

        # Plot Group A on self.ax_env
        if t_env_a:
            self.ax_env.plot(t_env_a, temp_a, color=COLOR_TEMP, linewidth=1.5, label="Temp A (°C)")
            title_env_a = "Grupo A - Variables Ambientales" if chunks_b else "Perfil de Variables Ambientales Secuencial (Multichunk)"
            self.ax_env.set_title(title_env_a, fontsize=10, color='#FFFFFF')
            self.ax_env.set_xlabel("Tiempo absoluto del experimento (s)", fontsize=9)
            self.ax_env.set_ylabel("Temperatura (°C)", fontsize=9, color=COLOR_TEMP)
            self.ax_env.tick_params(axis='y', labelcolor=COLOR_TEMP)
            self.ax_env.grid(True, linestyle=':', alpha=0.2)
            
            ax_hum_a = self.ax_env.twinx()
            ax_hum_a.plot(t_env_a, hum_a, color=COLOR_HUMID, linewidth=1.5, label="Hum A (%)")
            ax_hum_a.set_ylabel("Humedad (%)", fontsize=9, color=COLOR_HUMID)
            ax_hum_a.tick_params(axis='y', labelcolor=COLOR_HUMID)
            
            if t_signal_seq_a is not None:
                self.ax_env.axvline(t_signal_seq_a, color="#FFD700", linestyle="--", linewidth=1.2, label="Instante Señal")
                
            lines1, labels1 = self.ax_env.get_legend_handles_labels()
            lines2, labels2 = ax_hum_a.get_legend_handles_labels()
            if (lines1 + lines2):
                self.ax_env.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.6)

        # Plot Group B on self.ax_env_b
        if self.ax_env_b and t_env_b:
            self.ax_env_b.plot(t_env_b, temp_b, color=COLOR_TEMP_B, linewidth=1.5, label="Temp B (°C)")
            self.ax_env_b.set_title("Grupo B - Variables Ambientales", fontsize=10, color='#FFFFFF')
            self.ax_env_b.set_xlabel("Tiempo absoluto del experimento (s)", fontsize=9)
            self.ax_env_b.set_ylabel("Temperatura (°C)", fontsize=9, color=COLOR_TEMP_B)
            self.ax_env_b.tick_params(axis='y', labelcolor=COLOR_TEMP_B)
            self.ax_env_b.grid(True, linestyle=':', alpha=0.2)
            
            ax_hum_b = self.ax_env_b.twinx()
            ax_hum_b.plot(t_env_b, hum_b, color=COLOR_HUMID_B, linewidth=1.5, label="Hum B (%)")
            ax_hum_b.set_ylabel("Humedad (%)", fontsize=9, color=COLOR_HUMID_B)
            ax_hum_b.tick_params(axis='y', labelcolor=COLOR_HUMID_B)
            
            if t_signal_seq_b is not None:
                self.ax_env_b.axvline(t_signal_seq_b, color="#FFD700", linestyle="--", linewidth=1.2, label="Instante Señal")
                
            lines1, labels1 = self.ax_env_b.get_legend_handles_labels()
            lines2, labels2 = ax_hum_b.get_legend_handles_labels()
            if (lines1 + lines2):
                self.ax_env_b.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.6)

    def _plot_full_time_series(self):
        chunks_a = self.obtener_chunks_grupo_a()
        chunks_b = self.obtener_chunks_grupo_b()
        
        if not chunks_a and not chunks_b:
            return

        # Determine the global t_first
        info_c0 = dhn.obtener_info_chunk(self.test_group, self.chunks[0])
        t_first = info_c0['start_time']

        COLOR_VOLTAGE_B = "#FF1744"
        COLOR_FFT_B = "#FFD600"

        psds_a = []
        psds_b = []
        f_axis = None

        # 1. Plot Group A signals
        total_signals_a = 0
        chunk_info_a = []
        for i, cname in enumerate(chunks_a):
            if cname in self.test_group:
                info = dhn.obtener_info_chunk(self.test_group, cname)
                if info['n_signals'] > 0:
                    chunk_info_a.append((cname, info, total_signals_a))
                    total_signals_a += info['n_signals']

        if total_signals_a > 0:
            max_trazos = 250
            paso = max(1, total_signals_a // max_trazos)
            global_indices_a = set(range(0, total_signals_a, paso))

            for cname, info, start_idx in chunk_info_a:
                local_indices = [
                    gi - start_idx for gi in range(start_idx, start_idx + info['n_signals']) if gi in global_indices_a
                ]
                if not local_indices:
                    continue
                data, timestamps, triggers = dhn.obtener_todos_datos_senales(self.test_group, cname)
                n_signals, n_points = data.shape
                dt = 1.0 / 3e9
                tiempos_s = np.arange(n_points, dtype=np.float64) * dt
                
                for l_idx in local_indices:
                    t_offset = timestamps[l_idx] - t_first
                    v_raw = data[l_idx, :]
                    if self.ax_signal:
                        self.ax_signal.plot(
                            t_offset + tiempos_s, v_raw,
                            color=COLOR_VOLTAGE, alpha=0.3, linewidth=0.6
                        )
                    
                    f, psd = welch(v_raw, fs=3e9, nperseg=1024)
                    psds_a.append(psd)
                    if f_axis is None:
                        f_axis = f
                        
                    if self.ax_fft:
                        self.ax_fft.plot(
                            f / 1e6, psd,
                            color=COLOR_FFT, alpha=0.15, linewidth=0.5
                        )

        # 2. Plot Group B signals
        total_signals_b = 0
        chunk_info_b = []
        for i, cname in enumerate(chunks_b):
            if cname in self.test_group:
                info = dhn.obtener_info_chunk(self.test_group, cname)
                if info['n_signals'] > 0:
                    chunk_info_b.append((cname, info, total_signals_b))
                    total_signals_b += info['n_signals']

        if total_signals_b > 0:
            max_trazos = 250
            paso = max(1, total_signals_b // max_trazos)
            global_indices_b = set(range(0, total_signals_b, paso))

            for cname, info, start_idx in chunk_info_b:
                local_indices = [
                    gi - start_idx for gi in range(start_idx, start_idx + info['n_signals']) if gi in global_indices_b
                ]
                if not local_indices:
                    continue
                data, timestamps, triggers = dhn.obtener_todos_datos_senales(self.test_group, cname)
                n_signals, n_points = data.shape
                dt = 1.0 / 3e9
                tiempos_s = np.arange(n_points, dtype=np.float64) * dt
                
                for l_idx in local_indices:
                    t_offset = timestamps[l_idx] - t_first
                    v_raw = data[l_idx, :]
                    if self.ax_signal_b:
                        self.ax_signal_b.plot(
                            t_offset + tiempos_s, v_raw,
                            color=COLOR_VOLTAGE_B, alpha=0.3, linewidth=0.6
                        )
                    
                    f, psd = welch(v_raw, fs=3e9, nperseg=1024)
                    psds_b.append(psd)
                    if f_axis is None:
                        f_axis = f
                        
                    if self.ax_fft_b:
                        self.ax_fft_b.plot(
                            f / 1e6, psd,
                            color=COLOR_FFT_B, alpha=0.15, linewidth=0.5
                        )

        # Plot ghostly average FFT curves
        if total_signals_a > 0 and psds_a:
            mean_psd_a = np.mean(psds_a, axis=0)
            mean_val_a = np.mean(mean_psd_a)
            if self.ax_fft:
                self.ax_fft.plot(
                    f_axis / 1e6, mean_psd_a,
                    color=COLOR_FFT, linewidth=1.2, alpha=0.5, linestyle="--",
                    label=f"Promedio A (Media: {mean_val_a:.2e} V²/Hz)"
                )

        if total_signals_b > 0 and psds_b:
            mean_psd_b = np.mean(psds_b, axis=0)
            mean_val_b = np.mean(mean_psd_b)
            if self.ax_fft_b:
                self.ax_fft_b.plot(
                    f_axis / 1e6, mean_psd_b,
                    color=COLOR_FFT_B, linewidth=1.2, alpha=0.5, linestyle="--",
                    label=f"Promedio B (Media: {mean_val_b:.2e} V²/Hz)"
                )

        # 3. Setup Titles & Legends & Ratio
        if self.ax_ratio and psds_a and psds_b:
            mean_psd_a = np.mean(psds_a, axis=0)
            mean_psd_b = np.mean(psds_b, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = mean_psd_a / mean_psd_b
            valid_ratio = ratio[np.isfinite(ratio)]
            mean_ratio = np.mean(valid_ratio) if len(valid_ratio) > 0 else 0.0
            
            self.ax_ratio.axhline(1.0, color='#8A8A9E', linestyle=':', linewidth=1.0, alpha=0.7, label="Igualdad (1:1)")
            self.ax_ratio.plot(
                f_axis / 1e6, ratio,
                color='#FF4081', linewidth=1.5,
                label=f"Ratio Promedio A/B (Media: {mean_ratio:.2f})"
            )
            self.ax_ratio.set_title("Ratio de FFT Promedio - Serie Temporal (Grupo A / Grupo B)", fontsize=10, color='#FF4081', pad=5)
            self.ax_ratio.set_xlabel("Frecuencia (MHz)", fontsize=9)
            self.ax_ratio.set_ylabel("Ratio A/B", fontsize=9)
            if self.ax_ratio.get_legend_handles_labels()[0]:
                self.ax_ratio.legend(loc='upper right', framealpha=0.6)

        if self.ax_signal:
            title_a = "Grupo A - Serie Temporal" if self.ax_signal_b else "Serie Temporal Completa"
            self.ax_signal.set_title(f"{title_a} (Voltaje)", fontsize=10, color='#00E5FF', pad=5)
            self.ax_signal.set_xlabel("Tiempo relativo (s)", fontsize=9)
            self.ax_signal.set_ylabel("Voltaje (V)", fontsize=9)
            self.ax_signal.grid(True, linestyle=':', alpha=0.25)
            
        if self.ax_signal_b:
            self.ax_signal_b.set_title("Grupo B - Serie Temporal (Voltaje)", fontsize=10, color='#FF1744', pad=5)
            self.ax_signal_b.set_xlabel("Tiempo relativo (s)", fontsize=9)
            self.ax_signal_b.set_ylabel("Voltaje (V)", fontsize=9)
            self.ax_signal_b.grid(True, linestyle=':', alpha=0.25)

        if self.ax_fft:
            title_fft_a = "Grupo A - Welch PSD" if self.ax_fft_b else "Espectro de Potencia (Welch PSD) de la Serie Temporal"
            self.ax_fft.set_title(title_fft_a, fontsize=10, color='#E040FB', pad=5)
            self.ax_fft.set_xlabel("Frecuencia (MHz)", fontsize=9)
            self.ax_fft.set_ylabel("PSD (V²/Hz)", fontsize=9)
            self.ax_fft.grid(True, linestyle=':', alpha=0.25)
            if self.ax_fft.get_legend_handles_labels()[0]:
                self.ax_fft.legend(loc='upper right', framealpha=0.6)
            
        if self.ax_fft_b:
            self.ax_fft_b.set_title("Grupo B - Welch PSD", fontsize=10, color='#FFD600', pad=5)
            self.ax_fft_b.set_xlabel("Frecuencia (MHz)", fontsize=9)
            self.ax_fft_b.set_ylabel("PSD (V²/Hz)", fontsize=9)
            self.ax_fft_b.grid(True, linestyle=':', alpha=0.25)
            if self.ax_fft_b.get_legend_handles_labels()[0]:
                self.ax_fft_b.legend(loc='upper right', framealpha=0.6)

        # 4. Plot environment
        if self.ax_env:
            if self.chk_multichunk.isChecked():
                self._plot_environmental_sequential()
            else:
                step = max(1, len(self.chunks) // 120)
                t_env_all = []
                temp_all = []
                hum_all = []
                
                for i in range(0, len(self.chunks), step):
                    cname = self.chunks[i]
                    info = dhn.obtener_info_chunk(self.test_group, cname)
                    h, t, ts = dhn.obtener_datos_ambientales(self.test_group, cname)
                    if len(ts) > 0:
                        t_env_all.extend(ts - t_first)
                        temp_all.extend(t)
                        hum_all.extend(h)
                
                if t_env_all:
                    t_env_all = np.array(t_env_all)
                    temp_all = np.array(temp_all)
                    hum_all = np.array(hum_all)
                    
                    self.ax_env.plot(t_env_all, temp_all, color=COLOR_TEMP, linewidth=1.2, label="Temp (°C)")
                    self.ax_env.set_title("Historial de Variables Ambientales (Humedad y Temp)", fontsize=10, color='#FFFFFF')
                    self.ax_env.set_xlabel("Tiempo absoluto relativo (s)", fontsize=9)
                    self.ax_env.set_ylabel("Temperatura (°C)", fontsize=9, color=COLOR_TEMP)
                    self.ax_env.tick_params(axis='y', labelcolor=COLOR_TEMP)
                    self.ax_env.grid(True, linestyle=':', alpha=0.2)
                    
                    ax_hum = self.ax_env.twinx()
                    ax_hum.plot(t_env_all, hum_all, color=COLOR_HUMID, linewidth=1.2, label="Humedad (%)")
                    ax_hum.set_ylabel("Humedad (%)", fontsize=9, color=COLOR_HUMID)
                    ax_hum.tick_params(axis='y', labelcolor=COLOR_HUMID)
                    
                    lines1, labels1 = self.ax_env.get_legend_handles_labels()
                    lines2, labels2 = ax_hum.get_legend_handles_labels()
                    if (lines1 + lines2):
                        self.ax_env.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.6)

    def guardar_grafico(self):
        if not self.hdf_file:
            return

        sufijo = "completo" if self.combo_modo_vis.currentIndex() == 2 else self.combo_grupo.currentText()
        default_name = f"analisis_nativo_{sufijo}.png"
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
# Main entry point
# =============================================================================

if __name__ == '__main__':
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Segoe UI", 9))

    ventana = AnalizadorNuevoGUI()
    ventana.show()
    sys.exit(app.exec_())
