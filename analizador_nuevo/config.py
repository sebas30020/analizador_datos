"""
config.py
---------
Constantes globales de la aplicación: rutas por defecto, dimensiones,
colores de canales y configuración del tema oscuro de Matplotlib.
"""

# ---------------------------------------------------------------------------
# Rutas por defecto
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = r"C:\0_matrix\doctorado\proyectos\programacion\analizador_datos\data"

# ---------------------------------------------------------------------------
# Ventana principal
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Analizador de Señales HDF5 - Keysight Scope"
WINDOW_WIDTH  = 1250
WINDOW_HEIGHT = 850
SIDE_PANEL_WIDTH = 320

# ---------------------------------------------------------------------------
# Colores de canales (en graficación)
# ---------------------------------------------------------------------------
COLOR_CH3 = "#00E5FF"   # Cian eléctrico
COLOR_CH4 = "#FF007F"   # Rosa/magenta

# ---------------------------------------------------------------------------
# Tema oscuro de Matplotlib
# ---------------------------------------------------------------------------
MPLRC = {
    "figure.facecolor":   "#08080A",
    "axes.facecolor":     "#0B0B0D",
    "axes.edgecolor":     "#202028",
    "grid.color":         "#16161C",
    "axes.labelcolor":    "#A0A0AB",
    "xtick.color":        "#80808B",
    "ytick.color":        "#80808B",
    "legend.facecolor":   "#121216",
    "legend.edgecolor":   "#202028",
    "font.sans-serif":    ["Segoe UI", "Roboto", "Helvetica", "Arial"],
    "font.family":        "sans-serif",
}
