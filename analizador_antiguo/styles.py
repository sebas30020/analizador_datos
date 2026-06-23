"""
styles.py
---------
Hoja de estilos QSS para la aplicación AnalizadorGUI.
Modifica este archivo para cambiar el aspecto visual sin tocar la lógica.
"""

APP_QSS = """
    QMainWindow {
        background-color: #08080A;
    }
    QWidget#centralWidget {
        background-color: #08080A;
    }
    QFrame#PanelLateral {
        background-color: #121216;
        border: 1px solid #202028;
        border-radius: 10px;
    }
    QFrame#PlotContainer {
        background-color: #0B0B0D;
        border: 1px solid #202028;
        border-radius: 10px;
    }
    QLabel {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        font-size: 11px;
        color: #E1E1E6;
    }
    QLabel#title_horiz,
    QLabel#title_rango_y,
    QLabel#title_store {
        color: #8BA7C7;
        font-weight: bold;
        font-size: 12px;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    QComboBox {
        background-color: #181820;
        border: 1px solid #2D2D38;
        border-radius: 5px;
        color: #FFFFFF;
        padding: 4px 8px;
        min-height: 20px;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #121216;
        color: #FFFFFF;
        selection-background-color: #2D2D38;
        border: 1px solid #202028;
    }
    QSpinBox, QDoubleSpinBox {
        background-color: #181820;
        border: 1px solid #2D2D38;
        border-radius: 5px;
        color: #FFFFFF;
        padding: 4px 6px;
        min-height: 18px;
    }
    QCheckBox {
        color: #C8C8D4;
        spacing: 8px;
        font-size: 11px;
    }
    QCheckBox::indicator {
        width: 14px;
        height: 14px;
        border: 1px solid #52526A;
        border-radius: 3px;
        background: #2A2A3C;
    }
    QCheckBox::indicator:hover {
        border: 1px solid #8BA7C7;
        background: #32324A;
    }
    QCheckBox::indicator:checked {
        background: #8BA7C7;
        border-color: #8BA7C7;
    }
    QCheckBox::indicator:disabled {
        background: #181820;
        border-color: #2A2A38;
    }
    QPushButton {
        font-family: 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        font-weight: bold;
        border-radius: 5px;
        padding: 8px 12px;
        color: #FFFFFF;
    }
    QPushButton#BtnOpen {
        background-color: #2D2D38;
        border: 1px solid #3d3d4e;
    }
    QPushButton#BtnOpen:hover {
        background-color: #3D3D4E;
    }
    QPushButton#BtnPlot {
        background-color: #00C853;
        border: none;
    }
    QPushButton#BtnPlot:hover {
        background-color: #00E676;
    }
    QPushButton#BtnSave {
        background-color: #6200EA;
        border: none;
    }
    QPushButton#BtnSave:hover {
        background-color: #7C4DFF;
    }
    QToolTip {
        background-color: #1A1A24;
        color: #FFFFFF;
        border: 1px solid #2D2D38;
        border-radius: 4px;
        font-size: 11px;
    }
"""

TOOLBAR_QSS = """
    QToolBar {
        background-color: #121216;
        border: 1px solid #202028;
        border-radius: 5px;
        padding: 2px;
    }
    QToolButton {
        color: #FFFFFF;
        background-color: transparent;
        border: none;
        margin: 0px 2px;
    }
    QToolButton:hover {
        background-color: #2D2D38;
        border-radius: 3px;
    }
"""

PANEL_INDIVIDUAL_QSS = """
    QFrame#PanelIndividual {
        background-color: #121216;
        border: 1px solid #202028;
        border-radius: 8px;
        padding: 10px;
    }
    QLabel {
        color: #A0A0AB;
    }
    QLabel#ValMeta {
        color: #FFFFFF;
        font-weight: bold;
    }
    QTextEdit#TxtComentario {
        background-color: #0B0B0D;
        border: 1px solid #202028;
        border-radius: 5px;
        color: #E1E1E6;
        font-size: 11px;
    }
"""

BTN_NAV_QSS_TEMPLATE = """
    QPushButton#{obj_name} {{
        background-color: #2D2D38;
        border: 1px solid #3d3d4e;
        padding: 4px 8px;
        font-size: 11px;
    }}
    QPushButton#{obj_name}:hover {{
        background-color: #3D3D4E;
    }}
"""

SLIDER_QSS = """
    QSlider::groove:horizontal {
        border: 1px solid #2D2D38;
        height: 6px;
        background: #181820;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        background: #B388FF;
        border: 1px solid #B388FF;
        width: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }
    QSlider::handle:horizontal:hover {
        background: #D1C4E9;
        border-color: #D1C4E9;
    }
"""
