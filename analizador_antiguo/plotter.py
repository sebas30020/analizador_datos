"""
plotter.py
----------
Funciones altamente optimizadas de graficación de señales para evitar bloqueos de la GUI.
Reciben los ejes de Matplotlib como argumento para mantener el desacoplamiento.
"""

import numpy as np
import matplotlib
import matplotlib.ticker

from config import COLOR_CH3, COLOR_CH4


def configurar_eje_tiempo(ax, titulo: str, xlabel: str = "Tiempo relativo (s)"):
    """Aplica las configuraciones estéticas comunes al eje temporal."""
    ax.set_title(titulo, fontsize=11, color='#FFFFFF', pad=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Voltaje (V)", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.25)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))


def graficar_senal_individual(ax, group, tiempos: np.ndarray, idx_sel: int,
                              mostrar_ch3: bool, mostrar_ch4: bool):
    """
    Dibuja una única captura (idx_sel) sobre el eje `ax`.
    """
    if mostrar_ch3 and 'ch3_voltajes' in group:
        v_raw = group['ch3_voltajes'][:, idx_sel]
        if not np.all(np.isnan(v_raw)):
            ax.plot(tiempos, v_raw, color=COLOR_CH3, linewidth=1.2, label="CH3")

    if mostrar_ch4 and 'ch4_voltajes' in group:
        v_raw = group['ch4_voltajes'][:, idx_sel]
        if not np.all(np.isnan(v_raw)):
            ax.plot(tiempos, v_raw, color=COLOR_CH4, linewidth=1.2, label="CH4")


def graficar_superposicion(ax, hdf_file, grupos_a_procesar: list[str],
                            mostrar_ch3: bool, mostrar_ch4: bool,
                            alinear: bool, t_first: float):
    """
    Dibuja múltiples capturas superpuestas o en línea de tiempo sobre `ax`.
    Optimizado con carga perezosa (lazy loading) e indexación directa para evitar lentitud de E/S.
    """
    # 1. Obtener los tamaños de todos los grupos y pre-calcular índices globales
    group_info = []
    total_signals = 0
    
    for gname in grupos_a_procesar:
        if gname not in hdf_file:
            continue
        group = hdf_file[gname]
        
        # Obtener dimensiones
        if 'timestamps' in group:
            size = len(group['timestamps'])
        elif 'ch4_voltajes' in group:
            size = group['ch4_voltajes'].shape[1]
        elif 'ch3_voltajes' in group:
            size = group['ch3_voltajes'].shape[1]
        else:
            size = 0
            
        if size > 0:
            group_info.append((gname, total_signals, total_signals + size, size))
            total_signals += size

    if total_signals == 0:
        return

    # Límite máximo de trazos en pantalla para garantizar suavidad y no congelar la GUI (máx. 250 líneas)
    max_trazos = 250
    paso = max(1, total_signals // max_trazos)

    # Conjunto de índices globales que queremos graficar
    global_indices_to_plot = set(range(0, total_signals, paso))

    ch3_count = 0
    ch4_count = 0
    solo_una = (total_signals == 1)

    for gname, start_idx, end_idx, size in group_info:
        # Encontrar qué índices globales a graficar caen en este grupo
        local_indices_to_plot = [
            gi - start_idx for gi in range(start_idx, end_idx) if gi in global_indices_to_plot
        ]
        
        # Si no hay ningún trazo que graficar de este grupo, se salta por completo del disco
        if not local_indices_to_plot:
            continue
            
        group = hdf_file[gname]
        tiempos = group['tiempos'][:]
        timestamps_g = group['timestamps'][:] if 'timestamps' in group else np.zeros(size)
        
        # Referencias a datasets de HDF5 (sin cargar toda la matriz en RAM)
        ds_ch3 = group['ch3_voltajes'] if (mostrar_ch3 and 'ch3_voltajes' in group) else None
        ds_ch4 = group['ch4_voltajes'] if (mostrar_ch4 and 'ch4_voltajes' in group) else None

        # Graficar solo los índices seleccionados
        for l_idx in local_indices_to_plot:
            t_offset = (timestamps_g[l_idx] - t_first) if alinear else 0.0
            t_eje = t_offset + tiempos

            if ds_ch3 is not None and l_idx < ds_ch3.shape[1]:
                v_raw = ds_ch3[:, l_idx]
                if not np.all(np.isnan(v_raw)):
                    lbl = "CH3" if ch3_count == 0 else ""
                    ax.plot(t_eje, v_raw,
                            color=COLOR_CH3,
                            alpha=0.85 if solo_una else 0.4,
                            linewidth=0.8, label=lbl)
                    ch3_count += 1

            if ds_ch4 is not None and l_idx < ds_ch4.shape[1]:
                v_raw = ds_ch4[:, l_idx]
                if not np.all(np.isnan(v_raw)):
                    lbl = "CH4" if ch4_count == 0 else ""
                    ax.plot(t_eje, v_raw,
                            color=COLOR_CH4,
                            alpha=0.85 if solo_una else 0.4,
                            linewidth=0.8, label=lbl)
                    ch4_count += 1


def aplicar_leyenda(ax):
    """Agrega leyenda deduplicada al eje si hay elementos graficados."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(),
                  loc='upper right', framealpha=0.6)
