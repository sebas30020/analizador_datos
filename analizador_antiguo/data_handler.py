"""
data_handler.py
---------------
Funciones puras para leer y extraer datos de archivos HDF5 generados
por el capturador Keysight. Sin dependencias de PyQt5 ni Matplotlib.
"""

import os
import numpy as np
import h5py


def abrir_hdf5(filepath: str) -> tuple[h5py.File, list[str], str, str]:
    """
    Abre el archivo HDF5 y devuelve (hdf_file, grupos, fecha, comentario).

    Parameters
    ----------
    filepath : str
        Ruta completa al archivo .h5 / .hdf5.

    Returns
    -------
    hdf_file : h5py.File
        Objeto de archivo abierto en modo lectura.
    grupos : list[str]
        Lista ordenada de claves 'grupo_*'.
    fecha : str
        Atributo 'fecha_creacion' del archivo, o 'No especificada'.
    comentario : str
        Texto del dataset 'comentario', o 'Sin comentarios'.

    Raises
    ------
    Exception
        Si el archivo no puede abrirse o leerse.
    """
    hdf_file = h5py.File(filepath, 'r')
    fecha = hdf_file.attrs.get('fecha_creacion', 'No especificada')
    grupos = sorted([k for k in hdf_file.keys() if k.startswith('grupo_')])
    comentario = leer_comentario(hdf_file)
    return hdf_file, grupos, fecha, comentario


def leer_comentario(hdf_file: h5py.File) -> str:
    """Lee el dataset 'comentario' de nivel raíz del archivo."""
    if 'comentario' not in hdf_file:
        return "Sin comentarios"
    try:
        texto = hdf_file['comentario'].asstr()[()]
    except Exception:
        try:
            texto = hdf_file['comentario'][()]
            if isinstance(texto, bytes):
                texto = texto.decode('utf-8')
        except Exception:
            texto = ""
    return texto if texto else "Sin comentarios"


def contar_capturas(group: h5py.Group) -> int:
    """Devuelve el número de capturas en un grupo HDF5."""
    if 'timestamps' in group:
        return len(group['timestamps'])
    if 'ch4_voltajes' in group:
        return group['ch4_voltajes'].shape[1]
    if 'ch3_voltajes' in group:
        return group['ch3_voltajes'].shape[1]
    return 0


def leer_timestamp_str(group: h5py.Group, idx: int) -> str:
    """Devuelve la fecha/hora formateada del timestamp en el índice dado."""
    import datetime
    if 'timestamps' not in group:
        return "No disponible"
    t_val = group['timestamps'][idx]
    if t_val > 0:
        dt_obj = datetime.datetime.fromtimestamp(t_val)
        return dt_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return "No disponible"


def leer_trigger_str(group: h5py.Group, idx: int) -> str:
    """Devuelve una cadena descriptiva del nivel y canal de disparo."""
    trig_lvl = group['trigger_levels'][idx] if 'trigger_levels' in group else None
    trig_ch  = group['trigger_channels'][idx] if 'trigger_channels' in group else None
    if trig_lvl is not None and trig_ch is not None:
        return f"CH{trig_ch} @ {trig_lvl:.3f} V"
    return "No disponible"


def obtener_primer_timestamp(hdf_file: h5py.File, grupos: list[str]) -> float:
    """Retorna el primer timestamp encontrado entre todos los grupos, o 0.0."""
    for gname in grupos:
        g = hdf_file[gname]
        if 'timestamps' in g and len(g['timestamps']) > 0:
            return float(g['timestamps'][0])
    return 0.0
