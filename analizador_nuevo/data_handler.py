"""
data_handler_nuevo.py
---------------------
Funciones nativas para extraer datos del nuevo formato de experimento HDF5
(con chunks, señales y datos ambientales).
"""

import os
import datetime
import h5py
import numpy as np


def abrir_experimento(filepath: str) -> tuple[h5py.File, str, dict, list[str]]:
    """
    Abre el HDF5 y retorna (hdf_file, test_group_name, metadata, list_of_chunks).

    Parameters
    ----------
    filepath : str
        Ruta al archivo .hdf5.

    Returns
    -------
    hdf_file : h5py.File
        Archivo HDF5 abierto en modo lectura.
    test_group_name : str
        Nombre del grupo de test raíz.
    metadata : dict
        Atributos globales del experimento (fecha, descripción, etc.).
    chunks : list[str]
        Lista ordenada de nombres de chunks ('chunk_000000', etc.).
    """
    hdf_file = h5py.File(filepath, 'r')
    
    # Buscar el grupo raíz de prueba
    test_group_name = None
    for k in hdf_file.keys():
        obj = hdf_file[k]
        if isinstance(obj, h5py.Group):
            # Debe contener subgrupos que empiecen por chunk_
            subkeys = list(obj.keys())
            if any(sk.startswith('chunk_') for sk in subkeys):
                test_group_name = k
                break
                
    if not test_group_name:
        hdf_file.close()
        raise ValueError("El archivo no posee la estructura esperada de chunks y señales ambientales.")
        
    test_group = hdf_file[test_group_name]
    
    # Metadata del experimento
    metadata = {
        'date': test_group.attrs.get('date', 'No especificada'),
        'chunk_duration_s': float(test_group.attrs.get('chunk_duration_s', 10.0)),
        'initial_dead_time_s': float(test_group.attrs.get('initial_dead_time_s', 0.0)),
        'description': test_group.attrs.get('description', 'Sin descripción'),
        'version': int(test_group.attrs.get('version', 1))
    }
    
    # Decodificar bytes si es necesario
    if isinstance(metadata['date'], bytes):
        metadata['date'] = metadata['date'].decode('utf-8')
    if isinstance(metadata['description'], bytes):
        metadata['description'] = metadata['description'].decode('utf-8')
        
    chunks = sorted([k for k in test_group.keys() if k.startswith('chunk_')])
    
    return hdf_file, test_group_name, metadata, chunks


def obtener_info_chunk(test_group: h5py.Group, chunk_name: str) -> dict:
    """Extrae atributos de metadatos del chunk seleccionado."""
    c = test_group[chunk_name]
    info = {
        'is_baseline': bool(c.attrs.get('is_baseline', False)),
        'chunk_index': int(c.attrs.get('chunk_index', 0)),
        'start_time': float(c.attrs.get('start_time', 0.0)),
        'end_time': float(c.attrs.get('end_time', 0.0)),
        'n_signals': int(c.attrs.get('n_signals', 0)),
        'signal_offset': int(c.attrs.get('signal_offset', 0))
    }
    return info


def obtener_datos_senal(test_group: h5py.Group, chunk_name: str, idx: int) -> tuple[np.ndarray, float, float]:
    """
    Retorna la señal individual (voltaje), su timestamp y trigger.
    """
    c = test_group[chunk_name]
    data_ds = c['signals/data']
    t_ds = c['signals/timestamps']
    trig_ds = c['signals/triggers']
    
    v_raw = data_ds[idx, :]
    timestamp = float(t_ds[idx])
    trigger = float(trig_ds[idx])
    
    return v_raw, timestamp, trigger


def obtener_todos_datos_senales(test_group: h5py.Group, chunk_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna todos los datos de señales de un chunk.
    """
    c = test_group[chunk_name]
    return c['signals/data'][:], c['signals/timestamps'][:], c['signals/triggers'][:]


def obtener_datos_ambientales(test_group: h5py.Group, chunk_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (humedad, temperatura, timestamps) del chunk si existen.
    """
    c = test_group[chunk_name]
    if 'humidity' in c:
        hum_group = c['humidity']
        return hum_group['humidity'][:], hum_group['temperature'][:], hum_group['timestamps'][:]
    return np.array([]), np.array([]), np.array([])


def formatear_timestamp(t_val: float) -> str:
    """Convierte un timestamp UNIX en string legible."""
    if t_val > 0:
        dt_obj = datetime.datetime.fromtimestamp(t_val)
        return dt_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return "No disponible"
