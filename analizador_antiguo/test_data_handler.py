import unittest
import os
import h5py
import numpy as np
import data_handler as dh

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.ref_filepath = r"c:\0_matrix\doctorado\proyectos\programacion\analizador_datos\data\Primera medición pre-real_20260618_094112.h5"
        self.nuevo_filepath = r"c:\0_matrix\doctorado\proyectos\programacion\analizador_datos\data\nuevo_exp_ciclos_14.hdf5"

    def test_abrir_hdf5_reference(self):
        """Verifica que el archivo de referencia se abra y tenga la estructura esperada por el cargador."""
        self.assertTrue(os.path.exists(self.ref_filepath), "El archivo de referencia no existe.")
        
        hdf_file, grupos, fecha, comentario = dh.abrir_hdf5(self.ref_filepath)
        
        try:
            self.assertIsNotNone(hdf_file)
            self.assertEqual(len(grupos), 9, "El archivo de referencia debería tener 9 grupos.")
            self.assertTrue(grupos[0].startswith("grupo_"))
            self.assertEqual(fecha, "2026-06-18T09:41:14")
            
            # Verificar estructura interna del primer grupo
            first_group = hdf_file[grupos[0]]
            self.assertIn("tiempos", first_group)
            self.assertIn("ch3_voltajes", first_group)
            self.assertIn("ch4_voltajes", first_group)
            self.assertIn("timestamps", first_group)
            self.assertIn("trigger_levels", first_group)
            self.assertIn("trigger_channels", first_group)
            
            # Verificar formas de los conjuntos de datos
            n_points = len(first_group["tiempos"])
            n_captures = len(first_group["timestamps"])
            
            self.assertEqual(first_group["ch3_voltajes"].shape, (n_points, n_captures))
            self.assertEqual(first_group["ch4_voltajes"].shape, (n_points, n_captures))
            self.assertEqual(first_group["trigger_levels"].shape, (n_captures,))
            self.assertEqual(first_group["trigger_channels"].shape, (n_captures,))
            
        finally:
            hdf_file.close()

    def test_abrir_hdf5_nuevo_exp_converted(self):
        """Verifica que el archivo nuevo_exp_ciclos_14.hdf5 (una vez convertido) cumpla la estructura esperada."""
        self.assertTrue(os.path.exists(self.nuevo_filepath), "El archivo nuevo_exp no existe.")
        
        # Primero intentamos abrirlo. Si no está convertido, esta prueba fallará
        # ya que no encontrará grupos que comiencen con "grupo_".
        try:
            hdf_file, grupos, fecha, comentario = dh.abrir_hdf5(self.nuevo_filepath)
        except Exception as e:
            self.fail(f"Error al abrir el archivo HDF5: {e}")
            
        try:
            self.assertIsNotNone(hdf_file)
            self.assertGreater(len(grupos), 0, "El archivo convertido debería tener grupos tipo grupo_*.")
            self.assertTrue(grupos[0].startswith("grupo_"))
            
            # Verificar el primer grupo
            first_group = hdf_file[grupos[0]]
            self.assertIn("tiempos", first_group)
            self.assertIn("ch3_voltajes", first_group)
            self.assertIn("ch4_voltajes", first_group)
            self.assertIn("timestamps", first_group)
            self.assertIn("trigger_levels", first_group)
            self.assertIn("trigger_channels", first_group)
            
            n_points = len(first_group["tiempos"])
            n_captures = len(first_group["timestamps"])
            
            self.assertEqual(first_group["ch3_voltajes"].shape, (n_points, n_captures))
            self.assertEqual(first_group["ch4_voltajes"].shape, (n_points, n_captures))
            self.assertEqual(first_group["trigger_levels"].shape, (n_captures,))
            self.assertEqual(first_group["trigger_channels"].shape, (n_captures,))
            
        finally:
            hdf_file.close()

if __name__ == '__main__':
    unittest.main()
