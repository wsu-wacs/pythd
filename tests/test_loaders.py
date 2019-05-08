import os
import sys
import unittest

from pythd.loaders.wavefront import WavefrontOBJ


OBJ_FILE = os.path.join('data', 'capsule.obj')

class TestWavefront(unittest.TestCase):
    def setUp(self):
        try:
            self.obj = WavefrontOBJ(OBJ_FILE, normalize=True)
        except FileNotFoundError:
            path2 = os.path.join("tests", OBJ_FILE)
            self.obj = WavefrontOBJ(path2, normalize=True)
    
    def test_vertices(self):
        self.assertEqual(self.obj.get_num_vertices(), 5252)
        self.assertEqual(self.obj.get_vertices().shape[1], 4)
    
    def test_normals(self):
        self.assertEqual(self.obj.get_num_vertices(), self.obj.get_num_normals())
        self.assertEqual(self.obj.get_normals().shape[1], 3)
    
    def test_texcoords(self):
        self.assertEqual(self.obj.get_num_texcoords(), self.obj.get_num_vertices())
        self.assertEqual(self.obj.get_texcoords().shape[1], 3)
    
    def test_faces(self):
        self.assertEqual(self.obj.get_num_faces(), 10200)
