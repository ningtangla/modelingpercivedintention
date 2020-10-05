
import unittest
from ddt import ddt, data, unpack
import numpy as np 
import pandas as pd 
import pygame as pg
import MouseController as targetCode

class TestMouseController(unittest.TestCase):
	def setUp(self):
		self.controlByMouse=targetCode.MouseControlPolicy()
	def testMouseController(self):
		return

if __name__ == '__main__':
	mouseControllerTest = unittest.TestLoader().loadTestsFromTestCase(TestMouseController)
	unittest.TextTestRunner(verbosity = 2).run(mouseControllerTest)
