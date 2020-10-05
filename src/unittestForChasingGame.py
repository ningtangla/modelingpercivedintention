
import unittest
from ddt import ddt, data, unpack
import numpy as np
import ChasingGame as targetCode

@ddt
class TestCalDistanceBetweenStates(unittest.TestCase):
	def testDistance(self):
		state1=[30,40,1,2]
		state2=[0,0,2,3]
		self.assertEqual(targetCode.calDistanceBetweenStates(state1,state2),50)

	
if __name__ == '__main__':
	calDistanceTest = unittest.TestLoader().loadTestsFromTestCase(TestCalDistanceBetweenStates)
	unittest.TextTestRunner(verbosity = 2).run(calDistanceTest)

