import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np
import itertools as it
import math
import chasingDetection as targetCode

@ddt
class TestChasingDetectionFuncs(unittest.TestCase):
    def setUp(self): 
        self.dataIndex = pd.MultiIndex.from_product([[0, 1, 2],['x', 'y']], names=['Identity', 'coordinate'])
        self.hypothesisIndex = pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2], [50, 11, 3.3]], names = ['WolfIdentity', 'SheepIdentity', 'Subtlety'])
        self.beforeData = pd.DataFrame([3, 3, 5, 5, 3, 5], index =
                self.dataIndex).unstack('coordinate')
        self.nowData = pd.DataFrame([4, 4, 6, 6, 4, 6], index =
                self.dataIndex).unstack('coordinate')
        self.beforePosterior = pd.DataFrame([1.0/27]*27, index =
                self.hypothesisIndex, columns = ['logP'])
    @data((pd.DataFrame([[0, 0]], index = pd.Index(['x','y'])), 0),
          (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), 1),
          (pd.DataFrame([[2, 0]], index = pd.Index(['x','y'])), 2))
    @unpack
    def testCalVectorNorm(self, vector, normExpected):
        self.assertEqual(targetCode.calVectorNorm(vector).values[0], normExpected)
    
    @data((pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])),
         pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), 0),
          (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])),
              pd.DataFrame([[1, 0]], index = pd.Index(['x','y'])),
              math.pi/2), 
         (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])),
                 pd.DataFrame([[1, 1]], index = pd.Index(['x','y'])),
                 math.pi/4),
         (pd.DataFrame([[0, 1]], index = pd.Index(['x','y'])), pd.DataFrame([[-1, -1]], index = pd.Index(['x','y'])), math.pi*3/4))
    @unpack
    def testCalAngleBetweenVectors(self, vector1, vector2, angleExpected):
        self.assertAlmostEqual(targetCode.calAngleBetweenVectors(vector1,
            vector2).values[0], angleExpected, 3)

    def testCalLikelihood(self):
        likelihoodLogDf = targetCode.calLikelihoodLog(self.hypothesisIndex, self.beforeData, self.nowData)
        self.assertEqual(likelihoodLogDf['logP'].idxmax(), (0, 1, 50))
    
    def testCalPosterior(self):
        calPosteriorLog = targetCode.CalPosteriorLog(0.99)
        posteriorLogDf = calPosteriorLog(self.beforePosterior, self.beforeData, self.nowData)
        self.assertEqual(posteriorLogDf['logP'].idxmax(), (0, 1, 50))
    
    @data((pd.Series(np.log([0.2, 0.3])), pd.Series(np.log([0.4, 0.6]))))   
    @unpack 
    def testNormalizeLogP(self, originalDfLogP, normalizedLogP):
        self.assertTrue(np.allclose(targetCode.normalizeLogP(originalDfLogP), normalizedLogP))

    def tearDown(self):
        pass

if __name__ == '__main__':
   
    chasingDetectionSuit = unittest.TestLoader().loadTestsFromTestCase(TestChasingDetectionFuncs)
    unittest.TextTestRunner(verbosity = 2).run(chasingDetectionSuit) 
    
