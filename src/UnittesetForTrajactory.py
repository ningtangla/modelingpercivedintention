import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np
import itertools as it
import math

@ddt
class TestTrajactory(unittest.TestCase):
    def setUp(self):
        self.objectsNum = 3
        self.index = pd.index(list(range(self.objectsNum)))
        self.stateDimension = ['coordinateX','coordinateY','motionX','motionY']
        self.actionDimension = ['motionChangeX', 'motionChangeY']
        self.columnNames = self.stateDimension + self.actionDimension
        self.currFrameStateAndAction = pd.DataFrame([[0]*self.objectsNum]*len(self.stateDimension, columns = self.stateDimension), self.stateDimension)
        self.currFrameStateAndAction = pd
        #self.dataIndex = pd.MultiIndex.from_product([[0, 1, 2],['x', 'y']], names=['Identity', 'coordinate'])
        self.hypothesesIndex = pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2], [50, 11, 3.3]], names = ['wolfIdentity', 'sheepIdentity', 'chasingPrecision'])
        self.observedData = pd.DataFrame({'wolfDeviation': [3.14, 3.14, 3.14, 0.1, 0.1, 0.1, 0.78, 0.78, 0.78, 3.04, 3.04, 3.04, 3.14, 3.14, 3.14, 2.36, 2.36, 2.36, 2.36, 2.36, 2.36, 0.78, 0.78, 0.78, 3.14, 3.14, 3.14], 'sheepDeviation': [3.14, 3.14, 3.14, 0.1, 0.1, 0.1, 0.78, 0.78, 0.78, 3.04, 3.04, 3.04, 3.14, 3.14, 3.14, 2.36, 2.36, 2.36, 2.36, 2.36, 2.36, 0.78, 0.78, 0.78, 3.14, 3.14, 3.14]}, index = self.hypothesesIndex)
        self.hypothesesNum = len(self.hypothesesIndex)
        self.beforePosterior = pd.DataFrame(np.log([1.0/self.hypothesesNum]*self.hypothesesNum), index = self.hypothesesIndex, columns = ['logP'])
        self.beforePosterior['attentionStatus'] = [0] * self.hypothesesNum
    
    @data((pd.Series(np.log([0.2, 0.3])), pd.Series(np.log([0.4, 0.6]))))   
    @unpack 
    def testNormalizeLogP(self, originalDfLogP, normalizedLogP):
        self.assertTrue(np.allclose(targetCode.normalizeLogP(originalDfLogP), normalizedLogP))
from ddt import ddt, unpack, data


