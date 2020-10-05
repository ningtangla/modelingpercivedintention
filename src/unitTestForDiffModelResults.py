import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np
import itertools as it
import math
import calPosterior as targetCode


@ddt
class TestIdealObserveModel(unittest.TestCase):
    def setUp(self): 
        #self.dataIndex = pd.MultiIndex.from_product([[0, 1, 2],['x', 'y']], names=['Identity', 'coordinate'])
        self.hypothesesIndex = pd.MultiIndex.from_product([[0, 1, 2], [0, 1, 2], [50, 11, 3.3]], names = ['wolfIdentity', 'sheepIdentity', 'chasingPrecision'])
        self.observedData = pd.DataFrame({'wolfDeviation': [3.14, 3.14, 3.14, 0.1, 0.1, 0.1, 0.78, 0.78, 0.78, 3.04, 3.04, 3.04, 3.14, 3.14, 3.14, 2.36, 2.36, 2.36, 2.36, 2.36, 2.36, 0.78, 0.78, 0.78, 3.14, 3.14, 3.14], 'sheepDeviation': [3.14, 3.14, 3.14, 0.1, 0.1, 0.1, 0.78, 0.78, 0.78, 3.04, 3.04, 3.04, 3.14, 3.14, 3.14, 2.36, 2.36, 2.36, 2.36, 2.36, 2.36, 0.78, 0.78, 0.78, 3.14, 3.14, 3.14]}, index = self.hypothesesIndex)
        self.hypothesesNum = len(self.hypothesesIndex)
        self.beforePosterior = pd.DataFrame(np.log([1.0/self.hypothesesNum]*self.hypothesesNum), index = self.hypothesesIndex, columns = ['logP'])
        self.beforePosterior['attentionStatus'] = [0] * self.hypothesesNum

    @data((10000)) 
    @unpack
    def testIdealObserveModel(self, attention):
        self.highDecayData = self.beforePosterior.copy()
        self.highDecayData['perceptionPrecision'] = [1000] * self.hypothesesNum
        self.highDecayData['memoryDecay'] = [highDecay] * self.hypothesesNum
        highDecayDf = targetCode.calPosteriorLog(self.highDecayData, self.observedData)

        self.lowDecayData = self.beforePosterior.copy()
        self.lowDecayData['perceptionPrecision'] = [1000] * self.hypothesesNum
        self.lowDecayData['memoryDecay'] = [lowDecay] * self.hypothesesNum
        lowDecayDf = targetCode.calPosteriorLog(self.lowDecayData, self.observedData)
        
        highPHighDecay = np.max(highDecayDf['logP'])
        highPLowDecay = np.max(lowDecayDf['logP'])
        lowPHighDecay = np.min(highDecayDf['logP'])
        lowPLowDecay = np.min(lowDecayDf['logP'])
        self.assertLess(highPLowDecay, highPHighDecay)
        self.assertLess(lowPHighDecay, lowPLowDecay)
