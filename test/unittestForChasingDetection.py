import unittest
import pandas as pd
import numpy as np
import itertools as it
import ..src.chasingDetection as targetCode


class TestChansingAngleCalMethod(unittest.TestCase):
   
    def angleZero(self):
        self.assertEqual(targetCode.angleCal(zeroAngleLocDfBefore,
            zeroAngleLocDfNow), 0)

if __name__ == '__main__':
    multiIndex = pd.MultiIndex.from_product([['obj1', 'obj2'],['x',
    'y']], names=['objNum', 'coordinate'])
    
    zeroAngleLocFrameBefore = np.array([[3, 3, 5, 8]])
    zeroAngleLocFrameNow = np.array([[3, 4, 5, 9]])
    zeroAngleLocDfBefore = pd.DataFrame(zeroAngleLocFrameBefore, columns = multiIndex)
    zeroAngleLocDfNow = pd.DataFrame(zeroAngleLocFrameNow, columns
            = multiIndex)
    __import__('ipdb').set_trace()
    unittest.main()
    
