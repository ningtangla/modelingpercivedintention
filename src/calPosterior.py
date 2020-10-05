import pandas as pd
import numpy as np
import scipy.stats as stats
import math

def calPosteriorLog(hypothesesInformation, observedData):    
    escapingPrecision = 1.94
    hypothesesInformation['chasingLikelihoodLog'] = calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(observedData['wolfDeviation'], 1/(1/hypothesesInformation.index.get_level_values('chasingPrecision') + 1/hypothesesInformation['perceptionPrecision']))
    hypothesesInformation['escapingLikelihoodLog'] = calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(observedData['sheepDeviation'], 1/(1/escapingPrecision + 1/hypothesesInformation['perceptionPrecision']))
    hypothesesInformation['beforeLogPAfterDecay'] = hypothesesInformation['memoryDecay'] * hypothesesInformation['logP']
    hypothesesInformation['logP'] = hypothesesInformation['beforeLogPAfterDecay'] + hypothesesInformation['chasingLikelihoodLog'] + hypothesesInformation['escapingLikelihoodLog'] 
    return hypothesesInformation
    
def calAngleLikelihoodLogModifiedForPiRangeAndMemoryDecay(angle, center):
    return stats.vonmises.logpdf(angle, center) + np.log(2) + np.log(math.pi)



 
