
import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np
import pygame
import itertools as it
import LoadTrajectory
import calPosterior
import Attention
import Trial
import Perception as targetCode

@ddt
class TestObserveDF(unittest.TestCase):
	def setUp(self):
		self.subtletyHypothesis=[50]
		self.initialPrior=Trial.initialPrior(1)
		self.loadTrajectory=LoadTrajectory.loadTrajectory
		self.computePosterior=calPosterior.calPosteriorLog
		[self.trajectoryDF,self.realSubtlety]=self.loadTrajectory('0.pkl')
		self.frameLength=len(self.trajectoryDF.index)
		self.numberObjects=len(self.trajectoryDF.columns.unique(level='Identity'))

	def testIdeaObserveDF(self):
		hypothesisInformation=self.initialPrior(self.numberObjects,self.subtletyHypothesis)
		hypothesisInformation['perceptionPrecision']=np.array([1000000]*len(hypothesisInformation.index))
		positionOldTimeDF=self.trajectoryDF.loc[10].unstack('Coordinate')
		positionCurrentTimeDF=positionOldTimeDF.copy()
		positionCurrentTimeDF['x'][0]=positionOldTimeDF['x'][0]-74.74511933
		positionCurrentTimeDF['y'][0]=positionOldTimeDF['y'][0]+41.06284265
		observeDF=targetCode.computeObserveDF(hypothesisInformation, positionOldTimeDF, positionCurrentTimeDF)
		self.assertAlmostEqual(observeDF['wolfDeviation'].values[0],0,2)

	def testNoiseObserveDF(self):
		hypothesisInformation=self.initialPrior(self.numberObjects,self.subtletyHypothesis)
		hypothesisInformation['perceptionPrecision']=np.array([10]*len(hypothesisInformation.index))
		positionOldTimeDF=self.trajectoryDF.loc[10].unstack('Coordinate')
		positionCurrentTimeDF=positionOldTimeDF.copy()
		positionCurrentTimeDF['x'][0]=positionOldTimeDF['x'][0]-74.74511933
		positionCurrentTimeDF['y'][0]=positionOldTimeDF['y'][0]+41.06284265
		observeDF=targetCode.computeObserveDF(hypothesisInformation, positionOldTimeDF, positionCurrentTimeDF)
		self.assertLess(0,np.abs(observeDF['wolfDeviation'].values[0]))
		observeDFList=[targetCode.computeObserveDF(i, positionOldTimeDF, positionCurrentTimeDF) for i in [hypothesisInformation]*1000]
		wolfDeviationList=[i['wolfDeviation'].values[0] for i in observeDFList]
		self.assertAlmostEqual(np.var(wolfDeviationList),1/10,2)

@ddt
class TestPerception(unittest.TestCase):
	def setup(self):
		self.subtletyHypothesis=[50]
		self.initialPrior=Trial.initialPrior(1)
		self.loadTrajectory=LoadTrajectory.loadTrajectory
		self.computePosterior=calPosterior.calPosteriorLog
		[self.trajectoryDF,self.realSubtlety]=self.loadTrajectory('0.pkl')
		self.frameLength=len(self.trajectoryDF.index)
		self.numberObjects=len(self.trajectoryDF.columns.unique(level='Identity'))

	def testIdeaObserve(self):
		computePrecisionDecayFromAttention=Attention.AttentionToPrecisionAndDecay(1000, 1000, 1, 1)
		switchAttention=Attention.AttentionSwitch(0)
		perception=targetCode.Perception(computePrecisionDecayFromAttention, self.computePosterior, switchAttention, 480)
		currentTime=24
		positionOldTimeDF=self.trajectoryDF.loc[currentTime-12].unstack('Coordinate')
		positionCurrentTimeDF=self.trajectoryDF.loc[currentTime].unstack('Coordinate')
		newHypothesisInformation=perception(hypothesisInformation, positionOldTimeDF, positionCurrentTimeDF, currentTime)
		self.assertLess(0.5,np.exp(newHypothesisInformation['logP']).values[0])

	
if __name__ == '__main__':
	# observeDFTest = unittest.TestLoader().loadTestsFromTestCase(TestObserveDF)
	# unittest.TextTestRunner(verbosity = 2).run(observeDFTest)
	perceptionTest = unittest.TestLoader().loadTestsFromTestCase(TestPerception)
	unittest.TextTestRunner(verbosity = 2).run(perceptionTest)

