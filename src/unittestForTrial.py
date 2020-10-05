
import unittest
from ddt import ddt, data, unpack
import pygame as pg
import pandas as pd
import numpy as np
import itertools as it
import LoadTrajectory
import chasingDetection
import calPosterior
import Attention
import Visualization
import Perception
import Response
import Trial as targetCode

@ddt
class TestTrialResponse(unittest.TestCase):
	@data(('0.pkl',
		[50,11,3.3,1.83,0.92,0.31],
		12,
		36,
		4.0,
		0.0,
		0.7,
		0.0,
		8,
		Visualization.Visualize(circleSize=10, screenColor=[255,255,255], screen = pg.display.set_mode([800,800]), 
			saveImage=False, saveImageFile='image'),
		0.95,
		{50:5,11:30,3.3:60,1.83:90,0.92:120,0.31:150}))
	@unpack
	def testResponseForSimpleChasing(self,filename,subtletyHypothesis,updateFrequency,attentionSwitchFrequency,precisionPerSlot,precisionForUntracked,memoryratePerSlot,memoryrateForUntracked,attentionLimitation,visualize,responseRule,precisionToSubtletyDict):
		loadTrajectory=LoadTrajectory.loadTrajectory
		initialPrior=targetCode.initialPrior(attentionLimitation)
		computePosterior=calPosterior.calPosteriorLog
		attention=Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
		attentionSwitch=Attention.AttentionSwitch(attentionLimitation)
		perception=Perception.Perception(attention, computePosterior, attentionSwitch, attentionSwitchFrequency)
		response=Response.RuleBasedResponse(responseRule, precisionToSubtletyDict)
		trial=targetCode.Trial(subtletyHypothesis, updateFrequency, loadTrajectory, visualize, initialPrior, response, perception)
		trialResponse=trial(filename)
		# self.assertTrue(trialResponse['action'])
		self.assertLessEqual(trialResponse['RT'],8000)
		self.assertEqual(trialResponse['wolfIdentity'],0)
		self.assertEqual(trialResponse['sheepIdentity'],1)



if __name__ == '__main__':
	chasingDetectionSuit = unittest.TestLoader().loadTestsFromTestCase(TestTrialResponse)
	unittest.TextTestRunner(verbosity = 2).run(chasingDetectionSuit)

