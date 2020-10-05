
import unittest
from ddt import ddt, data, unpack
import pandas as pd
import numpy as np
import itertools as it
import Attention as targetCode

@ddt
class TestAttention(unittest.TestCase):
	@data((4.0, 2.4))
	@unpack
	def testModifyPrecisionForUntracked(self,precisionPerSlot,precisionForUntracked):
		self.assertEqual(targetCode.modifyPrecisionForUntracked(2,precisionPerSlot,precisionForUntracked),2)
		self.assertEqual(targetCode.modifyPrecisionForUntracked(0,precisionPerSlot,precisionForUntracked),precisionForUntracked/precisionPerSlot)
	
	@data((0.7,0.3))
	@unpack
	def testModifyDecayForUntracked(self,memoryratePerSlot,memoryrateForUntracked):
		self.assertEqual(targetCode.modifyDecayForUntracked(0.7, memoryratePerSlot, memoryrateForUntracked),0.7)
		self.assertAlmostEqual(targetCode.modifyDecayForUntracked(0,memoryratePerSlot,memoryrateForUntracked),(1 - memoryratePerSlot)/(1 - memoryrateForUntracked),2)

	@data((3,[50,30],2))
	@unpack
	def testAttentionSwitch(self,numberObjects,subtletyHypothesis,attentionLimitation):
		identityListOfTuple = list(it.permutations(range(numberObjects),2))
		numberPairs = len(identityListOfTuple)
		numberSubtlety = len(subtletyHypothesis)
		subtletyList=subtletyHypothesis*numberPairs
		subtletyList.sort()
		identityListOfTuple=identityListOfTuple*numberSubtlety
		hypothesisLevel=[identityListOfTuple[i]+tuple([subtletyList[i]]) for i in range(numberPairs*numberSubtlety)]
		name=['wolfIdentity','sheepIdentity','chasingPrecision']
		priorIndex=pd.MultiIndex.from_tuples(hypothesisLevel,names=name)
		p=[np.log(0.8)]+[np.log(0)]*(len(priorIndex)-1)
		hypothesisInformation=pd.DataFrame(p,priorIndex,columns=['logP'])
		allPairs = hypothesisInformation.groupby(['wolfIdentity','sheepIdentity']).mean().index
		attentionStatusForPair=np.random.multinomial(attentionLimitation,[1/len(allPairs)]*len(allPairs))
		attentionStatusForHypothesis=list(attentionStatusForPair)*numberSubtlety
		hypothesisInformation['attentionStatus']=attentionStatusForHypothesis
		attentionSwitch = targetCode.AttentionSwitch(attentionLimitation)
		newHypothesisInformationList=[attentionSwitch(i) for i in [hypothesisInformation]*3000]
		attentionOnWolfList=[i['attentionStatus'].values[6] for i in newHypothesisInformationList]
		self.assertAlmostEqual(np.sum(attentionOnWolfList)/attentionLimitation/3000,0.8,2)

	@data((3,[50],2,8.0,2.5,0.7,0.45))
	@unpack
	def testAttentionToPrecisionAndDecay(self,numberObjects,subtletyHypothesis,attentionLimitation,precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked):
		identityListOfTuple = list(it.permutations(range(numberObjects),2))
		numberPairs = len(identityListOfTuple)
		numberSubtlety = len(subtletyHypothesis)
		subtletyList=subtletyHypothesis*numberPairs
		subtletyList.sort()
		identityListOfTuple=identityListOfTuple*numberSubtlety
		hypothesisLevel=[identityListOfTuple[i]+tuple([subtletyList[i]]) for i in range(numberPairs*numberSubtlety)]
		name=['wolfIdentity','sheepIdentity','chasingPrecision']
		priorIndex=pd.MultiIndex.from_tuples(hypothesisLevel,names=name)
		p=[np.log(1)]+[np.log(0)]*(len(priorIndex)-1)
		hypothesisInformation=pd.DataFrame(p,priorIndex,columns=['logP'])
		allPairs = hypothesisInformation.groupby(['wolfIdentity','sheepIdentity']).mean().index
		attentionStatusForPair=np.random.multinomial(attentionLimitation,np.exp(p))
		attentionStatusForHypothesis=list(attentionStatusForPair)*numberSubtlety
		hypothesisInformation['attentionStatus']=attentionStatusForHypothesis
		computePrecisionAndDecay=targetCode.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
		[precisionHypothesisDF, decayHypothesisDF]=computePrecisionAndDecay(hypothesisInformation['attentionStatus'])
		self.assertEqual(precisionHypothesisDF['perceptionPrecision'].values[0],attentionLimitation*precisionPerSlot)
		self.assertEqual(precisionHypothesisDF['perceptionPrecision'].values[1],precisionForUntracked)
		self.assertAlmostEqual(decayHypothesisDF['memoryDecay'].values[0],1 - (1 - memoryratePerSlot)/attentionLimitation,2)
		self.assertAlmostEqual(decayHypothesisDF['memoryDecay'].values[1],memoryrateForUntracked,2)
		self.assertAlmostEqual(decayHypothesisDF['memoryDecay'].values[2],memoryrateForUntracked,2)
		self.assertAlmostEqual(decayHypothesisDF['memoryDecay'].values[3],memoryrateForUntracked,2)


if __name__ == '__main__':
	attentionTest = unittest.TestLoader().loadTestsFromTestCase(TestAttention)
	unittest.TextTestRunner(verbosity = 2).run(attentionTest)

