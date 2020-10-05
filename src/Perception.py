
import pandas as pd 
import numpy as np 

def computeDeviationFromTrajectory(vector1,vector2):
	def calVectorNorm(vector):
		return np.power(np.power(vector, 2).sum(axis = 1), 0.5)
	innerProduct = np.dot(vector1, vector2.T).diagonal()
	angle = np.arccos(innerProduct/(calVectorNorm(vector1)*calVectorNorm(vector2)))
	return angle

def computeObserveDF(hypothesisInformation,positionOldTimeDF,positionCurrentTimeDF):
	hypothesis = hypothesisInformation.index
	observeDF = pd.DataFrame(index=hypothesis,columns=['wolfDeviation'])
	wolfObjNums = hypothesis.get_level_values('wolfIdentity')
	sheepObjNums = hypothesis.get_level_values('sheepIdentity')
	wolfLocBefore = positionOldTimeDF.iloc[wolfObjNums]
	sheepLocBefore = positionOldTimeDF.iloc[sheepObjNums]
	wolfLocNow = positionCurrentTimeDF.iloc[wolfObjNums]
	sheepLocNow = positionCurrentTimeDF.iloc[sheepObjNums]
	wolfMotion = wolfLocNow - wolfLocBefore
	sheepMotion = sheepLocNow - sheepLocBefore
	seekingOrAvoidMotion = sheepLocBefore.values - wolfLocBefore.values
	chasingAngle = computeDeviationFromTrajectory(wolfMotion, seekingOrAvoidMotion)
	escapingAngle = computeDeviationFromTrajectory(sheepMotion, seekingOrAvoidMotion)
	deviationAngleForWolf = np.random.vonmises(0,hypothesisInformation['perceptionPrecision'].values)
	deviationAngleForSheep = np.random.vonmises(0,hypothesisInformation['perceptionPrecision'].values)
	observeDF['wolfDeviation']=pd.DataFrame(chasingAngle.values+deviationAngleForWolf,index=hypothesis,columns=['wolfDeviation'])
	observeDF['sheepDeviation']=pd.DataFrame(escapingAngle.values+deviationAngleForSheep,index=hypothesis,columns=['sheepDeviation'])
	return observeDF

def updateHypothesisInformation(hypothesisInformation,precisionHypothesisDF,decayHypothesisDF):
	hypothesisInformation['perceptionPrecision'] = precisionHypothesisDF.values
	hypothesisInformation['memoryDecay'] = decayHypothesisDF.values
	return hypothesisInformation

class Perception():
	def __init__(self,attention,computePosterior,attentionSwitch,attentionSwitchFrequency):
		self.attention = attention
		self.computePosterior=computePosterior
		self.attentionSwitch = attentionSwitch
		self.attentionSwitchFrequency=attentionSwitchFrequency
	def __call__(self,hypothesisInformation,positionOldTimeDF,positionCurrentTimeDF,currentTime):
		if np.mod(currentTime,self.attentionSwitchFrequency)==0:
			hypothesisInformation = self.attentionSwitch(hypothesisInformation)
		attentionStatusDF = hypothesisInformation['attentionStatus']
		[precisionHypothesisDF,decayHypothesisDF]=self.attention(attentionStatusDF)
		hypothesisInformation = updateHypothesisInformation(hypothesisInformation, precisionHypothesisDF, decayHypothesisDF)
		observeDF = computeObserveDF(hypothesisInformation, positionOldTimeDF, positionCurrentTimeDF)
		posteriorHypothesisDF = self.computePosterior(hypothesisInformation,observeDF)
		return posteriorHypothesisDF

if __name__=='__main__':
	print('end')
