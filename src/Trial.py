
import os
import itertools as it
import pandas as pd 
import numpy as np 
import pygame as pg 
import warnings
import LoadTrajectory
import Visualization
import calPosterior
import Attention
import Perception
import Response
warnings.simplefilter(action='ignore', category=FutureWarning)

class initialPrior():
    def __init__(self,attentionLimitation):
        self.attentionLimitation=attentionLimitation
    def __call__(self,numberObjects,subtletyHypothesis):
        identityListOfTuple = list(it.permutations(range(numberObjects),2))
        numberPairs = len(identityListOfTuple)
        numberSubtlety = len(subtletyHypothesis)
        subtletyList=subtletyHypothesis*numberPairs
        subtletyList.sort()
        identityListOfTuple=identityListOfTuple*numberSubtlety
        hypothesisLevel=[identityListOfTuple[i]+tuple([subtletyList[i]]) for i in range(numberPairs*numberSubtlety)]
        name=['wolfIdentity','sheepIdentity','chasingPrecision']
        priorIndex=pd.MultiIndex.from_tuples(hypothesisLevel,names=name)
        p=[np.log(1.0/len(priorIndex))]*len(priorIndex)
        initialPrior=pd.DataFrame(p,priorIndex,columns=['logP'])
        allPairs = initialPrior.groupby(['wolfIdentity','sheepIdentity']).mean().index
        attentionStatusForPair=np.random.multinomial(self.attentionLimitation,[1/len(allPairs)]*len(allPairs))
        attentionStatusForHypothesis=list(attentionStatusForPair)*numberSubtlety
        initialPrior['attentionStatus']=attentionStatusForHypothesis
        initialPrior['perceptionPrecision']=np.array([1]*len(priorIndex))
        initialPrior['memoryDecay']=np.array([1]*len(priorIndex))
        return initialPrior

class Trial():
    def __init__(self,subtletyHypothesis,updateFrequency,loadTrajectory,visualize,initialPrior,response,perception):
        self.updateFrequency=updateFrequency
        self.loadTrajectory=loadTrajectory
        self.visualize=visualize
        self.initialPrior=initialPrior
        self.subtletyHypothesis=subtletyHypothesis
        self.response=response
        self.perception=perception
    def __call__(self,filename):
        [trajectoryDF,realSubtlety]=self.loadTrajectory(filename)
        frameLength=len(trajectoryDF.index)
        numberObjects=len(trajectoryDF.columns.unique(level='Identity'))
        prior=self.initialPrior(numberObjects, self.subtletyHypothesis)
        for time in range(1,frameLength):
            positionCurrentTimeDF=trajectoryDF.loc[time].unstack('Coordinate')
            if np.mod(time,self.updateFrequency)==0:
                positionOldTimeDF=trajectoryDF.loc[time-self.updateFrequency].unstack('Coordinate')
                posteriorLogHypothesisDF=self.perception(prior,positionOldTimeDF,positionCurrentTimeDF,time)
            else:
                posteriorLogHypothesisDF=prior.copy()
            if (numberObjects == 4 and realSubtlety == 5) or (numberObjects == 4 and realSubtlety == 120) or (numberObjects == 9 and realSubtlety == 5): 
                self.visualize(posteriorLogHypothesisDF, positionCurrentTimeDF.to_dict('index'))
            prior=posteriorLogHypothesisDF.copy()
            response=self.response(frameLength,posteriorLogHypothesisDF,time)
            if response['action']==True:
                break
        response['RT']=np.multiply(time+1,1000.0/60.0)
        response['realSubtlety']=realSubtlety
        response['numberObjects']=numberObjects
        response['trajectoryFilename']=filename
        return response

if __name__ == '__main__':

	screenColor=[255,255,255]
	circleSize=10
	screenWide=800
	screenHeight=800
	filename='0.pkl'
	subtletyHypothesis=[50,11,3.3,1.83,0.92,0.31]
	precisionToSubtletyDict={50:5,11:30,3.3:60,1.83:90,0.92:120,0.31:150}
	updateFrequency=12
	attentionSwitchFrequency = 24
	responseRule=1.01
	precisionPerSlot=8.0
	precisionForUntracked=2.5
	memoryratePerSlot=0.7
	memoryrateForUntracked=0.45
	attentionLimitation=2
	outputpath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/data/'
	saveImage=False
	saveImageFile='image'

	initialPrior=initialPrior(attentionLimitation)
	screen = pg.display.set_mode([screenWide,screenHeight])
	loadTrajectory=LoadTrajectory.loadTrajectory
	visualize=Visualization.Visualize(circleSize, screenColor, screen, saveImage, outputpath+saveImageFile)
	computePosterior=calPosterior.calPosteriorLog
	attentionSwitch=Attention.AttentionSwitch(attentionLimitation)
	attention = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
	perception=Perception.Perception(attention, computePosterior, attentionSwitch, attentionSwitchFrequency)
	response=Response.RuleBasedResponse(responseRule, precisionToSubtletyDict)
	runOneTrial=Trial(subtletyHypothesis, updateFrequency, loadTrajectory, visualize, initialPrior, response, perception)

	response=runOneTrial(filename)
	pg.quit()
	print(response)
