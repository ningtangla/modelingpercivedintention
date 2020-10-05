
import os
import pandas as pd 
import numpy as np 
import pygame as pg 
import LoadTrajectory
import Visualization
import Trial
import calPosterior
import Perception
import Attention
import Writer
import ResultPlot
import Response

class Experiment():
	def __init__(self,trialListFilename,Trial,writer):
		self.filenameTrialSeries = pd.Series(pd.read_pickle(trialListFilename))
		self.Trial=Trial
		self.writer=writer
	def __call__(self):
		for i in range(len(self.filenameTrialSeries)):
			filename=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/data/trajectory/expt1/'+self.filenameTrialSeries.loc[i]
			response=self.Trial(filename)
			responseDF=pd.DataFrame(response,index=[i])
			self.writer(responseDF)
			print(i)
		pg.quit()

if __name__ =='__main__':
    trialListFilename='TrialList_expt1.pkl'
    screenColor=[0,0,0]
    baseLineWidth = 2
    circleSize=10
    screenWide=800
    screenHeight=800
    subtletyHypothesis=[50,11,3.3,1.83,0.92,0.31]
    precisionToSubtletyDict={50:5,11:30,3.3:60,1.83:90,0.92:120,0.31:150}
    updateFrequency=12
    attentionSwitchFrequency = 24
    responseRule=1
    precisionPerSlot=8.0
    precisionForUntracked=2.5
    memoryratePerSlot=0.7
    memoryrateForUntracked=0.45
    attentionLimitation=2
    outputpath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/data/'
    saveImage=True
    saveImageFile='image'
    saveResultFile='Hybird.csv'

    screen = pg.display.set_mode([screenWide,screenHeight])
    loadTrajectory = LoadTrajectory.loadTrajectory
    initialPrior=Trial.initialPrior(attentionLimitation)
    response=Response.RuleBasedResponse(responseRule, precisionToSubtletyDict)
    # response=Response.Response(precisionToSubtletyDict)
    writer=Writer.writeToCSV(outputpath+'results/'+saveResultFile)
    plotResult=ResultPlot.plotResult
    computePosterior = calPosterior.calPosteriorLog
    attentionSwitch=Attention.AttentionSwitch(attentionLimitation)
    attention = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
    perception=Perception.Perception(attention,computePosterior,attentionSwitch,attentionSwitchFrequency)
    visualize = Visualization.Visualize(baseLineWidth, circleSize, screenColor, screen, saveImage, saveImageFile)
    Trial=Trial.Trial(subtletyHypothesis, updateFrequency, loadTrajectory, visualize, initialPrior, response, perception)
    runExperiment = Experiment(trialListFilename, Trial, writer)
    
    runExperiment()
    plotResult(saveResultFile)
