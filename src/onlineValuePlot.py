
import matplotlib.pyplot as plt
import numpy as np
import chasingDetection
import Trial
import LoadTrajectory

class OnlinePlotDeviationAndLikelihood():
	def __init__(self,computeLikelihood,computeAngleBetweenVectors,subtletyHypothesis):
		self.computeLikelihood=computeLikelihood
		self.computeAngleBetweenVectors=computeAngleBetweenVectors
		self.subtletyHypothesis=subtletyHypothesis
	def __call__(self,posteriorHypothesisDF,positionOldTimeDF,positionCurrentTimeDF,line,fig):
		hypothesis=posteriorHypothesisDF.index
		likelihoodHypothesisDF=self.computeLikelihood(hypothesis,positionOldTimeDF,positionCurrentTimeDF)
		likelihoodHypothesisDF=np.exp(likelihoodHypothesisDF)
		likelihoodHypothesisDF=likelihoodHypothesisDF/likelihoodHypothesisDF.sum()
		wolfObjNums = hypothesis.get_level_values('WolfIdentity')
		sheepObjNums = hypothesis.get_level_values('SheepIdentity')
		wolfLocBefore = positionOldTimeDF.iloc[wolfObjNums]
		sheepLocBefore = positionOldTimeDF.iloc[sheepObjNums]
		wolfLocNow = positionCurrentTimeDF.iloc[wolfObjNums]
		sheepLocNow = positionCurrentTimeDF.iloc[sheepObjNums]
		wolfMotion = wolfLocNow - wolfLocBefore
		sheepMotion = sheepLocNow - sheepLocBefore
		seekingOrAvoidMotion = sheepLocBefore.values - wolfLocBefore.values
		chasingAngle = self.computeAngleBetweenVectors(wolfMotion, seekingOrAvoidMotion)
		escapingAngle = self.computeAngleBetweenVectors(sheepMotion, seekingOrAvoidMotion)
		for i in range(np.shape(line)[0]):
			for j in range(np.shape(line)[1]):
				xindex=list(line[i,j].get_xdata())
				likelihood=list(line[i,j].get_ydata())
				xindex.append(len(xindex)+1)
				likelihood.append(likelihoodHypothesisDF.loc[0,1,self.subtletyHypothesis[i*np.shape(line)[1]+j]])
				line[i,j].set_xdata(xindex)
				line[i,j].set_ydata(likelihood)
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.1)




if __name__=='__main__':
	computeLikelihood=chasingDetection.calLikelihoodLog
	computeAngleBetweenVectors=chasingDetection.calAngleBetweenVectors
	initialPrior=Trial.initialPrior
	subtletyHypothesis=[50,11,3.3,1.83,0.92,0.31]
	precisionToSubtletyDict={50:5,11:30,3.3:60,1.83:90,0.92:120,0.31:150}
	numberObjects=3
	filename='0.pkl'
	prior=initialPrior(numberObjects, subtletyHypothesis)
	posteriorHypothesisDF=prior
	[trajectoryDF,realSubtlety]=LoadTrajectory.loadTrajectory(filename)
	fig,ax=plt.subplots(2,np.int(len(subtletyHypothesis)/2),sharex='col',sharey='row')
	line=ax.copy()
	for i in range(np.shape(ax)[0]):
		for j in range(np.shape(ax)[1]):
			ax[i,j].set_xlim(0,len(trajectoryDF.index)/12)
			ax[i,j].set_ylim(-0.2,1.2)
			ax[i,j].set_title('subtlety ='+str(precisionToSubtletyDict[subtletyHypothesis[i*np.shape(line)[1]+j]]))
			line[i,j],=ax[i,j].plot([],[])

	onlinePlotDeviationAndLikelihood=OnlinePlotDeviationAndLikelihood(computeLikelihood, computeAngleBetweenVectors, subtletyHypothesis)
	for time in range(12,len(trajectoryDF.index),12):
		positionOldTimeDF=trajectoryDF.loc[time-12].unstack('Coordinate')
		positionCurrentTimeDF=trajectoryDF.loc[time].unstack('Coordinate')
		onlinePlotDeviationAndLikelihood(posteriorHypothesisDF, positionOldTimeDF, positionCurrentTimeDF, line, fig)







