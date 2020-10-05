
import numpy as np 
import pandas as pd 

class Response():
	def __init__(self,precisionToSubtletyDict):
		self.precisionToSubtletyDict=precisionToSubtletyDict
	def __call__(self,totalDuration,posteriorLogHypothesisDF,currentTime):
		response=dict()
		posteriorHypothesisDF=np.exp(posteriorLogHypothesisDF)
		posteriorMarginSubtlety=posteriorHypothesisDF.groupby(['wolfIdentity','sheepIdentity']).sum()
		posterior=posteriorMarginSubtlety['logP'].get_values()
		randomChoiceIndex=list(np.random.multinomial(1,posterior))
		chosenHypothesis=randomChoiceIndex.index(1)
		hypothesisSpace=posteriorMarginSubtlety.index.labels
		responseWolfIdentity=hypothesisSpace[0][chosenHypothesis]
		responseSheepIdentity=hypothesisSpace[1][chosenHypothesis]
		posteriorSubtlety=posteriorHypothesisDF.loc[responseWolfIdentity,responseSheepIdentity]
		responsePrecision=posteriorSubtlety.idxmax()[0]
		responseRule=np.random.uniform(0.99,1)
		if np.power(posterior[chosenHypothesis],(20.0-15.0*np.float(currentTime)/np.float(totalDuration)))>responseRule:
			action=True
		else:
			action=False
		response={'action':action,'wolfIdentity':responseWolfIdentity,'sheepIdentity':responseSheepIdentity,'precision':responsePrecision}
		response['responseSubtlety']=self.precisionToSubtletyDict[response['precision']]
		response['responsePosterior']=posterior[chosenHypothesis]
		return response

class RuleBasedResponse():
	def __init__(self,responseRule,precisionToSubtletyDict):
		self.responseRule=responseRule
		self.precisionToSubtletyDict=precisionToSubtletyDict
	def __call__(self,totalDuration,posteriorHypothesisDF,currentTime):
		response=dict()
		posterior=pd.DataFrame(np.exp(posteriorHypothesisDF['logP'])/np.exp(posteriorHypothesisDF['logP']).sum(),index=posteriorHypothesisDF.index,columns=['logP'])
		maxPosterior=posterior['logP'].max()
		maxHypothesis=posterior['logP'].idxmax()
		responseWolfIdentity=maxHypothesis[0]
		responseSheepIdentity=maxHypothesis[1]
		responsePrecision=maxHypothesis[2]
		if maxPosterior>=self.responseRule:
			action=True
		else:
			action=False
		response={'action':action,'wolfIdentity':responseWolfIdentity,'sheepIdentity':responseSheepIdentity,'precision':responsePrecision}
		response['responseSubtlety']=self.precisionToSubtletyDict[response['precision']]
		response['responsePosterior']=maxPosterior
		return response

# class GameResponse():
# 	def __init__(self,precisionToSubtletyDict):
# 		self.precisionToSubtletyDict=precisionToSubtletyDict
# 	def __call__(self,updateStateFunction):
# 		response = dict()
# 		response['ChasingSubtlety']=updateStateFunction.wolfPolicy.precision
# 		return response

def getResponseForGame(realSubtlety):
	response={'ChasingSubtlety':realSubtlety}
	return response


