
import numpy as np 
import pandas as pd 

def modifyPrecisionForUntracked(attentionStatus,precisionPerSlot,precisionForUntracked):
	attentionStatus = attentionStatus
	if attentionStatus==0:
		return precisionForUntracked/precisionPerSlot
	else:
		return attentionStatus

def modifyDecayForUntracked(attentionStatus,memoryratePerSlot,memoryrateForUntracked):
	attentionStatus = attentionStatus
	if attentionStatus==0:
		return (1 - memoryratePerSlot)/((1 - memoryrateForUntracked)+0.00000001)
	else:
		return attentionStatus

class AttentionSwitch():
    def __init__(self,attentionLimitation):
        self.attentionLimitation=attentionLimitation
    def __call__(self,hypothesisInformation):
        newHypothesisInformation=hypothesisInformation.copy()
        posteriorHypothesis=np.exp(hypothesisInformation['logP'])/np.exp(hypothesisInformation['logP']).sum()
        posterior=posteriorHypothesis.groupby(['wolfIdentity','sheepIdentity']).sum().values
        numOtherCondtionBeyondPair = hypothesisInformation.groupby(['wolfIdentity','sheepIdentity']).size().values[0]
        newAttentionStatus=list(np.random.multinomial(self.attentionLimitation,posterior))*numOtherCondtionBeyondPair
        newHypothesisInformation['attentionStatus']=np.array(newAttentionStatus)
        return newHypothesisInformation

class AttentionToPrecisionAndDecay():
    def __init__(self,precisionPerSlot,precisionForUntracked,memoryratePerSlot,memoryrateForUntracked):
        self.precisionPerSlot = precisionPerSlot
        self.precisionForUntracked = precisionForUntracked
        self.memoryratePerSlot = memoryratePerSlot
        self.memoryrateForUntracked = memoryrateForUntracked
    def __call__(self,attentionStatus):
        attentionForPrecision = list(map(lambda x: modifyPrecisionForUntracked(x,self.precisionPerSlot,self.precisionForUntracked),attentionStatus))
        attentionForDecay = list(map(lambda x: modifyDecayForUntracked(x,self.memoryratePerSlot,self.memoryrateForUntracked),attentionStatus))
        precisionHypothesis = np.multiply(self.precisionPerSlot , attentionForPrecision)+0.00000001
        decayHypothesis = 1 - np.divide((1 - self.memoryratePerSlot),np.add(attentionForDecay,0.00000001))
        precisionHypothesisDF = pd.DataFrame(precisionHypothesis,index=attentionStatus.index,columns=['perceptionPrecision'])
        decayHypothesisDF = pd.DataFrame(decayHypothesis,index=attentionStatus.index,columns=['memoryDecay'])
        return precisionHypothesisDF, decayHypothesisDF
