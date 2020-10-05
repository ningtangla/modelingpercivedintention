
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def computeACC(wolfIdentity,sheepIdentity):
	if (wolfIdentity==0)&(sheepIdentity==1):
		return 1
	else:
		return 0

def plotResult(saveResultFile):
	datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'/data/results/'
	filenameabs=datapath+saveResultFile
	resultsDF=pd.read_csv(filenameabs)
	accList=[computeACC(resultsDF.loc[i]['wolfIdentity'],resultsDF.loc[i]['sheepIdentity']) for i in resultsDF.index]
	resultsDF['ACC']=pd.Series(accList,index=resultsDF.index)
	plotResultsDF=resultsDF.groupby(['numberObjects','realSubtlety'])['ACC'].sum().unstack('numberObjects')
	plotResultsDF.plot(color=['blue','green','red','aqua'],marker='o')
	plt.show()

if __name__=='__main__':
	saveResultFile='Hybrid_8.csv'
	plotResult(saveResultFile)