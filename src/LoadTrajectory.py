
import os
import pandas as pd
import numpy as np

def loadTrajectory(filename):
	trajectoryDict = pd.read_pickle(filename)
	positionList = trajectoryDict['Position']
	def creatSingleObjectDF(position,identity):
		level=[[identity],['x','y']]
		name=['Identity','Coordinate']
		index = pd.MultiIndex.from_product(level,names=name)
		positionDF	= pd.DataFrame(position,columns=index).stack('Identity')
		return positionDF
	positionDFList=[creatSingleObjectDF(positionList[i],i) for i in range(len(positionList))]	
	trajectoryDF = pd.concat(positionDFList).stack('Coordinate')
	trajectoryDF = trajectoryDF.unstack(['Identity','Coordinate'])
	trajectoryDF.index.names=['Time']
	realSubtlety=trajectoryDict['Subtlety']
	return trajectoryDF,realSubtlety

def main():
	filename='0.pkl'
	[trajectoryDF,realSubtlety]=loadTrajectory(filename)
	# trajectoryDF.to_pickle("./testTrajectory.pkl")
	return

if __name__=="__main__":
	main()
