
import os
import pandas as pd

dataFilename = 'expt3'
currentDir = os.getcwd()
parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
dataDir=parentDir+'/data/trajectory/'+dataFilename

filenameList = os.listdir(dataDir)
pd.to_pickle(filenameList,currentDir+'/TrialList_'+dataFilename+'.pkl')