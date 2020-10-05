
import os
import pandas as pd
import numpy as np
import pygame as pg

def transPosteriorFromHypothesisToVisualization(posteriorHypothesisDF):
    posteriorNormalizedDF=pd.DataFrame(np.exp(posteriorHypothesisDF)/np.exp(posteriorHypothesisDF).sum(),index=posteriorHypothesisDF.index,columns=['logP'])
    posteriorWolf = posteriorNormalizedDF.groupby('wolfIdentity')['logP'].sum()
    posteriorWolf=pd.DataFrame(posteriorWolf.values, index = posteriorWolf.index, columns = ['posteriorWolf'])
    posteriorSheep = posteriorNormalizedDF.groupby('sheepIdentity')['logP'].sum()
    posteriorSheep=pd.DataFrame(posteriorSheep.values, index = posteriorSheep.index, columns = ['posteriorSheep'])
    posteriorDF = pd.concat([posteriorWolf,posteriorSheep],1)
    posteriorDict = posteriorDF.to_dict('index')
    return posteriorDict

def transPosteriorToColor(posteriorWolf,posteriorSheep,colorWolf=255,colorSheep=255):
    color = [posteriorWolf*colorWolf,posteriorSheep*colorSheep, int(255*(1-posteriorSheep-posteriorWolf))]
    return color

class Visualize():
    def __init__(self, baseLineWidth, circleSize,screenColor,screen,saveImage,saveImageFile):
        self.baseLineWidth = baseLineWidth
        self.circleSize = circleSize
        self.screenColor = screenColor
        self.screen = screen
        # self.clock = pg.time.Clock()
        self.saveImage=saveImage
        self.saveImageFile=saveImageFile
    def __call__(self,posteriorHypothesisDF,positionDict):
        posteriorDict=transPosteriorFromHypothesisToVisualization(posteriorHypothesisDF)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                done = True
        screen=self.screen
        screen.fill(self.screenColor)
        for i in positionDict.keys():
            color = transPosteriorToColor(posteriorDict[i]['posteriorWolf'],posteriorDict[i]['posteriorSheep'])
            pg.draw.circle(screen,color,[np.int(positionDict[i]['x']),np.int(positionDict[i]['y'])],self.circleSize)
        attentionStatusDF=posteriorHypothesisDF.groupby(['wolfIdentity','sheepIdentity'])['attentionStatus'].mean()
        attentionStatusIndex=attentionStatusDF.index[attentionStatusDF>0].tolist()
        for index in attentionStatusIndex:
            lineWidth = attentionStatusDF[index[0]][index[1]] * self.baseLineWidth
            pg.draw.line(screen, [128,128,128], [np.int(positionDict[index[0]]['x']),np.int(positionDict[index[0]]['y'])], [np.int(positionDict[index[1]]['x']),np.int(positionDict[index[1]]['y'])], np.int(lineWidth))
        pg.display.flip()
        currentDir = os.getcwd()
        parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
        saveImageDir=parentDir+'/data/'+self.saveImageFile
        if self.saveImage==True:
            filenameList = os.listdir(saveImageDir)
            pg.image.save(screen,saveImageDir+'/'+str(len(filenameList))+'.png')
        pg.time.wait(1)








