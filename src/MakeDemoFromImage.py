
import os
import ffmpy

currentDir = os.getcwd()
parentDir = os.path.abspath(os.path.join(currentDir, os.pardir))
imageFile=parentDir+'/data/image'
imageNameList = os.listdir(imageFile)
inputs=dict()
for imageName in imageNameList:
	inputs[imageName]=None
outputs={'output_demo.mp4': None}
ff=ffmpy.FFmpeg(inputs,outputs)
ff.cmd
ff.run()