# -*- coding: utf-8 -*-
# Author: yongyuan.name
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-query", required = True,
	help = "Path to query which contains image to be queried")
ap.add_argument("-index", required = True,
	help = "Path to index")
args = vars(ap.parse_args())


# read in indexed images' feature vectors and corresponding image names
h5f = h5py.File(args["index"],'r')

feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
        
print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")
    
# read and show query image
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
#print rank_ID
#print rank_score


# number of top retrieved images to show
maxres = 9
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]

print("top %d images in order are: " %maxres, imlist)
print (rank_score[0])


fig=plt.figure(figsize=(2, 5))

fig.add_subplot(2, 5, 1)
plt.imshow(queryImg)
plt.title("query" )

# show top #maxres retrieved result one by one
for i,im in enumerate(imlist):
    s= ('{:g}'.format(rank_score[i]))
    if rank_score[i] >=0.70:
	image = mpimg.imread(str(im))
	fig.add_subplot(2, 5, i+2)
	plt.title(s )

	plt.imshow(image)

plt.show()
