# P-123

from cgi import test
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.liner_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time
import numpy as np
import seaborn as sns
import pandas as pd
import cv2
import matplotlib.pyploy as plt

if(not os.environ.get('PYTHONHTTPSVERIFY' , '')and getattr(ssl,'_create_unverified_context' , None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X,Y = fetch_openml('mnist_784', version = 1, return_X_y = True)
cls = ['A' , 'B' , 'C' , 'D' , 'E' , 'F' , 'G' , 'H' , 'I' , 'J' , 'K' , 'L' , 'M' , 'N' ,'O' ,'P' ,'Q','R','S','T','U','V','W' ,'X','Y','Z']
nclasses = len(cls)

xtrain , xtest , ytrain , ytest = train_test_split(X,Y, random_state = 9 , train_size = 7500 , test_size = 2500)
xtrain_scale = xtrain/255.0
xtest_scale = xtest/255.0
clf = LogisticRegression(solver = 'saga' , multi_class = 'multinomial').fit(xtrain_scale , ytrain)
ypred = clf.predict(xtest_scale)
accuracy = accuracy_score(ytest , ypred)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret,frame = cap.read()
        gray = cv2.cvtCOLOR(frame,cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperLeft = (int(width/2 - 56), int(height/2 - 56))
        bottomright = (int(width/2 +56) , int(height/2 + 56))
        cv2.rectangle(gray,upperLeft,bottomright,(0,255,0),2)
        roi = gray[upperLeft[1]:bottomright[1],upperLeft[0]:bottomright[0]]
        impil = Image.fromarray(roi)
        imagebw = impil.convert('L')
        imagebwresize = imagebw.resize((28,28),Image.ANTIALIAS)
        imagebwresizeinverted = PIL.ImageOps.invert(imagebwresize)
        pixelfilter = 20
        minpixel = np.percentile(imagebwresizeinverted,pixelfilter)
        imagebwresizeinvertscale = np.clip(imagebwresizeinverted-minpixel,0,255)
        maxpixel = np.max(imagebwresizeinverted)
        imagebwresizeinvertscale = np.asarray(imagebwresizeinvertscale)/maxpixel
        testsample = np.array(imagebwresizeinvertscale).reshape(1,784)
        testpred = clf.predict(testsample)
        predict("predicted ==>" , testpred)
        cv2.imshow('frame' , gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
cap.release()
cv2.detroyAllWindows()
