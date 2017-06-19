### faceDetection.py
#
# Uses OpenCV face detection to detect faces
#
# Author: Allen Wang
###
import cv2
import sys
import errno
import os

import numpy as np

import dlib
from skimage import io
from skimage import transform

from openface.data import iterImgs

from util import rmP

SCORE_THRESH = 1.5

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
opencvModelDir = os.path.join(modelDir, 'opencv')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

faceCascadeDir = os.path.join(opencvModelDir, 'haarcascade_frontalface_default.xml')

# Get all relevant models first
faceCascade = cv2.CascadeClassifier(faceCascadeDir)

faceDetectionModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

detector = dlib.get_frontal_face_detector()

def detectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return False
    return True


"""
Extract the best faces from the folder and save only those
Assume that there will be at least one good face

Return the best face image

def processFaces(args, imgDirPath):
    bestFace = None
    bestScore = -100

    currentFrameNum = 1
    savedPictureNumber = 1

    imgs = list(iterImgs(imgDirPath))

    for imgObject in imgs:
        deleteImg = True
        rgbImg = imgObject.getRGB()

        dets, scores, idx = detector.run(rgbImg, 1, -1)

        if len(dets) != 0:
            # Face detected
            if args.verbose:
                print scores[0]

            if scores[0] > bestScore:
                bestFace = rgbImg
                bestScore = scores[0]
            if scores[0] > SCORE_THRESH:
                d = dets[0]
                if args.verbose:
                    print("Detection {}, score: {}, face_type:{}".format(d, scores[0], idx[0]))
                    print "We've saved {} pictures".format(savedPictureNumber)
                savedPictureNumber = savedPictureNumber + 1
                deleteImg = False
            if deleteImg:
                # Should only go down this path if the picture was not satisfactory
                rmP(imgObject.path) # This is technically not a private member of the class so...
                if args.verbose:
                    print "Removing {}".format(imgObject.path)

    return bestFace
"""
def processFaces(args, imgDirPath):
    imgs = list(iterImgs(imgDirPath))
    return imgs[0].getBGR()
