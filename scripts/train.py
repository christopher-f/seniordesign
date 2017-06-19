## train.py
# Author: Allen Wang
#
# Use
##
import sys
import errno
import os
import time

import numpy as np
import cv2

import argparse

import openface

from util import mkdirP

import dlib
from skimage import io
from skimage import transform

from openface.data import iterImgs

from util import mvP
from shutil import move

from faceRecognition import createModel

SCORE_THRESH = 1.5



# Useful file directories
fileDir = os.path.dirname(os.path.realpath(__file__))

modelDir = os.path.join(fileDir, '..', 'scripts', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

# Local data/training directories
dataDir = os.path.join(fileDir, '..', 'data')
trainingDir = os.path.join(dataDir, 'training-images')

# Server directories' paths
serverPath = os.path.dirname("/var/www/html/")
fullImgPath = os.path.join(serverPath, "img_full")
bestImgPath = os.path.join(serverPath, "img")
# TODO - verify that this is correct
serverTrainingPath = os.path.join(serverPath, 'train_known')



def getServerTrainingDirectories():
    return [name for name in os.listdir(serverTrainingPath)]

def waitForUpdate():
    # Continuously poll
    while True:
        print "Checking for new images"
        subDirectories = getServerTrainingDirectories()
        if len(subDirectories) != 0:
            return
        time.sleep(1)

def moveTrainingImages(args):
    """
    Moves the images from the training folder on the server
    to our local training images folder
    """
    for trainingFolder in getServerTrainingDirectories():
        src = os.path.join(serverTrainingPath, trainingFolder)
        if trainingFolder.endswith('jpg'):
            trainingFolder = trainingFolder[:-4]
        dest = os.path.join(trainingDir, os.path.basename(trainingFolder))
        if args.verbose:
            print "Moving {} to {}".format(src, dest)
        move(src, dest)


def trainFromImages(args):
    """
    Right now we assume that training folder in the server will be empty.
    Once the training folder gets updated, we can check for that here.
    We'll take that as a sign that a new person has been added and we need to classify for them
    """
    while True:
        waitForUpdate()

        if args.verbose:
            print "Detected a change in the server's training path at {}".format(serverTrainingPath)

        moveTrainingImages(args)

        # TODO - create back up models?
        createModel(args, False, args.clean)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true',
                        default=True)
    parser.add_argument('--clean', action='store_true',
                        default=False)
    args = parser.parse_args()
    trainFromImages(args)
