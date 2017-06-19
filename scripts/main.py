### main.py
#
# Main program that will be run on the local processing unit
# Author: Allen Wang
#
###
import errno
import os

import time
import copy

import numpy as np
import cv2

import argparse

import openface

from faceDetection import detectFace

import faceRecognition
import threading

from util import mkdirP
from util import rmdirP
from util import mvP

from test import testImages

# CONSTANTS
SLEEP_TIME = 0.02
NEEDED_IMAGES = 200
#NEEDED_IMAGES = 20
IDLE_WAIT_COUNT = 150


# Relevant file paths
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')

savedDataDir = os.path.join(fileDir, '..', 'data')
collectBufferDir = os.path.join(savedDataDir, 'collectBuffer')
sessionsDir = os.path.join(savedDataDir, 'sessions')



def eventDetector(args):
    # Make the data folder
    if args.verbose:
        print ("Creating {} if it does not exist already".format(savedDataDir))
        print ("Creating {} if it does not exist already".format(collectBufferDir))
        print ("Creating {} if it does not exist already".format(sessionsDir))
    mkdirP(savedDataDir)
    mkdirP(collectBufferDir)
    mkdirP(sessionsDir)

    # First load the feed
    if len(args.feed) == 0:
        if args.verbose:
            print("Loading feed from webcam")
        cap = cv2.VideoCapture(1)
    else:
        if args.verbose:
            print("Loading feed from {}".format(args.feed))
        cap = cv2.VideoCapture(args.feed)


    processedFrames = 1

    imageCount = 0
    idleCount = 0

    while (cap.isOpened()):
        ret = False
        time.sleep(SLEEP_TIME)
        while not ret:
            ret, frame = cap.read()

        processedFrames = processedFrames + 1


        # Detect a face
        faceDetected = detectFace(frame)
        if faceDetected:
            idleCount = 0
            if (imageCount < NEEDED_IMAGES):
                # Save module
                savedFacePath = os.path.join(collectBufferDir, "{}.jpg".format(imageCount))
                if args.verbose:
                    print "Saving image to {}".format(savedFacePath)
                cv2.imwrite(savedFacePath, frame)
            if (imageCount == NEEDED_IMAGES):
                newFolderName = time.strftime("%m%d%Y%H%M%S.jpg")
                newFolderPath = os.path.join(sessionsDir, newFolderName)

                mvP(collectBufferDir, newFolderPath)
                if args.verbose:
                    print "Saved enough images, moving {} to {}".format(collectBufferDir, newFolderPath, newFolderName)
                mkdirP(collectBufferDir)
                # Move all images in the folder somewhere else
                testingThread = threading.Thread(target=testImages, args=(args, newFolderPath))
                testingThread.start()
            imageCount += 1
        else:
            # No face detected, count idle
            idleCount += 1
            if idleCount >= IDLE_WAIT_COUNT:
                idleCount = 0
                if imageCount < NEEDED_IMAGES:
                    if args.verbose:
                        print "Detected no movement for awhile, deleting everything in the folder"
                    rmdirP(collectBufferDir)
                    mkdirP(collectBufferDir)
                imageCount = 0

        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        # Update optical flow info

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true',
                        default=True)
    parser.add_argument('--feed', type=str, help="Load feed from a file rather than webcam",
            default="")


    args = parser.parse_args()

    eventDetector(args)
