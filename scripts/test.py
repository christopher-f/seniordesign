## test.py
# Author: Allen Wang
#
# Use this script when image collection is over and
#
from faceRecognition import inferSession
from util import iterImgDirectory
from util import mvP
from util import chmodR
from util import chmodP
from faceDetection import processFaces
from shutil import move
import cv2
import glob
import os

"""
serverPath = "/var/www/html/"
fullImgPath = "/var/www/html/img_full/"
bestImgPath = "/var/www/html/img/"
"""
serverPath = os.path.dirname("/var/www/html/")
fullImgPath = os.path.join(serverPath, "img_full")
bestImgPath = os.path.join(serverPath, "img")


def testImages(args, imgDirPath):
    folderName = os.path.basename(imgDirPath)

    if args.verbose:
        print "Starting testing"
    # Assume imgDirPath is a directory with all of the images
    imgs = iterImgDirectory(imgDirPath)

    (person, confidence, scoreMap) = inferSession(args, imgs)

    if args.verbose:
        print "Classfication: {} with confidence {}".format(person, confidence)

    # person is the classification
    # confidence is the probability that this is correct
    # scoreMap shows the results
    if person == "unknown":
        if args.verbose:
            print "unknown person detected!"
        # imgs_in_dir = glob.glob(os.path.join(imgDirPath, '*.jpg'))
        bestFace = processFaces(args, imgDirPath)

        bestFacePath = os.path.join(bestImgPath, folderName)

        cv2.imwrite(bestFacePath, bestFace)
        move(imgDirPath, fullImgPath)
        chmodP(bestFacePath)
        chmodR(fullImgPath)
