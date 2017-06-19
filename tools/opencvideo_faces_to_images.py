#
# video_faces_to_images.py
#
# This script will take video(s) as input and extract and save all images that have a face detected.
# Author: Allen Wang
#
###
import errno
import os

import numpy as np
import cv2

import argparse

from util import mkdirP
from util import iterVids

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

fileDir = os.path.dirname(os.path.realpath(__file__))
defInputDir = os.path.join(fileDir, '..', 'videos', 'face_videos')
defOutputDir = os.path.join(fileDir, '..', 'images', 'extracted_faces_from_videos')


def extract_faces(args, videoObject, outDir):
    cap = videoObject.getCap()

    if not cap.isOpened():
        print ("Could not open :", videoObject.name)

    capLength = int(cap.get(7))

    currentFrameNum = 1
    savedPictureNumber = 1

    while(currentFrameNum < capLength):
        # Capture frame-by-frame
        ret, img = cap.read()

        # TODO - if video already processed then skip
        if args.verbose:
            print ("Processing frame {}/{}".format(currentFrameNum, capLength))
        currentFrameNum = currentFrameNum + 1

        # Operations on the frame come here
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        face_detected = False if len(faces) == 0 else True

        if face_detected:
            # Face detected
            cv2.imshow('frame', img) 
            savedImagePath = os.path.join(outDir, '{}.png'.format(savedPictureNumber))
            savedPictureNumber = savedPictureNumber + 1

            if args.verbose:
                print ("Face detected: Saving to {}".format(savedImagePath))
            cv2.imwrite(savedImagePath, img)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

 
    cap.release()
    cv2.destroyAllWindows()


def main(args):
    mkdirP(args.outputDir)
    videos = list(iterVids(args.inputDir))

    # TODO - consider shuffling

    for videoObject in videos: 
        print("=== {} ===".format(videoObject.path))
        classDir = os.path.join(args.outputDir, videoObject.cls)
        mkdirP(classDir)
        outDir = os.path.join(classDir, videoObject.name)
        mkdirP(outDir)

        extract_faces(args, videoObject, outDir)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputDir', type=str, help="Input video directory",
                        default=defInputDir)
    parser.add_argument('--outputDir', type=str, help="Output picture directory",
                        default=defOutputDir)
    parser.add_argument('--verbose', action='store_true',
                        default=True)

    parser.add_argument('--multiple', action='store_true', help="Detect multiple faces?",
                        default=False)

    args = parser.parse_args()

    main(args)

