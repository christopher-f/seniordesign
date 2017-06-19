#
# video_faces_to_images.py
#
# This script will take video(s) as input and extract and save all images that have a face detected.
# Author: Allen Wang
#
###
import sys
import errno
import os

import numpy as np
import cv2

import argparse

import openface

from util import mkdirP
from util import iterVids

import dlib
from skimage import io
from skimage import transform


SCORE_THRESH = 1.5
SAMPLE_PD = 5

# This value by default would be 95. A lower value means a smaller size and longer compression time.
IMG_QUAL = 80

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

fileDir = os.path.dirname(os.path.realpath(__file__))
#defInputDir = os.path.join(fileDir, '..', 'videos', 'face_videos')
defInputDir = os.path.join(fileDir, '..', 'videos', 'optical_flow')

defOutputDir = os.path.join(fileDir, '..', 'images', 'extracted_faces_from_videos')

#defInputDir = os.path.dirname('/media/sf_vboxshare/face_videos/')

modelDir = os.path.join(fileDir, '..', 'scripts', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

faceDetectionModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

def extract_faces(args, videoObject, outDir, net, align, videoName, multiple=False):
    cap = videoObject.getCap()

    if not cap.isOpened():
        print "Could not open :", videoObject.name

    capLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    currentFrameNum = 1
    savedPictureNumber = 1

    while(currentFrameNum < capLength):
        # Capture frame-by-frame
        ret, img = cap.read()


        if args.verbose:
            print "Processing frame {}/{}".format(currentFrameNum, capLength)
        currentFrameNum = currentFrameNum + 1

        if currentFrameNum % SAMPLE_PD != 0:
            if args.verbose:
                print "Skipping frame {}".format(currentFrameNum)
            continue


        # Operations on the frame come here
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        dets, scores, idx = detector.run(rgbImg, 1, -1)

        if len(dets) != 0:
            # Face detected
            print scores[0]
            if scores[0] > SCORE_THRESH:
                d = dets[0]
                print("Detection {}, score: {}, face_type:{}".format(d, scores[0], idx[0]))
                savedImagePath = os.path.join(outDir, '{}-{}.jpg'.format(videoName,savedPictureNumber))
                savedPictureNumber = savedPictureNumber + 1

                if args.verbose:
                    print "Face detected: Saving to {}".format(savedImagePath)
                cv2.imwrite(savedImagePath, img, [int(cv2.IMWRITE_JPEG_QUALITY), IMG_QUAL])
            win.clear_overlay()
            win.set_image(rgbImg)
            win.add_overlay(dets)


        """
        if multiple:
            bbs = align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]

        if not (len(bbs) == 0 or (not multiple and bb1 is None)):
            # Face detected
            savedImagePath = os.path.join(outDir, '{}-{}.png'.format(videoName,savedPictureNumber))
            savedPictureNumber = savedPictureNumber + 1

            if args.verbose:
                print "Face detected: Saving to {}".format(savedImagePath)
            cv2.imwrite(savedImagePath, img)
        """


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


def main(args):
    mkdirP(args.outputDir)
    videos = list(iterVids(args.inputDir))
    align = openface.AlignDlib(dlibFacePredictor)
    net = openface.TorchNeuralNet(faceDetectionModel, imgDim=96,cuda=False)

    # TODO - consider shuffling

    processList = [
#            "brendan",
#             "allen",
#             "chris",
#             "rohan",
#             "justin"
#            "eddie",
#            "nikita",
#            "tyler"
    ]

    for videoObject in videos:
        print("=== {} ===".format(videoObject.path))

#        if videoObject.cls not in processList:
#            continue
#        if "2" not in videoObject.name:
#            continue
        classDir = os.path.join(args.outputDir, videoObject.cls)
        mkdirP(classDir)

        extract_faces(args, videoObject, classDir, net, align, videoObject.name, args.multiple)





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputDir', type=str, help="Input video directory",
                        default=defInputDir)
    print defInputDir
    parser.add_argument('--outputDir', type=str, help="Output picture directory",
                        default=defOutputDir)
    parser.add_argument('--verbose', action='store_true',
                        default=True)

    parser.add_argument('--multiple', action='store_true', help="Detect multiple faces?",
                        default=False)

    args = parser.parse_args()

    main(args)
