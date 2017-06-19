### faceRecognition.py
#
# This script uses open face to train a model and to test if a person is recognized
# Author: Allen Wang
#
###
import os
import time

import cv2
import numpy as np
import pandas as pd
import random
import shutil
import pickle
from operator import itemgetter
import subprocess
import multiprocessing

import openface
from openface.data import iterImgs

from util import mkdirP
from util import rmP
from util import iterImgDirectory
from util import divideArray

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import shutil


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

dataDir = os.path.join(fileDir, '..', 'data')
trainingDir = os.path.join(dataDir, 'training-images')
testingDir = os.path.join(dataDir, 'testing-images')
alignedDir = os.path.join(dataDir, 'aligned-faces')
generatedDir = os.path.join(dataDir, 'generated-faces')


pklPath = os.path.join(modelDir, 'faceRecognitionClassifier.pkl')


dlibFacePredictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")
torchNetworkModel = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')


labelsPath = os.path.join(generatedDir, 'labels.csv')
repsPath = os.path.join(generatedDir, 'reps.csv')


imgDim = 96
imgSize = 96
landmarks = 'outerEyesAndNose'
ldaDim = 1

align_pred = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(torchNetworkModel, imgDim=imgDim,
                              cuda=True)


def getRep(imgPath, args, multiple=False):
    start = time.time()
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}\n".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    if multiple:
        bbs = align_pred.getAllFaceBoundingBoxes(rgbImg)
        print bbs
    else:
        bb1 = align_pred.getLargestFaceBoundingBox(rgbImg)
        bbs = [bb1]
        print bb1
    if len(bbs) == 0 or (not multiple and bb1 is None):
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    reps = []
    for bb in bbs:
        start = time.time()
        alignedFace = align_pred.align(
            imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if args.verbose:
            print("Alignment took {} seconds.".format(time.time() - start))
            print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

        start = time.time()
        rep = net.forward(alignedFace)
        if args.verbose:
            print("Neural network forward pass took {} seconds.".format(
                time.time() - start))
        reps.append((bb.center().x, rep))
    sreps = sorted(reps, key=lambda x: x[0])
    return sreps


def align(args, multi):
    # Aligning now

    start = time.time()

    mkdirP(alignedDir)
    imgs = list(iterImgs(trainingDir))
    # Shuffle so multiple versions can be run at once.
    random.shuffle(imgs)
    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if landmarks not in landmarkMap:
        # TODO: Avoid exceptions, find way to silently fail and skip image
        raise Exception("Landmarks unrecognized: {}".format(landmarks))

    landmarkIndices = landmarkMap[landmarks]
    align = openface.AlignDlib(dlibFacePredictor)
    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(alignedDir, imgObject.cls)
        mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        # TODO: output is still PNG?
        imgName = outputPrefix + ".png"

        if os.path.isfile(imgName):
            if args.verbose:
                print("  + Already found, skipping.")
        else:
            rgb = imgObject.getRGB()
            if rgb is None:
                if args.verbose:
                    print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align.align(imgSize, rgb,
                                     landmarkIndices=landmarkIndices,
                                     skipMulti=not multi)
                if outRgb is None and args.verbose:
                    print("  + Unable to align.")
            if outRgb is not None:
                if args.verbose:
                    print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)

# Run this before calling batch_align(...)
def prebatchAlign(num_processes, args, multi):
    mkdirP(alignedDir)
    imgs = list(iterImgs(trainingDir))
    np.random.shuffle(imgs)
    div_arrays = divideArray(imgs, num_processes)
    return div_arrays

# Align images in batches, allows multiprocessing
def batchAlign(imgs):
    multi = False
    start = time.time()

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if landmarks not in landmarkMap:
        # TODO: Avoid exceptions, find way to silently fail and skip image
        raise Exception("Landmarks unrecognized: {}".format(landmarks))

    landmarkIndices = landmarkMap[landmarks]
    nFallbacks = 0
    for imgObject in imgs:
        print("=== {} ===".format(imgObject.path))
        outDir = os.path.join(alignedDir, imgObject.cls)
        mkdirP(outDir)
        outputPrefix = os.path.join(outDir, imgObject.name)
        imgName = outputPrefix + ".png"

        '''            print("  + Already found, skipping.")
                else:'''
        if not os.path.isfile(imgName):
            rgb = imgObject.getRGB()
            if rgb is None:
                print("  + Unable to load.")
                outRgb = None
            else:
                outRgb = align_pred.align(imgSize, rgb,
                                     landmarkIndices=landmarkIndices,
                                     skipMulti=not multi)
                if outRgb is None:
                    print("  + Unable to align.")
            if outRgb is not None:
                print("  + Writing aligned file to disk.")
                outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(imgName, outBgr)

def generateRepresentations(args):
    # Open Face does this with lua so we'll use a subprocess to do this
    if args.verbose:
        print("Generating representations")
    # Need to remove the cache or sometimes we get errors...
    rmP(os.path.join(alignedDir, 'cache.t7'))
    subprocess.check_output(
        [
            './batch-represent/main.lua',
            '-outDir', generatedDir,
            '-data', alignedDir,
        ]
    )

def train(args, clean=False):
    if args.verbose:
        print("Training")

    labels = pd.read_csv(labelsPath, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    embeddings = pd.read_csv(repsPath, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    clf = SVC(C=1, kernel='linear', probability=True)

    if ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=ldaDim)),
                        ('clf', clf_final)])

    clf.fit(embeddings, labelsNum)

    print("Saving classifier to '{}'".format(pklPath))
    with open(pklPath, 'wb') as f:
        pickle.dump((le, clf), f)


# Everything underneath is public


def createModel(args, multi, clean=False):
    if args.verbose:
        print("Creating face recognition model")

    if clean:
        # TODO: Why only perform these actions if verbose?
        print("Cleaning...")
        print("Deleting generated-faces: {}".format(generatedDir))
        print("Deleting aligned-faces: {}".format(alignedDir))
        shutil.rmtree(alignedDir, ignore_errors = True)
        shutil.rmtree(generatedDir, ignore_errors = True)

    # Original single-threaded method
    # align(args, multi)
    # Proposed multithreading:
    img_splits = prebatchAlign(5, args, multi)
    pool = multiprocessing.Pool(5)
    pool.map(batchAlign, img_splits)
    generateRepresentations(args)
    print('Training SVM')
    train(args)

def prebatchTestAlign(imgs, num_processes):
    np.random.shuffle(imgs)
    return divideArray(imgs, num_processes)

def batchTestAlign(imgs):
    print('Starting batchTestAlign thread')
    multi = False

    landmarkMap = {
        'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
        'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
    }
    if landmarks not in landmarkMap:
        # TODO: Avoid exceptions, find way to silently fail and skip image
        raise Exception("Landmarks unrecognized: {}".format(landmarks))

    landmarkIndices = landmarkMap[landmarks]
    nFallbacks = 0
    out_aligns = []
    for imgObject in imgs:
        rgb = cv2.imread(imgObject)
        if rgb is None:
            print("  + Unable to load.")
            outRgb = None
        else:
            outRgb = align_pred.align(imgSize, rgb,
                                 landmarkIndices=landmarkIndices,
                                 skipMulti=not multi)
            if outRgb is None:
                print("  + Unable to align.")
        if outRgb is not None:
            out_aligns.append(outRgb)
    return out_aligns

def infer(args, imgs, multiple=False):
    with open(pklPath, 'rb') as f:
        (le, clf) = pickle.load(f)

    batch_imgs = prebatchTestAlign(imgs, 5)
    pool = multiprocessing.Pool(5)
    align_imgs_batch = pool.map(batchTestAlign, batch_imgs)
    align_imgs = []
    for img_batch in align_imgs_batch:
        align_imgs += img_batch

    sreps = []
    for aligned_face in align_imgs:
        rep = net.forward(aligned_face)
        sreps.append(rep)

    inferences = []
    for r in sreps:
        '''print("\n=== {} ===".format(img))'''
        try:
            rep = r.reshape(1, -1)
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            inferences.append(predictions)

            print("Predict {} with {:.2f} confidence.".format(person, confidence))
        except Exception as e:
            print e
            continue
    return (inferences , le.classes_)


# From a session we infer that there is only one person in the shot,
# and that it is the same person in every image.
# This function will average the confidence scores together and use the max as an inference.
def inferSession(args, imgs, multiple=False):
    # Silence inferences while collecting...

    # TODO - load pkl classifier here maybe
    verbose = args.verbose
    args.verbose = False

    (inferences, classes) = infer(args, imgs)
    args.verbose = verbose

    scoreMap = dict.fromkeys(classes, 0)

    for inference in inferences:
        for c, p in zip(classes, inference):
            scoreMap[c] += p

    # Average all values
    scoreMap = {k: v / len(inferences) for k, v in scoreMap.items()}

    print scoreMap

    person, confidence = max(scoreMap.iteritems(), key=lambda x:x[1])
    if args.verbose:
        print "Results of inference:"
        print scoreMap
    print "Classifier inferred {} with confidence: {}".format(person, confidence)
    return (person, confidence, scoreMap)



def modelExists():
    if os.path.isfile(pklPath):
        return True
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', action='store_true',
                        default=True)

    subparsers = parser.add_subparsers(dest='mode', help="Mode")

    # Training arguments
    trainParser = subparsers.add_parser(
            'train', help='Train the face recognition classifier.')

    trainParser.add_argument('--clean', action='store_true',
                        default=False)

    # Inference arguments
    inferParser = subparsers.add_parser(
        'infer', help='Predict who an image contains from a trained classifier.')

    inferParser.add_argument('imgs', type=str,
                             help="Input image.")
    inferParser.add_argument('--multi', help="Infer multiple faces in image",
                             action="store_true")

    args = parser.parse_args()

    if args.mode == 'train':
        createModel(args, False, args.clean)
    if args.mode == 'infer':
        imgs = iterImgDirectory(args.imgs)
        inferSession(args, imgs, args.multi)
