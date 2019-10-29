import pandas
import os
import pathlib
import numpy as np
from tensor import tensorUtil


FACES_DIR = "/resources/dataset/asian_men/"
RATING_FILE = "/resources/dataset/All_labels.txt"
PROJECT_ROOT = os.getcwd()


def getImageRatings(dirName):
  mapImageToRating = getRatingsMap()
  image_paths = getPathsContainedInDir(dirName);
  all_image_ratings = generateImageRatings(image_paths,mapImageToRating)
  return all_image_ratings 

def getDatasetImages(dirPath):
  images = getPathsContainedInDir(dirPath)
  return getImageList(images)

def getRatingsMap():
  filepath = PROJECT_ROOT+RATING_FILE
  df = pandas.read_csv(filepath, sep=" ", header=None)
  df.columns = ["fname","rating"]

  ratingMap = {}
  for row in df.values:
    ratingMap[row[0]] = row[1]
  return ratingMap

def generateImageRatings(allImagePaths, mapImageToRating):
  imageLabels = []
  for i in range(len(allImagePaths)):
    imageName = getImageFilename(allImagePaths[i])
    #imageLabels.append(roundToNearestTenths(float(mapImageToRating[imageName])))
    imageLabels.append(int(mapImageToRating[imageName]))
  return np.asarray(imageLabels)

def getImageFilename(image_path):
  lastIndex = -1;
  return image_path.split('/')[-1]

def roundToNearestTenths(floatVal):
   return round(floatVal,1)

def getPathsContainedInDir(dirName):
  datasetDir = pathlib.Path(PROJECT_ROOT + dirName)
  all_paths = list(datasetDir.iterdir())
  for i in range(len(all_paths)):
    all_paths[i] = str(all_paths[i])
  return all_paths

def getImageList(all_paths):
  all_images = []
  for path in all_paths:
    try:
      all_images.append(tensorUtil.preprocess_image(str(path)))
    except:
      print("Could not process image: ", getImageFilename(path))
  return np.asarray(all_images)
