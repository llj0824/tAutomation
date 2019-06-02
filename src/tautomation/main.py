from __future__ import absolute_import, division, print_function
import random
import IPython.display as display
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensor import tensorUtil
from util import FileReader

def main():
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  tf.enable_eager_execution()

  mapImageToRating = FileReader.getRatingsMap()
  all_image_paths = getImagePaths(FileReader.getDatasetDir())
  all_images = getImageList(all_image_paths)
  all_image_ratings = getImageRatings(all_image_paths, mapImageToRating)
  tfModel = tensorUtil.initTensorFlow()
  tensorUtil.trainModel(tfModel, all_images, all_image_ratings) 
  # showImages(all_image_paths, mapImageToRating) 

def getImageList(all_paths):
  all_images = []
  for path in all_paths:
    try:
      all_images.append(tensorUtil.preprocess_image(str(path)))
    except:
      print("Could not process image: ", getImageFilename(path))
  return np.asarray(all_images)

def getImageRatings(allImagePaths, mapImageToRating):
  imageLabels = []
  for i in range(len(allImagePaths)):
    imageName = getImageFilename(allImagePaths[i])
    imageLabels.append(roundToNearestTenths(float(mapImageToRating[imageName])))
  return np.asarray(imageLabels)

def roundToNearestTenths(floatVal):
   return round(floatVal,1)


def getImagePaths(dataSetDir):
  all_paths = list(dataSetDir.iterdir())
  for i in range(len(all_paths)):
    all_paths[i] = str(all_paths[i])
  return all_paths

def getImageFilename(image_path):
  lastIndex = -1;
  return image_path.split('/')[-1]
  
def getRandomImagePath(all_image_paths):
  return all_image_paths[random.randint(0,len(all_image_paths)-1)]

def showImages(all_image_paths, mapImageToRating):
  numImages = 25;
  numRows = 5;
  numCols = numImages/numRows;
  for i in range(25):
    img_path = getRandomImagePath(all_image_paths)
    plt.subplot(numRows, numCols,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(tensorUtil.preprocess_image(img_path))
    fileName = getImageFilename(img_path)
    rating = mapImageToRating[fileName]
    plt.xlabel("{}".format(round(float(rating),1)));
  plt.show()


if __name__ == "__main__":
  main()
