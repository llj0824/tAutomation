from __future__ import absolute_import, division, print_function
import random
import IPython.display as display
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from util import FileReader

IMAGE_SIZE = [300, 300]


def main():
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  tf.enable_eager_execution()
  mapImageFileToRating = FileReader.getRatingsMap()
  all_image_paths = getImageList(FileReader.getDatasetDir())


def getImageList(dataSetDir):
  return list(dataSetDir.iterdir())

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=0)
  image = tf.image.resize(image, IMAGE_SIZE)
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.read_file(path)
  return preprocess_image(image)


def getRandomImagePath(all_image_paths):
  return all_image_paths[random.randint(1,len(all_image_paths))]

def showImage(img_path):
  plt.imshow(load_and_preprocess_image(img_path))
  plt.grid(False)
  plt.title(label)
  plt.show()



if __name__ == "__main__":
  main()