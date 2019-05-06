from __future__ import absolute_import, division, print_function
import os
import random
import IPython.display as display
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

FACES_DIR = "/resources/dataset/Images/"
PROJECT_ROOT = os.getcwd()
IMAGE_SIZE = [300, 300]


def main():
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  tf.enable_eager_execution()
  all_image_paths = getImageList(getDatasetDir())
  demo(str(getRandomImagePath(all_image_paths)))



def getDatasetDir():
  return pathlib.Path(PROJECT_ROOT + FACES_DIR)


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

def demo(img_path):
  label = "Facial Attractiveness"

  plt.imshow(load_and_preprocess_image(img_path))
  plt.grid(False)
  plt.title(label)
  plt.show()
  print()


if __name__ == "__main__":
  main()