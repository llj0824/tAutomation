from __future__ import absolute_import, division, print_function
import random
import IPython.display as display
import tensorflow as tf
import matplotlib.pyplot as plt
from tensor import tensorUtil
from util import FileReader


# evaluate own photo using ASIAN_MEN dataset
# RESULT: model isn't returning right. Everyone's getting same results
FACES_DIR = "/resources/dataset/small_images/"
EVALUATE_DIR = "/resources/dataset/evaluate_test/"


def main():
    tf.data.experimental.AUTOTUNE
    tf.enable_eager_execution()

    all_images = FileReader.getDatasetImages(FACES_DIR)
    all_image_ratings = FileReader.getImageRatings(FACES_DIR)
    showImages(all_images, all_image_ratings)

    tfModel = tensorUtil.initTensorFlow()
    tensorUtil.trainModel(tfModel, all_images, all_image_ratings)

    evaluate_paths = FileReader.getPathsContainedInDir(EVALUATE_DIR)
    evaluate_images = FileReader.getDatasetImages(EVALUATE_DIR)
    breakpoint()
    results = evaluateImages(tfModel, evaluate_images, evaluate_paths)


def evaluateImages(tfModel, images, paths):
    results = tensorUtil.modelPredict(tfModel, images)
    for i in range(len(results)):
        result = results[i]
        name = FileReader.getImageFilename(paths[i])
        print(name, ":", result)
    return results


def getRandomImageIndex(all_images):
    return random.randint(0, len(all_images) - 1)


def showImages(all_images, all_image_ratings):
    numImages = 25
    numRows = 5
    numCols = numImages / numRows
    for i in range(25):
        rndIndex = getRandomImageIndex(all_images)
        img = all_images[rndIndex]
        plt.subplot(numRows, numCols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img)
        rating = all_image_ratings[rndIndex]
        plt.xlabel("{}".format(round(float(rating), 1)))
    plt.show()


if __name__ == "__main__":
    main()
