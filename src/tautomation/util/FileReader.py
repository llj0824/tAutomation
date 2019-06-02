import pandas
import os
import pathlib

FACES_DIR = "/resources/dataset/asian_men/"
RATING_FILE = "/resources/dataset/All_labels.txt"
PROJECT_ROOT = os.getcwd()

def getRatingsMap():
  filepath = PROJECT_ROOT+RATING_FILE
  df = pandas.read_csv(filepath, sep=" ", header=None)
  df.columns = ["fname","rating"]

  ratingMap = {}
  for row in df.values:
    ratingMap[row[0]] = row[1]
    
  return ratingMap

def getDatasetDir():
  return pathlib.Path(PROJECT_ROOT + FACES_DIR)
