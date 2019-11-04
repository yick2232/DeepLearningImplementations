# encoding: utf8

import os
import cv2
import numpy as np
from keras.utils import to_categorical


def load_data(data_dir:str, image_size:int=128):
  train_dir = os.path.join(data_dir, "train_data")
  test_dir = os.path.join(data_dir, "test_data")

  def read_data_from_dir(dir:str):
    print("data_dir: {}".format(dir))
    images, labels = [], []
    for label_name in os.listdir(dir):
      try:
        label_idx = int(label_name)
      except:
        print("error, label_name: {}".format(label_name))
        continue
      fnames = os.listdir(os.path.join(dir, label_name))
      print("label_name: {}, number: {}".format(label_name, len(fnames)))
      for fname in fnames:
        fpath = os.path.join(dir, label_name, fname)
        if fpath.endswith("gif"):
          continue
        try:
          image = cv2.imread(fpath)
          image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
          images.append(image)
          labels.append(label_idx)
        except:
          print("fpath: {}".format(fpath))
    return images, labels

  images_train, labels_train = read_data_from_dir(train_dir)
  images_test, labels_test = read_data_from_dir(test_dir)

  x_train = np.array(images_train)
  x_test = np.array(images_test)
  y_train = np.array(labels_train)
  y_test = np.array(labels_test)
  print("x_train.shape: {}, y_train.shape: {}".format(x_train.shape, y_train.shape))
  print("x_test.shape: {}, y_test.shape: {}".format(x_test.shape, y_test.shape))
  return (x_train, y_train), (x_test, y_test)