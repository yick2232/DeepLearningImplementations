# encoding: utf8

import os
import cv2
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import random


def load_data(data_dir:str, image_size:int=256):
  train_dir = os.path.join(data_dir, "train_data")
  test_dir = os.path.join(data_dir, "test_data")

  bad_labels = set()

  def read_data_from_dir(dir:str, test:bool=False):
    print("data_dir: {}".format(dir))
    images, labels = [], []
    lst = []
    for label_name in os.listdir(dir):
      try:
        label_idx = int(label_name)
      except:
        print("error, label_name: {}".format(label_name))
        continue
      lst.append((label_idx, label_name))
    lst.sort(key=lambda items: items[0])

    for label_idx, label_name in lst:
      fnames = os.listdir(os.path.join(dir, label_name))
      print("label_name: {}, number: {}".format(label_name, len(fnames)))
      if not test and len(fnames) < 20:
        bad_labels.add(label_idx)
      if label_idx in bad_labels:
        label_idx = 0
      # sample
      if not test:
        if 0 < len(fnames) < 20:
          tmp = [random.choice(fnames) for _ in range(20 - len(fnames))]
          fnames += tmp
        random.shuffle(fnames)
        fnames = fnames[:300]
        print("number by sample: {}".format(len(fnames)))
      for fname in fnames:
        fpath = os.path.join(dir, label_name, fname)
        if fpath.endswith("gif"):
          continue
        try:
          image = cv2.imread(fpath)
          image = cv2.resize(image, (image_size, image_size*2), cv2.INTER_LINEAR)
          images.append(image)
          labels.append(label_idx)
        except:
          print("fpath: {}".format(fpath))
    return images, labels

  images_train, labels_train = read_data_from_dir(train_dir)
  images_test, labels_test = read_data_from_dir(test_dir, True)

  x_train = np.array(images_train)
  x_test = np.array(images_test)
  y_train = np.array(labels_train)
  y_test = np.array(labels_test)
  print("x_train.shape: {}, y_train.shape: {}".format(x_train.shape, y_train.shape))
  print("x_test.shape: {}, y_test.shape: {}".format(x_test.shape, y_test.shape))
  return (x_train, y_train), (x_test, y_test)

def session_creater():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  session = tf.Session(config=config)
  set_session(session)
  return session