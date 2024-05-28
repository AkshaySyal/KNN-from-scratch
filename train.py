import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from KNN import KNN
from importlib import reload


reload(KNN)
train_images = idx2numpy.convert_from_file("dataset/train-images.idx3-ubyte")
train_images_copy = np.copy(train_images)
train_labels = idx2numpy.convert_from_file("dataset/train-labels.idx1-ubyte")

test_images = idx2numpy.convert_from_file("dataset/t10k-images.idx3-ubyte")
test_images_copy = np.copy(test_images)
test_labels = idx2numpy.convert_from_file("dataset/t10k-labels.idx1-ubyte")

clf = KNN(k=5)

clf.fit(train_images_copy,train_labels)

train_images_copy[0]

clf.predict_single(test_images_copy[0])
