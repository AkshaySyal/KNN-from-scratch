import numpy as np
import idx2numpy
import matplotlib.pyplot as plt



train_images = idx2numpy.convert_from_file("dataset/train-images.idx3-ubyte")
train_labels = idx2numpy.convert_from_file("dataset/train-labels.idx1-ubyte")
test_images = idx2numpy.convert_from_file("dataset/t10k-images.idx3-ubyte")
test_labels = idx2numpy.convert_from_file("dataset/t10k-labels.idx1-ubyte")


print(train_images.shape)
print(test_images.shape)
