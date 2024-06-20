# -*- coding: utf-8 -*-
"""
functions to parse the mnist dataset.
Convert the data into png format and save to the disk in the __main__ part
"""

from math import sqrt
import os
import struct
from matplotlib import axis
import numpy as np
import cv2


def parse_mnist_images(filename):
    with open(filename, 'rb') as fid:
        file_content = fid.read()
        item_number = struct.unpack('>i', file_content[4:8])[0]
        rows = struct.unpack('>i', file_content[8:12])[0]
        cols = struct.unpack('>i', file_content[12:16])[0]
        # 'item_number * rows * cols' is the number of bytes
        images = struct.unpack(
            '>%dB' % (item_number * rows * cols), file_content[16:])
        images = np.uint8(np.array(images))
        # np.reshape: the dimension assigned by -1 will be computed according
        # to the first input (images) and other dimensions (rows, cols)
        images = np.reshape(images, [-1, rows, cols])
    return images


def parse_mnist_labels(filename):
    with open(filename, 'rb') as fid:
        file_content = fid.read()
        item_number = struct.unpack('>i', file_content[4:8])[0]
        labels = struct.unpack('>%dB' % item_number, file_content[8:])
        labels = np.array(labels)
    return labels


def make_one_hot_labels(labels):
    classes = np.unique(labels)
    assert len(classes) == classes.argmax() - classes.argmin() + 1
    labels_one_hot = (labels[:, None] == np.arange(10)).astype(np.int32)
    return labels_one_hot


def shuffle_data(images, labels_one_hot):
    """
    Description:
        Random shuffle the images and labels_one_hot, note that image number
        should be placed at the first axis
    Inputs:
        images: images tensor, with number as the first axis
        labels_one_hot: one hot labels, with number as the first axis
    Outputs:
        Random shuffled images and labels_one_hot
    """
    images_shuffle = np.zeros_like(images)
    labels_shuffle = np.zeros_like(labels_one_hot)
    idx_shuffle = np.arange(images.shape[0])
    np.random.shuffle(idx_shuffle)
    images_shuffle = images[idx_shuffle]
    labels_shuffle = labels_one_hot[idx_shuffle]
    return images_shuffle, labels_shuffle


class BSActivation:
    @staticmethod
    def sigmoid(input: np.array):
        input_sigmoid = 1 / (1 + np.exp(-6*input))
        # return np.random.binomial(1, input_sigmoid)
        return sampling(input_sigmoid)
    
    @staticmethod
    def dsigmoid(input: np.array):
        input_sigmoid = 1 / (1 + np.exp(-6*input))
        p = 6*input_sigmoid * (1 - input_sigmoid)
        return sampling(p)


def softmax(data: np.array):
    return np.exp(data) / np.sum(np.exp(data), axis=1, keepdims=True)


def sampling(data: np.array):
    return data > np.random.rand(*data.shape)


def test_eval(images_test, labels, w1, w2, w3, b1, b2, b3):
    y1 = np.matmul(images_test, w1) + b1
    z1 = BSActivation.sigmoid(y1)
    y2 = np.matmul(z1, w2) + b2
    z2 = BSActivation.sigmoid(y2)
    z3 = np.matmul(z2, w3) + b3
    z3 = softmax(z3)
    # z3 = sampling(z3)
    predict_ok = z3.argmax(axis=1) == labels
    return np.mean(predict_ok)


images_train = parse_mnist_images("./data/MNIST/raw/train-images-idx3-ubyte")
# os.makedirs("./data/image", exist_ok=True)
# for index, image in enumerate(images_train):
#     cv2.imwrite(f"./data/image/{index}.png", image)
images_test = parse_mnist_images("./data/MNIST/raw/t10k-images-idx3-ubyte")
images_train = np.reshape(images_train, [images_train.shape[0], -1])
images_test = np.reshape(images_test, [images_test.shape[0], -1])
images_train = np.float32(images_train) / 255.0
images_test = np.float32(images_test) / 255.0

labels_train = parse_mnist_labels('./data/MNIST/raw/train-labels-idx1-ubyte')
labels_test = parse_mnist_labels('./data/MNIST/raw/t10k-labels-idx1-ubyte')
labels_train_one_hot = make_one_hot_labels(labels_train)

BATCH_SIZE = 64
EPOCH = 100
NUM_NODES = 256
LEARNING_RATE = 0.1
num_train = images_train.shape[0]
single_image_size = images_train.shape[1]
w1 = sqrt(2/512) * np.random.randn(784,512)
w2 = sqrt(2/256) * np.random.randn(512,256)
w3 = sqrt(2/10) * np.random.randn(256, 10)
# w1 = np.random.randn(784,512)
# w2 = np.random.randn(512,256)
# w3 = np.random.randn(256, 10)
b1 = np.zeros([1, 512])
b2 = np.zeros([1, 256])
b3 = np.zeros([1, 10])


for ep in range(EPOCH):
    print('epoch', ep + 1)
    images_shuffle, labels_shuffle = shuffle_data(images_train, labels_train_one_hot)
    loss = []
    for i in range(0, num_train, BATCH_SIZE):
        # get a batch of data
        images_batch = images_shuffle[i:i + BATCH_SIZE, :]
        # print(images_batch)
        labels_batch = labels_shuffle[i:i + BATCH_SIZE, :]
        images_batch = sampling(images_batch)
        y1 = np.matmul(images_batch, w1) + b1
        z1 = BSActivation.sigmoid(y1)
        dzdy1 = BSActivation.dsigmoid(y1)
        y2 = np.matmul(z1, w2) + b2
        z2 = BSActivation.sigmoid(y2)
        dzdy2 = BSActivation.dsigmoid(y2)
        z3 = np.matmul(z2, w3) + b3
        softmax_z3 = softmax(z3)

        # Loss
        loss.append(-np.sum(labels_batch * np.log10(softmax_z3)) / BATCH_SIZE)

        # backward propagation
        z3 = sampling(z3)
        dldy3 = np.sign(z3 - labels_batch)
        dldx3 = np.matmul(dldy3, w3.T)
        dldw3 = np.matmul(z2.T, dldy3)
        dldb3 = np.sum(dldy3)
        w3 = w3 - LEARNING_RATE * dldw3 / BATCH_SIZE
        b3 = b3 - LEARNING_RATE * dldb3 / BATCH_SIZE

        # print(np.size(z2))
        dldy2 = dldx3 * dzdy2
        dldx2 = np.matmul(dldy2, w2.T)
        dldw2 = np.matmul(z1.T, dldy2)
        dldb2 = np.sum(dldy2)
        w2 = w2 - LEARNING_RATE * dldw2 / BATCH_SIZE
        b2 = b2 - LEARNING_RATE * dldb2 / BATCH_SIZE

        dldy1 = dldx2 * dzdy1
        dldw1 = np.matmul(images_batch.T, dldy1)
        dldb1 = np.sum(dldy1)
        w1 = w1 - LEARNING_RATE * dldw1 / BATCH_SIZE
        b1 = b1 - LEARNING_RATE * dldb1 / BATCH_SIZE

    print(f"Loss : {np.mean(loss)}")
    acc = test_eval(sampling(images_test), labels_test, w1, w2, w3, b1, b2, b3)
    print('accuracy: ', acc)
