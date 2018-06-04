import sys, os
import struct
import numpy as np
# sys.path.append(os.pardir)

def _load_img(file_name):
    with open(file_name, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows, cols)
    return img.reshape(-1, 1, 28, 28)

def _load_lbl(file_name):
    with open(file_name, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)
    return lbl

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def load_mnist(path = ".", normalize=False, flatten=False, one_hot_label=True):
    fname_train_img = os.path.join(path, 'train-images-idx3-ubyte')
    fname_train_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    fname_test_img = os.path.join(path, 't10k-images-idx3-ubyte')
    fname_test_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    dataset = {}
    dataset['train_img'] = _load_img(fname_train_img)
    dataset['train_lbl'] = _load_lbl(fname_train_lbl)
    dataset['test_img'] = _load_img(fname_test_img)
    dataset['test_lbl'] = _load_lbl(fname_test_lbl)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_lbl'] = _change_one_hot_label(dataset['train_lbl'])
        dataset['test_lbl'] = _change_one_hot_label(dataset['test_lbl'])

    if flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 28*28)

    return (dataset['train_img'], dataset['train_lbl']), (dataset['test_img'], dataset['test_lbl'])