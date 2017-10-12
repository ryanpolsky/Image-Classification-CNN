"""
Multiview (MVCNN) ModelNet-40 dataset for Keras
"""

import os, os.path
import pprint
import glob
import random
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import numpy

DEFAULT_SRCDIR = 'modelnet40'
DEFAULT_TARGET_SIZE = (224, 224)  # Input size for ResNet-50
nclasses = 40
nviews = 12


def subdirs(dirname):
    return [x for x in os.listdir(dirname) if os.path.isdir(os.path.join(dirname, x))]


def read_image(filename, target_size, preprocess=None):
    #    print('read_image:', filename)
    x = image.load_img(filename, target_size=target_size)
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis=0)
    if preprocess is not None:
        x = preprocess(x)
    # print('read_image, shape:', x.shape)
    return x


def modelnet40_filenames(subset, src_dir=DEFAULT_SRCDIR):
    """
    List of models for ModelNet-40.

    Each model is a pair (class_index, filename_L).
    Here filename_L is a 12 length list of image filenames of views of the model.
    """
    src_dir = os.path.join(src_dir, 'classes')
    classes = sorted(subdirs(src_dir))
    ans = []
    for (icls, cls) in enumerate(classes):
        subset_dir = os.path.join(src_dir, cls, subset)
        model_dirs = subdirs(subset_dir)
        for model_dir in model_dirs:
            filenames = glob.glob(os.path.join(src_dir, cls, subset, model_dir, '*.png'))
            ans.append((icls, filenames))
    return ans


def modelnet40_generator(subset, src_dir=DEFAULT_SRCDIR, single=True, target_size=DEFAULT_TARGET_SIZE, repeats=None,
                         shuffle=True, verbose=0, frac=1.0, class_array=True, preprocess=preprocess_input):
    viewL = modelnet40_filenames(subset, src_dir)

    def generator_func():
        repeat = 0
        while repeats is None or repeat < repeats:
            if shuffle:
                random.shuffle(viewL)
            for (i, view) in enumerate(viewL[:int(len(viewL) * frac)]):
                if verbose == 1 and i % 100 == 0:
                    print('Loading %s data: %.1f%%' % (subset, i * 100.0 / (len(viewL) * frac)))
                (cls, view) = view
                if verbose == 2:
                    print('Loading data point %d, cls = %d' % (i, cls))
                if class_array:
                    cls_array = numpy.zeros((1, nclasses), 'float32')
                    cls_array[0, cls] = 1.0
                    cls = cls_array
                if single:
                    filename = random.choice(view)
                    yield (read_image(filename, target_size, preprocess), cls)
                else:
                    yield ([read_image(view_elem, target_size, preprocess) for view_elem in view], cls)
            repeat += 1

    return (generator_func(), len(viewL))


if __name__ == '__main__':
    (g, dataset_size) = modelnet40_generator('test')
    print('Loading first element from dataset')
    (x1, y1) = g.__next__()  # Python 3 syntax. Use .next() instead for Python 2.
    print(x1, y1)
    print('Loading second element from dataset')
    (x2, y2) = g.__next__()  # Python 3 syntax. Use .next() instead for Python 2.
    print(x2, y2)
    print('Done loading')