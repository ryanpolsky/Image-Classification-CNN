"""
batch_generator.py
Created by Ryan Polsky on 10/5/2017
The purpose of this class is to output data in batches of size n, using a
modelnet40_generator that produces data of batch size 1.
The modelnet40_generator object used was created by the University of Virginia
Department of Computer Science
"""

import os, os.path
import pprint
import glob
import random
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import numpy
from generator import *
from random import *


def hflip(image):
    willFlip = randint(0, 1)
    if willFlip == 1:
        return np.fliplr(image)
    else:
        return image

def batch_generator(subset, batch_size, src_dir=DEFAULT_SRCDIR):

    # initialize a modelnet40_generator object
    (data_gen, data_size) = modelnet40_generator(subset, src_dir)


    def generator_func():

        while True:


            (x, y) = data_gen.__next__()
            # horizontal flip with 50% probability
            x = hflip(x)
            for i in range(0, batch_size - 1):
                (x_next, y_next) = data_gen.__next__()
                x_next = hflip(x_next)
                x = np.concatenate((x, x_next), axis=0)
                y = np.concatenate((y, y_next), axis=0)

            yield (x, y)



    return (generator_func(),data_size)

