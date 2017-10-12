"""
multiview.py
Created by Ryan Polsky on 10/7/2017
University of Virginia, CS 6501: 3D Reconstruction and Understanding
The purpose of this class is to fine-tune and use a ResNet50 CNN
that has been pretrained on imagenet to classify 3D objects given
multiple 2D views the object. The model archeticture is based on
this paper: https://arxiv.org/abs/1505.00880, as well as the data set used.
The modelnet40_generator object used was created by the University of Virginia
Department of Computer Science
"""

from keras import applications
from keras import layers
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Input, Concatenate
from generator import *
from batch_generator import *



train_data_dir = 'view/list/train'
validation_data_dir = 'view/list/test'
batch_size = 16
epochs = 50

def flip(image):
    willFlip = randint(0, 1)
    if willFlip == 1:
        return np.fliplr(image)
    else:
        return image

def batch_generator(subset, batch_size, src_dir=DEFAULT_SRCDIR):

    # initialize a modelnet40_generator object
    (data_gen, data_size) = modelnet40_generator(subset, src_dir, single=False)


    def generator_func():

        while True:

            (x, y) = data_gen.__next__()
            # horizontal flip with 50% probability

            for i in range(0, batch_size - 1):
                (x_next, y_next) = data_gen.__next__()
                for i in range(0,12):
                    x_next[i] = hflip(x_next[i])
                    x[i] = np.concatenate((x[i], x_next[i]), axis=0)

                y = np.concatenate((y, y_next), axis=0)


            yield (x, y)

    return (generator_func(), data_size)



STOP_LAYER = 101

# Form CNN1 and truncate after 101 layers
cnn1 = applications.ResNet50(include_top=False, input_shape=(224,224,3))
cnn1 = Model(cnn1.input, cnn1.layers[STOP_LAYER].output)

# Create 12 Input instances
inputs = []
processed_inputs = []
for i in range(0,12):
    temp = Input(shape=(224,224,3), name='input_' + str(i))
    inputs.append(temp)
    processed_inputs.append(cnn1(temp))

# Get maximum tensor
max = layers.maximum(processed_inputs)

# Create 2 pairs of a convultional layer and a batch normalization layer
x = Conv2D(filters=16,kernel_size=[5,5], input_shape=(224,224,3))(max)
x = BatchNormalization()(x)

x = Conv2D(filters=16,kernel_size=[5,5], input_shape=(224,224,3))(x)
x = BatchNormalization()(x)

x = (Flatten(input_shape=(224,224,3)))(x)
x = Dense(40, activation='softmax')(x)

model = Model(input= inputs, output= x)




# Set the first p fraction of layers to non-trainable
p = 0.7
non_trainable_layers = len(model.layers) * p
for layer in model.layers[: int(non_trainable_layers)]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
sgd = optimizers.SGD(lr=.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


batch_train = batch_generator("train", batch_size, 'view')
batch_test = batch_generator("test", batch_size, 'view')

model.fit_generator(
    batch_train[0],
    steps_per_epoch = batch_train[1]/batch_size,
    epochs= epochs,
    validation_data= batch_test[0],
    validation_steps= batch_test[1]/batch_size,
    verbose=2)
