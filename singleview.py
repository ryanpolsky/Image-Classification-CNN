"""
singleview.py
Created by Ryan Polsky on 10/5/2017
University of Virginia, CS 6501: 3D Reconstruction and Understanding
The purpose of this class is to fine-tune and use a ResNet50 CNN
that has been pretrained on imagenet to classify 2D images.
Used this tutorial for examples of fine-tuning:
https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html
"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from generator import *
from batch_generator import *

train_data_dir = 'view/list/train'
validation_data_dir = 'view/list/test'
batch_size = 16
epochs = 50



# 1) build the ResNet network with no top layers
model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))


# add the model on top of the convolutional base


# build a classifier model to put on top of the convolutional model
top_model = Sequential()
# 2a) add flat top layer
top_model.add(Flatten(input_shape=model.output_shape[1:]))

# 2b) add fully conneted softmax layer
top_model.add(Dense(40, activation='softmax'))




model = Model(input= model.input, output= top_model(model.output))



# 3) set the first p fraction of layers to non-trainable
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

