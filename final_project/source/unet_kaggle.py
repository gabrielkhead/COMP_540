import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from skimage.transform import rotate, resize
from skimage import data
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1, help="# of GPUs to use for training")
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)

    # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

    up6 = merge(
        [Convolution2D(256, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv5)), conv4],
        mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge(
        [Convolution2D(128, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv3],
        mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge(
        [Convolution2D(64, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv7)), conv2],
        mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge(
        [Convolution2D(32, 2, 2, activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv8)), conv1],
        mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model

# do i need to customize the output layer?


#Parameters and Data go here
img_rows = 160

img_cols = 224

smooth = 1.

epochs = 20

batch_size = 50

#load numpy arrays and resie to correct dims

print("[INFO] training with {} GPUs...".format(G))
# store a copy of the model on *every* GPU and then combine
# the results from the gradient updates on the CPU

with tf.device("/cpu:0"):
#initialize
    model = get_unet()
#make parallel
model = ModelMGPU(model, gpus = G)

model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

train_datagen = ImageDataGenerator(
#rescale = 1./255,
horizontal_flip = True,
fill_mode = "wrap",
zoom_range = 0.2,
width_shift_range = 0.25,
height_shift_range=0.25,
rotation_range=30)

test_datagen = ImageDataGenerator()
#rescale = 1./255,
#horizontal_flip = True,
#fill_mode = "nearest",
#zoom_range = 0.3,
#width_shift_range = 0.3,
#height_shift_range=0.3,
#rotation_range=30)

train_generator = train_datagen.flow(x = X_train, y =y_train, batch_size= batch_size * G)

validation_generator = test_datagen.flow( x = X_test, y = y_test)

# Save the model according to the conditions, there are problems saving multi-GPU models
checkpoint = ModelCheckpoint(filepath= args["output"]+'vgg19_model.h5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only =True, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=15, verbose=1, mode='auto')

# Train the model
H = model_final.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples // (batch_size * G),
epochs = epochs,
validation_data = validation_generator,
validation_steps = nb_validation_samples // (batch_size * G),
verbose = 1,
class_weight = class_weight,
callbacks = [checkpoint,early])

