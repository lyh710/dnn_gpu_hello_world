# ***Reference***
# 
# https://towardsdatascience.com/prototyping-an-anomaly-detection-system-for-videos-step-by-step-using-lstm-convolutional-4e06b7dcdd29
# https://github.com/hashemsellat/Video-Anomaly-Detection/blob/master/lstmautoencoder.ipynb

import warnings
warnings.filterwarnings('ignore')

# Enforce GPU
import tensorflow as tf
print(tf.test.is_gpu_available())

import numpy as np
import os, pickle, shutil
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, ConvLSTM2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
# from keras_layer_normalization import LayerNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, Callback

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(32)

from os import listdir
from os.path import isfile, join, isdir

from utils import get_clips_by_stride, get_train_set
from utils import TRAIN_PATH, TEST_PATH

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times_per_sample = []
        self.times_per_epoch  = []
        self.train_time_start = time.time()

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times_per_sample.append((time.time() - self.epoch_time_start)/60000)
        self.times_per_epoch.append(time.time() - self.epoch_time_start)
        
    def on_train_end(self, logs={}):
        self.train_time = time.time() - self.train_time_start

# Set up
version = 'ucsd.gpu'

# Prepar folder for output
out_dir = join('out', version)
if os.path.isdir(out_dir):
    shutil.rmtree(out_dir)
    os.mkdir(out_dir)
#     None
else:
    os.mkdir(out_dir)

train = get_train_set()

seq = Sequential()
seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same", activation='selu'), batch_input_shape=(None, 10, 256, 256, 1)))
seq.add(BatchNormalization())
seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same", activation='selu')))
seq.add(BatchNormalization())
# # # # #
seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
seq.add(BatchNormalization())
# # # # #
seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same", activation='selu')))
seq.add(BatchNormalization())
seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same", activation='selu')))
seq.add(BatchNormalization())
seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
print(seq.summary())
seq.compile(loss='mse', optimizer=optimizers.Adam(lr=1e-4, decay=1e-5, epsilon=1e-6))

# Prepar folder for logs
log_dir = join('logs', version)
tensorboard = TensorBoard(log_dir=log_dir)
history = LossHistory()

time_callback = TimeHistory()

seq.fit(train, train, batch_size=2, epochs=3, shuffle=False, callbacks=[tensorboard, history, time_callback])

times_per_sample, times_per_epoch, train_time = time_callback.times_per_sample, time_callback.times_per_epoch, time_callback.train_time
pickle.dump((times_per_sample, times_per_epoch, train_time), open('time_gpu.pkl', 'wb'))

pkl_fn = open(join(log_dir, 'log.pkl'), 'wb')
pickle.dump(history.losses, pkl_fn)
pkl_fn.close()
print(history.losses)

# Save the model
model_name = join(out_dir, version)
seq.save(model_name)