# import the needed libraries

import numpy as np
from matplotlib import pyplot as plt
import random
import soundfile as sf
from python_speech_features import mfcc
# Library for machine learning
import keras
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Sequential,Model
from keras.layers import Input,Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Embedding,LSTM,GaussianNoise, GRU,CuDNNLSTM,CuDNNGRU
from keras.layers import Conv1D,SimpleRNN,MaxPooling1D,GlobalAveragePooling1D
import librosa
from librosa.feature import mfcc 
import keras.backend as K
import tensorflow as tf
######################################
# this library is to load the model and use it for testing
from keras.models import load_model
######################################

# load the data 
# then compute the mfcc as a feature vector of dimension (49,39)
# 49 corresponds to the time steps and 39 the features in each time step
# librosa.effects.trim though not used is used for trimming the sound file for useful information

data  = []
label = []
for i in range(0,79):
  for j in range(0,24):

    back, sr = sf.read("back_"+str(i) +"_"+str(j)+".wav")
#     back, index = librosa.effects.trim(back)
    x=mfcc(y = back, sr = sr, n_mfcc=39)      
    i=49-x.shape[1]
    tp=np.zeros((39,i))
    x=np.append(x,tp,axis=1);
    data.append(x.T)
    label.append(0)

    forward, sr = sf.read("forward_"+str(i) +"_"+str(j)+".wav")
#     forward, index = librosa.effects.trim(forward)
    x=mfcc(y = forward, sr = sr, n_mfcc=39)      
    i=49-x.shape[1]
    tp=np.zeros((39,i))
    x=np.append(x,tp,axis=1);
    data.append(x.T)
    label.append(1)
    
    left, sr = sf.read("left_"+str(i) +"_"+str(j)+".wav")
#     left, index = librosa.effects.trim(left)
    x=mfcc(y = left, sr = sr, n_mfcc=39)      
    i=49-x.shape[1]
    tp=np.zeros((39,i))
    x=np.append(x,tp,axis=1);
    data.append(x.T)
    label.append(2)
    
    right, sr = sf.read("right_"+str(i) +"_"+str(j)+".wav")
#     right, index = librosa.effects.trim(right)
    x=mfcc(y = right, sr = sr, n_mfcc=39)      
    i=49-x.shape[1]
    tp=np.zeros((39,i))
    x=np.append(x,tp,axis=1);
    data.append(x.T)
    label.append(3) 

  
    stop, sr = sf.read("stop_"+str(i) +"_"+str(j)+".wav")
#     stop, index = librosa.effects.trim(stop)
    x=mfcc(y = stop, sr = sr, n_mfcc=39)      
    i=49-x.shape[1]
    tp=np.zeros((39,i))
    x=np.append(x,tp,axis=1);
    data.append(x.T)
    label.append(4)
#print(mfcc(y = stop, sr = sr, n_mfcc=39).shape[1])
#print(mfcc(y = right, sr = sr, n_mfcc=39).shape)
print(stop.shape[0])
print(right.shape)

label = np.array(label)
label = to_categorical(label)

data = np.array(data)

# print the shapes of the input and label arrays
print(data.shape)
print(label.shape)
########################################################################################

# Def Loss function

# this is the implementation of categorical loss function from scratch
# the input to the function is predicted probability and a one hot vector
# it computes the loss by summing over the expression -ylog(p) over the tuple
# this is summed over a batch and the final loss is given
def categorical_cross_entropy(ytrue, ypred, axis=-1):
    return -1.0*tf.reduce_mean(tf.reduce_sum(ytrue * tf.math.log(ypred), axis))

#######################################################################################
# LSTM model

# the model contains a lstm layer to exploit the sequential nature of sound files 
# Followed by maxpooling for eliminating the unnecesary information
# then it is flattened and sent to the dense layer with 5 nodes with softmax as activation 
# layer for probability prediction

model = Sequential()

model.add(LSTM(units = 128, return_sequences = True, input_shape = (data.shape[1],39)))
#print("Them Data")
#print(data.shape[0])

model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(5, activation='softmax'))

model.summary() 


# compile the keras model
model.compile(loss=categorical_cross_entropy, optimizer='Rmsprop', metrics=['accuracy'])
print(data.shape)

# fit the keras model on the dataset
model.fit(data, label, validation_split=0.1, epochs=5, batch_size=5)


# evaluate the keras model
_, accuracy = model.evaluate(data, label)
print('Accuracy: %.2f' % (accuracy*100))

model.save('model.h5')  # creates a HDF5 file 'model.h5'
