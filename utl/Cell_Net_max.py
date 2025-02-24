import sys
import time
from random import shuffle
import numpy as np
import argparse
import tensorflow as tf

#from keras.utils import multi_gpu_model
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
from .metrics import bag_accuracy, bag_loss
from .custom_layers_max import Mil_Attention, Last_Sigmoid
import keras as K

def cell_net(input_dim, args, useMulGpu=False):

    lr = args.init_lr
    weight_decay = args.init_lr
    momentum = args.momentum

    data_input = Input(shape=input_dim, dtype='float32', name='input')
    conv1 = Conv2D(36, kernel_size=(4,4), kernel_regularizer=l2(weight_decay), activation='relu')(data_input)
    conv1 = MaxPooling2D((2,2))(conv1)

    conv2 = Conv2D(48, kernel_size=(3,3),  kernel_regularizer=l2(weight_decay), activation='relu')(conv1)
    conv2 = MaxPooling2D((2,2))(conv2)
    x = Flatten()(conv2)

    fc1 = Dense(512, activation='relu',kernel_regularizer=l2(weight_decay), name='fc1')(x)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(512, activation='relu', kernel_regularizer=l2(weight_decay), name='fc2')(fc1)      #(N_patches,512)
    fc2 = Dropout(0.5)(fc2)
    
    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=args.useGated)(fc2)    #(N_patches,1)
    x_mul = multiply([alpha, fc2]) #(N_patches,512)

    #out1 = Last_Sigmoid(output_dim=5, name='FC1_sigmoid')(fc2)
    #out= tf.keras.backend.mean(out1,axis=0,keepdims=True)
    #out= tf.keras.backend.max(out1,axis=1,keepdims=True)
    out1 = 4*Last_Sigmoid(output_dim=1, name='FC1_sigmoid')(fc2)
    out= tf.keras.backend.mean(out1,axis=0,keepdims=True)
    
    model = Model(inputs=[data_input], outputs=[out])

    # model.summary()

    if useMulGpu == True:
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
        parallel_model = model


    return parallel_model



