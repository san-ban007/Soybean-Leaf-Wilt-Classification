#!/usr/bin/env python
'''
This is a re-implementation of the following paper:
"Attention-based Deep Multiple Instance Learning"
I got very similar results but some data augmentation techniques not used here
https://128.84.21.199/pdf/1802.04712.pdf
*---- Jiawen Yao--------------*
'''


import numpy as np
import cv2
import time
from utl import Cell_Net_max
from random import shuffle
import argparse
from keras.models import Model
from utl.dataset import load_dataset
from utl.data_aug_op import random_flip_img, random_rotate_img
import glob
import os
from os.path import join
#import scipy.misc as sci
#import imageio
import tensorflow as tf
import sys
from PIL import Image
import os

from keras import backend as K
#from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os


def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay', 
                        help='weight decay',
                        default=0.0005, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=100, type=int)
    parser.add_argument('--useGated', dest='useGated',
                        help='use Gated Attention',
                        default=False, type=int)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def create_heatmaps(bag_patches, att_wts, label, main_img, num_img,val):
    """
    This creates a heatmap and saves it in ./Results/ folder.
    bag_patches: str: the path to the images that are int he bag as patches.
    att_wts: list: The calculated weights from the model.
    """

    print('Main image being analysed: ', main_img)
    all_patches = [join(main_img, p_name) for p_name in bag_patches]
    #print('Num of patches in this bag: ', len(all_patches),
     #' is same as attn weight: ', len(all_patches) == len(att_wts))
    img_name=os.path.split(main_img)[1]
    h_img_strips = []
    patch_num = 0
    for row in range(0, 480, 80):
      
      for col in range(0, 600,120 ):
        if col == 0:
          img = att_wts[patch_num] * cv2.imread(all_patches[patch_num])
        else:
          img = np.hstack((img, att_wts[patch_num] * cv2.imread(all_patches[patch_num])))
        patch_num += 1

      h_img_strips.append(img)

    for h_num, horiz in enumerate(h_img_strips):
      if h_num == 0:
        new_img = horiz
      else:
        new_img = np.vstack((new_img, horiz))

    print('Print the new image shape: ', new_img.shape)

    if not os.path.isdir('/home/sbaner24/Soybean/Final_Tests/Output/ResultsB/Heatmaps/'+str(val)+'/'):
      os.mkdir('/home/sbaner24/Soybean/Final_Tests/Output/ResultsB/Heatmaps/'+str(val)+'/')

    cv2.imwrite(join('/home/sbaner24/Soybean/Final_Tests/Output/ResultsB/Heatmaps/'+str(val)+'/'+str(img_name)+'_attimg'+'.png'), np.uint8(new_img))      
    print('Created Heatmap!')
    print('------')

def create_heatmaps1(bag_patches, att_wts, label, main_img, num_img,val):
    print('Main image being analysed: ', main_img)
    all_patches = [join(main_img, p_name) for p_name in bag_patches]
    #print('Num of patches in this bag: ', len(all_patches),
     #' is same as attn weight: ', len(all_patches) == len(att_wts))
    img_name=os.path.split(main_img)[1]
    new_im=Image.new('RGB',(600,450))
    patch_num = 0
    for row in range(0,450,90):
        for col in range(0,600,120):
            im =  Image.open(all_patches[patch_num])
            np_array=att_wts[patch_num]*np.array(im)
            im_new=Image.fromarray(np_array)
            new_im.paste(im, (col,row))
            patch_num+=1
    new_im.save('/home/sbaner24/Soybean/Final_Tests/Output/ResultsB/Heatmaps/'+str(val)+'/'+str(img_name)+'_attimg'+'.png')
    
def generate_batch(path):
    print('generate_batch')
    bags = []
    #print('Path: ', len(path))

    # Iterates over the chosen image folders for training/testing.
    for bag_cnt, each_path in enumerate(path):
        name_img = []
        img = []
        img_path = glob.glob(each_path + '/*.jpg') 
        img_path.sort()
        num_ins = len(img_path)
        print('Num of patches in bag {} is {}'.format(bag_cnt, num_ins))
        

        label = int(each_path.split('/')[-2]) # 0/1/2/3/4

        if label == 0:
            curr_label = np.zeros(num_ins,dtype=np.float64) # was np.uint8
        elif label==1:
            curr_label = np.ones(num_ins, dtype=np.float64) # was np.uint8
        elif label==2:
            curr_label = 2*np.ones(num_ins, dtype=np.float64)
        elif label==3:
            curr_label = 3*np.ones(num_ins, dtype=np.float64)
        elif label==4:
            curr_label = 4*np.ones(num_ins, dtype=np.float64)
        # Iterates over the patches in the chosen image folder.
        for each_img in img_path:
            
            img_data = cv2.imread(each_img).astype(np.float64)
            #img_data -= 255
            #img_data[:, :, 0] -= 123.68
            #img_data[:, :, 1] -= 116.779
            #img_data[:, :, 2] -= 103.939
            img_data /= 255
            # sci.imshow(img_data)
            img.append(np.expand_dims(img_data,0))
            name_img.append(each_img.split('/')[-1])
        
        stack_img = np.concatenate(img, axis=0)
        bags.append((stack_img, curr_label, name_img))
    #assert len(bags) == len(path), 'Bags'
    return bags



def Get_train_valid_Path(Train_set, train_percentage=0.85):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    print('Get_train_valid_path')
    import random
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)
    
    num_train = int(train_percentage*len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def test_eval(model, test_set):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """
    print('test_eval')
    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    predictions=[]
    for ibatch, batch in enumerate(test_set):
        #print("batch in test_batch=",batch)
        result = model.test_on_batch(x=batch[0], y=batch[1])
        preds= model.predict_on_batch(x=batch[0])
        #print("preds shape=",preds.shape)
        #print("preds[0][0]=",preds[0][0])
        test_loss[ibatch] = result[0]
        test_acc[ibatch] = result[1]
        predic=preds[0][0]
        predictions.append(predic)
    #print("predictions:",predictions)
    return np.mean(test_loss), np.mean(test_acc),predictions

def train_eval(model, train_set, irun):
    """Evaluate on training set. Use Keras fit_generator
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    print('train_eval')
    batch_size = 1
    model_train_set, model_val_set = Get_train_valid_Path(train_set,train_percentage=0.9 )

    from utl.DataGenerator import DataGenerator
    train_gen = DataGenerator(batch_size=1, shuffle=True).generate(model_train_set)
    val_gen = DataGenerator(batch_size=1, shuffle=False).generate(model_val_set)

    model_name = "/home/sbaner24/Soybean/Final_Tests/Output/Saved_model/" + "_Batch_size_" + str(batch_size) + "epoch_" + "best.hd5"

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto', period=1)

    EarlyStop = EarlyStopping(monitor='val_loss', patience=20)

    callbacks = [checkpoint_fixed_name, EarlyStop]

    history = model.fit_generator(generator=train_gen, steps_per_epoch=len(model_train_set)//batch_size, epochs=args.max_epoch, validation_data=val_gen, validation_steps=len(model_val_set)//batch_size, callbacks=callbacks)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['bag_accuracy']
    val_acc = history.history['val_bag_accuracy']

    fig = plt.figure()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = '/home/sbaner24/Soybean/Final_Tests/Output/Results/' + str(irun)  + "_loss_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)


    fig = plt.figure()
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    save_fig_name = '/home/sbaner24/Soybean/Final_Tests/Output/Results/' + str(irun) + '_'  + "_acc_batchsize_" + str(batch_size) + "_epoch"  + ".png"
    fig.savefig(save_fig_name)

    return model_name


def model_training(input_dim, train_bags,test_bags, irun):

    train_bags = train_bags
    test_bags=test_bags
    print('Num of training samples: ', len(train_bags))
    #print("test_bags=",test_bags[0])
    #print("test_bags=",cv2.imread(str(test_bags[0])+'.png'))
    print('Num of test samples: ', len(test_bags))

    # convert bag to batch
    train_set = generate_batch(train_bags)
    test_set=generate_batch(test_bags)
    #print("test_set=",test_set)

    model = Cell_Net_max.cell_net(input_dim, args, useMulGpu=False)

    # train model
    
    num_batch = len(train_set)
    # for epoch in range(args.max_epoch):
    model_name = train_eval(model, train_set,irun)

    #model_name = 
    print("load saved model weights")
    model.load_weights(model_name)

    
  # Generating heatmap
    max_weight_train=[]
    min_weight_train=[]
    main_image=[]
    image_label=[]
    for num_img, whole_img in enumerate(train_bags):
      main_img = train_bags[num_img]
      img_name = train_set[num_img][2]
      img_label = int(train_set[num_img][1][0])
    
      # The firs bag's concatenated patches.
      ak_x = train_set[num_img][0] 
      print('Train data size: ', ak_x.shape)
    
      # This creates a function out of the input and output model layers into tensors.
      ak = K.function([model.layers[0].input], [model.layers[10].output])
      
      # This gets the attention weights from the model.
      ak_output = ak([ak_x])
      ak_output=np.max(ak_output[0],axis=1,keepdims=True)
      ak_output = np.array(ak_output).reshape((ak_x.shape[0]))
      #ak_output_f = np.array(ak_output_f[0][1]).reshape((ak_x.shape[0]))
      # For my dataset, there are 48 images in a bag.
      # rescale the weight as described in the paper
      minimum = ak_output.min()
      maximum = ak_output.max()
      max_weight_train.append(maximum)
      #min_weight_train.append(minimum)
      main_image.append(os.path.basename(main_img))
      image_label.append(img_label)
      ak_output = ( ak_output-minimum ) / ( maximum-minimum )


      #n_largest_idx = np.argpartition(ak_output, -30)[-30:]
      # print("10_largest_idx {}".format(n_largest_idx))
      # print("ak_output[n_largest_idx] {}".format(ak_output[n_largest_idx]))
      val='train'
      file=open("/home/sbaner24/Soybean/Final_Tests/Output/Results/train/train_"+str(os.path.basename(main_img))+'.txt',"w")
      file.write("att_weight_train="+str(ak_output))
      file.close
      create_heatmaps(img_name, ak_output, img_label, main_img, num_img,val)
      
    train_loss, train_acc,train_predictions=test_eval(model,train_set)
    file = open("/home/sbaner24/Soybean/Final_Tests/Output/Results/train_"+str(irun)+'.txt', "w")
    file.write("\n"+"train_acc="+ str(train_acc) + "," +"train_loss="+ str(train_loss)+ "\n" +"predictions="+ str(train_predictions)+"\n"+"gt_labels="+str(image_label)   
               +"\n"+"max_weight_train)="+str(max_weight_train)+"\n"+"main_img_names="+str(main_image)+"\n"+"len(main_img_names)="+
               str(len(main_image)))
    file.close
    t1 = time.time()
    max_weight_test=[]  
    min_weight_test=[]
    main_image=[]
    image_label=[]
    for num_img, whole_img in enumerate(test_bags):
        main_img = test_bags[num_img]
        img_name = test_set[num_img][2]
        img_label = int(test_set[num_img][1][0])
        
        # The firs bag's concatenated patches.
        ak_x = test_set[num_img][0] 
        print('Test data size: ', ak_x.shape)
    
        # This creates a function out of the input and output model layers into tensors.
        ak = K.function([model.layers[0].input], [model.layers[10].output])
        ak_output = ak([ak_x])
        ak_output=np.max(ak_output[0],axis=1,keepdims=True)
        ak_output = np.array(ak_output).reshape((ak_x.shape[0]))
        # For my dataset, there are 48 images in a bag.
        # rescale the weight as described in the paper
        minimum = ak_output.min()
        maximum = ak_output.max()
        max_weight_test.append(maximum)
        #min_weight_test.append(minimum)
        main_image.append(os.path.basename(main_img))
        image_label.append(img_label)
        ak_output = ( ak_output-minimum) / ( maximum-minimum)

        val='test'
        file=open("/home/sbaner24/Soybean/Final_Tests/Output/Results/test_"+str(os.path.basename(main_img))+'.txt',"w")
        file.write("att_weight_test="+str(ak_output))
        file.close
        create_heatmaps(img_name, ak_output, img_label, main_img, num_img,val)
    
    test_loss, test_acc,predictions= test_eval(model, test_set)
    

    
    file = open("/home/sbaner24/Soybean/Final_Tests/Output/Results/test_"+str(irun)+'.txt', "w")
    file.write("\n"+"test_acc="+ str(test_acc) + "," +"test_loss="+ str(test_loss)+ "\n" +"predictions="+ str(predictions)+"\n"+"gt_labels="+str(image_label)
               +"\n"+"max_weight_test="+str(max_weight_test)+"\n"+"main_img_names="+str(main_image)+"\n"+"len(main_img_names)="+
               str(len(main_image)))
    file.close
    t2 = time.time()

    print ('run time:', (t2 - t1) / 60.0, 'min')
    print ('test_acc={:.3f}'.format(test_acc))
 
    return test_acc

def load_testdata(test_path):
    # load test datapath
    path0 = glob.glob(test_path+'/0/*')
    path1 = glob.glob(test_path+'/1/*')
    path2 = glob.glob(test_path+'/2/*')
    path3 = glob.glob(test_path+'/3/*')
    path4 = glob.glob(test_path+'/4/*')

    all_path = path0+path1+path2+path3+path4
    return all_path

if __name__ == "__main__":

    args = parse_args()

    print ('Called with args:')
    print (args)

    input_dim = (80,120,3) 

    run = 1
    n_folds =1 # Runs 10 times
    acc = np.zeros((run), dtype=float)
    data_path = '/home/sbaner24/Soybean/Data/Patches/11-46/overlap_00/80x120/train' # dataset is in Patches.
    test_path=  '/home/sbaner24/Soybean/Data/Patches/11-46/overlap_00/80x120/test'
    

    for irun in range(run): # Runs 1 time
        # load_dataset is in 
        train_bags = load_dataset(data_path) 
        test_bags=load_dataset(test_path)
        acc[irun] = model_training(input_dim, train_bags,test_bags, irun)
        print ('mi-net mean accuracy = ', np.mean(acc))
        print ('std = ', np.std(acc))
        file = open("/home/sbaner24/Soybean/Final_Tests/Output/Results/test_final.txt", "w")
        file.write("\n"+"mi-net mean accuracy="+str(np.mean(acc))+"\n" +"std ="+str(np.std(acc)))
        file.close

    

