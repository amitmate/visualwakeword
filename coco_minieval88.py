#!/usr/bin/env python
# coding: utf-8

# In[1]:
from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
import os
import cv2
from coco import COCO
import numpy as np
import skimage.io as io
import random
#import matplotlib
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import pylab
#from IPython.display import display, Image
from skimage.filters import threshold_otsu

from keras.models import Sequential # To build a sequential model
from keras.layers import MaxPooling3D, LeakyReLU, MaxPooling2D, Conv3D, Conv2D, Dense, Flatten, Reshape,Activation, LSTM, CuDNNLSTM, SimpleRNN, CuDNNGRU , ConvLSTM2D,Dropout, BatchNormalization # For the layers in the model
#from keras.layers import Dense, LSTM, SimpleRNN,  ConvLSTM2D,Dropout, BatchNormalization # For the layers in the model
#from keras.callbacks import EarlyStopping, TensorBoard # For our early stopping and tensorboard callbacks
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.utils.generic_utils import get_custom_objects

#stopping_criterions =[
#            EarlyStopping(monitor='loss', min_delta=0, patience = 1000),
#                EarlyStopping(monitor='acc', base_line=.9992, patience =0) ]

IMAGESIZE = 96
PERSONTHR = 2048
#lrt = [100, 120, 140, 180]
#lrt = [50, 100, 150, 180]

def logErrorIndices(filename, model, x_test, y_test,imgIndices):
    print(y_test.shape, x_test.shape)
    pred = model.predict(x_test)
    print(y_test[1:10])
    print(pred[1:10])
    pred= (pred > 0.5).astype(np.uint8)
    print(pred[1:10])
    indices = [i for i,v in enumerate(pred) if pred[i]!=y_test[i]]
    #indices = [i for i,v in enumerate(pred) ]
    #subset_of_wrongly_predicted = [x_test[i] for i in indices,if pred[i]!=y_test[i] ]
    windices = [imgIndices[i] for i in indices ]
    print(len(windices))
    fp = open(filename,'w')
    print(*windices,file=fp,sep='\n')

def read_integers(filename):
    with open(filename) as f:
        return [int(x) for x in f]

def lr_schedule1(epoch):

    lr = 1e-3

    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 90:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 30:
        lr *= 1e-1
    
    print('Learning rate: ', lr)
    return lr

def lr_schedule(epoch):

    lr = 1e-3

    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 140:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    
    print('Learning rate: ', lr)
    return lr

def focal_loss(gamma=2., alpha=.5):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

def resnet_layer(inputs,
                 num_filters=8,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 l2reg=1e-4):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(l2reg))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x



def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 8
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3): 
        for res_block in range(num_res_blocks):
            activation = 'elu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    y = Dense(num_classes*8, kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)
    y = Activation('elu')(y)

    outputs = Dense(num_classes,
                    activation='sigmoid',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0: #6
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 8
    num_res_blocks = int((depth - 2) / 6) #6

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    l2reg =1e-4

    # Instantiate the stack of residual units
    for stack in range(3): #6,3
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            if stack ==2:
                l2reg=1e-3
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,l2reg=l2reg)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2  #6,2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    y = Dense(num_classes*8, kernel_initializer='he_normal')(y)
    y = BatchNormalization()(y)
    y = Dropout(0.1)(y)
    y = Activation('relu')(y)

    outputs = Dense(num_classes,
                    activation='sigmoid',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


#get_ipython().run_line_magic('pylab', 'inline')
drate = 0.2
def CNNModel(numUnits, numLayers):

  model = Sequential()

  model.add(Reshape((640,640,1), input_shape=(640,640))) 

  model.add(Conv2D(numUnits, (3, 3), use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(drate))

  model.add(Conv2D(numUnits, (3, 3), use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(drate))
  #model.add(Reshape((1,319,319,4), input_shape=(319,319,4))) 
  #model.add(Conv3D(1, (1, 1,numUnits), use_bias=False))
  #model.add(Reshape((319,316,1), input_shape=(1,319,316,1))) 
  #model.add(BatchNormalization())
  #model.add(Activation("relu"))
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  #model.add(Dropout(drate))
  model.add(Conv2D(numUnits, (3, 3), use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(drate))


  model.add(Conv2D(numUnits, (3, 3), use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(drate))
  #model.add(Reshape((1264,1264), input_shape=(158,158,64))) 
  #model.add(CuDNNLSTM(1, return_sequences=False)) 
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  return model

def ConvLSTMModel(numUnits, numLayers):

  model = Sequential()
  model.add(Reshape((1,640,640,1), input_shape=(640,640))) 
  model.add(ConvLSTM2D(numUnits, (3, 3), use_bias=False))
  #model.add(Conv2D(numUnits, (3, 3), use_bias=False))
  model.add(BatchNormalization())
  model.add(Activation("relu"))
  #model.add(Conv2D(numUnits, (3, 3), use_bias=False))
  #model.add(BatchNormalization())
  #model.add(Activation("relu"))
  #model.add(Conv2D(numUnits, (3, 3), use_bias=False))
  #model.add(BatchNormalization())
  #model.add(Activation("relu"))
  #model.add(LSTM(numUnits, return_sequences=rs))  
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  return model



def RNNModel(numUnits, numLayers):
  
  model = Sequential()
  
  rs = True
  
  if numLayers == 1:
    rs = False
  
  model.add(CuDNNLSTM(numUnits, input_shape=(input_shape), return_sequences=rs))
  model.add(BatchNormalization())
  
  if numLayers == 2:
    rs = False
    
  if numLayers > 1:
    model.add(CuDNNLSTM(numUnits, input_shape=(input_shape), return_sequences=rs))
    model.add(BatchNormalization())
    
  if numLayers > 2:
    model.add(CuDNNLSTM(numUnits, input_shape=(input_shape), return_sequences=False))
    model.add(BatchNormalization())
  
  model.add(Dense(1, activation='sigmoid'))
  return model

def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)

def rgb2gray(rgb):
    if len(rgb.shape) == 3:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = rgb
    return gray

def checkPersonThreshold(anns):
    for i in range(len(anns)):
        if anns[i]['area'] > PERSONTHR:
            return 1
    return 0

def cocovw_loadMini(train,numimg,num_classes):
    train = False
    mini = True
    dataDir='/home/amit_mate2009/coco/raw-data/'
    if train:
        dataType='train2014'
    else:
        dataType ='val2014'
        
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)   
    
    coco=COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
    
    catIds = coco.getCatIds(catNms=['person']);
    imgIds = coco.getImgIds(catIds=catIds );

    if not mini:
        imgIdsNonPersons = list(set(coco.getImgIds()) -set(imgIds) );
    else:
        imgMVIds = read_integers('/home/amit_mate2009/coco/mscocominival.txt')
        imgIds = list(set(imgMVIds)&set(imgIds))
        imgIdsNonPersons = list(set(imgMVIds) -set(imgIds) );
        
    if train == False:
#        b = np.arange(len(imgIds))
#        np.random.shuffle(b)
        imgIds = random.sample(imgIds,len(imgIds))
#
#        c = np.arange(len(imgIdsNonPersons))
#        np.random.shuffle(c)
        imgIdsNonPersons = random.sample(imgIdsNonPersons,len(imgIdsNonPersons))

    print(len(imgIdsNonPersons))
    print(len(imgIds))
    samples = len(imgIds)
    numValidPersons = 3580
    NUMIMAGES = samples 
    img3= np.ones((np.int( numValidPersons + len(imgIdsNonPersons)) ,IMAGESIZE,IMAGESIZE,3), np.uint8)
    type3 = np.zeros((numValidPersons + len(imgIdsNonPersons),1),np.uint8)
    imgIndices = np.zeros((numValidPersons + len(imgIdsNonPersons),1),np.uint32)
    k = 0  
    for i in range(NUMIMAGES):
        if i%1000 == 0:
            print (i, imgIds[i])
        img = coco.loadImgs(imgIds[i])[0]
        
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
        anns = coco.loadAnns(annIds)
        pFLAG = checkPersonThreshold(anns)
        if pFLAG == 0:
            continue
        
        name = dataDir+dataType+'/'+img['file_name']
        img1=mpimg.imread(name)
        if len(img1.shape) != 3:
            #print("found gray") 
            img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        #print(i, name)
        #img2 = rgb2gray(img1)
        #img2= ( (img2 < threshold_otsu(img2)).astype('float32') )                
        #img1 = (img1 -np.mean(img1))/np.std(img1)
        img3[k] = resize2SquareKeepingAspectRation(img1, IMAGESIZE, cv2.INTER_AREA)
        #img3[i][0:img2.shape[0],0:img2.shape[1]] = img2[0:img2.shape[0],0:img2.shape[1]]
        

		
        type3[k] = pFLAG
        imgIndices[k] = img['id']
        k = k +1 
        #if type3[i] == 0:
        #    imgplot = plt.imshow(img3[i])
        #    coco.showAnns(anns)
        #    plt.show()
    print("Valid Persons", k)

    for i in range(len(imgIdsNonPersons)):
        if i%1000 == 0:
            print (i, imgIdsNonPersons[i])
        img = coco.loadImgs(imgIdsNonPersons[i])[0]
        
        #annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        #anns = coco.loadAnns(annIds)
        
        name = dataDir+dataType+'/'+img['file_name']
        img1=mpimg.imread(name)
        if len(img1.shape) != 3:
            #print("found gray") 
            img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        #print(i, name)
        #img2 = rgb2gray(img1)
        #img2= ( (img2 < threshold_otsu(img2)).astype('float32') )                
        #img1 = (img1 -np.mean(img1))/np.std(img1)
        img3[i+numValidPersons] = resize2SquareKeepingAspectRation(img1, IMAGESIZE, cv2.INTER_AREA)
        #img3[i+NUMIMAGES][0:img2.shape[0],0:img2.shape[1]] = img2[0:img2.shape[0],0:img2.shape[1]]
        


        imgIndices[i+numValidPersons] = img['id']
        type3[i+numValidPersons] = 0

    a = np.arange(img3.shape[0])
    np.random.shuffle(a)
    img3 = img3[a]
    type3 = type3[a]
    imgIndices = imgIndices[a]
    #print("randomized")
    print(np.sum(type3))

    #type3 = keras.utils.to_categorical(type3, num_classes)
    img3 = img3.astype('float32')
    img3 = img3/127.5 -1.0
    #img3 = (img3 -np.mean(img3))/np.std(img3)
    print(np.sum(type3))
    return (img3,type3,imgIndices)

def cocovw_load(train,numimg,num_classes):
    #FACTOR =0.7027 
    #FACTOR =0.83254 
    #FACTOR = 1
    dataDir='/home/amit_mate2009/coco/raw-data/'
    if train:
        dataType='train2014'
    else:
        dataType ='val2014'
        
    annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)   
    
    coco=COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print('COCO supercategories: \n{}'.format(' '.join(nms)))
    
    catIds = coco.getCatIds(catNms=['person']);
    imgIds = coco.getImgIds(catIds=catIds );
    
    imgIdsNonPersons = list(set(coco.getImgIds()) -set(imgIds) );

    if train == False:
#        b = np.arange(len(imgIds))
#        np.random.shuffle(b)
        imgIds = random.sample(imgIds,len(imgIds))
#
#        c = np.arange(len(imgIdsNonPersons))
#        np.random.shuffle(c)
        imgIdsNonPersons = random.sample(imgIdsNonPersons,len(imgIdsNonPersons))

    print(len(imgIdsNonPersons))
    print(len(imgIds))

    numValidPersons = 0

    if train:
        samples = len(imgIds)
        numValidPersons = 38459 
        numValidNonPersons = len(imgIdsNonPersons)
    else:
        for i in range(numimg):
            img = coco.loadImgs(imgIds[i])[0]
            if train: 
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            else:
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)

            anns = coco.loadAnns(annIds)

            numValidPersons =numValidPersons+checkPersonThreshold(anns)

        samples = numimg
        print(numValidPersons) 
        numValidNonPersons = numValidPersons

    NUMIMAGES = samples 
    img3= np.ones((np.int( numValidPersons + numValidNonPersons) ,IMAGESIZE,IMAGESIZE,3), np.uint8)
    type3 = np.zeros((numValidPersons + numValidNonPersons,1),np.uint8)
    print(np.mean(img3), np.std(img3))

    k =0 
    for i in range(NUMIMAGES):
        if i%1000 == 0:
            print (i, imgIds[i])
        img = coco.loadImgs(imgIds[i])[0]
        
        if train: 
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        else:
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)

        anns = coco.loadAnns(annIds)

        pFLAG = checkPersonThreshold(anns)
        if  pFLAG == 0:
            continue
        
        name = dataDir+dataType+'/'+img['file_name']
        img1=mpimg.imread(name)
        if len(img1.shape) != 3:
            #print("found gray") 
            img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        #print(i, name)
        #img2 = rgb2gray(img1)
        #img2= ( (img2 < threshold_otsu(img2)).astype('float32') )                
        #img1 = (img1 -np.mean(img1))/np.std(img1)
        img3[k] = resize2SquareKeepingAspectRation(img1, IMAGESIZE, cv2.INTER_AREA)
        #img3[i][0:img2.shape[0],0:img2.shape[1]] = img2[0:img2.shape[0],0:img2.shape[1]]
        

		
        type3[k] = pFLAG
        k=k+1
        
        #if type3[i] == 0:
        #    imgplot = plt.imshow(img3[i])
        #    coco.showAnns(anns)
        #    plt.show()
    print(k)
    #for i in range(np.int(NUMIMAGES*FACTOR)):
    for i in range(numValidNonPersons):
        if i%1000 == 0:
            print (i, imgIdsNonPersons[i])
        img = coco.loadImgs(imgIdsNonPersons[i])[0]
        
        #annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        #anns = coco.loadAnns(annIds)
        
        name = dataDir+dataType+'/'+img['file_name']
        img1=mpimg.imread(name)
        if len(img1.shape) != 3:
            #print("found gray") 
            img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2RGB)
        #print(i, name)
        #img2 = rgb2gray(img1)
        #img2= ( (img2 < threshold_otsu(img2)).astype('float32') )                
        #img1 = (img1 -np.mean(img1))/np.std(img1)
        img3[i+numValidPersons] = resize2SquareKeepingAspectRation(img1, IMAGESIZE, cv2.INTER_AREA)
        #img3[i+NUMIMAGES][0:img2.shape[0],0:img2.shape[1]] = img2[0:img2.shape[0],0:img2.shape[1]]
        


        type3[i+numValidPersons] = 0

    a = np.arange(img3.shape[0])
    np.random.shuffle(a)
    img3 = img3[a]
    type3 = type3[a]
    #print("randomized")
    print(np.sum(type3))

    #type3 = keras.utils.to_categorical(type3, num_classes)
    img3 = img3.astype('float32')
    img3 = img3/127.5 -1.0
    #img3 = (img3 -np.mean(img3))/np.std(img3)

    print(np.sum(type3))
    return (img3,type3)

NUMEPOCHS = 150
NUMUNITS = 4
NUMLAYERS = 1
BSIZE = 64 
NTRAIN=45174
NVAL = 9000
#NTRAIN=256
#NVAL = 256
NCLASSES = 1
(x_train, y_train) = cocovw_load(True,NTRAIN,NCLASSES)
print(x_train.shape)
(x_val, y_val) = cocovw_load(False,NVAL,NCLASSES)
#PERSONTHR = 2048
#(x_val, y_val,imgIndices) = cocovw_loadMini(False,NVAL*2,NCLASSES)
print(x_val.shape)
CNNM = False
input_shape = (x_train.shape[1], x_train.shape[2],x_train.shape[3])
#model = RNNModel (NUMUNITS,NUMLAYERS)
img_rows, img_cols = x_train.shape[1], x_train.shape[2]

if CNNM:
	model = CNNModel (NUMUNITS,NUMLAYERS)
else:
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        #x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        x_val = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
        #x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 3)
        input_shape = (img_rows, img_cols, 3)

    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_val = x_val.astype('float32')
    
    n = 2

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2 #6
    elif version == 2:
        depth = n * 9 + 2

    # Model name, depth and version
    model_type = 'ResNet%dv%d' % (depth, version)
    print(model_type)
    print(input_shape)
    if version == 1:
        model = resnet_v1(input_shape=input_shape, depth=depth,num_classes=NCLASSES)
    else:
        model = resnet_v2(input_shape=input_shape, depth=depth,num_classes=NCLASSES)

#filepath1 = '/home/amit_mate2009/coco/SubResNet14v1_model87_63ME.h5'

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=Adam(), loss=focal_loss(gamma=2.0,alpha=0.75), metrics=['accuracy'])

#model.load_weights(filepath1)
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Sub%s_model_ZM.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,monitor='val_acc',verbose=1,save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
#lr_scheduler = LearningRateScheduler(lr_schedule1)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,min_lr=0.5e-6)
#lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.001, cooldown=0, min_lr=0.5e-6)


callbacks = [checkpoint, lr_reducer, lr_scheduler]


DATA_AUGMENT = True

if not DATA_AUGMENT:
	hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=callbacks,shuffle= True, batch_size= BSIZE, epochs=NUMEPOCHS)
else :
#if False:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    
    #datagen = ImageDataGenerator(preprocessing_function=get_random_eraser(v_l=0, v_h=255))
    
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        preprocessing_function=get_random_eraser(v_l=0, v_h=0),
        # preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        #brightness_range=[1.0, 1.2],
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=BSIZE),
        epochs=NUMEPOCHS, callbacks=callbacks,shuffle= True,
        validation_data=(x_val, y_val),steps_per_epoch = x_train.shape[0]/BSIZE,
        workers=4)



#model.summary()
del model
#loss = focal_loss(gamma =2, alpha=0.25)
#get_custom_objects().update({"focal_loss_fixed": loss})

(x_test, y_test,imgIndices) = cocovw_loadMini(False,NVAL*2,NCLASSES)
#(x_test, y_test) = (x_val,y_val)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
#filepath = '/home/amit_mate2009/SubResNet14v1ZM.h5'
model = load_model(filepath)
print(filepath)
#model.load_weights(filepath)
scores = model.evaluate(x_test, y_test)
logErrorIndices(filename='logErrors3', model=model, x_test=x_test, y_test=y_test,imgIndices=imgIndices)
print(scores)

#filepath1 = '/home/amit_mate2009/coco/SubResNet14v1_model87_63ME.h5'
#filepath1 = '/home/amit_mate2009/coco/SubResNet14v1_model87_5_86.8_Final.h5'
#model = load_model(filepath1)
#model.load_weights(filepath)
#scores = model.evaluate(x_test, y_test)
#y_test = model.predict(x_test)
#print(scores)
#logErrorIndices(filename='logErrors2', model=model, x_test=x_test, y_test=y_test,imgIndices=imgIndices)

#filepath1 = '/home/amit_mate2009/coco/SubResNet14v1_model87.6_87.8.h5'
#model = load_model(filepath1)
#scores = model.evaluate(x_test, y_test)
#print(scores)



#filepath1 = '/home/amit_mate2009/coco/SubResNet14v1_87.9_88.2.h5'
#model = load_model(filepath1)
#scores = model.evaluate(x_test, y_test)
#print(scores)
# In[ ]:




