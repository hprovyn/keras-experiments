# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 12:30:44 2017

@author: Big Pigeon
"""
import pdb
import os
import keras
import h5py

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import numpy as np
import imagenet_utils
import time
from scipy import ndimage
from scipy import misc
import pickle
import shutil

#Upodate Theano
#sudo pip install git+git://github.com/Theano/Theano.git --upgrade --no-deps

#To change backend: edit .keras/keras.json, use either theano or tensorflow


#Implementation of the VGG16 model using Keras
#Uses pre-trained weights that need to be laoded from a specific path
#Drop out pops off the classification and dropout layers to gain access to embedding vectors, enable with True flag
def keras_VGG16(weights_path, dropFinalCNN=False):    
    print("loading neural net")
    fstart = time.time() 
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    if not dropFinalCNN:        
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
        
    fend = time.time()
    
    print(fend - fstart)    
    return model
        
#Preprocessing steps for VGG models so that test images are resized according to
#the size images the model was trained on
def preprocess_image(img_path, model):      
    # scale the image, according to the format used during VGG training on ImageNet
   # if model == 'InceptionV3':
    #    im = image.load_img(img_path, target_size=(299, 299))
 #   else:
    im = image.load_img(img_path, target_size=(224, 224))
#    plt.figure(figsize=(4, 4))
#    plt.axis("off")
#    plt.imshow(im)
    x = image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    return x
                     
datagen = ImageDataGenerator(
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')

def dataGen(srcdir, targetdir, n): 
    shutil.rmtree(targetdir, ignore_errors=True, onerror=None)
    os.mkdir(targetdir)    

    for a in set([im.split("_")[0] for im in os.listdir(srcdir)]):        
        os.mkdir(os.path.join(targetdir, a))
        
    for im in os.listdir(srcdir):
        img = load_img(os.path.join(srcdir, im))
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        splits = im.split("_")        
        cap_class = im.split("_")[0]
        if len(splits) == 3:
            SAVE_PREFIX = cap_class + "_" + splits[1] + "_" + splits[2].split(".")[0]
        else:
            SAVE_PREFIX = cap_class + "_" + splits[1].split(".")[0]

        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=targetdir + "\\" + cap_class, save_prefix=SAVE_PREFIX, save_format='jpeg'):
            i += 1
            if i > n - 1:
                break

def getTraining(imgRootDir):
    trainX = []
    trainY = []            

    for im in os.listdir(imgRootDir):
        preprocessed = []            
        preprocessed = np.ndarray((len(os.listdir(imgRootDir + "\\" + im)), 3, 224, 224), dtype=np.float32)
        count = 0
        for img in os.listdir(imgRootDir + "\\" + im):
            scaledIm = preprocess_image(imgRootDir + "\\" + im + "\\" + img, "th")
            preprocessed[count] = scaledIm
            count = count + 1
                    
        trainX.extend(preprocessed)
        ys = np.tile(im, len(preprocessed))
        trainY.extend(ys)                
    trainData = {"data": trainX,
                 "label": trainY}                             
    return trainData
    
def computeBottleneck(model, imgRootDir, pklFileName):        
    trainX = []
    trainY = []            

    for im in os.listdir(imgRootDir):
        preprocessed = []            
        preprocessed = np.ndarray((len(os.listdir(imgRootDir + "\\" + im)), 3, 224, 224), dtype=np.float32)
        count = 0
        for img in os.listdir(imgRootDir + "\\" + im):
            scaledIm = preprocess_image(imgRootDir + "\\" + im + "\\" + img, "th")
            preprocessed[count] = scaledIm
            count = count + 1
            
        print (preprocessed.shape)
        p = model.predict(preprocessed, verbose=1)            
        trainX.extend(p)
        ys = np.tile(im, len(preprocessed))
        trainY.extend(ys)                
        
    trainData = {"data": trainX,
                 "label": trainY}                             
        
    
    output = open(pklFileName, 'wb')
    pickle.dump(trainData, output)            
    output.close
    return trainData

def computeBottleneckTest(model, imgRootDir, pklFileName): 
    trainX = []
    trainY = []            
    preprocessed = []
    preprocessed = np.ndarray((len(os.listdir(imgRootDir)), 3, 224, 224), dtype=np.float32)        
    count = 0  
    for im in os.listdir(imgRootDir):                                         
        scaledIm = preprocess_image(imgRootDir + "\\" + im, "th")
        preprocessed[count] = scaledIm
        ys = np.tile(im.split("_")[0], 1)
        trainY.extend(ys)
        count = count + 1
            
    print (preprocessed.shape)
    p = model.predict(preprocessed, verbose=1)            
    trainX.extend(p)        
        
    trainData = {"data": trainX,
                 "label": trainY}                             
    output = open(pklFileName, 'wb')
    pickle.dump(trainData, output)            
    output.close
    return trainData

def createClassWeights(trainDir, images):
    classWeights = {}
    count = 0
    classes = len(os.listdir(trainDir))
    for im in os.listdir(trainDir):
        classWeights[count] = images / classes / len(os.listdir(trainDir + "\\" + im))
        count = count + 1
    return classWeights
        
def train_top_model(trainPklFile, testPklFiles, topModelWeightsPath, trainOutDir, epochs):        
    trainInput = open(trainPklFile, 'rb')
    trainData = pickle.load(trainInput)
                        
    y_all = LabelEncoder().fit_transform(trainData["label"])
    y_all = np_utils.to_categorical(y_all)
    print(len(y_all), "label size")

    trainNp = np.array(trainData["data"])
    model = Sequential()
    model.add(Flatten(input_shape=trainNp.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(os.listdir(trainOutDir)), activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    trainNp, y_all = shuffle(trainNp, y_all, random_state=0)
            
    model.fit(trainNp, y_all,
              nb_epoch=epochs, batch_size=1024,validation_split=0.2, shuffle=True,
              verbose=1, class_weight = createClassWeights(trainOutDir, len(y_all)))    
    for testPklFile in testPklFiles:
        testInput = open(testPklFile, 'rb')
        testData = pickle.load(testInput)
                
        preds = model.predict(np.array(testData["data"]), verbose = 1)
        model.save_weights(topModelWeightsPath)
        
        class_names = sorted(set(testData["label"]))
        correct = 0
        class_truth = {}
        for cn in class_names:
            class_truth[cn] = {'correct':0, 'incorrect':0}
        for i in range(len(preds)):
            maxIndex = np.argmax(preds[i])
            this_truth = testData["label"][i]
            if (this_truth == class_names[maxIndex]):
                correct += 1
                class_truth[this_truth]['correct'] += 1            
            else:
                class_truth[this_truth]['incorrect'] += 1
        print(" ")
        for i in class_names:
            print(i, " accuracy: ", class_truth[i]['correct'] / (class_truth[i]['correct'] + class_truth[i]['incorrect']))    
        print("accuracy: ", correct / len(testData["label"]))