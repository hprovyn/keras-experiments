# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 18:47:02 2017

@author: Big Pigeon
"""

import hough_img_prep as hip
import model as m
import os

vgg_weights_path = "C:\\Users\\Big Pigeon\\Downloads\\vgg16_weights.h5"
def computeAllEmbeddings(trainRootDir, testRootDirs, trainPkl):
    model = m.keras_VGG16(vgg_weights_path, dropFinalCNN=False)
    m.computeBottleneck(model, trainRootDir, trainPkl)
    count = 0
    for testRootDir in testRootDirs:
        count = count + 1
        m.computeBottleneckTest(model, testRootDir[0], testRootDir[1])

root_dir = "C:\\Users\\Big Pigeon\\Documents\\Python Scripts\\pigeons"

#adds carneau labels to labeled image dir, creates train/test sets, crop/resize all to 224 by 224
def create_training_test(root_dir):
    all_dir = os.path.join(root_dir, "all")
    hip.create_labels(os.path.join(root_dir, "carneau"), all_dir, "carneau")
    hip.create_train_test(root_dir, all_dir, ["barb", "carneau", "frillback", "pouter"], 0.2)
    raw_train_dir = os.path.join(root_dir, "raw_train")
    raw_test_dir = os.path.join(root_dir, "raw_test")
    hip.crop_resize_to_squares(raw_train_dir, root_dir)
    hip.crop_resize_to_squares(raw_test_dir, root_dir, 320) #subsequent rotation step will resize to [224 x 224]
    test_320px_dir = os.path.join(root_dir, "square320px")
    hip.perform_rotations(90, test_320px_dir, root_dir)

def experiment_harness(root_dir, augmentations, epochs):
    to_augment_name = "square224px"
    augmentation_desc = "imagedatagen_" + to_augment_name + "_" + str(augmentations)
    train_dir = os.path.join(root_dir, augmentation_desc)
    train_pkl = os.path.join(root_dir, to_augment_name + ".pkl")
    test_dir = os.path.join(root_dir, "90_rotations")
    test_dirs = [[test_dir, os.path.join(root_dir, "test.pkl")]]
    m.dataGen(os.path.join(root_dir, to_augment_name), train_dir, augmentations)
    computeAllEmbeddings(train_dir, test_dirs, train_pkl)
    weights_path = os.path.join(root_dir, augmentation_desc + "model")
    m.train_top_model(train_pkl, [test_dirs[0][1]], weights_path, train_dir, epochs)
    
create_training_test(root_dir)
experiment_harness(root_dir, 40, 5)