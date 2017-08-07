# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 20:12:52 2016

@author: Hunter
"""

from scipy import misc
from scipy import ndimage
import os
import numpy as np

def crop(img_array, size):
    lx, ly, lz = img_array.shape
    if lx > ly + 1:        
        crop_img = img_array[int((lx-ly)/2): - int((lx-ly)/2), 0: ly]
    else:
        if ly > lx + 1:                    
            crop_img = img_array[0: lx, int((ly-lx)/2): - int((ly-lx)/2)]
        else: crop_img = img_array
    newlx, newly, newlz = crop_img.shape
    if newlx != size and newly != size:
        crop_img = misc.imresize(crop_img, (size, size))
    return crop_img
    
def crop_around_circle(img, c = None, default_radius = 0.2, default_background=[103.939,116.779,123.68]):
    sx, sy, sz = img.shape
    if c is None:
        cx = sx / 2
        cy = cx
        radius = sx * default_radius
    else:
        cy, cx, radius = c
    r2 = radius * radius
    background = np.array(default_background)
    for i in range(sx):
        iDist = (i - cx) * (i - cx)
        for j in range(sy):
            jDist = (j - cy) * (j - cy)
            if iDist + jDist > r2:
                img[i,j] = background 
    return img

import cv2
import numpy as np

def get_best_hough_circle(img_path, radius_min, radius_max):
    img = cv2.imread(img_path,0)
    x,y = img.shape    
    img = cv2.medianBlur(img,5)
    try:
        circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=int(x * radius_min),maxRadius=int(x * radius_max))
        circles = np.uint16(np.around(circles))
        i = circles[0][0]
    except:
        i = None    
    return i
    
def zoom_crop(img, scale):        
    lx, ly, lz = img.shape
    img = img[int((lx - lx * scale) / 2):int((lx + lx * scale) / 2), int((ly - ly * scale) / 2):int((ly + ly * scale) / 2)]
    return img
            
def create_rotated(rotations, rotation_dir, img_array, name):
    for i in range(rotations):
        angle_deg = int(i * 360 / rotations)
        class_name = name.split('.')[0]
        suffix = name.split('.')[1]
        rotated = ndimage.rotate(img_array, angle_deg, reshape=False)
        rotated = zoom_crop(rotated, 0.7)
        file_name = class_name + "_" + str(angle_deg) + "." + suffix
        misc.imsave(os.path.join(rotation_dir, file_name), rotated)

import shutil

def crop_hough(imgPath, radius_min, radius_max):
    bestCircle = get_best_hough_circle(imgPath, radius_min, radius_max)
    img = misc.imread(imgPath)
    return crop_around_circle(img, bestCircle)
    
def crop_resize_to_squares(img_folder, target_root_dir, size = 224):
    squares_folder = os.path.join(target_root_dir, "square" + str(size) + "px")
    shutil.rmtree(squares_folder, ignore_errors=True, onerror=None)
    os.mkdir(squares_folder)    
    print(img_folder)
    print(os.listdir(img_folder))
    for each_img in os.listdir(img_folder):
        img_path = os.path.join(img_folder, each_img)
        print (each_img)        
        cropped = crop(misc.imread(img_path),size)
        misc.imsave(os.path.join(squares_folder, each_img), cropped)
    
def create_background_cropped_training_set(img_folder, target_root_dir, background=True, crop_circle=True, zoom_factor=None, radius_min=0.13, radius_max=0.3):
    description = "trainset"
    if background:
        description += "_original_background"
    if crop_circle:
        description += "_removed_background"
    if zoom_factor:
        description += "_zoom_" + str(zoom_factor)
        
    background_crop_folder = os.path.join(target_root_dir, description)
    
    shutil.rmtree(background_crop_folder, ignore_errors=True, onerror=None)
    os.mkdir(background_crop_folder)    
    print(img_folder)
    print(os.listdir(img_folder))
    for each_img in os.listdir(img_folder):
        img_path = os.path.join(img_folder, each_img)
        print (each_img)        
        class_name = each_img.split(".")[0]
        
        if background:
            target_file_path = os.path.join(background_crop_folder, class_name + "_orig.jpg")
            if zoom_factor is None:
                shutil.copy(img_path, target_file_path)
            else:            
                img = misc.imread(img_path)
                img = zoom_crop(img, zoom_factor)        
                misc.imsave(target_file_path, img)
        if crop_circle:
            circle = crop_hough(img_path, radius_min, radius_max)
            target_file_path = os.path.join(background_crop_folder, class_name + "_circle.jpg")
            if zoom_factor is not None:
                circle = zoom_crop(circle, zoom_factor)
            misc.imsave(target_file_path, circle)
            
def perform_rotations(rotations, source_dir, target_root_dir):
    rotation_dir = os.path.join(target_root_dir, str(rotations) + "_rotations")
    shutil.rmtree(rotation_dir, ignore_errors=True, onerror=None)
    os.mkdir(rotation_dir)
    for each_img in os.listdir(source_dir):
        img_path = os.path.join(source_dir, each_img)
        create_rotated(rotations, rotation_dir, misc.imread(img_path), each_img)

from PIL import Image

def create_labels(source_dir, target_dir, label):    
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    count = 0
    for each_img in os.listdir(source_dir):                      
        img_path = os.path.join(source_dir, each_img)
        labeled_path = os.path.join(target_dir, label + "_" + str(count) + ".jpg")
        im = Image.open(img_path)
        im.convert('RGB').save(labeled_path, "JPEG")        
        count += 1

import random        
def create_train_test(target_root_dir, source_dir, classes, test_percent=0.2):
    train_dir = os.path.join(target_root_dir, "raw_train")    
    test_dir = os.path.join(target_root_dir, "raw_test")
        
    shutil.rmtree(train_dir, ignore_errors=True, onerror=None)
    os.mkdir(train_dir)
    shutil.rmtree(test_dir, ignore_errors=True, onerror=None)
    os.mkdir(test_dir)
    
    print(os.listdir(source_dir))
    for each_img in os.listdir(source_dir):
        if each_img.split("_")[0] in classes:
            thisfile = os.path.join(source_dir, each_img)
            if random.random() > test_percent:            
                shutil.copy(thisfile, train_dir)
            else:
                shutil.copy(thisfile, test_dir)