"""
In this script, I have made use of annotation from a json file to determine which is positive image and which is
negative image. This was done for the crop dataset. Which had both damaged(posiitve) and non-damaged(negative) images.
The main part which is of interest however is the same as what i told you the other day.
The create_patches function is the key one that is of interest. What you pass to it is upto you. You can modify
it based on the comments given here.

The following is the file structure to be followed:
--> 0 (if this class exists)
    --> IMG_001
        -->IMG_001_patch1.jpg
        -->IMG_001_patch2.jpg
        .
        .
        .
        -->IMG_001_patchN.jpg
    --> IMG_009
        -->IMG_009_patch1.jpg
        -->IMG_009_patch2.jpg
        .
        .
        .
        -->IMG_009_patchN.jpg
    .
    .
    -->IMG_100
        -->IMG_100_patch1.jpg
        .
        .
        -->IMG_100_patchN.jpg

--> 1
    --> IMG_003
        --> IMG_003_patch1.jpg
        --> IMG_003_patch2.jpg
        .
        .
        --> IMG_003_patchN.jpg
    --> IMG_007
        -->
        -->
    .
    .
"""
import os
import glob
from os.path import basename, join, isdir, isfile
import json

import cv2
import numpy as np

def create_patches(cur_img_file, img_dir_save, patch_h,patch_w,overlap):
    """
    This function is to create (24, 24, 3) patches for the crop dataset.
    These crops have to be stored into folders 0/1 based on whether they have a damaged region or not.

    cur_img_file: str: path to the current image that needs to split into patches.
    img_dir_save: str: the location to the dir in which we save the patches.
    patch_dim: int: the dimesnion of the patch. gives len of one side. Its a square patch.
    :return: - Saves the pacthes in the respective mg_dir_save path.
    """

    cur_img = cv2.imread(cur_img_file)
    print('Original image shape: ', cur_img.shape)

    # iterating over the dimensions of the image to patch it up.
    patch_num = 0  # keeps track of the patch count
    img_h=cur_img.shape[0]
    img_w=cur_img.shape[1]
    row_interval=int((1-overlap)*patch_h)
    col_interval=int((1-overlap)*patch_w)
    row_final=img_h-patch_h
    col_final=img_w-patch_w
    for row in range(0, row_final+1, row_interval):
        for col in range(0, col_final+1, col_interval):
            patch = cur_img[row:row+patch_h, col:col+patch_w]
            patch_name = join(img_dir_save, f'{patch_num:04}'+'.jpg')
            #print(patch_name)

            # writing the image in save loc.
            cv2.imwrite(patch_name, patch)
            patch_num += 1
            assert isfile(patch_name) == True, 'Patch image not saved.'

    print('Num of patches in the dir is: ', len(os.listdir(img_dir_save)))
    print('Patch image shape: ', cv2.imread(join(img_dir_save, os.listdir(img_dir_save)[0])).shape)
    print('Patches creation done!')
    print('-----------')


if __name__ == "__main__":
    folder='test/'
    label='0/'
    cur_img_dir='/home/sbaner24/Soybean/Final_Tests/Data/11-46_resized/'+str(folder)+str(label)
    dst_dir='/home/sbaner24/Soybean/Final_Tests/Data/11-46_resized_patches/overlap_25/80x120/'+str(folder)
    if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
    dst_img_dir=dst_dir+str(label)
    if not os.path.exists(dst_img_dir):
                os.mkdir(dst_img_dir)
    
    patch_h=80
    patch_w=120
    overlap=0.25
    for file in os.listdir(cur_img_dir):
        file_name=file.split('.')[0]
        cur_img_file=cur_img_dir+str(file)
        img_folder=dst_img_dir+str(file_name)
        if not isdir(img_folder):
                os.mkdir(img_folder)
        create_patches(cur_img_file, img_folder, patch_h, patch_w, overlap)