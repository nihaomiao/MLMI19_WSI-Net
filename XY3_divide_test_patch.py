# Dividing testing patches
# To preprocess slide, firstly simply divide the slide into
# several patches(500 * 500), secondly delete those fat patches
# Author: Haomiao Ni
# Created on Sept 28, 2018
# # For some reasons, we name In-Situ as DG, Invasive as JR.

import os
from libtiff import TIFF
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
from time import time
from PIL import Image
import threading
from scipy.sparse import *

def open_slide(tif_path, set_current_level):
    slide = TIFF.open(tif_path)
    set_res = slide.SetDirectory(set_current_level)
    if set_res == 1:
        tiff_array = slide.read_image()
    else:
        set_res = slide.SetDirectory(set_current_level - 1)
        assert set_res == 1
        tiff_array = slide.read_image()
        height = tiff_array.shape[0]
        width = tiff_array.shape[1]
        tiff_array = tiff_array[0:height:2, 0:width:2]
    return tiff_array


def divide_Normal_slide(level0_img, filename, imgpath, thread_index, ranges, r_list, c_list, log):
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        shard = s
        r = r_list[shard]
        c = c_list[shard]
        try:
            topy = r*8-patch_size/2
            buttomy = topy+patch_size
            leftx = c*8-patch_size/2
            rightx = leftx+patch_size
            if topy<0 or leftx<0 or buttomy>level0_img.shape[0] or rightx>level0_img.shape[1]:
                print 'Out of Index: ('+ str(leftx) + ',' + str(topy) + ") for " + filename
                log.writelines('Out of Index: ('+ str(leftx) + ',' + str(topy) + ") for " + filename+'\n')
                continue

            imgarr = level0_img[topy:buttomy, leftx:rightx]
            image = Image.fromarray(imgarr)
            image = image.convert('RGB')
        except:
            print "Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename
            log.writelines("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename + '\n')
            continue
        else:
            imagename = os.path.join(imgpath, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
            image.save(imagename, "JPEG")


def process_normal_tif(file, filename, images, log):
    start = time()
    imgpath = os.path.join(images, filename[:-4])
    if not os.path.exists(imgpath):
        os.mkdir(imgpath)
    else:
        return
    set_current_level = 3
    low_dim_array = open_slide(file, set_current_level)
    low_dim_img = Image.fromarray(low_dim_array)

    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()
    low_s_array = np.array(low_s)

    # --OSTU threshold
    low_s_thre = filters.threshold_otsu(low_s_array)
    low_s_bin = low_s_array > low_s_thre

    # divide low_s
    h = low_s.height
    w = low_s.width
    assert h > 500
    assert w > 500
    # h OR w = 500 + 450k + R
    h_k = (h - patch_size / 8) // ((patch_size - overlap_size) / 8)
    h_R = (h - patch_size / 8) % ((patch_size - overlap_size) / 8)
    if h_R <= overlap_size / 8:
        h_flag = 0
    else:
        h_flag = 1
    w_k = (w - patch_size / 8) // ((patch_size - overlap_size) / 8)
    w_R = (w - patch_size / 8) % ((patch_size - overlap_size) / 8)
    if w_R <= overlap_size / 8:
        w_flag = 0
    else:
        w_flag = 1

    h_list = []
    h_list.append(patch_size / 16)

    for i in range(1, h_k + 1):
        h_list.append(patch_size / 16 + ((patch_size - overlap_size) / 8) * i)

    if not h_flag:
        if h_k >= 1: h_list.remove(patch_size / 16 + ((patch_size - overlap_size) / 8) * h_k)
        h_list.append(h - patch_size / 16)
    else:
        h_list.append(h - patch_size / 16)

    w_list = []
    w_list.append(patch_size / 16)

    for j in range(1, w_k + 1):
        w_list.append(patch_size / 16 + ((patch_size - overlap_size) / 8) * j)

    if not w_flag:
        if w_k >= 1: w_list.remove(patch_size / 16 + ((patch_size - overlap_size) / 8) * w_k)
        w_list.append(w - patch_size / 16)
    else:
        w_list.append(w - patch_size / 16)

    [r_list, c_list] = np.meshgrid(h_list, w_list)
    r_list = r_list.flatten()
    c_list = c_list.flatten()

    tar_r_list = []
    tar_c_list = []

    for i in range(len(r_list)):
        r = r_list[i]
        c = c_list[i]
        topy = r - patch_size / 16
        buttomy = r + patch_size / 16
        leftx = c - patch_size / 16
        rightx = c + patch_size / 16
        low_patch = low_s_bin[topy:buttomy, leftx:rightx]
        if np.sum(low_patch):
            tar_r_list.append(r)
            tar_c_list.append(c)

    r_list = tar_r_list
    c_list = tar_c_list

    level0_img = open_slide(file, 0)
    print('shape:', level0_img.shape)
    log.writelines('shape:'+str(level0_img.shape[0])+' '+str(level0_img.shape[1])+'\n')
    print (time() - start, 's')
    num_threads = 64
    num_patches = len(r_list)
    print ('num_patches : ', num_patches)
    log.writelines('num_patches : ' + str(num_patches) + '\n')

    spacing = np.linspace(0, num_patches, num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    threads = []
    for thread_index in range(len(ranges)):
        args = (level0_img, filename, imgpath, thread_index, ranges, r_list, c_list, log)
        t = threading.Thread(target=divide_Normal_slide, args=args)
        t.setDaemon(True)
        threads.append(t)

    for t in threads:
        t.start()

    # Wait for all the threads to terminate.
    for t in threads:
        t.join()

    stop = time()
    print ('processing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')


def divide_Tumor_slide(level0_img, level0_mask, filename, imgpath, labelpath, thread_index, ranges, r_list, c_list,
                       log):
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        shard = s
        r = r_list[shard]
        c = c_list[shard]
        try:
            topy = r * 8 - patch_size / 2
            buttomy = topy+patch_size
            leftx = c * 8 - patch_size/ 2
            rightx = leftx+patch_size
            if topy < 0 or leftx < 0 or buttomy > level0_img.shape[0] or rightx > level0_img.shape[1]:
                print 'Out of Index: ('+ str(leftx) + ',' + str(topy) + ") for " + filename
                log.writelines('Out of Index: ('+ str(leftx) + ',' + str(topy) + ") for " + filename+'\n')
                continue
            imgarr = level0_img[topy:buttomy, leftx:rightx]
            image = Image.fromarray(imgarr)
            image = image.convert('RGB')
            mask = Image.fromarray(level0_mask[topy:buttomy, leftx:rightx])
            mask = mask.convert('L')
        except:
            print "Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename
            log.writelines("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename + '\n')
            continue
        else:
            imagename = os.path.join(imgpath, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
            image.save(imagename, "JPEG")
            maskname = os.path.join(labelpath, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.png')
            mask.save(maskname, 'PNG')

def process_tumor_tif(file, filename, maskpath, images, labels, log):
    start = time()
    imgpath = os.path.join(images, filename[:-4])
    labelpath = os.path.join(labels, filename[:-4])
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)
    else:
        return
    if not os.path.exists(labelpath):
        os.makedirs(labelpath)

    set_current_level = 3
    low_dim_array = open_slide(file, set_current_level)
    low_dim_img = Image.fromarray(low_dim_array)

    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()
    low_s_array = np.array(low_s)

    # --OSTU threshold
    low_s_thre = filters.threshold_otsu(low_s_array)
    low_s_bin = low_s_array > low_s_thre

    # divide low_s
    h = low_s.height
    w = low_s.width
    assert h > 500
    assert w > 500
    # h OR w = 500 + 450k + R
    h_k = (h - patch_size / 8) // ((patch_size - overlap_size) / 8)
    h_R = (h - patch_size / 8) % ((patch_size - overlap_size) / 8)
    if h_R <= overlap_size / 8:
        h_flag = 0
    else:
        h_flag = 1
    w_k = (w - patch_size / 8) // ((patch_size - overlap_size) / 8)
    w_R = (w - patch_size / 8) % ((patch_size - overlap_size) / 8)
    if w_R <= overlap_size / 8:
        w_flag = 0
    else:
        w_flag = 1

    h_list = []
    h_list.append(patch_size / 16)

    for i in range(1, h_k + 1):
        h_list.append(patch_size / 16 + ((patch_size - overlap_size) / 8) * i)

    if not h_flag:
        if h_k >= 1: h_list.remove(patch_size / 16 + ((patch_size - overlap_size) / 8) * h_k)
        h_list.append(h - patch_size / 16)
    else:
        h_list.append(h - patch_size / 16)

    w_list = []
    w_list.append(patch_size / 16)

    for j in range(1, w_k + 1):
        w_list.append(patch_size / 16 + ((patch_size - overlap_size) / 8) * j)

    if not w_flag:
        if w_k >= 1: w_list.remove(patch_size / 16 + ((patch_size - overlap_size) / 8) * w_k)
        w_list.append(w - patch_size / 16)
    else:
        w_list.append(w - patch_size / 16)

    [r_list, c_list] = np.meshgrid(h_list, w_list)
    r_list = r_list.flatten()
    c_list = c_list.flatten()
    tar_r_list = []
    tar_c_list = []

    for i in range(len(r_list)):
        r = r_list[i]
        c = c_list[i]
        topy = r - patch_size/16
        buttomy = r + patch_size/16
        leftx = c - patch_size/16
        rightx = c + patch_size/16
        low_patch = low_s_bin[topy:buttomy, leftx:rightx]
        if np.sum(low_patch):
            tar_r_list.append(r)
            tar_c_list.append(c)

    r_list = tar_r_list
    c_list = tar_c_list

    level0_img = open_slide(file, 0)
    maskfile = os.path.join(maskpath, filename[:-4]+'_Mask.tif')
    level0_mask = open_slide(maskfile, 0)
    assert level0_img.shape[0]==level0_mask.shape[0] and level0_img.shape[1]==level0_mask.shape[1]
    print('shape:', level0_img.shape)
    log.writelines('shape:'+str(level0_img.shape[0])+' '+str(level0_img.shape[1])+'\n')
    print (time() - start, 's')
    num_threads = 64
    num_patches = len(r_list)
    print ('num_patches : ', num_patches)
    log.writelines('num_patches : ' + str(num_patches) + '\n')

    spacing = np.linspace(0, num_patches, num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    threads = []
    for thread_index in range(len(ranges)):
        args = (level0_img, level0_mask, filename, imgpath, labelpath, thread_index, ranges, r_list, c_list, log)
        t = threading.Thread(target=divide_Tumor_slide, args=args)
        t.setDaemon(True)
        threads.append(t)

    for t in threads:
        t.start()

    # Wait for all the threads to terminate.
    for t in threads:
        t.join()

    stop = time()
    print ('processing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')


if __name__ == '__main__':
    # please check the following 2 parameters can be divided by 8
    # (It may be better that first parameter can be divided by 16).
    patch_size = 2048
    overlap_size = 192
    tifpath = '/disk8t-1/Xiangya2/test'
    maskpath = '/disk8t-1/Xiangya2/Mask_test'  # just for evaluation
    savepath = '/disk8t-1/deeplab-xiangya2/new_2048_test_stride_192_XY3c'
    logpath = '/disk8t-1/deeplab-xiangya2/logfiles/new_2048_test_stride_192_XY3c.log'

    images = os.path.join(savepath, 'images')
    labels = os.path.join(savepath, 'labels')

    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(labels):
        os.makedirs(labels)

    NormalMask = os.path.join(labels, 'All_Normal_Mask.png')
    if not os.path.exists(NormalMask):
        NMask = Image.new('L', (patch_size, patch_size))
        NMask.save(NormalMask, 'PNG')

    log = open(logpath, 'w')

    tiflist = os.listdir(tifpath)
    tiflist.sort()
    total_start = time()

    for filename in tiflist:
        if filename.split('.')[-1] != 'tif':
            continue
        file = os.path.join(tifpath, filename)
        print file
        log.write(file + '\n')
        maskfile = os.path.join(maskpath, filename.split('.')[0] + '_Mask.tif')
        if os.path.exists(maskfile):
            process_tumor_tif(file, filename, maskpath, images, labels, log)
        else:
            process_normal_tif(file, filename, images, log)

    total_stop = time()
    print "total processing time:", total_stop - total_start
    log.writelines("total processing time : " + str(total_stop - total_start) + '\n')
    log.close()

    TestPath = '/disk8t-1/deeplab-xiangya2/new_2048_test_stride_192_XY3c/images'
    TxtPath = '/disk8t-1/deeplab-xiangya2/xiangya-test-text/2048_s192'
    if not os.path.exists(TxtPath):
        os.makedirs(TxtPath)
    SlideList = os.listdir(TestPath)
    for Slide in SlideList:
        fname = Slide + '.txt'
        ftxt = os.path.join(TxtPath, fname)
        f = open(ftxt, 'w')
        SlideDir = os.path.join(TestPath, Slide)
        PatchList = os.listdir(SlideDir)
        for PatchName in PatchList:
            PatchFile = os.path.join(SlideDir, PatchName)
            f.write(PatchFile + '\n')
