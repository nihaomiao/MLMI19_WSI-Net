# Preprocessing WSI and Dividing it
# into 1280*1280 training patches with
# the stride of 640
# Author: Haomiao Ni
# For some reasons, we name In-Situ as DG, Invasive as JR.

import os
from libtiff import TIFF, TIFF3D, TIFFfile, TIFFimage
from skimage import filters, io, img_as_uint
import numpy as np
from time import time, localtime
from PIL import Image
import threading
from scipy.sparse import coo_matrix


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


def read_region(tiff, x, y, width, height, channel=3):
    tilew = tiff.GetField("TileWidth")
    tileh = tiff.GetField("TileLength")

    xoffset = x % tilew
    yoffset = y % tileh

    xtile_num = (width + xoffset) / tilew + 1
    ytile_num = (height + yoffset) / tileh + 1

    region = np.zeros((ytile_num * tileh, xtile_num * tilew, channel), dtype=np.uint8)
    if channel == 1:
        region = np.zeros((ytile_num * tileh, xtile_num * tilew), dtype=np.uint8)

    for i in range(0, xtile_num):
        for j in range(0, ytile_num):
            region[j * tileh:j * tileh + tileh, i * tilew:i * tilew + tilew] = tiff.read_one_tile(x + i * tilew,
                                                                                                  y + j * tileh)
    return region[yoffset:yoffset + height, xoffset:xoffset + width]


def divide_Normal_slide(level0_img, filename, images, thread_index, ranges, sparse_s_bin, log):
    Normaldir = os.path.join(images, 'Normal')
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        shard = s
        r = sparse_s_bin.row[shard]
        c = sparse_s_bin.col[shard]
        try:
            topy = r * 32 - 1280 / 2
            buttomy = topy + 1280
            leftx = c * 32 - 1280 / 2
            rightx = leftx + 1280
            if topy < 0 or leftx < 0 or buttomy > level0_img.shape[0] or rightx > level0_img.shape[1]:
                continue
            image = Image.fromarray(level0_img[topy:buttomy, leftx:rightx])
            image = image.convert('RGB')
        except:
            print "Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename
            log.writelines("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename + '\n')
            continue
        else:
            imagename = os.path.join(Normaldir, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
            image.save(imagename, "JPEG")


def process_normal_tif(file, filename, images, log):
    start = time()
    set_current_level = 5
    low_dim_img = Image.fromarray(open_slide(file, set_current_level))

    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()

    # --OSTU threshold
    low_s_thre = filters.threshold_otsu(np.array(low_s))
    low_s_bin = low_s > low_s_thre  # row is y and col is x

    del low_dim_img
    del low_hsv_img
    del low_s

    level0_img = open_slide(file, 0)
    print level0_img.shape

    sample_bin = np.zeros(low_s_bin.shape, dtype=np.int)
    for r in range(0, low_s_bin.shape[0], 20):
        for c in range(0, low_s_bin.shape[1], 20):
            if low_s_bin[r, c] != 0:
                sample_bin[r, c] = 1

    print(time() - start, 's')
    num_threads = 64
    num_patches = np.sum(sample_bin)
    sparse_s_bin = coo_matrix(sample_bin)
    assert num_patches == len(sparse_s_bin.data)
    print('num_patches : ', num_patches)
    log.writelines('num_patches : ' + str(num_patches) + '\n')

    spacing = np.linspace(0, len(sparse_s_bin.data), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    threads = []
    for thread_index in range(len(ranges)):
        args = (level0_img, filename, images, thread_index, ranges, sparse_s_bin, log)
        t = threading.Thread(target=divide_Normal_slide, args=args)
        t.setDaemon(True)
        threads.append(t)

    for t in threads:
        t.start()

    # Wait for all the threads to terminate.
    for t in threads:
        t.join()

    stop = time()
    print('processing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')


def divide_Tumor_slide(level0_img, level0_mask, filename, images, labels, thread_index, ranges, sparse_s_bin,
                       saveNormal, log):
    Normaldir = os.path.join(images, 'Normal')
    Tumordir = os.path.join(images, 'Tumor')
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        shard = s
        r = sparse_s_bin.row[shard]
        c = sparse_s_bin.col[shard]
        try:
            topy = r * 32 - 1280 / 2
            buttomy = topy + 1280
            leftx = c * 32 - 1280 / 2
            rightx = leftx + 1280
            if topy < 0 or leftx < 0 or buttomy > level0_img.shape[0] or rightx > level0_img.shape[1]:
                continue
            image = Image.fromarray(level0_img[topy:buttomy, leftx:rightx])
            image = image.convert('RGB')
            array_mask = level0_mask[topy:buttomy, leftx:rightx]
            if 100 in array_mask:
                continue
        except:
            print "Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename
            log.writelines("Can not read the point (" + str(leftx) + ',' + str(topy) + ") for " + filename + '\n')
            continue
        else:
            tumorid = np.argwhere(array_mask >= 130)
            IsTumor = (len(tumorid) > 0)
            if IsTumor:  # Tumor, need to save mask
                # rewrite the mask
                # Normal 0, In Situ/DG 127, Invasive/JR 254
                # Note that 'T-' Slide can't contain Normal pixels
                if filename[0] == 'T':
                    normal_bin = array_mask < 100
                    if np.sum(normal_bin) != 0:
                        continue
                    else:
                        JR_bin = array_mask >= 130
                        array_mask[JR_bin] = 254
                else:
                    normal_bin = array_mask < 100
                    DG_bin = np.logical_and(array_mask >= 130, array_mask < 230)
                    JR_bin = array_mask >= 230
                    array_mask[normal_bin] = 0
                    array_mask[DG_bin] = 127
                    array_mask[JR_bin] = 254
                imagename = os.path.join(Tumordir, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
                image.save(imagename, "JPEG")
                maskname = os.path.join(labels, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.png')
                mask = Image.fromarray(array_mask)
                mask = mask.convert('L')
                mask.save(maskname, 'PNG')
            else:  # normal
                if saveNormal == True:
                    imagename = os.path.join(Normaldir, filename[:-4] + '_' + str(leftx) + '_' + str(topy) + '.jpg')
                    image.save(imagename, "JPEG")


def process_tumor_tif(file, filename, maskpath, images, labels, Incomplete_slide, log):
    start = time()
    saveNormal = True
    if filename in Incomplete_slide:
        print 'should not save normal patch from ', filename
        log.writelines('should not save normal patch from ' + filename + '\n')
        saveNormal = False

    set_current_level = 5
    low_dim_img = Image.fromarray(open_slide(file, set_current_level))

    low_hsv_img = low_dim_img.convert('HSV')
    _, low_s, _ = low_hsv_img.split()

    # --OSTU threshold
    low_s_thre = filters.threshold_otsu(np.array(low_s))
    low_s_bin = low_s > low_s_thre  # row is y and col is x

    del low_dim_img
    del low_hsv_img
    del low_s

    level0_img = open_slide(file, 0)

    maskfile = os.path.join(maskpath, filename[:-4] + '_Mask.tif')
    level0_mask = open_slide(maskfile, 0)
    assert level0_img.shape[0] == level0_mask.shape[0] and level0_img.shape[1] == level0_mask.shape[1]

    sample_bin = np.zeros(low_s_bin.shape, dtype=np.int)
    for r in range(0, low_s_bin.shape[0], 20):
        for c in range(0, low_s_bin.shape[1], 20):
            if low_s_bin[r, c] != 0:
                sample_bin[r, c] = 1

    print(time() - start, 's')
    num_threads = 64
    num_patches = np.sum(sample_bin)
    sparse_s_bin = coo_matrix(sample_bin)
    assert num_patches == len(sparse_s_bin.data)
    print('num_patches : ', num_patches)
    log.writelines('num_patches : ' + str(num_patches) + '\n')

    spacing = np.linspace(0, len(sparse_s_bin.data), num_threads + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    threads = []
    for thread_index in range(len(ranges)):
        args = (level0_img, level0_mask, filename, images, labels, thread_index, ranges, sparse_s_bin, saveNormal, log)
        t = threading.Thread(target=divide_Tumor_slide, args=args)
        t.setDaemon(True)
        threads.append(t)

    for t in threads:
        t.start()

    # Wait for all the threads to terminate.
    for t in threads:
        t.join()

    stop = time()
    print('processing time : ' + str(stop - start))
    log.writelines('processing time : ' + str(stop - start) + '\n')


if __name__ == '__main__':
    tifpath = '/disk8t-1/Xiangya2/train'
    maskpath = '/disk8t-1/Xiangya2/Mask_train'
    savepath = '/disk8t-1/deeplab-xiangya2/1280_train_stride_640_XY3c'
    logpath = '/disk8t-1/deeplab-xiangya2/logfiles/1280_train_stride_640_XY3c.log'
    Incomplete_slide_txt = '/disk8t-1/Xiangya2/Incomplete_annotation_slide.txt'

    Incomplete_slide = []
    for name in open(Incomplete_slide_txt, 'r').readlines():
        Incomplete_slide.append(name.strip())

    images = os.path.join(savepath, 'images')
    labels = os.path.join(savepath, 'labels')

    if not os.path.exists(images):
        os.makedirs(images)
    if not os.path.exists(labels):
        os.makedirs(labels)

    Normaldir = os.path.join(images, 'Normal')
    if not os.path.exists(Normaldir):
        os.makedirs(Normaldir)
    Tumordir = os.path.join(images, 'Tumor')
    if not os.path.exists(Tumordir):
        os.makedirs(Tumordir)

    NormalMask = os.path.join(labels, 'All_Normal_Mask.png')
    if not os.path.exists(NormalMask):
        NMask = Image.new('L', (1280, 1280))
        NMask.save(NormalMask, 'PNG')

    log = open(logpath, 'w')

    tiflist = os.listdir(tifpath)
    total_start = time()

    for filename in tiflist:
        if not os.path.splitext(filename)[1] == '.tif':
            continue
        maskfile = os.path.join(maskpath, filename[:-4] + '_Mask.tif')
        if os.path.exists(maskfile):
            file = os.path.join(tifpath, filename)
            print('Tumor', file)
            process_tumor_tif(file, filename, maskpath, images, labels, Incomplete_slide, log)
        else:
            file = os.path.join(tifpath, filename)
            print('Normal', file)
            process_normal_tif(file, filename, images, log)

    total_stop = time()
    print "total processing time:", total_stop - total_start
    log.writelines("total processing time : " + str(total_stop - total_start) + '\n')
    log.close()

    Tumor = os.path.join(images, 'Tumor')
    tumorlist = os.listdir(Tumor)
    tumortxt = open(os.path.join(savepath, 'Tumor.txt'), 'w')
    for t in tumorlist:
        tumorname = os.path.join(Tumor, t)
        labelname = os.path.join(labels, t[:-4] + '.png')
        if os.path.exists(labelname):
            tumortxt.writelines(tumorname + ' ' + labelname + '\n')
    tumortxt.close()

    Normal = os.path.join(images, 'Normal')
    normallist = os.listdir(Normal)
    normaltxt = open(os.path.join(savepath, 'Normal.txt'), 'w')
    labelname = os.path.join(labels, 'All_Normal_Mask.png')
    for n in normallist:
        normalname = os.path.join(Normal, n)
        normaltxt.writelines(normalname + ' ' + labelname + '\n')
    normaltxt.close()

    # Combining Tumor.txt with Normal.txt
    TumorTxt = '/disk8t-1/deeplab-xiangya2/1280_train_stride_640_XY3c/Tumor.txt'
    NormalTxt = '/disk8t-1/deeplab-xiangya2/1280_train_stride_640_XY3c/Normal.txt'
    TrainTxt = '/disk8t-1/deeplab-xiangya2/1280_train_stride_640_XY3c/train.txt'

    TumorFile = open(TumorTxt, 'r')
    NormalFile = open(NormalTxt, 'r')
    TrainFile = open(TrainTxt, 'w')

    TumorLines = TumorFile.readlines()
    NormalLines = NormalFile.readlines()

    for line in TumorLines:
        TrainFile.write(line)
    for line in NormalLines:
        TrainFile.write(line)
