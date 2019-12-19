# Merging the segmentation results of each patch
# and outputing the entire segmentation map for WSI
# Author: Haomiao Ni
# Created on 10/30/2018
# Note that You can merge and segment patches at the same time
from __future__ import print_function
from datetime import datetime
import os
import sys
from PIL import Image
from libtiff import TIFF
from time import time, sleep
import matplotlib.pyplot as plt
import numpy as np
import openslide
import threading
from scipy.sparse import *

mutex = threading.Lock()
coomats = []
jpgdict = {}


def merge_npz_thread(dirpath, npzfiles, thread_index, ranges, width, height):
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        npz = npzfiles[s]
        if npz.split('.')[-1] != 'npz':
            continue
        lefttopx = int(int(npz.split('_')[-4]) / 8)
        lefttopy = int(int(npz.split('_')[-3]) / 8)
        npzfile = load_npz(os.path.join(dirpath, npz))
        npzfile = npzfile.todense()
        npzfile = coo_matrix(npzfile)
        for i in range(len(npzfile.data)):
            npzfile.row[i] = lefttopy + npzfile.row[i]
            npzfile.col[i] = lefttopx + npzfile.col[i]
            if npzfile.row[i] >= height / 8 or npzfile.col[i] >= width / 8:
                npzfile.row[i] = 0
                npzfile.col[i] = 0
                npzfile.data[i] = 0
        npzfile = coo_matrix((npzfile.data, (npzfile.row, npzfile.col)), shape=(height / 8, width / 8))
        global coomats, mutex
        mutex.acquire()
        coomats.append(npzfile)

        mutex.release()
        print('finish ' + npz + ' \t' + str(s) + '/' + str(len(npzfiles)))
        del npzfile


def merge_npz(npzpath, dir, npzname, width, height):
    dirpath = os.path.join(npzpath, dir)
    npzfileslist = [os.listdir(dirpath)]
    num_npz = len(npzfileslist[0])
    segment = 1
    if num_npz > 1000:
        segment = 8
        l = (num_npz / segment)
        npzfileslist = [npzfileslist[0][l * i:min(num_npz, l * (i + 1))] for i in range(segment)]

    dokmat = dok_matrix((height / 8, width / 8))
    dokdict = dok_matrix((height / 8, width / 8))
    for seg in range(segment):
        npzfiles = npzfileslist[seg]
        num_threads = len(npzfiles)
        spacing = np.linspace(0, len(npzfiles), num_threads + 1).astype(np.int)
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])
        threads = []
        global coomats
        coomats = []
        for thread_index in range(len(ranges)):
            args = (dirpath, npzfiles, thread_index, ranges, width, height)
            t = threading.Thread(target=merge_npz_thread, args=args)
            t.setDaemon(True)
            threads.append(t)

        for t in threads:
            t.start()

        # Wait for all the threads to terminate.
        for t in threads:
            t.join()

        for thread_index in range(len(ranges)):
            print('thread_index = ', thread_index, len(ranges))
            for i in range(len(coomats[thread_index].data)):
                dokmat[coomats[thread_index].row[i], coomats[thread_index].col[i]] += coomats[thread_index].data[i]
                dokdict[coomats[thread_index].row[i], coomats[thread_index].col[i]] += 1
        coomats = []

    coomat = coo_matrix(dokmat)
    coodict = dokdict

    del dokmat, dokdict
    for i in range(len(coomat.data)):
        r = coomat.row[i]
        c = coomat.col[i]
        if coodict[r, c] > 1:
            assert coomat.data[i] % 127 == 0
            _254_num = (coomat.data[i] - 127 * coodict[r, c]) / 127
            _127_num = coodict[r, c] - _254_num
            assert _254_num >= 0
            assert _127_num >= 0
            if _254_num >= _127_num:
                coomat.data[i] = 254
            elif _254_num < _127_num:
                coomat.data[i] = 127
        else:
            assert coomat.data[i] <= 254

    print('saving ' + npzname)
    save_npz(npzname, coomat.tocsr())


def merge_mul_npz_fun(srcpath, path, model_id, logfile):
    # start = time()
    npzpath = os.path.join(path, model_id)
    savenpz = os.path.join(path, 'whole_npz/' + model_id + '_debug')
    if not os.path.exists(savenpz):
        os.makedirs(savenpz)
    if not os.path.exists(npzpath):
        return
    listdir = os.listdir(npzpath)
    listdir.sort()
    if len(listdir) != dirlen:
        listdir = listdir[:-1]
    if len(listdir) == 0:
        return
    cnt = 0
    for dir in listdir:
        # dir = 'GDG-2018-04-10_18_59_15'
        npzname = os.path.join(savenpz, dir + '_Map.npz')
        if os.path.exists(npzname):
            cnt += 1
            continue
        print(npzname)
        logfile.writelines(npzname)
        slide = openslide.open_slide(os.path.join(srcpath, dir + '.tif'))
        width, height = slide.level_dimensions[0]

        merge_npz(npzpath, dir, npzname, width, height)
    if cnt == dirlen:
        return True


if __name__ == '__main__':
    dirlen = 100 # the total number of testing dataset
    patchsource = 'new_2048_test_stride_192_XY3c'
    jpgpath = '/disk8t-1/deeplab-xiangya2/' + patchsource + '/images'
    jpgdir = os.listdir(jpgpath)
    jpgdir.sort()
    for dir in jpgdir:
        jpgnum = len(os.listdir(os.path.join(jpgpath, dir)))
        jpgdict[dir] = jpgnum

    srcpath = '/disk8t-1/Xiangya2/test'
    path = '/disk8t-1/unet-xiangya2/xiangya-test-npz'

    model_id = 'Xiangya_Deeplab_B4_S280000_Frozen_BN_test2048'
    logfile = open('/home/nihaomiao/PycharmProjects/research/deeplabForXiangya2/log/merge_npz_' + model_id + '.log',
                   'w')
    while True:
        final = merge_mul_npz_fun(srcpath, path, model_id, logfile)
        print('sleep 300s')
        if not final:
            sleep(300)
        else:
            break

    # merge_mul_npz_fun(srcpath, path, model_id, logfile)
