# Metrics for evaluating our algorithms
# Author: Haomiao Ni
# We design some metrics with the reference
# to Section 5 in paper FCN
import os
from libtiff import TIFF
import numpy as np
from scipy.sparse import load_npz
import glob

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

    MatList = os.listdir(MatPath)
    Normal_IoU_Sum = 0.0
    NCnt = 0.0
    DG_IoU_Sum = 0.0
    DCnt = 0.0
    JR_IoU_Sum = 0.0
    JCnt = 0.0
    for MatName in MatList:
        if MatName[-4:] == '.npz':
            print MatName
            MatFile = os.path.join(MatPath, MatName)
            PredMsk = load_npz(MatFile)
            PredMsk = PredMsk.todense()
            MaskName = MatName.replace('_Map.npz', '_Mask.tif')
            MaskFile = os.path.join(MaskPath, MaskName)
            if os.path.exists(MaskFile):
                low_dim_array = open_slide(MaskFile, set_current_level)
                normal_bin = low_dim_array < 100
                low_dim_array[normal_bin] = 0
                if MaskName[0] == 'T':
                    JR_bin = low_dim_array>= 130
                    low_dim_array[JR_bin] = 254
                else:
                    DG_bin = np.logical_and(low_dim_array >= 130, low_dim_array < 230)
                    JR_bin = low_dim_array >= 230
                    low_dim_array[normal_bin] = 0
                    low_dim_array[DG_bin] = 127
                    low_dim_array[JR_bin] = 254
            else:
                low_dim_array = np.zeros(PredMsk.shape)
            # calculate Pixel Accuracy
            Normal_out = PredMsk==0
            Normal_tar = low_dim_array==0
            DG_out = PredMsk==127
            DG_tar = low_dim_array==127
            JR_out = PredMsk==254
            JR_tar = low_dim_array==254
            Normal_IoU = np.sum(np.logical_and(Normal_out, Normal_tar), dtype=np.float32)\
                         /np.sum(np.logical_or(Normal_out, Normal_tar))
            DG_sum = np.sum(np.logical_or(DG_out, DG_tar))
            DG_IoU = 0
            if DG_sum:
                DG_IoU = np.sum(np.logical_and(DG_out, DG_tar), dtype=np.float32) \
                         / DG_sum
            JR_sum = np.sum(np.logical_or(JR_out, JR_tar))
            JR_IoU = 0
            if JR_sum:
                JR_IoU = np.sum(np.logical_and(JR_out, JR_tar), dtype=np.float32) \
                         / JR_sum
            print Normal_IoU, DG_IoU, JR_IoU
            if MatName[0] == 'N':
                Normal_IoU_Sum += Normal_IoU
                NCnt += 1
            elif MatName[0] == 'J' or MatName[0] == 'T':
                JR_IoU_Sum += JR_IoU
                JCnt += 1
            else:
                DG_IoU_Sum += DG_IoU
                DCnt += 1
    print "total:", Normal_IoU_Sum/NCnt, DG_IoU_Sum/DCnt, JR_IoU_Sum/JCnt

def Metrics(MaskPath, MatPath):
    MatList = os.listdir(MatPath)
    N_00 = 0.0
    N_01 = 0.0
    N_02 = 0.0
    N_10 = 0.0
    N_11 = 0.0
    N_12 = 0.0
    N_20 = 0.0
    N_21 = 0.0
    N_22 = 0.0
    t_0 = 0.0
    t_1 = 0.0
    t_2 = 0.0
    assert len(MatList) == MatSize
    for MatName in MatList:
        if MatName[-4:] == '.npz':
            print MatName
            MatFile = os.path.join(MatPath, MatName)
            PredMsk = load_npz(MatFile)
            PredMsk = PredMsk.todense()
            MaskName = MatName.replace('_Map.npz', '_Mask.tif')
            MaskFile = os.path.join(MaskPath, MaskName)
            if os.path.exists(MaskFile):
                low_dim_array = open_slide(MaskFile, set_current_level)
                normal_bin = low_dim_array < 100
                low_dim_array[normal_bin] = 0
                if MaskName[0] == 'T':
                    JR_bin = low_dim_array >= 130
                    low_dim_array[JR_bin] = 254
                else:
                    DG_bin = np.logical_and(low_dim_array >= 130, low_dim_array < 230)
                    JR_bin = low_dim_array >= 230
                    low_dim_array[normal_bin] = 0
                    low_dim_array[DG_bin] = 127
                    low_dim_array[JR_bin] = 254
            else:
                low_dim_array = np.zeros(PredMsk.shape)
            # calculate Pixel Accuracy
            Normal_out = PredMsk == 0
            Normal_tar = low_dim_array == 0
            DG_out = PredMsk == 127
            DG_tar = low_dim_array == 127
            JR_out = PredMsk == 254
            JR_tar = low_dim_array == 254

            N_00 += np.sum(np.logical_and(Normal_out, Normal_tar))
            N_01 += np.sum(np.logical_and(DG_out, Normal_tar))
            N_02 += np.sum(np.logical_and(JR_out, Normal_tar))

            N_10 += np.sum(np.logical_and(Normal_out, DG_tar))
            N_11 += np.sum(np.logical_and(DG_out, DG_tar))
            N_12 += np.sum(np.logical_and(JR_out, DG_tar))

            N_20 += np.sum(np.logical_and(Normal_out, JR_tar))
            N_21 += np.sum(np.logical_and(DG_out, JR_tar))
            N_22 += np.sum(np.logical_and(JR_out, JR_tar))

            t_0 += np.sum(Normal_tar, dtype=np.float32)
            t_1 += np.sum(DG_tar, dtype=np.float32)
            t_2 += np.sum(JR_tar, dtype=np.float32)

    pixel_acc = (N_00+N_11+N_22)/(t_0+t_1+t_2)
    tumor_pixel_acc = (N_11+N_22)/(t_1+t_2)
    mean_acc = (N_00/t_0+N_11/t_1+N_22/t_2)/3.0
    mean_tumor_acc = (N_11/t_1+N_22/t_2)/2.0
    DG_pixel_acc = N_11/t_1
    JR_pixel_acc = N_22/t_2

    mIoU = (1/3.0)*(N_00/(t_0+N_10+N_20)+N_11/(t_1+N_01+N_21)+N_22/(t_2+N_02+N_12))
    tumor_mIoU = (1/2.0)*(N_11/(t_1+N_01+N_21)+N_22/(t_2+N_02+N_12))
    fw_mIoU = (1/(t_0+t_1+t_2))*(t_0*N_00/(t_0+N_10+N_20)+t_1*N_11/(t_1+N_01+N_21)+
                                 t_2*N_22/(t_2+N_02+N_12))
    DG_mIoU = N_11/(t_1+N_01+N_21)
    JR_mIoU = N_22/(t_2+N_02+N_12)

    print MatPath.split('/')[-1]
    print "pixel_acc:{0:.4f}".format(pixel_acc)
    print "tumor_pixel_acc:{0:.4f}".format(tumor_pixel_acc)
    print "mean_acc:{0:.4f}".format(mean_acc)
    print "mean_tumor_acc:{0:.4f}".format(mean_tumor_acc)
    print "DG_pixel_acc:{0:.4f}".format(DG_pixel_acc)
    print "JR_pixel_acc:{0:.4f}".format(JR_pixel_acc)
    print "mIoU:{0:.4f}".format(mIoU)
    print "tumor_mIoU:{0:.4f}".format(tumor_mIoU)
    print "fw_mIoU:{0:.4f}".format(fw_mIoU)
    print "DG_mIoU:{0:.4f}".format(DG_mIoU)
    print "JR_mIoU:{0:.4f}".format(JR_mIoU)

    print MatPath.split('/')[-1]
    print "{0:.4f}".format(pixel_acc)
    print "{0:.4f}".format(tumor_pixel_acc)
    print "{0:.4f}".format(mean_acc)
    print "{0:.4f}".format(mean_tumor_acc)
    print "{0:.4f}".format(DG_pixel_acc)
    print "{0:.4f}".format(JR_pixel_acc)
    print "{0:.4f}".format(mIoU)
    print "{0:.4f}".format(tumor_mIoU)
    print "{0:.4f}".format(fw_mIoU)
    print "{0:.4f}".format(DG_mIoU)
    print "{0:.4f}".format(JR_mIoU)

    MatList = os.listdir(MatPath)
    TFP = 0.0
    TT = 0.0
    assert len(MatList) == MatSize
    for MatName in MatList:
        if MatName[-4:] == '.npz':
            # print MatName
            MatFile = os.path.join(MatPath, MatName)
            PredMsk = load_npz(MatFile)
            PredMsk = PredMsk.todense()
            MaskName = MatName.replace('_Map.npz', '_Mask.tif')
            MaskFile = os.path.join(MaskPath, MaskName)
            if os.path.exists(MaskFile):
                low_dim_array = open_slide(MaskFile, set_current_level)
                normal_bin = low_dim_array < 100
                low_dim_array[normal_bin] = 0
                if MaskName[0] == 'T':
                    JR_bin = low_dim_array >= 130
                    low_dim_array[JR_bin] = 254
                else:
                    DG_bin = np.logical_and(low_dim_array >= 130, low_dim_array < 230)
                    JR_bin = low_dim_array >= 230
                    low_dim_array[normal_bin] = 0
                    low_dim_array[DG_bin] = 127
                    low_dim_array[JR_bin] = 254
            else:
                low_dim_array = np.zeros(PredMsk.shape)

            tumor_mask = low_dim_array!=0
            tumor_pred = PredMsk!=0
            FP = np.sum(np.logical_and(~tumor_mask, tumor_pred))
            T = tumor_pred.size
            TFP += FP
            TT += T
    print TFP/TT

if __name__ == '__main__':
    set_current_level = 3  # 5 for 10X resolution
    MatSize = 100
    MaskPath = '/disk8t-1/Xiangya2/Mask_test'
    MatPath = '/disk8t-1/deeplab-xiangya2-v2/xiangya-test-npz/whole_npz/Xiangya_Deeplab_B3_S370000_Frozen_BN_test2048'
    Metrics(MaskPath, MatPath)







