# Using bootstrap to evaluate metrics
# Author: Haomiao Ni
import os
from libtiff import TIFF
import numpy as np
from scipy.sparse import load_npz
import glob
from scipy.misc import imread, imresize

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

def Metrics(MaskPath, MatPath, MatList, BootstrapName):
    n_00_list = []
    n_01_list = []
    n_02_list = []
    n_10_list = []
    n_11_list = []
    n_12_list = []
    n_20_list = []
    n_21_list = []
    n_22_list = []
    tt_0_list = []
    tt_1_list = []
    tt_2_list = []

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

            n_00 = np.sum(np.logical_and(Normal_out, Normal_tar))
            n_00_list.append(n_00)
            N_00 += np.sum(np.logical_and(Normal_out, Normal_tar))

            n_01 = np.sum(np.logical_and(DG_out, Normal_tar))
            n_01_list.append(n_01)
            N_01 += np.sum(np.logical_and(DG_out, Normal_tar))

            n_02 = np.sum(np.logical_and(JR_out, Normal_tar))
            n_02_list.append(n_02)
            N_02 += np.sum(np.logical_and(JR_out, Normal_tar))

            n_10 = np.sum(np.logical_and(Normal_out, DG_tar))
            n_10_list.append(n_10)
            N_10 += np.sum(np.logical_and(Normal_out, DG_tar))

            n_11 = np.sum(np.logical_and(DG_out, DG_tar))
            n_11_list.append(n_11)
            N_11 += np.sum(np.logical_and(DG_out, DG_tar))

            n_12 = np.sum(np.logical_and(JR_out, DG_tar))
            n_12_list.append(n_12)
            N_12 += np.sum(np.logical_and(JR_out, DG_tar))

            n_20 = np.sum(np.logical_and(Normal_out, JR_tar))
            n_20_list.append(n_20)
            N_20 += np.sum(np.logical_and(Normal_out, JR_tar))

            n_21 = np.sum(np.logical_and(DG_out, JR_tar))
            n_21_list.append(n_21)
            N_21 += np.sum(np.logical_and(DG_out, JR_tar))

            n_22 = np.sum(np.logical_and(JR_out, JR_tar))
            n_22_list.append(n_22)
            N_22 += np.sum(np.logical_and(JR_out, JR_tar))

            tt_0 = np.sum(Normal_tar, dtype=np.float32)
            tt_0_list.append(tt_0)
            t_0 += np.sum(Normal_tar, dtype=np.float32)

            tt_1 = np.sum(DG_tar, dtype=np.float32)
            tt_1_list.append(tt_1)
            t_1 += np.sum(DG_tar, dtype=np.float32)

            tt_2 = np.sum(JR_tar, dtype=np.float32)
            tt_2_list.append(tt_2)
            t_2 += np.sum(JR_tar, dtype=np.float32)

    mIoU = (1/3.0)*(N_00/(t_0+N_10+N_20)+N_11/(t_1+N_01+N_21)+N_22/(t_2+N_02+N_12))
    tumor_mIoU = (1/2.0)*(N_11/(t_1+N_01+N_21)+N_22/(t_2+N_02+N_12))
    DG_mIoU = N_11/(t_1+N_01+N_21)
    JR_mIoU = N_22/(t_2+N_02+N_12)

    print MatPath.split('/')[-1]
    print "mIoU:{0:.4f}".format(mIoU)
    print "C_mIoU:{0:.4f}".format(tumor_mIoU)
    print "D_mIoU:{0:.4f}".format(DG_mIoU)
    print "I_mIoU:{0:.4f}".format(JR_mIoU)
    np.save(os.path.join(BootstrapName, 'n_00.npy'), n_00_list)
    np.save(os.path.join(BootstrapName, 'n_01.npy'), n_01_list)
    np.save(os.path.join(BootstrapName, 'n_02.npy'), n_02_list)
    np.save(os.path.join(BootstrapName, 'n_10.npy'), n_10_list)
    np.save(os.path.join(BootstrapName, 'n_11.npy'), n_11_list)
    np.save(os.path.join(BootstrapName, 'n_12.npy'), n_12_list)
    np.save(os.path.join(BootstrapName, 'n_20.npy'), n_20_list)
    np.save(os.path.join(BootstrapName, 'n_21.npy'), n_21_list)
    np.save(os.path.join(BootstrapName, 'n_22.npy'), n_22_list)
    np.save(os.path.join(BootstrapName, 't_0.npy'), tt_0_list)
    np.save(os.path.join(BootstrapName, 't_1.npy'), tt_1_list)
    np.save(os.path.join(BootstrapName, 't_2.npy'), tt_2_list)

# Evaluating the performance of inception model
def InceptionMetrics(MaskPath, MatPath, MatList, BootstrapName):
    n_00_list = []
    n_01_list = []
    n_02_list = []
    n_10_list = []
    n_11_list = []
    n_12_list = []
    n_20_list = []
    n_21_list = []
    n_22_list = []
    tt_0_list = []
    tt_1_list = []
    tt_2_list = []

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
    for MatName in MatList:
        if MatName[-4:] == '.png':
            print MatName
            MatFile = os.path.join(MatPath, MatName)
            PredMsk = imread(MatFile)
            postfix = '_' + MatName.split('_')[-1]
            MaskName = MatName.replace(postfix, '_Mask.tif')
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
            if PredMsk.shape != low_dim_array.shape:
                PredMsk = imresize(PredMsk, low_dim_array.shape)

            # calculate Pixel Accuracy
            Normal_out = PredMsk == 0
            Normal_tar = low_dim_array == 0
            DG_out = np.logical_or(PredMsk == 127, PredMsk == 128)
            DG_tar = low_dim_array == 127
            JR_out = np.logical_or(PredMsk == 254, PredMsk == 255)
            JR_tar = low_dim_array == 254

            n_00 = np.sum(np.logical_and(Normal_out, Normal_tar))
            n_00_list.append(n_00)
            N_00 += np.sum(np.logical_and(Normal_out, Normal_tar))

            n_01 = np.sum(np.logical_and(DG_out, Normal_tar))
            n_01_list.append(n_01)
            N_01 += np.sum(np.logical_and(DG_out, Normal_tar))

            n_02 = np.sum(np.logical_and(JR_out, Normal_tar))
            n_02_list.append(n_02)
            N_02 += np.sum(np.logical_and(JR_out, Normal_tar))

            n_10 = np.sum(np.logical_and(Normal_out, DG_tar))
            n_10_list.append(n_10)
            N_10 += np.sum(np.logical_and(Normal_out, DG_tar))

            n_11 = np.sum(np.logical_and(DG_out, DG_tar))
            n_11_list.append(n_11)
            N_11 += np.sum(np.logical_and(DG_out, DG_tar))

            n_12 = np.sum(np.logical_and(JR_out, DG_tar))
            n_12_list.append(n_12)
            N_12 += np.sum(np.logical_and(JR_out, DG_tar))

            n_20 = np.sum(np.logical_and(Normal_out, JR_tar))
            n_20_list.append(n_20)
            N_20 += np.sum(np.logical_and(Normal_out, JR_tar))

            n_21 = np.sum(np.logical_and(DG_out, JR_tar))
            n_21_list.append(n_21)
            N_21 += np.sum(np.logical_and(DG_out, JR_tar))

            n_22 = np.sum(np.logical_and(JR_out, JR_tar))
            n_22_list.append(n_22)
            N_22 += np.sum(np.logical_and(JR_out, JR_tar))

            tt_0 = np.sum(Normal_tar, dtype=np.float32)
            tt_0_list.append(tt_0)
            t_0 += np.sum(Normal_tar, dtype=np.float32)

            tt_1 = np.sum(DG_tar, dtype=np.float32)
            tt_1_list.append(tt_1)
            t_1 += np.sum(DG_tar, dtype=np.float32)

            tt_2 = np.sum(JR_tar, dtype=np.float32)
            tt_2_list.append(tt_2)
            t_2 += np.sum(JR_tar, dtype=np.float32)

    mIoU = (1/3.0)*(N_00/(t_0+N_10+N_20)+N_11/(t_1+N_01+N_21)+N_22/(t_2+N_02+N_12))
    tumor_mIoU = (1/2.0)*(N_11/(t_1+N_01+N_21)+N_22/(t_2+N_02+N_12))
    DG_mIoU = N_11/(t_1+N_01+N_21)
    JR_mIoU = N_22/(t_2+N_02+N_12)

    print MatPath.split('/')[-1]
    print "mIoU:{0:.4f}".format(mIoU)
    print "C_mIoU:{0:.4f}".format(tumor_mIoU)
    print "D_mIoU:{0:.4f}".format(DG_mIoU)
    print "I_mIoU:{0:.4f}".format(JR_mIoU)
    np.save(os.path.join(BootstrapName, 'n_00.npy'), n_00_list)
    np.save(os.path.join(BootstrapName, 'n_01.npy'), n_01_list)
    np.save(os.path.join(BootstrapName, 'n_02.npy'), n_02_list)
    np.save(os.path.join(BootstrapName, 'n_10.npy'), n_10_list)
    np.save(os.path.join(BootstrapName, 'n_11.npy'), n_11_list)
    np.save(os.path.join(BootstrapName, 'n_12.npy'), n_12_list)
    np.save(os.path.join(BootstrapName, 'n_20.npy'), n_20_list)
    np.save(os.path.join(BootstrapName, 'n_21.npy'), n_21_list)
    np.save(os.path.join(BootstrapName, 'n_22.npy'), n_22_list)
    np.save(os.path.join(BootstrapName, 't_0.npy'), tt_0_list)
    np.save(os.path.join(BootstrapName, 't_1.npy'), tt_1_list)
    np.save(os.path.join(BootstrapName, 't_2.npy'), tt_2_list)

def BootstrapMetrics(BootstrapDir, BootstrapNum, MatSize):
    BootstrapDirList = os.listdir(BootstrapDir)
    BootstrapDirList.sort()
    for BootstrapDirName in BootstrapDirList:
        print BootstrapDirName
        BootstrapName = os.path.join(BootstrapDir, BootstrapDirName)

        n_00_list = np.load(os.path.join(BootstrapName, 'n_00.npy'))
        n_01_list = np.load(os.path.join(BootstrapName, 'n_01.npy'))
        n_02_list = np.load(os.path.join(BootstrapName, 'n_02.npy'))
        n_10_list = np.load(os.path.join(BootstrapName, 'n_10.npy'))
        n_11_list = np.load(os.path.join(BootstrapName, 'n_11.npy'))
        n_12_list = np.load(os.path.join(BootstrapName, 'n_12.npy'))
        n_20_list = np.load(os.path.join(BootstrapName, 'n_20.npy'))
        n_21_list = np.load(os.path.join(BootstrapName, 'n_21.npy'))
        n_22_list = np.load(os.path.join(BootstrapName, 'n_22.npy'))
        tt_0_list = np.load(os.path.join(BootstrapName, 't_0.npy'))
        tt_1_list = np.load(os.path.join(BootstrapName, 't_1.npy'))
        tt_2_list = np.load(os.path.join(BootstrapName, 't_2.npy'))

        mIoU = np.zeros(BootstrapNum)
        C_mIoU = np.zeros(BootstrapNum)
        D_mIoU = np.zeros(BootstrapNum)
        I_mIoU = np.zeros(BootstrapNum)

        for j in range(BootstrapNum):
            temp_n_00_list = np.zeros(MatSize)
            temp_n_01_list = np.zeros(MatSize)
            temp_n_02_list = np.zeros(MatSize)
            temp_n_10_list = np.zeros(MatSize)
            temp_n_11_list = np.zeros(MatSize)
            temp_n_12_list = np.zeros(MatSize)
            temp_n_20_list = np.zeros(MatSize)
            temp_n_21_list = np.zeros(MatSize)
            temp_n_22_list = np.zeros(MatSize)
            temp_tt_0_list = np.zeros(MatSize)
            temp_tt_1_list = np.zeros(MatSize)
            temp_tt_2_list = np.zeros(MatSize)

            for i in range(MatSize):
                tar = np.random.randint(0, MatSize)
                temp_n_00_list[i] = n_00_list[tar]
                temp_n_01_list[i] = n_01_list[tar]
                temp_n_02_list[i] = n_02_list[tar]
                temp_n_10_list[i] = n_10_list[tar]
                temp_n_11_list[i] = n_11_list[tar]
                temp_n_12_list[i] = n_12_list[tar]
                temp_n_20_list[i] = n_20_list[tar]
                temp_n_21_list[i] = n_21_list[tar]
                temp_n_22_list[i] = n_22_list[tar]
                temp_tt_0_list[i] = tt_0_list[tar]
                temp_tt_1_list[i] = tt_1_list[tar]
                temp_tt_2_list[i] = tt_2_list[tar]

            N_00 = np.sum(temp_n_00_list)
            N_01 = np.sum(temp_n_01_list)
            N_02 = np.sum(temp_n_02_list)
            N_10 = np.sum(temp_n_10_list)
            N_11 = np.sum(temp_n_11_list)
            N_12 = np.sum(temp_n_12_list)
            N_20 = np.sum(temp_n_20_list)
            N_21 = np.sum(temp_n_21_list)
            N_22 = np.sum(temp_n_22_list)
            t_0 = np.sum(temp_tt_0_list)
            t_1 = np.sum(temp_tt_1_list)
            t_2 = np.sum(temp_tt_2_list)

            mIoU[j] = (1 / 3.0) * (N_00 / (t_0 + N_10 + N_20) + N_11 / (t_1 + N_01 + N_21) + N_22 / (t_2 + N_02 + N_12))
            C_mIoU[j] = (1 / 2.0) * (N_11 / (t_1 + N_01 + N_21) + N_22 / (t_2 + N_02 + N_12))
            D_mIoU[j] = N_11 / (t_1 + N_01 + N_21)
            I_mIoU[j] = N_22 / (t_2 + N_02 + N_12)
        print np.percentile(mIoU, [2.5, 97.5])
        print np.percentile(C_mIoU, [2.5, 97.5])
        print np.percentile(D_mIoU, [2.5, 97.5])
        print np.percentile(I_mIoU, [2.5, 97.5])

if __name__ == '__main__':
    # set_current_level = 3  # 5 for 10X resolution
    MatSize = 100
    MaskPath = '/disk8t-1/Xiangya2/Mask_test'
    BootstrapDir = '/home/nihaomiao/PycharmProjects/research/deeplabForXiangya2/ICME/BootstrapDir'

    BootstrapMetrics(BootstrapDir, 2000, MatSize)


