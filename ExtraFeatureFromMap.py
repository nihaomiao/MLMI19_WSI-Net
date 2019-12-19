# Extracting features from the WSI segmentation map
# Author: Haomiao Ni

import os
from scipy.sparse import load_npz
from skimage import measure
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import label_binarize
from libtiff import TIFF, TIFF3D, TIFFfile, TIFFimage
from PIL import Image

Image.MAX_IMAGE_PIXELS = 10 ** 9


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


def ExtractRegFea(heatmap, TumorArea, TotalArea, threshold):
    MapBin = heatmap == threshold
    if not np.any(MapBin):
        MapFea = [0, 0, 0, 0, 0, 0, 0, 0]
        return MapFea
    label = measure.label(np.array(MapBin), connectivity=2)

    region = measure.regionprops(label)
    Area = []
    AxisLength = []
    Extent = []
    Eccent = []
    MapFea = [0, 0, 0, 0, 0, 0, 0, 0]
    if region != []:
        for reg in region:
            Area.append(reg.area)
            AxisLength.append(reg.major_axis_length)
            Extent.append(reg.extent)
            Eccent.append(reg.eccentricity)

        # the major axis length, eccentricity, extent of the max tumor
        MaxAreaInd = np.argmax(Area)
        MaxTumorArea = Area[MaxAreaInd]
        MaxTumorEcc = Eccent[MaxAreaInd]
        MaxTumorExt = Extent[MaxAreaInd]

        MinArea = np.min(Area)
        AvgArea = np.average(Area)
        SumArea = np.sum(Area)
        TumorPro = SumArea / float(TumorArea)
        TotalPro = SumArea / float(TotalArea)

        MapFea = [MaxTumorArea, MaxTumorEcc, MaxTumorExt, MinArea, AvgArea, SumArea, TumorPro, TotalPro]

    return MapFea


def ExtractTrainFea(MapPath):
    TrainFea = []
    TrainLabel = []
    DirList = os.listdir(MapPath)
    for MapName in DirList:
        MapFile = os.path.join(MapPath, MapName)
        print MapFile
        heatmap = load_npz(MapFile).todense()

        MapLabel = MapName.split('_')[0]
        if MapLabel[0] == 'N':
            MapLabel = 0
        elif MapLabel[0] == 'J' or MapLabel[0] == 'T':
            MapLabel = 2
        else:
            MapLabel = 1

        TumorArea = np.sum(heatmap != 0)
        TotalArea = heatmap.shape[0] * heatmap.shape[1]
        TumorFea = float(TumorArea) / TotalArea
        JRRegFea = ExtractRegFea(heatmap, TumorArea, TotalArea, 254)
        DGRegFea = ExtractRegFea(heatmap, TumorArea, TotalArea, 127)
        MapFea = JRRegFea + DGRegFea
        MapFea.append(TumorFea)
        MapFea.append(TumorArea)
        assert len(MapFea) == 18
        TrainFea.append(MapFea)
        TrainLabel.append(MapLabel)

    TrainFea = np.vstack(TrainFea)
    TrainLabel = np.hstack(TrainLabel)
    return TrainFea, TrainLabel


def ExtractTestFea(MapPath):
    TrainFea = []
    TrainLabel = []
    DirList = os.listdir(MapPath)
    assert len(DirList) == MapLen
    for MapName in DirList:
        MapFile = os.path.join(MapPath, MapName)
        print MapFile
        heatmap = load_npz(MapFile).todense()

        MapLabel = MapName.split('_')[0]
        if MapLabel[0] == 'N':
            MapLabel = 0
        elif MapLabel[0] == 'J' or MapLabel[0] == 'T':
            MapLabel = 2
        else:
            MapLabel = 1

        TumorArea = np.sum(heatmap != 0)
        TotalArea = heatmap.shape[0] * heatmap.shape[1]
        TumorFea = float(TumorArea) / TotalArea
        JRRegFea = ExtractRegFea(heatmap, TumorArea, TotalArea, 254)
        DGRegFea = ExtractRegFea(heatmap, TumorArea, TotalArea, 127)
        MapFea = JRRegFea + DGRegFea
        MapFea.append(TumorFea)
        MapFea.append(TumorArea)
        assert len(MapFea) == 18
        TrainFea.append(MapFea)
        TrainLabel.append(MapLabel)

    TrainFea = np.vstack(TrainFea)
    TrainLabel = np.hstack(TrainLabel)
    return TrainFea, TrainLabel


def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print "confusion_matrix(left labels: y_true, up labels: y_pred):"
    print "labels\t",
    for i in range(len(labels)):
        print labels[i], "\t",
    print
    for i in range(len(conf_mat)):
        print i, "\t",
        for j in range(len(conf_mat[i])):
            print conf_mat[i][j], '\t',
        print
    print


if __name__ == '__main__':
    TrainMapPath = '/disk8t-1/deeplab-xiangya2/xiangya-train-npz/whole_npz/Xiangya_Deeplab_B2_S555000_Frozen_BN_test2048'
    TrainMaskPath = '/disk8t-1/XiangYa2/Mask_train'

    # extracting training features
    TrainFea, TrainLabel = ExtractTrainFea(TrainMapPath)

    # training classifier
    clf = RandomForestClassifier(random_state=0, n_estimators=25)
    print clf

    clf.fit(TrainFea, TrainLabel)
    TrainRes = clf.predict(TrainFea)
    TrainAcc = np.sum(TrainRes == TrainLabel, dtype=np.float) / len(TrainRes)
    print TrainAcc

    # testing
    SrcPath = '/disk8t-1/deeplab-xiangya2-IC/xiangya-test-res/whole_npz'
    model_id = 'Xiangya_Deeplab_B4_S28000_IC_test2048_0.5'
    MapLen = 100
    TestMapPath = '/disk8t-1/deeplab-xiangya2-IC/xiangya-test-res/whole_npz/Xiangya_Deeplab_B4_S28000_IC_test2048_wo_pl_0.5'
    TestFea, TestLabel = ExtractTrainFea(TestMapPath)
    TestRes = clf.predict(TestFea)
    TestProb = clf.predict_proba(TestFea)
    for i in range(len(TestProb)):
        if TestProb[i, 1] == TestProb[i, 0] and TestProb[i, 1] == np.max(TestProb[i]):
            TestRes[i] = 1
        if TestProb[i, 2] == TestProb[i, 1] and TestProb[i, 1] == np.max(TestProb[i]):
            TestRes[i] = 2
        if TestProb[i, 2] == TestProb[i, 0] and TestProb[i, 2] == np.max(TestProb[i]):
            TestRes[i] = 2
    TestAcc = np.sum(TestRes == TestLabel, dtype=np.float) / len(TestRes)
    print "TestAcc:", TestAcc

    my_confusion_matrix(TestLabel, TestRes)
