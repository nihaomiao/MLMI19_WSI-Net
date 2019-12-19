# Outputing the heatmap and mask of patches
# Author: Haomiao Ni

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.sparse import load_npz


def SavePatchMap(MatPath, FigPath):
    MatDir = os.listdir(MatPath)
    for SubMatDirName in MatDir:
        SubFigDir = os.path.join(FigPath, SubMatDirName)
        if not os.path.exists(SubFigDir):
            os.mkdir(SubFigDir)

        SubMatDir = os.path.join(MatPath, SubMatDirName)
        MatList = os.listdir(SubMatDir)
        for MatName in MatList:
            MatFile = os.path.join(SubMatDir, MatName)
            SegRes = load_npz(MatFile)
            SegRes = SegRes.todense()
            SegRes = 254 - SegRes
            Fig = Image.fromarray(SegRes.astype(dtype=np.uint8))
            del SegRes
            Fig.convert('L')
            FigName = MatName.replace('.npz', '.png')
            FigFile = os.path.join(SubFigDir, FigName)
            Fig.save(FigFile, 'PNG')

def SaveMap(MatPath, FigPath):
    MatDir = os.listdir(MatPath)
    dpi = 1000.0
    plt.ioff()
    fig = plt.figure(frameon=False)
    for MatName in MatDir:
        MatFile = os.path.join(MatPath, MatName)
        MatRes = load_npz(MatFile)
        MatRes = MatRes.todense()

        fig.clf()
        fig.set_size_inches(MatRes.shape[1] / dpi, MatRes.shape[0] / dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        cm = plt.cm.get_cmap('gray')
        ax.imshow(MatRes, cmap=cm, aspect='auto')

        FigName = MatName.replace('.npz', '.png')
        FigFile = os.path.join(FigPath, FigName)
        fig.savefig(FigFile, dpi=int(dpi))

def SavePatchFigMap(OriginalPath, FigPath, FigMatPath):
    NormalFig = Image.new('L', (patch_size, patch_size), 255)
    ImagePath = os.path.join(OriginalPath, 'images')
    LabelPath = os.path.join(OriginalPath, 'labels')
    ImageDir = os.listdir(ImagePath)
    for SubImageDirName in ImageDir:
        print "processing:", SubImageDirName
        SubLabelDir = os.path.join(LabelPath, SubImageDirName)
        SubImageDir = os.path.join(ImagePath, SubImageDirName)
        SubImageList = os.listdir(SubImageDir)
        for SubImageName in SubImageList:
            SubImageFile = os.path.join(SubImageDir, SubImageName)
            SubImage = Image.open(SubImageFile)
            SubImage = SubImage.resize((patch_size, patch_size))

            SubLabelName = SubImageName.replace('.jpg', '.png')
            SubLabelFile = os.path.join(SubLabelDir, SubLabelName)
            if os.path.exists(SubLabelFile):
                SubLabel = Image.open(SubLabelFile)
            else:
                SubLabel = Image.open(NormalLabelPath)
            SubLabel = SubLabel.resize((patch_size, patch_size))

            DirName = SubImageDir.split('/')[-1]
            SubFigDir = os.path.join(FigPath, DirName)
            SubFigName = SubImageName.replace('.jpg', '*')
            SubFigFile = os.path.join(SubFigDir, SubFigName)
            SubFigFile = glob.glob(SubFigFile)
            if len(SubFigFile)==0:
                SubFig = NormalFig
            else:
                SubFigFile = SubFigFile[0]
                SubFig = Image.open(SubFigFile)

            # splicing
            FigMapDir = os.path.join(FigMapPath, DirName)
            if not os.path.exists(FigMapDir):
                os.mkdir(FigMapDir)
            FigMapName = SubImageName.replace('.jpg','_FIGMAP.jpg')
            FigMapFile = os.path.join(FigMapDir, FigMapName)

            new_im = Image.new('RGB', (patch_size*3, patch_size))
            new_im.paste(SubImage, (0, 0))
            new_im.paste(SubLabel, (patch_size, 0))
            new_im.paste(SubFig, (patch_size*2, 0))
            new_im.save(FigMapFile)


if __name__ == '__main__':
    patch_size = 2048 / 8 + 1
    MatPath = '/disk8t-1/deeplab-xiangya2/xiangya-test-npz/Xiangya_Deeplab_B2_S555000_Frozen_BN_test2048'
    FigPath = '/disk8t-1/deeplab-xiangya2/xiangya-test-fig/Xiangya_Deeplab_B2_S555000_Frozen_BN_test2048'
    if not os.path.exists(FigPath):
        os.mkdir(FigPath)
    FigMapPath = '/disk8t-1/deeplab-xiangya2/xiangya-test-figmap/Xiangya_Deeplab_B2_S555000_Frozen_BN_test2048'
    if not os.path.exists(FigMapPath):
        os.mkdir(FigMapPath)
    OriginalPath = '/disk8t-1/deeplab-xiangya2/new_2048_val_stride_192_XY3c'
    NormalLabelPath = '/disk8t-1/deeplab-xiangya2/new_2048_val_stride_192_XY3c/labels/All_Normal_Mask.png'
    SavePatchMap(MatPath, FigPath)
    SavePatchFigMap(OriginalPath, FigPath, FigMapPath)
