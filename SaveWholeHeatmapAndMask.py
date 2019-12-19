# Outputing the entire heatmap and mask of WSI
# Author: Haomiao Ni

import os
from scipy.sparse import load_npz
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
from libtiff import TIFF, TIFF3D, TIFFfile, TIFFimage
Image.MAX_IMAGE_PIXELS = 933120000

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


def SavePatchMap(MatPath, FigPath):
    MatList = os.listdir(MatPath)
    for MatName in MatList:
        FigName = MatName.replace('.npz', '.png')
        FigFile = os.path.join(FigPath, FigName)
        if os.path.exists(FigFile):
            continue
        MatFile = os.path.join(MatPath, MatName)
        SegRes = load_npz(MatFile)
        SegRes = SegRes.todense()
        SegRes = 254 - SegRes
        Fig = Image.fromarray(SegRes.astype(dtype=np.uint8))
        del SegRes
        Fig.convert('L')
        Fig.save(FigFile, 'PNG')


def SavePatchFigMap(OriginalPath, LabelPath, FigPath, FigMapPath):
    ImageList = os.listdir(OriginalPath)
    for ImageName in ImageList:
        FigMapName = ImageName.replace('.tif','_FIGMAP.jpg')
        FigMapFile = os.path.join(FigMapPath, FigMapName)
        if os.path.exists(FigMapFile):
            continue
        print "processing:", ImageName
        ImageFile = os.path.join(OriginalPath, ImageName)
        ImageArr = open_slide(ImageFile, set_current_level)
        (h, w, _) = ImageArr.shape
        Img = Image.fromarray(ImageArr)

        LabelName = ImageName.replace('.tif', '_Mask.tif')
        LabelFile = os.path.join(LabelPath, LabelName)
        if os.path.exists(LabelFile):
            LabelArr = open_slide(LabelFile, set_current_level)
        else:
            LabelArr = np.zeros(ImageArr.shape[:2])
        Label = Image.fromarray(LabelArr)

        FigName = ImageName.replace('.tif', '_Map.png')
        FigFile = os.path.join(FigPath, FigName)
        Fig = Image.open(FigFile)

        # splicing
        if w < h:
            new_im = Image.new('RGB', (w*3, h))
            new_im.paste(Img, (0, 0))
            new_im.paste(Label, (w, 0))
            new_im.paste(Fig, (w*2, 0))
            new_im.save(FigMapFile)
        else:
            new_im = Image.new('RGB', (w, h*3))
            new_im.paste(Img, (0, 0))
            new_im.paste(Label, (0, h))
            new_im.paste(Fig, (0, h*2))
            new_im.save(FigMapFile)


if __name__ == '__main__':
    SetType = 'train'
    resolution = '40'
    type = 'train'
    set_current_level = 3
    LabelPath = '/disk8t-1/Xiangya2/Mask_train'
    SrcPath = '/disk8t-1/deeplab-xiangya2/'

    MatPath = SrcPath + 'xiangya-'+type+'-npz/whole_npz/Xiangya_Deeplab_B2_S555000_Frozen_BN_test2048'
    FigPath = SrcPath + 'xiangya-'+type+'-fig/whole_fig/Xiangya_Deeplab_B2_S555000_Frozen_BN_test2048'
    FigMapPath = SrcPath + 'xiangya-'+type+'-figmap/whole_figmap/Xiangya_Deeplab_B2_S555000_Frozen_BN_test2048'
    # if not os.path.exists(FigPath):
    #     os.makedirs(FigPath)
    # SavePatchMap(MatPath, FigPath)
    if not os.path.exists(FigMapPath):
        os.makedirs(FigMapPath)
    OriginalPath = '/disk8t-1/Xiangya2/'+SetType
    SavePatchFigMap(OriginalPath, LabelPath, FigPath, FigMapPath)


    # thre = [0.1, 0.2, 0.3, 0.4, 0.5]
    # pl = ['', 'wo_pl_']
    # for j in pl:
    #     for i in thre:
    #         MatPath = SrcPath + 'xiangya-'+type+'-res/whole_npz/Xiangya_Deeplab_B4_S28000_IC_test2048_'+str(j)+str(i)
    #         FigPath = SrcPath + 'xiangya-'+type+'-fig/whole_fig/Xiangya_Deeplab_B4_S28000_IC_test2048_'+str(j)+str(i)
    #         FigMapPath = SrcPath + 'xiangya-'+type+'-figmap/whole_figmap/Xiangya_Deeplab_B4_S28000_IC_test2048_'+str(j)+str(i)
    #         if not os.path.exists(FigPath):
    #             os.makedirs(FigPath)
    #         SavePatchMap(MatPath, FigPath)
    #         if not os.path.exists(FigMapPath):
    #             os.makedirs(FigMapPath)
    #         OriginalPath = '/disk8t-1/Xiangya2/'+SetType
    #         SavePatchFigMap(OriginalPath, LabelPath, FigPath, FigMapPath)

