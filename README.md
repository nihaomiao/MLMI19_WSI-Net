WSI-Net
====

The Pytorch implementation of our MLMI19 oral paper [WSI-Net: Branch-based and Hierarchy-aware Network for Segmentation and Classification of Breast Histopathological Whole-slide Images](https://drive.google.com/open?id=1XBRQUKKxkAYxywSY5EqqfSDaFTy9qvmr).

Dependencies
----
Python 2.7, Pytorch 1.0.0, opencv, libtiff, etc.

Data Preparation
----
Our benchmark dataset is provided by Xiangya Hospital, [Central South University](http://en.csu.edu.cn/index.htm), which is still a private dateset now. The detailed descrption of our dataset can be found in our paper. You may also utilize the other similar public datasets, such as [BACH](https://iciar2018-challenge.grand-challenge.org/), [Camelyon16](https://camelyon16.grand-challenge.org/) and [Camelyon17](https://camelyon17.grand-challenge.org/).

About Training
----
Our models mainly include: the original DeepLab, DeepLab-HA (DeepLab plus our hierarchy-aware loss), and WSI-Net (DeepLab-HA plus our classification branch).

### A. Training DeepLab
**TODO
