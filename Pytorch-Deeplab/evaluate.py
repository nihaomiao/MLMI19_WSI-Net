# Running models
# Author: Haomiao Ni
# Modified by Haomiao Ni on 20 Sept, 2018
# Track set True for Normal BN (Frozen BN when training) or False for batch stats
# I recommend you set this parameter to be True. 
# Details about BatchNorm: https://pytorch.org/docs/stable/nn.html#normalization-layers
from __future__ import print_function
import argparse
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils import data
from deeplab.model import Res_Deeplab
from deeplab.datasets import XiangyaTest
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.sparse import save_npz, coo_matrix

IMG_MEAN = np.array((188.3417403, 139.62816465, 178.39104105), dtype=np.float32)  # BGR

resolution = '40p'
SetType = 'test'
if resolution == '10':
    res = '-10X'
elif resolution == '40':
    res = ''
elif resolution == '40p':
    res = '-v2'
elif resolution == '10p':
    res = '-v2-10X'

DATA_DIRECTORY = '/disk8t-1/Xiangya2/'+SetType
DATA_LIST_PATH = '/disk8t-1/deeplab-xiangya2'+res+'/xiangya-'+SetType+'-text/2048_s192/'
if not os.path.exists(DATA_LIST_PATH):
    os.mkdir(DATA_LIST_PATH)
NUM_CLASSES = 3
RESTORE_FROM = '/disk8t-1/deeplab-xiangya2-IC/snapshots-FrozenBN-IC-joint-wo-patch-loss/Xiangya_Deeplab_B4_S28000.pth'
NPZ_PATH = '/disk8t-1/deeplab-xiangya2'+res+'/xiangya-'+SetType+'-npz-temp/Xiangya_Deeplab_B4_S28000_Frozen_BN_test2048_0.0_wo_pl'
if not os.path.exists(NPZ_PATH):
    os.makedirs(NPZ_PATH)
# else:
#     raise Exception("Directory has existed!", NPZ_PATH)
MAP_PATH = '/disk8t-1/deeplab-xiangya2'+res+'/xiangya-'+SetType+'-map-temp/Xiangya_Deeplab_B4_S28000_Frozen_BN_test2048_0.0_wo_pl'
if not os.path.exists(MAP_PATH):
    os.makedirs(MAP_PATH)
# else:
#     raise Exception("Directory has existed!", MAP_PATH)
BATCH_SIZE = 4
INPUT_SIZE = (2048, 2048)
LOG_PATH = '/disk8t-1/deeplab-xiangya2'+res+'/logfiles'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--npz-path", default=NPZ_PATH)
    parser.add_argument("--map-path", default=MAP_PATH)
    parser.add_argument("--load-seg-from-jointly-train", default=True)
    parser.add_argument("--track-running-stats", default=True) # set false to use current batch_stats when eval
    parser.add_argument("--momentum", default=0) # set 0 to freeze running mean and var, useless when eval
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", default='2,3',
                        help="choose gpu device.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--log-path', default=LOG_PATH)
    return parser.parse_args()


def main():
    preprocess_start_time = time.time()
    """Create the model and start the evaluation process."""
    args = get_arguments()
    print("Restored from:", args.restore_from)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    localtime = time.localtime(time.time())
    LogName = "Test_HeatMap_log_" + str(localtime.tm_year) + "_" + str(localtime.tm_mon) + "_" + str(localtime.tm_mday) \
              + "_" + str(localtime.tm_hour) + "_" + str(localtime.tm_min) + "_" + str(localtime.tm_sec)
    LogFile = os.path.join(args.log_path, LogName)
    log = open(LogFile, 'a', 0)
    log.writelines('batch size:' + str(args.batch_size) + ' ' + 'gpu:' + args.gpu + '\n')
    log.writelines(args.data_list + '\n')
    log.writelines('restore from ' + args.restore_from + '\n')

    model = Res_Deeplab(num_classes=args.num_classes,
                        track_running_stats=args.track_running_stats, momentum=args.momentum)
    model = nn.DataParallel(model)
    model.cuda()
    saved_state_dict = torch.load(args.restore_from)
    num_examples = saved_state_dict['example']
    if args.track_running_stats:
        print("using running mean and running var")
        log.writelines("using running mean and running var\n")
        if args.load_seg_from_jointly_train:
            print("Single output though loading from the seg model with cls branch.")
            log.writelines("single output though loading from the seg model with cls branch.")
            new_params = saved_state_dict['state_dict'].copy()
            for i in saved_state_dict['state_dict']:
                i_parts = i.split('.')
                if "AuxLogits" in i_parts:
                    del new_params[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict['state_dict'])
    else:
        print("using current batch stats instead of running mean and running var")
        log.writelines("using current batch stats instead of running mean and running var\n")
        print("if you froze BN when training, maybe you are wrong now!!!")
        log.writelines("if you froze BN when training, maybe you are wrong now!!!\n")
        new_params = saved_state_dict['state_dict'].copy()
        for i in saved_state_dict['state_dict']:
            i_parts = i.split('.')
            if ("running_mean" in i_parts) or ("running_var" in i_parts):
                del new_params[i]
        model.load_state_dict(new_params)

    model.eval()
    log.writelines('preprocessing time: ' + str(time.time() - preprocess_start_time) + '\n')

    TestDir = os.listdir(args.data_dir)
    TestDir.sort()
    for TestName in TestDir:
        print('Processing '+TestName)
        log.writelines('Processing ' + TestName + '\n')
        TestName = TestName[:-4]

        TestTxt = os.path.join(args.data_list, TestName + '.txt')
        testloader = data.DataLoader(XiangyaTest(TestTxt, crop_size=INPUT_SIZE, mean=IMG_MEAN),
                                     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


        TestNpzPath = os.path.join(args.npz_path, TestName)
        TestMapPath = os.path.join(args.map_path, TestName)
        if not os.path.exists(TestNpzPath):
            os.mkdir(TestNpzPath)
        if not os.path.exists(TestMapPath):
            os.mkdir(TestMapPath)

        batch_time = AverageMeter()
        with torch.no_grad():
            end = time.time()
            for index, (image, name) in enumerate(testloader):

                output = model(Variable(image).cuda())
                del image
                Softmax = torch.nn.Softmax2d()
                pred = torch.max(Softmax(output), dim=1, keepdim=True)
                del output

                for ind in range(0, pred[0].size(0)):
                    prob = torch.squeeze(pred[0][ind]).data.cpu().numpy()
                    prob = coo_matrix(prob)
                    if len(prob.data) == 0:
                        continue
                    mapname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MAP.npz')
                    mapfile = os.path.join(TestMapPath, mapname)
                    save_npz(mapfile, prob.tocsr())

                    msk = torch.squeeze(pred[1][ind]).data.cpu().numpy()
                    msk = msk * 127
                    msk = coo_matrix(msk)
                    if len(msk.data) == 0:
                        continue
                    npzname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MSK.npz')
                    npzfile = os.path.join(TestNpzPath, npzname)
                    save_npz(npzfile, msk.tocsr())

                batch_time.update(time.time() - end)
                end = time.time()

                if index % args.print_freq == 0:
                    print('Test:[{0}/{1}]\t'
                          'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                          .format(index, len(testloader), batch_time=batch_time))

        print('The total test time for '+TestName+' is '+str(batch_time.sum))
        log.writelines('batch num:' + str(len(testloader)) + '\n')
        log.writelines('The total test time for ' + TestName + ' is ' + str(batch_time.sum) + '\n')
    log.writelines('The total running time is '+str(time.time()-preprocess_start_time)+'\n')
    log.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
