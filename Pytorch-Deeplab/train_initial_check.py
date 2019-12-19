# Freezing the front layers and just training the branch classifier
# Author: Haomiao Ni
# For some reasons, we name In-Situ as DG, Invasive as JR.

from __future__ import print_function
import argparse
import torch
from torch.utils import data
import numpy as np
import cv2
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from deeplab.model_initial_check import Res_Deeplab
from deeplab.datasets import XiangyaTrain
import timeit
import math
from tensorboardX import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

start = timeit.default_timer()

IMG_MEAN = np.array((188.3417403, 139.62816465, 178.39104105), dtype=np.float32)  # BGR

BATCH_SIZE = 4
MAX_EPOCH = 40
DATA_LIST_PATH = '/disk8t-1/deeplab-xiangya2/4480_train_stride_2240_XY3c/train.txt'
INPUT_SIZE = '4096,4096'
NUM_CLASSES = 2
RANDOM_SEED = 1234
RESTORE_FROM = '/disk8t-1/deeplab-xiangya2-IC/snapshots-FrozenBN-IC-v3/Xiangya_Deeplab_B4_S30000.pth'
SAVE_PRED_EVERY = 3000
SNAPSHOT_DIR = '/disk8t-1/deeplab-xiangya2-IC/snapshots-FrozenBN-IC-v3'
NUM_EXAMPLES_PER_EPOCH = 16583
# MAX_ITER should exactly be EE*EP + (BS-R)*EP
# where EE = exapmle per epoch, EP = epoch, BS = batch size
# EE = N*BS + R (R < BS)
MAX_ITER = NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + BATCH_SIZE*MAX_EPOCH
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--set-start", default=True,
                        help="If False, directly set the start step to be 0")
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0.00004, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--alpha', default=0.9, type=float, metavar='A',
                        help='alpha')
    parser.add_argument('--eps', default=1.0, type=float, metavar='E',
                        help='eps')
    parser.add_argument("--img-dir", type=str, default='/disk8t-1/deeplab-xiangya2-IC/imgshots-FrozenBN-IC',
                        help="Where to save intermediate images of the model.")
    parser.add_argument("--num-workers", default=16)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default="0, 1, 2, 3",
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=200, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the text file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--random-mirror", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-jitter", default=True)
    parser.add_argument("--random-rotate", default=True)
    parser.add_argument("--random-scale", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


args = get_arguments()


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    return criterion(pred, label)


def adjust_learning_rate(optimizer, actual_step):
    epoch = actual_step//NUM_STEPS_PER_EPOCH
    lr = args.lr * (0.5 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def patch_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred).long().cuda())

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    min_input_side = 321
    cudnn.enabled = True
    torch.manual_seed(args.random_seed)

    model = Res_Deeplab(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model)

    # freeze all the layers except InceptionAux
    for i in model.named_parameters():
        if 'AuxLogits' not in i[0].split('.'):
            i[1].requires_grad = False

    # I will use the optimizer just for aux classifier
    optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    alpha=args.alpha,
                                    eps=args.eps,
                                    weight_decay=args.weight_decay)

    trainloader = data.DataLoader(XiangyaTrain(args.data_list,
                                               crop_size=input_size, scale=args.random_scale,
                                               mirror=args.random_mirror, color_jitter=args.random_jitter,
                                               rotate=args.random_rotate,
                                               mean=IMG_MEAN),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

    if args.fine_tune:
        # fine tune from some trained model
        print("=> fine-tuning from checkpoint '{}'".format(args.restore_from))
        saved_state_dict = torch.load(args.restore_from)
        saved_state_dict = saved_state_dict['state_dict']
        new_params = model.state_dict().copy()

        for i in saved_state_dict:
            if 'fc' in i.split('.'):
                continue
            if 'layer3' in i.split('.'):
                continue
            new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            try:
                if args.set_start:
                    args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (step {})"
                      .format(args.restore_from, args.start_step))
            except:
                model.load_state_dict(checkpoint)
                print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))

    model.train()
    model.cuda()

    cudnn.benchmark = True

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    writer = SummaryWriter(args.snapshot_dir)

    cnt = 0
    actual_step = args.start_step
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)

            data_time.update(timeit.default_timer() - iter_end)

            images, labels, patch_name = batch

            # set random size images and labels here. [min_input_size, h]
            # crop_side [0, h-min_input_size]
            crop_side = np.random.randint(0, (h-min_input_side+1)/2)
            images = F.pad(images, (-crop_side, -crop_side, -crop_side, -crop_side))
            labels = F.pad(labels, (-crop_side, -crop_side, -crop_side, -crop_side))
            cls_labels = labels.sum(dim=1).sum(dim=1) > 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            _, aux_pred = model(images)

            del images, labels

            # 0 Normal 1 Tumor
            # get label
            loss = loss_calc(aux_pred, cls_labels)
            losses.update(loss.item(), aux_pred.size(0))
            acc = patch_accuracy(aux_pred, cls_labels)
            accuracy.update(acc[0].item(), aux_pred.size(0))
            loss.backward()
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Patch Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    cnt, actual_step, args.final_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, accuracy=accuracy))
                writer.add_scalar("train_loss", losses.avg, actual_step)
                writer.add_scalar("patch_accuracy", accuracy.avg, actual_step)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], actual_step)

            if actual_step % args.save_pred_every == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'state_dict': model.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'Xiangya_Deeplab_B' + str(args.batch_size) + '_S' + str(actual_step) + '.pth'))
            cnt += 1

    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'state_dict': model.state_dict()},
               osp.join(args.snapshot_dir,
                        'Xiangya_Deeplab_B' + str(args.batch_size) + '_S' + str(actual_step) + '.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')


def _pixel_accuracy(pred, target):
    accuracy_sum = 0.0
    for i in range(0, pred.shape[0]):
        out = pred[i].argmax(axis=0)
        accuracy = np.sum(out == target[i], dtype=np.float32) / out.size
        accuracy_sum += accuracy
    return accuracy_sum / args.batch_size



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
