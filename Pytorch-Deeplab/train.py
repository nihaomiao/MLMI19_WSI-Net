# Deeplab-ResNet-V2 for Xiangya2 Dataset
# Modified by Haomiao Ni on 20, Sept, 2018
# Frozen BN when set is_training as false
# we freeze BN due to the small batch size
# It heavily borrows code from:
# https://github.com/speedinghzl/Pytorch-Deeplab
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
from deeplab.model import Res_Deeplab
from deeplab.datasets import XiangyaTrain
import timeit
import math
from tensorboardX import SummaryWriter
from PIL import Image

start = timeit.default_timer()

IMG_MEAN = np.array((188.3417403, 139.62816465, 178.39104105), dtype=np.float32)  # BGR

BATCH_SIZE = 4
MAX_EPOCH = 5
DATA_LIST_PATH = '/disk8t-1/deeplab-xiangya2/1280_train_stride_640_XY3c/train.txt'
# due to the constraint of GPU memory, we reset the size of training patch to be 1140.
INPUT_SIZE = '1140,1140'
LEARNING_RATE = 2.5e-5
MOMENTUM = 0.9
NUM_CLASSES = 3
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/disk8t-1/deeplab-xiangya2-v2/snapshots-FrozenBN/Xiangya_Deeplab_B1_S105000.pth'
SAVE_PRED_EVERY = 5000  # about 2h
SNAPSHOT_DIR = '/disk8t-1/deeplab-xiangya2/snapshots-FrozenBN'
WEIGHT_DECAY = 0.0005
NUM_EXAMPLES_PER_EPOCH = 222360
MAX_ITER = NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--set-start", default=True,
                        help="If False, directly set the start step to be 0")
    parser.add_argument("--start-step", default=105000, type=int)
    parser.add_argument("--is-training", default=False,
                        help="Whether to freeze BN layers, False for Freezing")
    parser.add_argument("--img-dir", type=str, default='/disk8t-1/deeplab-xiangya2/imgshots-FrozenBN',
                        help="Where to save intermediate images of the model.")
    parser.add_argument("--num-workers", default=16)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--gpu", default="0, 1",
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
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
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
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
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


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []

    b.append(model.conv1)
    b.append(model.bn1)
    b.append(model.layer1)
    b.append(model.layer2)
    b.append(model.layer3)
    b.append(model.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    b.append(model.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def adjust_learning_rate(optimizer, actual_step):
    """Original Author: Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, actual_step * args.batch_size, MAX_ITER, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.float32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    torch.manual_seed(args.random_seed)

    model = Res_Deeplab(num_classes=args.num_classes)
    model = torch.nn.DataParallel(model)

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model.module), 'lr': args.learning_rate},
                           {'params': get_10x_lr_params(model.module), 'lr': 10 * args.learning_rate}],
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.fine_tune:
        # fine tune from coco dataset
        saved_state_dict = torch.load(args.restore_from)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if i_parts[1] != 'layer5':
                new_params[i.replace('Scale', 'module')] = saved_state_dict[i]
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

    if not args.is_training:
        # Frozen BN
        # when training, the model will use the running means and the
        # running vars of the pretrained model.
        # But note that eval() doesn't turn off history tracking.
        print("Freezing BN layers")
        model.eval()
    else:
        model.train()
    model.cuda()

    cudnn.benchmark = True

    trainloader = data.DataLoader(XiangyaTrain(args.data_list,
                                               crop_size=input_size, scale=args.random_scale,
                                               mirror=args.random_mirror, color_jitter=args.random_jitter,
                                               rotate=args.random_rotate,
                                               mean=IMG_MEAN),
                                  batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)

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
            images = Variable(images).cuda()

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            pred = model(images)
            image = images.data.cpu().numpy()[0]
            del images
            # 0 Normal 1 DG 2 JR
            labels = resize_target(labels, pred.size(2))

            loss = loss_calc(pred, labels)
            losses.update(loss.item(), pred.size(0))
            acc = _pixel_accuracy(pred.data.cpu().numpy(), labels.data.cpu().numpy())
            accuracy.update(acc, pred.size(0))
            loss.backward()
            optimizer.step()

            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Pixel Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    cnt, actual_step, args.final_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, accuracy=accuracy))
                writer.add_scalar("train_loss", losses.avg, actual_step)
                writer.add_scalar("pixel_accuracy", accuracy.avg, actual_step)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], actual_step)

            if actual_step % args.save_img_freq == 0:
                msk_size = pred.size(2)
                image = image.transpose(1, 2, 0)
                image = cv2.resize(image, (msk_size, msk_size), interpolation=cv2.INTER_NEAREST)
                image += IMG_MEAN
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label = labels.data.cpu().numpy()[0]
                label = np.repeat(254, msk_size) - label * 127
                single_pred = pred.data.cpu().numpy()[0].argmax(axis=0)
                single_pred = single_pred * 127
                new_im = Image.new('RGB', (msk_size * 3, msk_size))
                new_im.paste(Image.fromarray(image.astype('uint8'), 'RGB'), (0, 0))
                new_im.paste(Image.fromarray(single_pred.astype('uint8'), 'L'), (msk_size, 0))
                new_im.paste(Image.fromarray(label.astype('uint8'), 'L'), (msk_size * 2, 0))
                new_im_name = 'B' + str(args.batch_size) + '_S' + str(actual_step) + '_' + patch_name[0]
                new_im_file = os.path.join(args.img_dir, new_im_name)
                new_im.save(new_im_file)

            if actual_step % args.save_pred_every == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,  # TODO it may be (actual_step+1)*args.batch_size
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
