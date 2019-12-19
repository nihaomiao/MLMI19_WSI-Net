# Deeplab-HA for Xiangya2 Dataset
# Author: Haomiao Ni
# Modified by Haomiao Ni on 1, Oct, 2018
# Frozen BN + patch-level label_loss
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

BATCH_SIZE = 3
MAX_EPOCH = 5
DATA_LIST_PATH = '/disk8t-1/deeplab-xiangya2/1280_train_stride_640_XY3c/train.txt'
INPUT_SIZE = '1140,1140'
LEARNING_RATE = 2.5e-5
MOMENTUM = 0.9
NUM_CLASSES = 3
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/disk8t-1/deeplab-xiangya2-v2/snapshots-FrozenBN/Xiangya_Deeplab_B2_S480000.pth'
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = '/disk8t-1/deeplab-xiangya2-v2/snapshots-FrozenBN'
WEIGHT_DECAY = 0.0005
NUM_EXAMPLES_PER_EPOCH = 222360
MAX_ITER = NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + BATCH_SIZE * MAX_EPOCH
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--set-start", default=True, help="If False, directly set the start step to be 0")
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--is-training", default=False,
                        help="Whether to freeze BN layers, False for Freezing")
    parser.add_argument("--img-dir", type=str, default='/disk8t-1/deeplab-xiangya2-v2/imgshots-FrozenBN',
                        help="Where to save images of the model.")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--num-workers", default=16)
    parser.add_argument("--loss-coeff", default=2.3)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--gpu", default="0, 1, 2",
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
    label_cpu = label.data.cpu().numpy()
    pred_cpu = pred.data.cpu().numpy()
    msk_cpu = pred_cpu.argmax(axis=1)
    patch_label = np.zeros(label_cpu.shape[0])  # (N,)
    patch_pred = np.zeros(pred_cpu.shape[0:2])  # (N, C)
    for i in range(msk_cpu.shape[0]):
        if np.sum(label_cpu[i] == 2) != 0:
            patch_label[i] = 2
        elif np.sum(label_cpu[i] == 1) != 0:
            patch_label[i] = 1
        else:
            assert np.sum(label_cpu[i] != 0) == 0
            patch_label[i] = 0
        if np.sum(msk_cpu[i] == 2) != 0:
            pred_cpu[i][2][msk_cpu[i] != 2] = -1000
            ind = np.unravel_index(pred_cpu[i][2].argmax(axis=None), pred_cpu[i][2].shape)
            patch_pred[i, :] = pred_cpu[i][:][:, ind[0], ind[1]]
            assert patch_pred[i, :].argmax() == 2
        elif np.sum(msk_cpu[i] == 1) != 0:
            pred_cpu[i][1][msk_cpu[i] != 1] = -1000
            ind = np.unravel_index(pred_cpu[i][1].argmax(axis=None), pred_cpu[i][1].shape)
            patch_pred[i, :] = pred_cpu[i][:][:, ind[0], ind[1]]
            assert patch_pred[i, :].argmax() == 1
        else:
            assert np.sum(msk_cpu[i] != 0) == 0
            ind = np.unravel_index(pred_cpu[i][0].argmax(axis=None), pred_cpu[i][0].shape)
            patch_pred[i, :] = pred_cpu[i][:][:, ind[0], ind[1]]
            assert patch_pred[i, :].argmax() == 0
    patch_label = torch.tensor(patch_label).long().cuda()
    patch_pred = torch.tensor(patch_pred).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    pixel_loss = criterion(pred, label)
    patch_loss = criterion(patch_pred, patch_label)
    return pixel_loss, patch_loss


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
            # Scale.layer5.conv2d_list.3.weight
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
        print("Freezing BN layers, that is taking BN as linear transform layer")
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
    pixel_losses = AverageMeter()
    patch_losses = AverageMeter()
    accuracy = AverageMeter()
    writer = SummaryWriter(args.snapshot_dir)

    cnt = 0
    actual_step = args.start_step
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, (images, labels, patch_name) in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)

            data_time.update(timeit.default_timer() - iter_end)

            images = Variable(images).cuda()

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            # pred = interp(model(images))
            pred = model(images)
            image = images.data.cpu().numpy()[0]
            del images
            # 0 Normal 1 DG 2 JR
            labels = resize_target(labels, pred.size(2))

            pixel_loss, patch_loss = loss_calc(pred, labels)
            loss = pixel_loss.double() + args.loss_coeff * patch_loss
            losses.update(loss.item(), pred.size(0))
            pixel_losses.update(pixel_loss.item(), pred.size(0))
            patch_losses.update(patch_loss.item(), pred.size(0))

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
                      'Pixel Loss {pixel_loss.val:.4f} ({pixel_loss.avg:.4f})\t'
                      'Patch_Loss {patch_loss.val:.4f} ({patch_loss.avg:.4f})\t'
                      'Pixel Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    cnt, actual_step, args.final_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, pixel_loss=pixel_losses, patch_loss=patch_losses,
                    accuracy=accuracy))
                writer.add_scalar("train_loss", losses.avg, actual_step)
                writer.add_scalar("pixel_loss", pixel_losses.avg, actual_step)
                writer.add_scalar("patch_loss", patch_losses.avg, actual_step)
                writer.add_scalar("pixel_accuracy", accuracy.avg, actual_step)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], actual_step)

            # TODO complete this part using writer
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
