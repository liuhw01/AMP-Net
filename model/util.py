from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import shutil
import cv2
from PIL import Image
import torch.nn as nn
import torch
import threading
import math
from torch.nn import functional as F



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()



def random_choose_data(label_path):
    random.seed(1)
    file = open(label_path)
    lines = file.readlines()
    slice_initial = random.sample(lines, len(lines))  # if don't change this ,it will be all the same
    train_label = slice_initial[:int(len(lines)*0.8)]
    test_label = slice_initial[int(len(lines)*0.8):len(lines)]
    return train_label, test_label  # output the list and delvery it into ImageFolder


def choose_data(label_path):

    file = open(label_path)
    lines = file.readlines()
    train_label=[]
    test_label=[]
    for i in range(len(lines)):
        if lines[i][0:3]=='tra':
            num=int(lines[i][-2])-1
            s1=list(lines[i])
            s1[-2]=str(num)
            s=''.join(s1)
            train_label.append(s)
        if lines[i][0:3]=='tes':
            num=int(lines[i][-2])-1
            s1=list(lines[i])
            s1[-2]=str(num)
            s=''.join(s1)
            test_label.append(s)

    return train_label, test_label  # output the list and delvery it into ImageFolder

def rotate(ps,m):
    pts = np.float32(ps).reshape([-1, 2])
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    target_point = np.dot(m, pts)
    target_point = [[target_point[0][x],target_point[1][x]] for x in range(len(target_point[0]))]
    return target_point
def rotate_img_and_point(img,points,angle,center_x,center_y,resize_rate=1.0):
    h,w,c = img.shape
    M = cv2.getRotationMatrix2D((center_x,center_y), angle, resize_rate)
    res_img = cv2.warpAffine(img, M, (w, h))
    out_points = rotate(points,M)
    return res_img,out_points

class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        #self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
    def run(self):
        self.result = self.func(*self.args)



def pre_processing(images, facial, p, pp, l):
    for i in range(len(images)):
        im = images[i].transpose((1, 2, 0))


        if random.random()<p:

            resize_rate=round(random.uniform(0.9,1.1),2)
            angle=random.randint(-15, 15)
            im, point=rotate_img_and_point(im,facial[i],angle,112,112,resize_rate )
            facial[i]=point

        if random.random() < p:
            horizon=random.randint(-20, 20)
            vertial=random.randint(-20, 20)
            mat_translation = np.float32([[1, 0, horizon], [0, 1, vertial]])
            im = cv2.warpAffine(im, mat_translation, (224, 224))
            for j in range(len(facial[i])):
                facial[i][j][0]= facial[i][j][0]+horizon
                facial[i][j][1] = facial[i][j][1] + vertial
                if facial[i][j][0]<0:
                    facial[i][j][0]=0
                if facial[i][j][0]>224:
                    facial[i][j][0]=224
                if facial[i][j][1] < 0:
                    facial[i][j][1] = 0
                if facial[i][j][1] > 224:
                    facial[i][j][1] = 224


        if random.random()<pp:
            im=np.fliplr(im)
            landmark=facial[i]
            facial[i][0]=  [224-landmark[1][0] ,  landmark[1][1]]
            facial[i][1] = [224 - landmark[0][0], landmark[0][1]]
            facial[i][2] = [224 - landmark[2][0], landmark[2][1]]
            facial[i][3] = [224 - landmark[4][0], landmark[4][1]]
            facial[i][4] = [224 - landmark[3][0], landmark[3][1]]

        images[i] = im.transpose((2, 0, 1))

    rect_all=[]
    facial=np.array(facial)
    facial[facial<0]=0
    facial[facial > 224] = 224


    for i in range(len(facial)):
        rect = []
        rect_local = []

        land_resize=np.around(facial[i]*(28/224)).astype(int)

        a_width = int((land_resize[0][0] + land_resize[1][0]) / 2)
        a_high = int(land_resize[2][1])
        min_length=min(a_high, a_width,28-a_width)
        if min_length>=28/3:
            rect.append([a_width - min_length, a_high - min_length, a_width, a_high])
            rect.append([a_width + min_length, a_high - min_length, a_width, a_high])
        if min_length<28/3:
            eyemin=np.array([a_high, a_width,28-a_width])
            a_width_ind=np.where(eyemin==eyemin.min())[0][0]
            if a_width_ind==0:
                rect.append([a_width - min_length, a_high - min_length, a_width, a_high])
                rect.append([a_width + min_length, a_high - min_length, a_width, a_high])
            if a_width_ind==1:
                rect.append([a_width - min_length, a_high - min_length, a_width, a_high])
                min_eye_length1 = min(a_high,28-a_width, 14)
                rect.append([a_width + min_eye_length1, a_high - min_eye_length1, a_width, a_high])
            if a_width_ind==2:
                min_eye_length2 = min(a_high,a_width, 14)
                rect.append([a_width - min_eye_length2, a_high - min_eye_length2, a_width, a_high])
                rect.append([a_width + min_length, a_high - min_length, a_width, a_high])


        b_width = int((land_resize[3][0] + land_resize[4][0]) / 2)
        min_mou_length=min(28-a_high, b_width,28-b_width)
        if min_mou_length>=28/3:
            rect.append([b_width - min_mou_length, a_high + min_mou_length, b_width, a_high])
            rect.append([b_width + min_mou_length, a_high + min_mou_length, b_width, a_high])
        if min_mou_length<28/3:
            moumin=np.array([28-a_high, b_width,28-b_width])
            moumin_ind=np.where(moumin==moumin.min())[0][0]
            if moumin_ind==0:
                rect.append([b_width - min_mou_length, a_high + min_mou_length, b_width, a_high])
                rect.append([b_width + min_mou_length, a_high + min_mou_length, b_width, a_high])
            if moumin_ind==1:
                rect.append([b_width - min_mou_length, a_high + min_mou_length, b_width, a_high])
                min_mou_length1 = min(28 - a_high, 28 - b_width,14)
                rect.append([b_width + min_mou_length1, a_high + min_mou_length1, b_width, a_high])
            if moumin_ind==2:
                min_mou_length2 = min(28 - a_high, b_width,14)
                rect.append([b_width - min_mou_length2, a_high + min_mou_length2, b_width, a_high])
                rect.append([b_width + min_mou_length, a_high + min_mou_length, b_width, a_high])


        land=land_resize
        eye1 = land[0]
        eye2 = land[1]
        eye_midle = (eye1 + eye2) / 2
        mouth1 = land[3]
        mouth2 = land[4]
        landmark = np.array([eye1, eye2, eye_midle, mouth1, mouth2]).astype(int)

        for j in range(len(landmark)):
            if landmark[j][0] < l:
                landmark[j][0] = l
            if landmark[j][0] + l > 28:
                landmark[j][0] = 28 - l
            if landmark[j][1] < l:
                landmark[j][1] = l
            if landmark[j][1] > 28 - l:
                landmark[j][1] = 28 - l

        rect.append(landmark)
        rect_all.append(rect)

    return rect_all




def pre_pro(images,facial,p,pp,l,num):
    images=images.numpy()
    imag = [[] for j in range(num)]
    facia = [[] for j in range(num)]
    length = len(images)
    for n in range(num):
        imag[n] = images[math.floor(n / num * length):math.floor((n + 1) / num * length), :, :, :]
        facia[n] = facial[math.floor(n / num * length):math.floor((n + 1) / num * length)]
    threads = []  #
    for i in range(num):
        t = MyThread(pre_processing, (imag[i], facia[i], p,pp,l,), pre_processing.__name__)
        threads.append(t)  #
    for i in range(num):
        threads[i].start()
    for i in range(num):
        threads[i].join()
    rect_all=[]
    for i in range(num):
        for j in range(len(threads[i].get_result())):
            rect_all.append(threads[i].get_result()[j])
    rect = [np.concatenate((rect_all[i][0], rect_all[i][1], rect_all[i][2], rect_all[i][3]), 0) for i in range(len(rect_all))]
    rect_local = [[rect_all[i][4][0][0] - l, rect_all[i][4][0][0] + l,
                   rect_all[i][4][0][1] - l, rect_all[i][4][0][1] + l,
                   rect_all[i][4][1][0] - l, rect_all[i][4][1][0] + l,
                   rect_all[i][4][1][1] - l, rect_all[i][4][1][1] + l,
                   rect_all[i][4][2][0] - l, rect_all[i][4][2][0] + l,
                   rect_all[i][4][2][1] - l, rect_all[i][4][2][1] + l,
                   rect_all[i][4][3][0] - l, rect_all[i][4][3][0] + l,
                   rect_all[i][4][3][1] - l, rect_all[i][4][3][1] + l,
                   rect_all[i][4][4][0] - l, rect_all[i][4][4][0] + l,
                   rect_all[i][4][4][1] - l, rect_all[i][4][4][1] + l, ] for i in
                  range(len(rect_all))]
    images = torch.tensor(images)

    return images, rect,rect_local




def default_loader(path):
    return Image.open(path).convert('RGB')  # operation object is the PIL image object


class myImageFloder(datasets.ImageFolder):  # Class inheritance
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
         # fh = open(label)
        c = 0
        imgs = []
        class_names = ['anger','disgust','fear','happy','neural','sad','surprise']
        for line in label:  # label is a list
            cls = line.split()  # cls is a lis
            fn = cls.pop(0)
            la=int(cls[0])
            if os.path.isfile(os.path.join(root, fn)):
                #imgs.append((fn, tuple([float(v) for v in cls[:len(cls)]])))
                imgs.append((fn, la))
                # access the last label
                # images is the list,and the content is the tuple, every image corresponds to a label
                # despite the label's dimension
                # we can use the append way to append the element for list
            c = c + 1
        print('the total image is',c)
        print(class_names)
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        fn, label = self.imgs[index]  # even though the imgs is just a list, it can return the elements of it
        # in a proper way
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label), fn


    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes


class myImageFloder1(datasets.ImageFolder):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
         # fh = open(label)
        c = 0
        imgs = []
        class_names = ['anger','disgust','fear','happy','neural','sad','surprise']
        for line in label:  # label is a list
            cls = line.split()  # cls is a lis
            fn = cls.pop(0)
            la=int(cls[0])
            if os.path.isfile(os.path.join(root, fn)):
                #imgs.append((fn, tuple([float(v) for v in cls[:len(cls)]])))
                imgs.append((fn, la))
                # access the last label
                # images is the list,and the content is the tuple, every image corresponds to a label
                # despite the label's dimension
                # we can use the append way to append the element for list
            c = c + 1
        print('the total image is',c)
        print(class_names)
        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):

        fn, label = self.imgs[index]  # even though the imgs is just a list, it can return the elements of it
        # in a proper way
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label), fn


    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)





def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res







class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""

    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 30
        self.epoch_losses[idx, 1] = val_loss * 30
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1800, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 5
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x30', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print('Saved figure')
        plt.close(fig)

def save_checkpoint(state, is_best, kwargs):
    torch.save(state, kwargs.checkpoint_path)
    if is_best:
        shutil.copyfile(kwargs.checkpoint_path, kwargs.best_checkpoint_path)

