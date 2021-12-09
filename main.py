#from model_mcnet import mcnet
import numpy as np
import torch
from torchsummary import summary
import random
import time
from torchvision import datasets, transforms
import torch.utils.data as data
import torchvision.models as models
import os
import torch.nn as nn
import torch.backends.cudnn as cudnn
import datetime
import matplotlib.pyplot as plt
from util import *
from PIL import Image
############
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#device = torch.device('cuda:1')
#torch.cuda.set_device(1)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        txt_name = './log/' + time_str + 'log.txt'
        with open(os.path.join(save_path,txt_name), 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def train(train_loader, model,data_facial, criterion, optimizer, epoch, kwargs,txt_name,time_str):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    top1_1 = AverageMeter('Accuracy', ':6.3f')
    top1_2 = AverageMeter('Accuracy', ':6.3f')
    #top1_3 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    l=kwargs['range']
    data_name=[]
    for i in range(len(data_facial)):
        data_name.append(data_facial[i][0])
    for i, (images, target, fn) in enumerate(
            train_loader):  # the first i in index, and the () is the content
        # search 相应的面部地标
        facial_indx = []
        for j in range(len(fn)):
            facial_indx.append(data_name.index(fn[j]))
        facial=data_facial[facial_indx,1]
        facial = np.stack(facial, axis=0)
        images,rect,rect_local= pre_pro(images,facial,0.8,0.5,l,kwargs['num_workers'])

        #fan zhuan

        images = images.cuda()
        target = target.cuda()

        model.set_rect(rect)
        model.set_rect_local(rect_local)

        # compute output  1:自适应 2：局部 3：全局
        #output = model(images)
        output1, output2 = model(images)
        #output1 = (0.25 * output1_1+ 0.25 * output1_2 + 0.25 * output1_3 + 0.25 * output1_4)
        output =  ( kwargs['beta1'] * output1) + ( (1-kwargs['beta1']) * output2)
        loss = (kwargs['beta1'] * criterion(output1, target)) + ((1-kwargs['beta1'])* criterion(output2, target))
        #loss = criterion(output, target)
        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        acc2, _ = accuracy(output1, target, topk=(1, 5))
        acc3, _ = accuracy(output2, target, topk=(1, 5))
        #acc4, _ = accuracy(output3, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top1_1.update(acc2[0], images.size(0))
        top1_2.update(acc3[0], images.size(0))
        #top1_3.update(acc4[0], images.size(0))



        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % kwargs['print_freq'] == 0:
            progress.display(i)

    ### ia zai
    name1='log/' + time_str + 'log_err_out.txt'
    with open(os.path.join(kwargs['save_path'],txt_name), 'a') as f:
        f.write(' * training  Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
    with open(os.path.join(kwargs['save_path'],name1), 'a') as f:
        f.write(' * training Accuracy,output1: {top1_1.avg:.3f}'.format(top1_1=top1_1) + ' * Accuracy,output2: {top1_2.avg:.3f}'.format(top1_2=top1_2)+ '\n')


    return top1.avg, losses.avg


def validate(val_loader, model, data_facial,criterion, kwargs, txt_name,time_str,):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    top1_1 = AverageMeter('Accuracy', ':6.3f')
    top1_2 = AverageMeter('Accuracy', ':6.3f')
    #top1_3 = AverageMeter('Accuracy', ':6.3f')

    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    l=kwargs['range']
    with torch.no_grad():
        data_name = []
        for i in range(len(data_facial)):
            data_name.append(data_facial[i][0])
        for i, (images, target, fn) in  enumerate(val_loader):
            # search 相应的面部地标
            # search 相应的面部地标
            facial_indx = []
            for j in range(len(fn)):
                facial_indx.append(data_name.index(fn[j]))
            facial = data_facial[facial_indx, 1]
            facial = np.stack(facial, axis=0)
            images, rect, rect_local = pre_pro(images, facial, 0, 0, l, kwargs['num_workers'])

            # fan zhuan

            images = images.cuda()
            target = target.cuda()

            model.set_rect(rect)
            model.set_rect_local(rect_local)

            # compute output  1:自适应 2：局部 3：全局output2 = model(images)
            #output = model(images)
            output1, output2 = model(images)
            #output1=(0.25* output1_1+ 0.25 * output1_2+ 0.25* output1_3+ 0.25*output1_4 )
            output = (kwargs['beta1'] * output1) + ((1 - kwargs['beta1']) * output2)
            loss = (kwargs['beta1'] * criterion(output1, target)) + ((1 - kwargs['beta1']) * criterion(output2, target))

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc, _ = accuracy(output, target, topk=(1, 5))
            acc1, _ = accuracy(output1, target, topk=(1, 5))
            acc2, _ = accuracy(output2, target, topk=(1, 5))
           # acc3, _ = accuracy(output3, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc[0], images.size(0))
            top1_1.update(acc1[0], images.size(0))
            top1_2.update(acc2[0], images.size(0))
           # top1_3.update(acc3[0], images.size(0))

            if i % kwargs['print_freq'] == 0:
                progress.display(i)

        print(' **** Accuracy {top1.avg:.3f} *** '.format(top1=top1))

        name1='log/' + time_str + 'log_err_out.txt'
        with open(os.path.join(kwargs['save_path'],txt_name), 'a') as f:
            f.write(' * vail Accuracy {top1.avg:.3f}'.format(top1=top1) + '\n')
        with open(os.path.join(kwargs['save_path'],name1), 'a') as f:
            f.write(' * vail Accuracy,output1: {top1_1.avg:.3f}'.format(top1_1=top1_1) + ' * vail  Accuracy,output2: {top1_2.avg:.3f}'.format(top1_2=top1_2)  +'\n')

    return top1.avg, losses.avg





############# 读取



now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")
best_acc = 0
print('Training time: ' + now.strftime("%m-%d %H:%M"))

kwargs = {'num_workers': 16,
          'pin_memory': True,
          'beta1':0.5,
          'beta2':0.4,
          'beta3':0.3,
          'print_freq':10,
          'checkpoint_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/recoder_new/' + time_str + 'model.pth',
          'best_checkpoint_path': '/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/recoder_new/' + time_str +'model_best.pth',
          'save_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/recoder_new',
          'data_root':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/re_dataset', # 新的数据集
          'data_label':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index/data_label.txt', #新的数据集label
          'data_mark_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index/land_marks.npy', #面部地标坐标
          'land_marks':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index/land_marks.npy', # 5征点
          'lr':0.01,
          'momentum':0.9,
          'weight_decay':1e-4,
          'epochs': 100,
          'batch_size':300,
          'resume':False,
          'start_epoch':0,
          'range':6,
          'dataset': 'RAF',
          }

save_path=kwargs['save_path']
lr=kwargs['lr']
momentum=kwargs['momentum']
weight_decay=kwargs['weight_decay']
epochs=kwargs['epochs']
batch_size = kwargs['batch_size']





#加载模型
model = mcnet()
# 加载预训练模型

#resnet50 = models.resnet50(pretrained=True)
#pretrained_dict = resnet50.state_dict()

checkpoint_load=r'/home/lighting/liuhanwei/Occlusion FER/my occusion/face_dataset/faces_vgg_112x112/recoder/[10-14]-[10-52]-model_best.pth'
checkpoint = torch.load(checkpoint_load)
# 加载预训练模型
#我们的
model_dict = model.state_dict()

# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}

model.load_state_dict(checkpoint['state_dict'])
model.fc_5= torch.nn.Linear(512 * 2, 7)
model.fc_3= torch.nn.Linear(512, 7)
model.fc_1_1=torch.nn.Linear(512, 7)


model.cuda()
#
torch.cuda.device_count()
# 查看可用GPU型号
torch.cuda.current_device()
# GPU并行计算
#model = torch.nn.DataParallel(model).cuda()

# define loss function (criterion) and optimizer
#criterion = LabelSmoothingCrossEntropy().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
recorder = RecorderMeter(epochs)
################可选从检查点恢复

# xuan

if kwargs['resume']:
    if os.path.isfile(checkpoint_load):
        print("=> loading checkpoint '{}'".format(checkpoint_load))
        checkpoint = torch.load(checkpoint_load)
        kwargs['start_epoch'] = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        recorder = checkpoint['recorder']
        best_acc = best_acc.to()
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_load, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_load))

############

cudnn.benchmark = True
# data_root 数据集  data_label 数据集标签
data_root = kwargs['data_root']
data_label =  kwargs['data_label']

data_facial_path=kwargs['land_marks']
data_facial= np.load(data_facial_path,allow_pickle=True)

mytransform = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                  transforms.RandomGrayscale(p=0.2),
                                  transforms.ToTensor()])  # transform [0,255] to [0,1]


mytransform1 = transforms.Compose([ transforms.ToTensor()])  # transform [0,255] to [0,1]

#划分数据集和测试集  8:2
if kwargs['dataset']=='RAF':
    train_label, test_label = choose_data(data_label)
else:
    train_label, test_label = random_choose_data(data_label)
train_data=myImageFloder(root=data_root, label=train_label, transform=mytransform)
test_data=myImageFloder(root=data_root, label=test_label, transform=mytransform1)
val_data=myImageFloder(root=data_root, label=test_label, transform=mytransform1)

# 读取训练集
train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True,num_workers= kwargs['num_workers'], pin_memory=kwargs['pin_memory'])
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size, shuffle=True,num_workers= kwargs['num_workers'], pin_memory=kwargs['pin_memory'])
val_loader = torch.utils.data.DataLoader(val_data,batch_size=batch_size, shuffle=True,num_workers= kwargs['num_workers'], pin_memory=kwargs['pin_memory'])




for epoch in range(kwargs['start_epoch'], epochs):
    start_time = time.time()
    current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']
    print('Current learning rate: ', current_learning_rate)
    txt_name = 'log/' + time_str + 'log.txt'
    with open(os.path.join(save_path,txt_name), 'a') as f:
        f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

    # train for one epoch
    train_acc, train_los = train(train_loader, model, data_facial, criterion, optimizer, epoch, kwargs,txt_name,time_str)

    # evaluate on validation set
    val_acc, val_los = validate(val_loader, model, data_facial, criterion, kwargs,txt_name,time_str)

    scheduler.step()

    recorder.update(epoch, train_los, train_acc, val_los, val_acc)
    curve_name = time_str + 'cnn.png'

    recorder.plot_curve(os.path.join(save_path,'log/', curve_name))

    # remember best acc and save checkpoint
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)

    print('Current best accuracy: ', best_acc.item())
    txt_name = 'log/' + time_str + 'log.txt'
    with open(os.path.join(save_path,txt_name), 'a') as f:
        f.write('Current best accuracy: ' + str(best_acc.item()) + '\n')

    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'best_acc': best_acc,
                     'optimizer': optimizer.state_dict(),
                     'recorder': recorder}, is_best, kwargs)
    end_time = time.time()
    epoch_time = end_time - start_time
    print("An Epoch Time: ", epoch_time)
    txt_name = 'log/' + time_str + 'log.txt'
    with open(os.path.join(save_path,txt_name), 'a') as f:
        f.write(str(epoch_time) + '\n')





















