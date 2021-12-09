import torch
import torch.nn.functional as F
import numpy as np
import skimage
from skimage import io
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import eval_widerface
import torchvision_model



# Create torchvision model
return_layers = {'layer2':1,'layer3':2,'layer4':3}
RetinaFace = torchvision_model.create_retinaface(return_layers)

kwargs ={
  'image_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/aligned', # 这里是你的图片路径,
    'image_save':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/dataset',
    'image_list':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/list_patition_label.txt',
  'err_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index_old/err.csv', # 未检测图像
  'label_mark':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index_old/data_label.txt',
    'label_mark_new':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index/data_label.txt',
  'land_marks':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index/land_marks.npy',
    'model_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/model/model.pt',
}

# Load trained model
retina_dict = RetinaFace.state_dict()
model_path=kwargs['model_path']
pre_state_dict = torch.load(model_path)
pretrained_dict = {k[7:]: v for k, v in pre_state_dict.items() if k[7:] in retina_dict}
RetinaFace.load_state_dict(pretrained_dict)

RetinaFace = RetinaFace.cuda()
RetinaFace.eval()

data_save=kwargs['image_save']






image_path = kwargs['image_path']   # 这里是你的图片路径
label_mark=kwargs['label_mark']



txt_path = kwargs['image_list']
file = open(txt_path)
label_old = file.readlines()
image_name=[]
label_l=[]
for line in label_old:  # label is a list
    cls = line.split()  # cls is a lis
    image_name.append(str(cls[0][0:-4]+'_aligned.jpg'))
    label_l.append(cls[1])


# 保存 index\data_label.txt
data_label=kwargs['label_mark']

file = open(data_label,'w');
for i in range(len(image_name)):
    name=str(image_name[i])+' '+str(label_l[i])+'\n'
    file.write(name)
file.close()


land_marks=[]
rect_all = []  # rect=[name,[x1,y1,x2,y2]*4,[xx1,xx2,yy1,yy2]*4]
rect_all_err=[]




for i in range(len(image_name)):
    image=image_name[i]
    img_path=os.path.join(image_path,image)
    img = cv2.imread(img_path)
    if img is not None:
        #cv2.imwrite(os.path.join(data_save, image), img)

        img1 = torch.from_numpy(img)
        img1 = img1.permute(2,0,1)
        input_img = img1.unsqueeze(0).float().cuda()
        picked_boxes, picked_landmarks, picked_scores = eval_widerface.get_detections(input_img, RetinaFace, score_threshold=0.7, iou_threshold=0.8)
        # anduan
        if picked_landmarks[0] is not None:
            # np_img = resized_img.cpu().permute(1,2,0).numpy()
            rect = []
            mark = []
            mark.append(image)
            rect.append(image)
            print(image)

            land=picked_landmarks[0].cpu().numpy()
            if land.shape[0]>1:
                land=land[0]
            land=land.reshape(5,2)

            point_size = int(img.shape[0] / 100)
            point_color = (0, 0, 255)  # BGR
            thickness = -1  # 可以为 0 、4、8

            # resize
            point_size = 5
            img_resize = cv2.resize(img, (224, 224))
            land = np.array(land)
            land_resize = land * (224 / len(img))
            # io.imsave(os.path.join(img_save1,image), img_resize)
            cv2.imwrite(os.path.join(data_save, image), img_resize)
            mark.append(land_resize)
            land_marks.append(mark)
        else:
            rect_all_err.append(image)
    else:
        rect_all_err.append(image)



from retinaface import RetinaFace
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

rect_all_err_new=[]

for i in range(len(rect_all_err)):
    image = rect_all_err[i]
    img_path = os.path.join(image_path, image)
    img = cv2.imread(img_path)
    #img = cv2.resize(img, (224, 224))
    top_size, bottom_size, left_size, right_size = (img.shape[0], img.shape[0], img.shape[0], img.shape[0])
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT, value=0)

    obj = RetinaFace.detect_faces(constant)

    # 判断是否存在
    face_exists = 'face_1' in obj
    if face_exists==True:

        for key in obj.keys():
          identity = obj[key]
          #print(identity)

          facial_area = identity["facial_area"]
          lanmarks=identity["landmarks"]
          # np_img = resized_img.cpu().permute(1,2,0).numpy()
        land=[]
        for row, key in lanmarks.items():
            land.append(key)
        land=np.array(land)
        # 判断面部地标是否在有效位置
        land_false = False
        for i in range(len(land)):
            land[i][0] = land[i][0] - top_size
            land[i][1] = land[i][1] - top_size
            if land[i][0] < 0 or land[i][0] > top_size:
                land_false = True
            if land[i][1] < 0 or land[i][1] > top_size:
                land_false = True
        # 满足条件则跳出循环
        if land_false:
            rect_all_err_new.append(image)
            continue
        # np_img = resized_img.cpu().permute(1,2,0).numpy()
        else:
            rect = []
            mark = []
            mark.append(image)
            rect.append(image)
            print(image)

            img_resize = cv2.resize(img, (224, 224))
            land_resize = land * (224 / len(img))
            # io.imsave(os.path.join(img_save1,image), img_resize)
            cv2.imwrite(os.path.join(data_save, image), img_resize)
            mark.append(land_resize)
            land_marks.append(mark)

    else:
        rect_all_err_new.append(image)










#保存csv

import pandas as pd
err=pd.DataFrame(rect_all_err_new,columns=None)

err.to_csv(kwargs['err_path'],columns=None,header=None,index=None)
np.save(kwargs['land_marks'],land_marks)





####################################################


import numpy as np


########## 1. 生成 data_label.txt数据集：[name,label]







### 不符合要求的图片，err.csv
import csv
csv_path = kwargs['err_path']
# 通过with语句读取，以列表类型读取
with open(csv_path,'r',encoding='utf8')as fp:
    # 使用列表推导式，将读取到的数据装进列表
    err_label = [i for i in csv.reader(fp)]  # csv.reader 读取到的数据是list类型




label_path=kwargs['label_mark']
file = open(label_path)
lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

### 剔除
err_num=[i  for i in range(len(lines)) for j in range(len(err_label)) if lines[i][0]==err_label[j][0]]
print(len(err_label))
print(len(err_num))
err_num=np.array(err_num)

for counter, index in enumerate(err_num):
    index = index - counter
    lines.pop(index)

# 保存 index\data_label.txt
new_data_label=kwargs['label_mark_new']
file = open(new_data_label,'w');
for line in lines:
    name=str(line[0])+' '+str(line[1])+'\n'
    file.write(name)
file.close()

