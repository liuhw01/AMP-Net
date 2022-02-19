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
import pandas as pd


# Create torchvision model
return_layers = {'layer2':1,'layer3':2,'layer4':3}
RetinaFace = torchvision_model.create_retinaface(return_layers)


kwargs ={
  'image_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/aligned', # This is the path to your dataset,
    'image_save':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/dataset', # The path to save the image after cropping
    'image_list':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/list_patition_label.txt', #List of pictures of the RAF dataset
  'err_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index_old/err.csv', # Images with no keypoints detected
  'label_mark':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index/data_label.txt', # image list
  'land_marks':'/home/lighting/liuhanwei/Occlusion FER/my occusion/RAF/index/land_marks.npy', # key point
    'model_path':'/home/lighting/liuhanwei/Occlusion FER/my occusion/model/model.pt', # RetinaFace model
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




image_path = kwargs['image_path']   #  This is the path to your dataset,
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
            thickness = -1

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





#save err csv

err=pd.DataFrame(rect_all_err,columns=None)

err.to_csv(kwargs['err_path'],columns=None,header=None,index=None)
np.save(kwargs['land_marks'],land_marks)

### Images that do not meet the requirements
import csv
csv_path = kwargs['err_path']
with open(csv_path,'r',encoding='utf8')as fp:
    err_label = [i for i in csv.reader(fp)]  # csv.reader


file = open(txt_path)
lines = file.readlines()
for i in range(len(lines)):
    lines[i] = lines[i].split()

### Eliminate wrong pictures
err_num=[i  for i in range(len(lines)) for j in range(len(err_label)) if lines[i][0]==err_label[j][0]]
print(len(err_label))
print(len(err_num))
err_num=np.array(err_num)

for counter, index in enumerate(err_num):
    index = index - counter
    lines.pop(index)

# save
new_data_label=kwargs['label_mark']
file = open(new_data_label,'w');
for line in lines:
    name=str(line[0])+' '+str(line[1])+'\n'
    file.write(name)
file.close()

