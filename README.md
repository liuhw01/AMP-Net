AMP-Net
===

This is the source code for paper

IEEE Transactions on Circuits and Systems for Video Technology

Adaptive multilayer perceptual attention network for facial expression recognition
---

![The structure of the proposed method](https://github.com/liuhw01/AMP-Net/blob/main/checkpoint/proposed%20method.png)
This paper proposed an adaptive multilayer perceptual attention network (AMP-Net) to extract global, local, and salient facial emotional features with different fine-grained features to learn the underlying diversity and key information of facial emotions. AMP-Net can adaptively guide the network to focus on multiple finer and distinguishable local patches with robustness to occlusion and variant pose, improving the effectiveness of learning potential facial diversity information. In addition, the proposed global perception module can learn different receptive field features in the global perception domain, and AMP-Net also supplements salient facial regions features with high emotion correlation based on prior knowledge to capture key texture details and avoid important information loss. 

## Environment requirest
Ubuntu 20.04 LTS, Python 3.8, PyTorch 1.9.0


## Training
* Step 1: To train your model, you need to first obtain the labels of the dataset and the coordinates of the five-point landmarks. Use the [RetinaFace toolkit](https://github.com/biubug6/Pytorch_Retinaface) to get dataset list `data_label.txt` and facial key point `land_marks.npy` through `detect_torch.py` and place in `./index`.
    
    Download [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset  and the prepared file format is as follows:
```
./index/data_label.txt
    train_00001.jpg 5
    train_00002.jpg 5
    train_00003.jpg 4
    train_00004.jpg 4
./index/land_marks.npy
    'train_00001.jpg', [[65,80],[156,73],[113,130],[93,157],[134,152]]
    'train_00002.jpg', [[79,98],[152,98],[125,133],[88,165],[141,165]]

[Note] land_marks.npy:  
''image_name', [[Xeye1,Yeye1],[Xeye2,Yeye2],[Xnose,Ynose],[Xmouth1,Ymouth1],[Xmouth2,Ymouth2]]'
```




* Step 2: download pre-trained model from [Google Drive](https://drive.google.com/file/d/11KU4hI-kTgIsgmZqSj8Mt-WvTjFAFld0/view?usp=sharing), and put it into `./checkpoint`.

* Step 3: change data_path in `main.py` to your path

* Step 4: run `python main.py`

Training results can be downloaded in [Google Drive](https://drive.google.com/drive/folders/1isqsXl-6sQ0N2CK4MPpiS9ZtZ4oSwrL_?usp=sharing).

## If you use this work, please cite our paper

```
@ARTICLE{9750079,
  author={Liu, Hanwei and Cai, Huiling and Lin, Qingcheng and Li, Xuefeng and Xiao, Hui},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Adaptive multilayer perceptual attention network for facial expression recognition}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3165321}}
```

