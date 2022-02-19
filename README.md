AMP-Net
===

This is the source code for paper

Adaptive multi-layer perceptual attention network for facial expression recognition
---

![The structure of the proposed method](https://github.com/liuhw01/dd/blob/main/occulsion.jpg)
This paper proposed an adaptive multilayer perceptual attention network (AMP-Net) to extract global, local, and salient facial emotional features with different fine-grained features to obtain robust attributes. Aiming at the irregular facial appearance caused by occlusion and variant pose, AMP-Net can guide the network to adaptively focus on the fine subregions with a more reasonable allocation of facial organs and supplement facial features with a substantial emotional correlation to avoid the loss of network information.

## Environment requirest
Ubuntu 16.04 LTS, Python 3.5, PyTorch 1.3


## Training
* Step 1: In order to train your model, you need to first obtain the labels of the dataset and the coordinates of the five-point landmarks. Use the [RetinaFace toolkit](https://github.com/biubug6/Pytorch_Retinaface) to get dataset list `data_label.txt` and facial key point `land_marks.npy` through `detect_torch.py` and place in `./index`.
    
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




* Step 2: download pre-trained model from Google Drive, and put it into ./checkpoint.

* Step 3: change data_path in main.py to your path

* Step 4: run python main.py


## If you use this work, please cite our paper



## Contributors

For any questions, feel free to open an issue or contact us:

* liuhw1@tongji.edu.cn
