## AMP-Net

This is the source code for paper

Adaptive multi-layer perceptual attention network for facial expression recognition


## Environment requirest
Ubuntu 16.04 LTS, Python 3.5, PyTorch 1.3


## Training
* Step 1: In order to train your model, you need to first obtain the labels of the dataset and the coordinates of the five-point landmarks. Use the [RetinaFace toolkit](https://github.com/biubug6/Pytorch_Retinaface) to get 'data_label.txt' and 'land_marks.npy' through 'detect_torch.py'

* Step 2: download pre-trained model from Google Drive, and put it into ./checkpoint.

* Step 3: change data_path in main.py to your path

* Step 4: run python main.py


## If you use this work, please cite our paper



## Contributors

For any questions, feel free to open an issue or contact us:

* liuhw1@tongji.edu.cn
