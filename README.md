AMP-Net
===

This is the source code for paper

Adaptive multi-layer perceptual attention network for facial expression recognition
---
In complex real-world situations, problems, such as illumination changes, facial occlusion, and variant pose make facial expression recognition (FER) a challenging task. The current popular methods are mainly based on multilandmark-based patches and image-based patches to extract local and global facial information. However, these two methods either very much rely on the reliability of the landmark detector under occlusion or lack adaptability to facial organ allocation under variant poses. To solve the robustness problem, inspired by the coarse to fine facial perception mechanism of the human visual system, this paper proposed an adaptive multilayer perceptual attention network (AMP-Net) to extract global, local, and salient facial emotional features with different fine-grained features to obtain robust attributes. Aiming at the irregular facial appearance caused by occlusion and variant pose, AMP-Net can guide the network to adaptively focus on the fine subregions with a more reasonable allocation of facial organs and supplement facial features with a substantial emotional correlation to avoid the loss of network information.



## Environment requirest
Ubuntu 16.04 LTS, Python 3.5, PyTorch 1.3


## Training
* Step 1: In order to train your model, you need to first obtain the labels of the dataset and the coordinates of the five-point landmarks. Use the [RetinaFace toolkit](https://github.com/biubug6/Pytorch_Retinaface) to get  `data_label.txt` and `land_marks.npy` through `detect_torch.py`, 



* Step 2: download pre-trained model from Google Drive, and put it into ./checkpoint.

* Step 3: change data_path in main.py to your path

* Step 4: run python main.py


## If you use this work, please cite our paper



## Contributors

For any questions, feel free to open an issue or contact us:

* liuhw1@tongji.edu.cn
