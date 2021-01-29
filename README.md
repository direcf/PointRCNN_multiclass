# PointRCNN

## 3D Object Proposal Generation and Detection from Point Cloud [Multiclass]

Reference paper **PointRCNN:3D Object Proposal Generation and Detection from Point Cloud**, CVPR 2019.

**Authors**: [Shaoshuai Shi](https://sshaoshuai.github.io/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Hongsheng Li](http://www.ee.cuhk.edu.hk/~hsli/).

[[arXiv]](https://arxiv.org/abs/1812.04244)&nbsp;  [[Project Page]](#)&nbsp;

## Installation
### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04)
* Python 3.6+
* PyTorch 1.0

### Install PointRCNN 

a. Clone the PointRCNN_multiclass repository.
```shell
git clone https://github.com/sshaoshuai/PointRCNN.git
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.
```shell
conda env create --file environment.yaml
conda activate pointrcnn
```

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:
```shell
sh build_and_install.sh
```

## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
PointRCNN
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```
Here the images are only used for visualization and the [road planes](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing) are optional for data augmentation in the training.

## Training
a. Create ground truth Multiclass File (Need to change tools/default.yaml CLASSES: Car -> Multiclass)
```shell
python generate_gt_database.py --class_name 'Multiclass' --split train
```
b. RPN training (Create proposal candidates)
```shell
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200
```
c. RCNN training (Create target proposals)
```shell
python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn --epochs 100 --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth
```


## Inference
```shell
python eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_100.pth --eval_mode rcnn --test
```
If you want to test, not eval, change TEST.SPLIT: val => TEST.SPLIT: test in tools/default.yaml general training config part


## Citation
If you find this work useful in your research, please consider cite:
```
@InProceedings{Shi_2019_CVPR,
    author = {Shi, Shaoshuai and Wang, Xiaogang and Li, Hongsheng},
    title = {PointRCNN: 3D Object Proposal Generation and Detection From Point Cloud},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
}
```

## To Do List
- [x] Implement multiclass one shot learning and Release
- [ ] Increase Pedestrain, Cyclist Detection Accuracy (Data unbalance problem)
- [ ] Annotation arrangement
