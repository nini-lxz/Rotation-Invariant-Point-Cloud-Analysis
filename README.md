# A Rotation-Invariant Framework for Deep Point Cloud Analysis
by [Xianzhi Li](https://nini-lxz.github.io/), [Ruihui Li](https://liruihui.github.io/), Guangyong Chen, [Chi-Wing Fu](https://www.cse.cuhk.edu.hk/~cwfu/), [Daniel Cohen-Or](https://www.cs.tau.ac.il/~dcor/) and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/).

### Introduction
This repository is for our IEEE Transactions on Visualization and Computer Graphics (TVCG) 2021 paper [A Rotation-Invariant Framework for Deep Point Cloud Analysis](https://arxiv.org/pdf/2003.07238.pdf). In this paper, we introduce a new low-level purely rotation-invariant representation to replace common 3D Cartesian coordinates as the network inputs. Also, we present a network architecture to embed these representations into features, encoding local relations between points and their neighbors, and the global shape structure. To alleviate inevitable global information loss caused by the rotation-invariant representations, we further introduce a region relation convolution to encode local and non-local information. We evaluate our method on multiple point cloud analysis tasks, including (i) shape classification, (ii) part segmentation, and (iii) shape retrieval. Extensive experimental results show that our method achieves consistent, and also the best performance, on inputs at arbitrary orientations, compared with all the state-of-the-art methods.

### Usage
The code is tested under TF1.9.0, Python 2.7, and CUDA10.0 on Ubuntu 16.04.

For TF operators included under `3d_interpolation`, `grouping`, `nn_distance`, and `sampling` folders, you may need to compile them first. For details, please refer to the [PointNet++](https://github.com/charlesq34/pointnet2) GitHub page.

(1) You can directly test our previously-trained classification network (with z rotation augmentation) using our provided testing code (test_rotinv_cls.py). Here, we provide the [ModelNet40 data](https://gocuhk-my.sharepoint.com/:u:/g/personal/xianzhili_cuhk_edu_hk/ERIlBDvVyaZOufpRHGPEyJYB6IfiNns6t5TCEF0A16IxCA?e=ZFsW40) for testing.

(2) You can also re-train our network either using your own training data or our provided [training dataset](https://gocuhk-my.sharepoint.com/:u:/g/personal/xianzhili_cuhk_edu_hk/ERIlBDvVyaZOufpRHGPEyJYB6IfiNns6t5TCEF0A16IxCA?e=ZFsW40) using our provided training code (train_rotinv_cls.py). 

(3) You can directly test our previously-trained segmentation network using our provided testing code (test_rotinv_seg.py). In `log_so_seg` folder, the network is trained under arbitrary rotations. In `log_z_seg` folder, the network is trained under z rotations. For the object part segmentation dataset, please refer to the [PointNet++](https://github.com/charlesq34/pointnet2) GitHub page.

(4) You can also re-train our segmentation network using our provided training code (train_rotinv_seg.py).

### Questions
Please contact 'lixianzhi123@gmail.com'.

