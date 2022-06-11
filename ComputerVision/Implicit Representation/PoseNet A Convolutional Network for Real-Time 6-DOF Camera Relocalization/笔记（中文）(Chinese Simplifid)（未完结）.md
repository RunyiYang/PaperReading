# PoseNet: 基于深度学习的拍照定位

![image-20220611185334007](/Users/xieyang/Library/Application Support/typora-user-images/image-20220611185334007.png)

Figure 1:**PoseNet: Convolutional neural network monocular camera relocalization**.Relocalization results for an inputimage (top), the predicted camera pose of a visual reconstruction (middle), shown again overlaid in red on the original image(bottom). Our system relocalizes to within approximately 2m and 3◦ for large outdoor scenes spanning50,000m2. For anonline demonstration, please see our project webpage:mi.eng.cam.ac.uk/projects/relocalisation/

# 摘要

我们提出了一种鲁棒、实时的单目六自由度重定位系统。我们的系统训练卷积神经网络，从端到端的单个RGB图像中回归6自由度相机位姿，无需额外的工程或图形优化。该算法可以在室内和室外实时运行，每帧计算时间为5ms。保证了大约2米和3度大规模场景的精度和0.5m和5度的室内精度。这是使用一个高效的23层深度卷积神经网络实现的，证明了CNN可以用于解决复杂的图像平面外回归问题。这是通过利用大规模分类数据的转移学习实现的。PoseNet从高级特征进行本地化，并且对难以照明、运动模糊和关键点检测registration fails的不同相机本质具有鲁棒性。此外，我们还展示了姿势特征是如何产生的，并将其推广到其他场景，从而使我们能够仅通过几十个训练示例来恢复位姿。

**我们可以总结一下这篇文章主要干了什么**

- 一个实时的、单目、六自由度重定位系统，保证了大约2米和3度大规模场景的精度和0.5m和5度的室内精度
- 23层卷积神经网络，输入是一个单目RGB图像，输出我猜测应该是位姿信息
- 证明了CNN可以用于解决复杂的图像平面外回归问题（我不知道这个证明的意义是什么，我刚入门，并不太了解背景）
- 通过利用大规模分类数据的转移学习实现的（具体什么是这个xxxx学习我并不是非常了解）
- 可以推广到其他场景，仅需要几十个训练实例来恢复位姿信息

## 1. 简介

### 背景

​	在传统的计算机视觉领域，许多研究基于三维点云方法Structure from Motion (SfM) ，在离线阶段采集大量的训练照片，进行两两匹配构造一个三维点云。在定位阶段，将用户的查询照片输入到系统中。系统将查询照片注册到点云，最终推测相机位置。虽然这种方法可以获得比较高的精度，但是受制于较大的计算量和较长的匹配时间。

### 本文主要贡献

- 主要贡献

​	一个基于深度卷积神经网络的相机位姿回归器

​	用了2个方法来实现：

​		1. 超大规模的分类数据集

​		2. 场景视频自动生成训练标签

- 次要贡献

​	理解卷积神经网络的生成表示，可以学习到容易映射到位姿的特征向量，并且很容易推广到其他场景。

### 基于外观的重定位和SLAM是两个重要的传统方法

### CNN的训练

​	训练CNN的过程总是需要很大的标签数据集，成本很高。

- 解决方案
  - 一种基于Structure from Motion的自动标记数据方法
  - 转移学习