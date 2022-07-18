# Geometric loss functions for camera pose regression with deep learning



## Abstract

Deep learning has shown to be effective for robust and real-time monocular image relocalisation. In particular, PoseNet is a deep convolutional neural network which learns to regress the 6-DOF camera pose from a single image. It learns to localize using high level features and is robust to difficult lighting, motion blur and unknown camera intrinsics, where point based SIFT registration fails. However, it was trained using a naive loss function, with hyperparameters which require expensive tuning. In this paper, we give the problem a more fundamental theoretical treatment. **We explore a number of novel loss functions for learning camera pose which are based on geometry and scene reprojection error.** Additionally we show how to automatically learn an optimal weighting to simultaneously regress position and orientation. By leveraging geometry, we demonstrate that our technique significantly improves PoseNet’s performance across datasets ranging from indoor rooms to a small city.



**What did the paper mainly talking about**

- Background: mentioned PoseNet, using a deep CNN to learn; **Baseline is PoseNet**
- We did: Exploration a number of novel loss functions **based on geometry and scene reprojection error** for learning task
- Additionally: how to automatically learn an optimal weighting to simultaneously to do regression task.
- All tasks are focusing on **Orientation and Position**



## Conclusion

- Investigated a number of loss functions for learning
- Presented an algorithm for training PoseNet which does not require any hyper-parameter tuning
- Prooved on 3 large scale datasets, narrowed the performance gap to traditional point-feature approaches.



## Results



![Results](/Users/xieyang/Desktop/PaperReading/ComputerVision/3D SLAM/Geometric loss functions for camera pose regression with deep learning/Results.png)

