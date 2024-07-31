# Basic-Information-Consolidation
A collection of papers from **Binocular Stereo Vision** to **Multi-View Stereo Vision**.

**Mastery of Basic Terms in This Field: Some Examples**

![Example Image](pic/examples.png "examples1")

- **Epipolar Geometry:** Epipolar geometry
- **Baseline (baseline):** The straight line Oc-Oc’ is the baseline.
- **Epipolar Pencil:** A pencil of planes with the baseline as the axis.
- **Epipolar Plane:** Any plane that includes the baseline is called an epipolar plane.
- **Epipole:** The intersection of the camera's baseline with each image. For example, points e and e’ in the above figure.
- **Epipolar Line:** The intersection line between the epipolar plane and the image. For example, lines l and l’ in the above figure.
- **Coplanarity of Five Points:** Points x, x’, camera centers Oc and Oc’, and the space point X are coplanar.
- **Epipolar Constraint:** The correspondence relationship between points on the epipolar lines.
  - Explanation: Line l is the epipolar line corresponding to point x’, and line l’ is the epipolar line corresponding to point x. The epipolar constraint means that point x’ must lie on the epipolar line l’ corresponding to x, and point x must lie on the epipolar line l corresponding to x’.

- **Stereo Matching:**
  - After rectification, the next step is correspondence, also known as stereo matching. 
  - Intuitively, this involves finding points in the left and right images that correspond to the same point in reality. Through the disparity between these two points, the depth information of this point in reality can be obtained.

- **Matching Cost:**
  - Since it involves finding the same points in two images, it is necessary to determine the similarity between the two points. This similarity description is called matching cost.
  - However, it is unreasonable to consider only a single point because there are definitely relationships between pixels in the image. Therefore, it is necessary to consider the correlation between pixels to optimize the initial cost.

- **Disparity:** It is the calculation of |XR - XT| in the diagram below.

![Example Image](pic/examples.png "examples2")

## Surveys or Papers
- Stereo vision algorithms and applications [[paper](https://pan.baidu.com/s/1aCDHkCDUd7gjEFX2l3m-LA)]
- An invitation to 3D vision [[paper](https://www.academia.edu/5458357/An_invitation_to_3D_vision)]
- FADNet: A Fast and Accurate Network for Disparity Estimation [[paper](https://arxiv.org/abs/2003.10758)]
- FADNet++: Real-Time and Accurate Disparity Estimation with Configurable Networks [[paper](https://arxiv.org/abs/2110.02582)]
- CF-NeRF: Camera Parameter Free Neural Radiance Fields with Incremental Learning [[paper](https://arxiv.org/abs/2312.08760)]
- Rethinking Disparity: A Depth Range Free Multi-View Stereo Based on Disparity [[paper](https://arxiv.org/abs/2211.16905)]

## Datasets
- KITTI： Vision meets Robotics-The KITTI Dataset [[paper](https://www.cvlibs.net/publications/Geiger2013IJRR.pdf)]
- Middlebury: Stereo datasets with ground truth [[paper](https://vision.middlebury.edu/stereo/data/)]
- Scene Flow： A Large Dataset to Train Convolutional Networks for Disparity, Optical Flow, and Scene Flow Estimation [[paper](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)]
- IRS: A Large Synthetic Indoor Robotics Stereo Dataset for Disparity and Surface Normal Estimation [[paper](https://arxiv.org/abs/1912.09678)]
- ETH3D: A Multi-View Stereo Benchmark With High-Resolution Images and Multi-Camera Videos [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Schops_A_Multi-View_Stereo_CVPR_2017_paper.pdf)]
- DTU: Large Scale Multi-view Stereopsis Evaluation [[paper](https://paperswithcode.com/dataset/dtu)]

## Progress Of Specific 3D Reconstruction Task 
Please check the doc to get the newest progress.

## Weekly Paper Report of Some Papers
Please check the doc to get the newest progress.
