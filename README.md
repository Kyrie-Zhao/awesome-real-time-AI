# Awesome EdgeAI Inference
This is a list of awesome EdgeAI inference related projects & papers.

![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)]

## Contents
- [Open Source Projects](#open-source-projects)
- [Papers](#papers)
  - [Survey](#survey)
  - [DNN Compiler](#dnn-compiler)
  - [Edge-Cloud Collaborative Inference](#edge-cloud-collaborative-inference)
  - [Concurrent DNN Inference](#concurrent-dnn-inference)
  - [Latency Predictor](#latency-predictor)
  - [TinyML](#tinyml)
  - [Multi-modality Inference](#multi-modality-inference)
  - [Sparse Inference](#sparse-inference)
  - [Others](#others)
 
 
## Open Source Projects
- [TVM](https://tvm.apache.org/)
- [KubeEdge](https://github.com/kubeedge/kubeedge)

## Papers

### Survey
- [Edge Intelligence: Architectures, Challenges, and Applications](https://arxiv.org/pdf/2003.12172) by Xu, Dianlei, et al., arxiv 2020
- [A Survey of Multi-Tenant Deep Learning Inference on GPU](https://arxiv.org/pdf/2203.09040) by Yu, Fuxun, et al., arxiv 2022

### DNN Compiler
- [TASO: The Tensor Algebra SuperOptimizer for Deep Learning](https://dl.acm.org/doi/abs/10.1145/3341301.3359630) by Zhihao Jia et al., SOSP 2019
- [AStitch: Enabling a New Multi-dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures](https://dl.acm.org/doi/10.1145/3503222.3507723) by Zhen Zheng et al., ASPLOS 2022
- [PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections](https://www.usenix.org/conference/osdi21/presentation/wang) by Haojie Wang et al., OSDI 2021
- [Rammer: Enabling Holistic Deep Learning Compiler Optimizations with rTasks](https://www.usenix.org/conference/osdi20/presentation/ma) by Lingxiao Ma et al., OSDI 2020
- [TASO: The Tensor Algebra SuperOptimizer for Deep Learning](https://dl.acm.org/doi/abs/10.1145/3341301.3359630) by Zhihao Jia et al., SOSP 2019
- [Bolt: Bridging the Gap between Auto-tuners and Hardware-native Performance](https://proceedings.mlsys.org/paper/2022/hash/38b3eff8baf56627478ec76a704e9b52-Abstract.html) by Jiarong Xing et al., MLSys 2022
- [Ansor: Generating High-Performance Tensor Programs for Deep Learning](https://arxiv.org/abs/2006.06762) by Lianmin Zheng et al., OSDI 2020
- [TenSet: A Large-scale Program Performance Dataset for Learned Tensor Compilers](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/a684eceee76fc522773286a895bc8436-Abstract-round1.html) by Lianmin Zheng., NeurIPS 2021
- [Romou: Rapidly Generate High-Performance Tensor Kernels for Mobile GPUs](https://www.microsoft.com/en-us/research/uploads/prod/2022/02/mobigpu_mobicom22_camera.pdf) by Liang, Rendong, et al., MobiCom 2022
- [Asymo: scalable and efficient deep-learning inference on asymmetric mobile cpus](https://dl.acm.org/doi/abs/10.1145/3447993.3448625) by Wang, Manni, et al., MobiCom 2021
- [Ios: Inter-operator scheduler for cnn acceleration](https://proceedings.mlsys.org/paper/2021/file/38b3eff8baf56627478ec76a704e9b52-Paper.pdf) by Ding, Yaoyao, et al., MLSys 2021
- [Moses: Efficient Exploitation of Cross-device Transferable Features for Tensor Program Optimization](https://arxiv.org/pdf/2201.05752) by Zhao, Zhihe, et al., arxiv 2022


### Edge-Cloud Collaborative Inference
- [EdgeML: An AutoML framework for real-time deep learning on the edge](https://dl.acm.org/doi/10.1145/3450268.3453520) by Zhao, Zhihe, et al., IoTDI 2021
- [SPINN: synergistic progressive inference of neural networks over device and cloud](https://dl.acm.org/doi/pdf/10.1145/3372224.3419194) by Laskaridis, Stefanos, et al., MobiCom 2020
- [Clio: Enabling automatic compilation of deep learning pipelines across iot and cloud](https://dl.acm.org/doi/pdf/10.1145/3372224.3419215?casa_token=XuIaaSOhXj8AAAAA:yC9swMPqSUSKBfe8yFelcaUvzBb3VHrpYroB87OFI0XgbEDZ6-EQipQFpnP9aduYdB3kjOu0MdNcRg) by Huang, Jin, et al., MobiCom 2020
- [Neurosurgeon: Collaborative intelligence between the cloud and mobile edge](https://dl.acm.org/doi/pdf/10.1145/3093337.3037698) by Kang, Yiping, et al., ASPLOS 2017
- [Mistify: Automating DNN Model Porting for On-Device Inference at the Edge](https://www.usenix.org/system/files/nsdi21-guo.pdf) by Guo, Peizhen et al., NSDI 2021

### Concurrent DNN Inference
- [VELTAIR: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling](https://dl.acm.org/doi/pdf/10.1145/3503222.3507752?casa_token=4WEkiHRYJHEAAAAA:Ae8yvTck-swW5LDJ3Cx3spak5Q2IzfRIVvPAvEG3zHkCudGBF0R4-XxeJk1hBaS4LmzCmxaAqNenCg) by Liu, Zihan, et al., ASPLOS 2021
- [RT-mDL: Supporting Real-Time Mixed Deep Learning Tasks on Edge Platforms](https://dl.acm.org/doi/pdf/10.1145/3485730.3485938?casa_token=gmqsW0h-7TUAAAAA:MqWvebvOkaT0kCORWXPCoIN5VEN0EBToR36zucKe63d1Exf9m-a5H8ebHVO2-OZZh6YB7DzvytW8QQ) by Ling, Neiwen, et al., SenSys 2021
- [Horus: Interference-aware and prediction-based scheduling in deep learning systems](https://ieeexplore.ieee.org/iel7/71/4359390/09428512.pdf) by Yeung, Gingfung, et al., IEEE TPDS 2021
- [Automated Runtime-Aware Scheduling for Multi-Tenant DNN Inference on GPU](https://ieeexplore.ieee.org/iel7/9643423/9643432/09643501.pdf) by Yu, Fuxun, et al., ICCAD 2021

### Latency Predictor
- [MAPLE-Edge: A Runtime Latency Predictor for Edge Devices](https://openaccess.thecvf.com/content/CVPR2022W/EVW/papers/Nair_MAPLE-Edge_A_Runtime_Latency_Predictor_for_Edge_Devices_CVPRW_2022_paper.pdf) by Nair, Saeejith, et al., CVPR 2022
- [nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices](https://dl.acm.org/doi/pdf/10.1145/3458864.3467882?casa_token=XqdJMALxIdkAAAAA:VhfaRoo7fWSlLOERqfqPcTjW4NfsBB0EXG7AKTw-s_eUFjMscdzTK6oU1kAPREY6nlF2jqO43PTk6g) by Zhang, Li Lyna, et al., MobiSys 2021

### TinyML
- [Mcunet: Tiny deep learning on iot devices](https://proceedings.neurips.cc/paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf) by Lin, Ji, et al. , Neurips 2020

### Multi-modality Inference
- [Dynamic Multimodal Fusion](https://arxiv.org/abs/2204.00102) by Xue, Zihui, and Radu Marculescu., arxiv 2022

### Sparse Inference
- [SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute](https://www.usenix.org/system/files/osdi22-zheng-ningxin.pdf) by Zheng, Ningxin, et al., OSDI 2022

### Others
