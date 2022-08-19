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
  - [Heterogeneous Platforms](#heterogeneous-platform)
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
- [DeepCuts: A Deep Learning Optimization Framework for Versatile GPU Workloads](https://dl.acm.org/doi/pdf/10.1145/3453483.3454038) by Jung, Wookeun, Thanh Tuan Dao, and Jaejin Lee., PLDI 2021
- [CASE: a compiler-assisted SchEduling framework for multi-GPU systems](https://dl.acm.org/doi/pdf/10.1145/3503221.3508423?casa_token=c738_W6lYCkAAAAA:QcpPBlGQcRjbfbnr99-N7fJXV5ssZMVL9odTdrfRFRLXtaOcAAQOeh3JwhEeaQvOEpX1IRJquURmTg) by Chen, Chao, Chris Porter, and Santosh Pande., PPoPP 2022


### Edge-Cloud Collaborative Inference
- [EdgeML: An AutoML framework for real-time deep learning on the edge](https://dl.acm.org/doi/10.1145/3450268.3453520) by Zhao, Zhihe, et al., IoTDI 2021
- [SPINN: synergistic progressive inference of neural networks over device and cloud](https://dl.acm.org/doi/pdf/10.1145/3372224.3419194) by Laskaridis, Stefanos, et al., MobiCom 2020
- [Clio: Enabling automatic compilation of deep learning pipelines across iot and cloud](https://dl.acm.org/doi/pdf/10.1145/3372224.3419215?casa_token=XuIaaSOhXj8AAAAA:yC9swMPqSUSKBfe8yFelcaUvzBb3VHrpYroB87OFI0XgbEDZ6-EQipQFpnP9aduYdB3kjOu0MdNcRg) by Huang, Jin, et al., MobiCom 2020
- [Neurosurgeon: Collaborative intelligence between the cloud and mobile edge](https://dl.acm.org/doi/pdf/10.1145/3093337.3037698) by Kang, Yiping, et al., ASPLOS 2017
- [Mistify: Automating DNN Model Porting for On-Device Inference at the Edge](https://www.usenix.org/system/files/nsdi21-guo.pdf) by Guo, Peizhen et al., NSDI 2021
- [Deep compressive offloading: Speeding up neural network inference by trading edge computation for network latency.](https://dl.acm.org/doi/pdf/10.1145/3384419.3430898) by Yao, Shuochao, et al., SenSys 2020

### Concurrent DNN Inference
- [VELTAIR: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling](https://dl.acm.org/doi/pdf/10.1145/3503222.3507752?casa_token=4WEkiHRYJHEAAAAA:Ae8yvTck-swW5LDJ3Cx3spak5Q2IzfRIVvPAvEG3zHkCudGBF0R4-XxeJk1hBaS4LmzCmxaAqNenCg) by Liu, Zihan, et al., ASPLOS 2021
- [RT-mDL: Supporting Real-Time Mixed Deep Learning Tasks on Edge Platforms](https://dl.acm.org/doi/pdf/10.1145/3485730.3485938?casa_token=gmqsW0h-7TUAAAAA:MqWvebvOkaT0kCORWXPCoIN5VEN0EBToR36zucKe63d1Exf9m-a5H8ebHVO2-OZZh6YB7DzvytW8QQ) by Ling, Neiwen, et al., SenSys 2021
- [Horus: Interference-aware and prediction-based scheduling in deep learning systems](https://ieeexplore.ieee.org/iel7/71/4359390/09428512.pdf) by Yeung, Gingfung, et al., IEEE TPDS 2021
- [Automated Runtime-Aware Scheduling for Multi-Tenant DNN Inference on GPU](https://ieeexplore.ieee.org/iel7/9643423/9643432/09643501.pdf) by Yu, Fuxun, et al., ICCAD 2021
- [Interference-aware scheduling for inference serving](https://dl.acm.org/doi/pdf/10.1145/3437984.3458837?casa_token=CWMAG53X-EoAAAAA:FXA3ZHvAzasGqkaUPoVoXNREe1qEzYkaFQramYjfG_mu0MjJVJdQzH_IjBPGAIuhkmq9CgPuAQ) by Mendoza, Daniel, et al., EuroMLSys
- [Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences](https://www.usenix.org/system/files/osdi22-han.pdf) by Han, Mingcong, et al., OSDI 2022
- [Planaria: Dynamic architecture fission for spatial multi-tenant acceleration of deep neural networks](https://ieeexplore.ieee.org/iel7/9251289/9251849/09251939.pdf?casa_token=JK4fOIekLU0AAAAA:zLGfIHmDMqPICBSEjAYEa1xHuFRBXTN4Kc-bjO8eZD-VhVIuNlV2B_UCCSXdFY-bWUSOC1xbJqM) by Ghodrati, Soroush, et al., MICRO 2020
- [Heimdall: mobile GPU coordination platform for augmented reality applications](https://dl.acm.org/doi/pdf/10.1145/3372224.3419192?casa_token=TDdZgOLYfioAAAAA:w_HWQIpOiic498lYa-z6kRUYp4RxS92RiVoxNP6P6vBR7QvK2wyAYS4h77wEIh9ogv6B1n29zwfedw) by Yi, Juheon, and Youngki Lee., MobiCom 2020
- [Deepeye: Resource efficient local execution of multiple deep vision models using wearable commodity hardware](https://dl.acm.org/doi/pdf/10.1145/3081333.3081359) by Mathur, Akhil, et al., MobiSys 2017
- [PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications](https://www.usenix.org/system/files/osdi20-bai.pdf) by Bai, Zhihao, et al., OSDI 2020

### Heterogeneous Platforms
- [Lalarand: Flexible layer-by-layer cpu/gpu scheduling for real-time dnn tasks](https://ieeexplore.ieee.org/iel7/9622323/9622324/09622325.pdf?casa_token=HtllgvtNt8wAAAAA:hasDDmtMmf8uVWgtrevp1XT2Ldh4u-0bWMjup4VPqT1PsCbq77cgwOMOAeAboXs_J_Goklo) by Kang, Woosung, et al., RTSS 2021
- [DUET: A Compiler-Runtime Subgraph Scheduling Approach for Tensor Programs on a Coupled CPU-GPU Architecture](https://dl.acm.org/doi/pdf/10.1145/3442442.3452297?casa_token=rNEtNHavbREAAAAA:U4wL45EN3WD8dh_CW41BVZSAcdKCzDO1KRvNRJsCIMTk6SoeaUa2NvvosOkN6vLnCLcZHt7sMg) by Zhang, Minjia, Zehua Hu, and Mingqin Li., IPDPS 2021
- [Band: coordinated multi-DNN inference on heterogeneous mobile processors](https://dl.acm.org/doi/pdf/10.1145/3498361.3538948) by Jeong, Joo Seong, et al., MobiSys 2022

### Latency Predictor
- [MAPLE-Edge: A Runtime Latency Predictor for Edge Devices](https://openaccess.thecvf.com/content/CVPR2022W/EVW/papers/Nair_MAPLE-Edge_A_Runtime_Latency_Predictor_for_Edge_Devices_CVPRW_2022_paper.pdf) by Nair, Saeejith, et al., CVPR 2022
- [Maple: Microprocessor a priori for latency estimation](https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Abbasi_MAPLE_Microprocessor_a_Priori_for_Latency_Estimation_CVPRW_2022_paper.pdf) by Abbasi, Saad, Alexander Wong, and Mohammad Javad Shafiee., CVPR 2022
- [nn-Meter: towards accurate latency prediction of deep-learning model inference on diverse edge devices](https://dl.acm.org/doi/pdf/10.1145/3458864.3467882?casa_token=XqdJMALxIdkAAAAA:VhfaRoo7fWSlLOERqfqPcTjW4NfsBB0EXG7AKTw-s_eUFjMscdzTK6oU1kAPREY6nlF2jqO43PTk6g) by Zhang, Li Lyna, et al., MobiSys 2021


### TinyML
- [Mcunet: Tiny deep learning on iot devices](https://proceedings.neurips.cc/paper/2020/file/86c51678350f656dcc7f490a43946ee5-Paper.pdf) by Lin, Ji, et al. , NeurIPS 2020
- [TinyML: Current Progress, Research Challenges, and Future Roadmap](https://ieeexplore.ieee.org/iel7/9585997/9586083/09586232.pdf?casa_token=OkZ2VanNSfoAAAAA:VepU071luwbGdzcLFA_4bHRlLXZ-CRo2Xwsw-7kLKF8BBLw_2eI83V52l8kc8XP93MwUwHxArgI) by Shafique, Muhammad, et al., DAC 2021
- [Benchmarking TinyML systems: Challenges and direction](https://arxiv.org/pdf/2003.04821) by Banbury, Colby R., et al., arxiv 2020
- [μNAS: Constrained Neural Architecture Search for Microcontrollers](https://dl.acm.org/doi/pdf/10.1145/3437984.3458836) by Liberis, Edgar, Łukasz Dudziak, and Nicholas D. Lane., EuroMLSys 2021
- [Memory-efficient Patch-based Inference for Tiny Deep Learning](https://proceedings.neurips.cc/paper/2021/file/1371bccec2447b5aa6d96d2a540fb401-Paper.pdf) by Lin, Ji, et al., NeurIPS 2021

### Multi-modality Inference
- [Dynamic Multimodal Fusion](https://arxiv.org/abs/2204.00102) by Xue, Zihui, and Radu Marculescu., arxiv 2022

### Sparse Inference
- [SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute](https://www.usenix.org/system/files/osdi22-zheng-ningxin.pdf) by Zheng, Ningxin, et al., OSDI 2022

### Others 
Compression related papers :)
