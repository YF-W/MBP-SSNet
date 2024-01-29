# Multi-Bottleneck Progressive Semantic Segmentation
Lesion segmentation in medical images presents challenges, including limitations in data processing at the decoding end and insufficient long-distance information dependency at the encoding-decoding intersection. Traditional networks often perform semantic transmission at the deepest layers, leading to a single channel for expressing rich semantic information. Additionally, the encoding-decoding intersection lacks dependency on long-distance information, especially when dealing with deep semantic and small-sized, multi-channel medical images using small-kernel convolutions, which may overlook concealed lesions. Similarly, the decoding end lacks dependency on short-distance information, often neglected in network layouts, yet crucial for fine-tuning at the final image enlargement stage and addressing key points in lesions.

# Paper:MBP-SSNet（Multi-Bottleneck Progressive Semantic Segmentation Network）
Yuefei Wang,*, Yutong Zhang, Li Zhang, Yuquan Xu, Ronghui Feng, Haoyue Cai, Zuwei Zhao, Jiajing Xue, Zixu Wang, Siyi Qiu, Yixi Yang, Xi Yu

# 1. Architecture Overview 
![image](https://github.com/YF-W/MBP-SSNet/assets/66008255/a136bfcf-7ed9-4523-8f52-eb40a93096ff)
MBP-SSNet is an innovative multi-branch encoder-decoder network designed to address the balance and feature extraction limitations in traditional networks during semantic information processing. The main branch assimilates global ViT encoding and ResNet structures, utilizing MPBFE to construct continuous network blocks, exploiting the "local + global" perspective of large-kernel convolutions. Multiple bottlenecks receive information from different semantic depths, achieving "batch-wise, layer-wise" feature absorption. The decoding process adopts a progressive feature fusion approach, departing from the traditional parallelism of multi-branch decoders. A fine-tuning module is introduced at the network's tail to address edge artifacts and enhance segmentation performance for small-scale objects.

# 2. Comparison chart between traditional u-decode and multi-head decode
![image](https://github.com/YF-W/MBP-SSNet/assets/66008255/314f0471-c93f-4198-90ff-12e8c52d6476)

# 3. Traditional multi-branch baseline and our network baseline
![image](https://github.com/YF-W/MBP-SSNet/assets/66008255/6b6df667-45f6-490e-b92b-e74ee6eeb91c)
In response to the limited semantic information capacity of traditional U-shaped single encoder-decoder architectures, we propose an innovative multi-branch encoder-decoder network named MB-PPFG. In contrast to the conventional triple encoder-single decoder structure, MB-PPFG incorporates three bottlenecks at different semantic depths, dispersing enriched semantic information for multi-level information preservation. Additionally, a Progressive Propulsion Feature Guidance mechanism is introduced in the decoder to progressively fuse semantics from different depths. This design demonstrates enhanced accuracy and efficiency in medical image processing.

# 4. Module 1：MPBFE
![image](https://github.com/YF-W/MBP-SSNet/assets/66008255/62bc34b3-c1ab-4fab-be66-4f61db4191ab)

In the network architecture of MBP-SSNet, the MacroFocus Pre-Bottleneck Feature Enhancer (MPBFE) is cleverly reused three times, positioned at different depths within the Backbone Branch to receive feature collections transitioning towards deeper semantics. Each module consists of three parallel branches: Attention Branch, Macro-Kernel Extraction Branch, and Summarization Branch, each serving a distinct purpose. (1) The Attention Branch fine-tunes channel attention, employing depth-wise separable convolutions and pixel filtering through average and max pooling, processing results differently over various execution times. (2) The Macro-Kernel Extraction Branch utilizes 7x7 and 9x9 convolutions for feature extraction, integrating semantic information from same-level ViT and ResNet. This branch merges with the Attention Branch before passing into the Summarization Branch. (3) The Summarization Branch performs straightforward feature extraction, primarily for aggregating same-level semantic feature channels, ultimately combining with the first two branches.

| Layer Name  | Main Branch                                                  |  Auxiliary Branch  |
| ----------- | ------------------------------------------------------------ | :----------------: |
| Layer 1     | M(x1)=Conv[7*7,64]   M'(x1)=DwConv[3*3,64]                   | A(x1)=A(x0)+M(x0)  |
| Output size | [4, 64, 112, 112]                                            | [4, 64, 112,  112] |
| Layer 2     | M(x2)=(M(x1)+A(x1))   Conv[5*5,64]   M'(x2)=Concat(A(X1),M'(x1))   Conv[1*1,64] |  A(x2)=DAB(A(x1))  |
| Output size | [4, 64, 112, 112]                                            | [4, 64, 112, 112]  |
| Layer 3     | M(x3)=Concat(A(x2),M'(x1))                                   |  A(x3)=DAB(M(x3))  |
| Output size | [4, 128, 112, 112]                                           | [4, 128, 112, 112] |
| Layer 4     | out=A(x3)+M'(x2)                                             |                    |
| Output size | [4, 128, 112, 112]                                           |                    |

# 5. Module 2: MPDFR 
![image](https://github.com/YF-W/MBP-SSNet/assets/66008255/e7cff1f5-a47d-4a26-b93a-201ef9562239)

The MPDFR, positioned at the end of the network, serves as a feature fine-tuning module that employs three methods to refine features passed from the front-end. 1) Image Compression and Expansion: Within two FlexiScale Branches, the incoming image undergoes pooling to increase the receptive field while resisting displacement changes. It then combines with features refined through the HybridFocus Block in the previous stage, ultimately restoring lost pixel information. 2) Fine-Grained Feature Extraction: This module conducts fine-grained feature extraction in two TwinCore Branches and the Refine Block. As the MPDFR image has a larger size and global information has been correlated in the frontend, there's an increased need at this stage to control local subtle features. 3) Memory Point Connection: At the early decoding phase, two critical memory points are preset to retain deeper semantic information. In the MPDFR, the content of these memory points is transmitted to the step before image restoration, supplementing complex deep semantic information.

| Layer Name  | Main  Branch(D(3)D(4)D(5))                                   |
| ----------- | ------------------------------------------------------------ |
| Layer 1     | D'(3)=DAB+Conv[3*3,64]+Conv[5*5,64]                          |
| Output size | [4, 64, 112, 112]                                            |
| Layer 2     | D'(4)=[Maxpool,64]+D'(3),UpConv[3*3,64]   D'(5)=[Maxpool,64]+D'(3),UpConv[3*3,64]   D''(4)=Conv[3*3,64],Conv[5*5,64]   D''(5)=Conv[3*3,64],Conv[5*5,64] |
| Output size | D'(4)=D'(5)=[4, 64, 224, 224]   D''(4)=D''(5)=[4, 64, 224, 224] |
| Layer 3     | p_out(1)=Concat(D'(4),D''(4)),Conv[1*1,64]   p_out(2)=Concat(D'(5),D''(5)),Conv[1*1,64] |
| Output size | p_out(1)=[4, 64, 224, 224]   p_out(2)=[4, 64, 224, 224]      |
| Layer 4     | out=p_out(1)+Dw_Conv[3*3,64]   +p_out(2)+Dw_Conv[3*3,64]     |
| Output size | out=[4, 64, 224, 224]                                        |

# Datasets:

1.The LUNG dataset: https://www.kaggle.com/datasets/kmader/finding-lungs-in-ct-data.

2.The Skin Lesions dataset: https://www.kaggle.com/datasets/ojaswipandey/skin-lesion-dataset.

3.The NEVUS dataset: https://challenge.isic-archive.com/data/#2017.

4.The SIC2017-Seborrheic Keratosis dataset: https://challenge.isic-archive.com/data/#2017.

5.The MICCAI2015-CVC ClinicDB dataset: https://polyp.grand-challenge.org/CVCClinicDB/.

6.The TNSCUI2020 - Thyroid Nodule dataset: https://github.com/WAMAWAMA/TNSCUI2020-Seg-Rank1st.
