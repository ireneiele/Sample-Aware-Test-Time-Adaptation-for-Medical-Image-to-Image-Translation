# Sample-Aware-Test-Time-Adaptation-for-Medical-Image-to-Image-Translation

## Introduction
This repository contains the official implementation of our dynamic Test-Time Adaptation (TTA) framework for medical image-to-image translation under distribution shift.
Our method enhances pretrained translation models by inserting lightweight, trainable adapters that are dynamically updated at test time. A set of autoencoder-based reconstruction modules is used to estimate sample-wise domain shift, enabling selective and configuration-specific adaptation only for Out-of-Distribution (OOD) samples. The adaptation is guided by reconstruction errors and optimized via different search strategies.

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b)]()


## Abstract
Image-to-image translation has emerged as a powerful technique in medical imaging, enabling tasks such as image denoising and cross-modality conversion.
However, it suffers from limitations in handling Out-Of-Distribution (OOD) samples without causing performance degradation.
To address this limitation, we propose a novel Test-Time Adaptation (TTA) framework that dynamically adjusts the translation process based on the characteristics of each test sample.
Our method introduces a Reconstruction Module to quantify the domain shift and a Dynamic Adaptation Block that selectively modifies the internal features of a pretrained translation model to mitigate the shift without compromising the performance on In-Distribution (ID) samples that do not require adaptation.
We evaluate our approach on two medical image-to-image translation tasks: low-dose CT denoising and $T_1$ to $T_2$ MRI translation, showing consistent improvements over both the baseline translation model without TTA and prior TTA methods.
Our analysis highlights the limitations of the state-of-the-art that uniformly apply the adaptation to both OOD and ID samples, demonstrating that dynamic, sample-specific adjustment offers a promising path to improve model resilience in real-world scenarios.

![alt text](https://raw.githubusercontent.com/ireneiele/Sample-Aware-Test-Time-Adaptation-for-Medical-Image-to-Image-Translation/refs/heads/main/method_v6_page.jpg)

