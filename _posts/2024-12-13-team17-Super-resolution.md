---
layout: post
comments: true
title: Super-resolution
author: Nicholas Chu, James Feeney, Tyler Nguyen, Jonny Xu
date: 2024-01-01
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction

![Example of Low-Resolution vs. High-Resolution Images](/assets/images/UCLAdeepvision/SR.png)
*Figure 1: An illustration of SR enhancing image clarity.*

Super-resolution (SR) is a transformative task in computer vision, aimed at enhancing the spatial resolution of images or videos by reconstructing high-resolution (HR) content from low-resolution (LR) inputs. The problem of SR is not only about visual appeal; it is fundamental to extracting meaningful details from data where high-quality inputs are unavailable or impractical to acquire.

### Applications of Super-Resolution
SR has far-reaching applications across various domains:
- **Medical Imaging:** Provides enhanced clarity in CT scans, MRI images, and X-rays, aiding more accurate diagnosis.
- **Satellite Imagery:** Improves spatial detail for monitoring environmental changes, urban growth, and disaster management.
- **Video Enhancement:** Plays a crucial role in restoring legacy video footage, streaming high-definition content, and ensuring a better viewing experience.
- **Security and Surveillance:** Enhances the resolution of footage from low-quality cameras, making it easier to identify objects and people.
- **Consumer Electronics:** Powers upscaling technologies in modern televisions, ensuring that content matches the resolution of high-definition displays.

### The Super-Resolution Problem
At its core, SR addresses the challenge of inferring missing high-frequency information from LR data. Mathematically, the relationship between the HR and LR image can be modeled as:
$$\ HR = f(LR) + \epsilon\$$
where $f$ represents the SR mapping function that is learned and $\epsilon$ is the reconstruction error. The goal of SR techniques is to learn an $f$ that minimizes $\epsilon$ while preserving fine details.

The main challenges in SR include:
1. **Ill-Posed Nature:** A single LR image can correspond to multiple possible HR images, making the problem inherently ambiguous.
2. **Trade-Off Between Quality and Computational Cost:** High-quality SR often requires complex models and heavy computational resources.
3. **Realism vs. Perception:** Balancing perceptually pleasing outputs with faithful HR reconstruction remains a significant hurdle.

With advances in machine learning and deep learning, SR techniques have evolved to overcome these limitations, delivering results that were previously unattainable.



---

## History of Classical Methods

Before the rise of deep learning, classical super-resolution methods relied on mathematical models and statistical assumptions. These methods laid the groundwork for modern SR techniques, offering insights into how high-frequency details could be recovered from low-resolution inputs. They can be broadly categorized as interpolation-based, reconstruction-based, and example-based approaches.

![](/assets/images/UCLAdeepvision/bicubic.png)
*Figure 2: Different interpolation methods: black dot is interpolated estimate.*

### Interpolation-Based Methods
Interpolation methods, such as **nearest neighbor**, **bilinear**, and **bicubic interpolation**, estimate missing pixel values by averaging surrounding pixels. While these methods are computationally simple and fast, they often produce overly smooth results and lack the ability to recover finer textures or details.

### Reconstruction-Based Methods
Reconstruction-based techniques, like **iterative back-projection (IBP)**, utilize knowledge of the degradation process (e.g., downscaling and blurring) to iteratively refine the high-resolution output. These methods aim to maintain consistency between the LR input and the reconstructed HR image. Despite their promise, they are computationally expensive and sensitive to noise, limiting their applicability in real-world scenarios.

### Example-Based Methods
Example-based methods introduced the idea of leveraging external datasets to enhance SR. Techniques such as **sparse coding** involve training a dictionary of image patches to model the mapping between LR and HR pairs. These methods achieve better detail reconstruction compared to interpolation-based and reconstruction-based approaches but require extensive datasets and manual feature engineering.

### Transition to Deep Learning and Upsampling Techniques
Classical methods' limitations, such as reliance on handcrafted features, led to the adoption of deep learning. Early models like the **Super-Resolution Convolutional Neural Network (SRCNN)** demonstrated that neural networks could learn the mapping from LR to HR directly from data.

#### Key Components of Deep Learning SR:
1. **Feature Extraction:** Convolutional layers extract spatial features from the LR input.
2. **Non-Linear Mapping:** Hidden layers learn complex transformations to model high-frequency details.
3. **Upsampling Layers:** Deep learning-based SR methods typically include upsampling to increase the spatial dimensions of the image:
   - **Deconvolution (Transposed Convolution):** Applies learned filters to increase resolution while adding meaningful features.
   - **Sub-Pixel Convolution:** Rearranges features from depth to spatial dimensions, offering efficient upscaling.
   - **Interpolation + Convolution:** Combines simple interpolation with subsequent convolution layers to refine details.
4. **Reconstruction:** The final layers generate the HR output from learned features.

#### Example: SRCNN Workflow
1. The LR image is first interpolated to the desired resolution using bicubic interpolation.
2. Convolutional layers extract features from this upscaled image.
3. Additional layers map these features to high-resolution space.
4. The final layer reconstructs the HR image.

Deep learning SR models surpassed classical methods by automating feature learning and capturing complex patterns, paving the way for advanced architectures like GANs and attention mechanisms.

---
## Diffusion-Based Methods: Super-Resolution via Iterative Refinement (SR3)
SR3 (Super-Resolution via Iterative Refinement) is a novel approach to super-resolution using diffusion probabilistic models that was proposed by Chitwan Saharia et al. in 2021. It adapts these models for conditional image generation, achieving high-quality results through a stochastic denoising process.

SR3 operates in two key phases: Forward Diffusion and Reverse Denoising. It works as follows:
   1. **Forward Diffusion: Adding noise**
      
         - The forward process adds Gaussian noise to the HR target image step-by-step using a Markovian-based process. This transforms the HR image into pure noise over $T$ steps.

         - At the final step $T$, $y_T$ is approximately pure Gaussian noise.

   3. **Reverse Denoising: Removing noise**
      
        -  The reverse process begins with a noisy image $y_T \sim \mathcal{N}(0, I)$, and iteratively denoises it to produce $y_0$, the reconstructed HR image.

         - A U-Net-based denoising model $f_{\theta}$ is used to estimate the noise at each step. The denoising objective ensures the predicted noise is subtracted at every iteration:

            $$y_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( y_t - \frac{1 - \alpha_t}{\sqrt{1 - \gamma_t}} f_\theta(x, y_t, \gamma_t) \right) + \sqrt{1 - \alpha_t} z$$

            where $z \sim \mathcal{N}(0, I)$ is Gaussian noise added for stochasticity.

insert image here!!

For the model architecture, SR3 uses a modified U-Net backbone is used to process both the noisy HR image and the bicubic-upsampled LR image. These are concatenated channel-wise as input. Additionally, SR3 adds residual blocks and skip connections to improve gradient flow and learning efficiency. For efficient inference SR3 sets the maximum inference budget to 100 diffusion steps, and hyper-parameter searches over the inference noise schedule.

As a result of these model architecture optimizations by SR3, the model is able to achieve state-of-the-art performance on multiple super-resolution tasks across datasets and domains (faces, natural images). 

inser image here!!

As seen in figure 5 just above, the images that SR3 produced in this example for a 16x16 --> 128x128 face super-resolution task contain finer details, such as skin and hair texture texture, outperforming other GAN-based methods (FSRGAN, PULSE) and a regression baseline trained with MSE while also avoiding GAN artifacts like mode collapse. 

SR3 was able to achieve a fool rate close to 54%, meaning that it produces outputs that are nearly indistringuishable from real images. As a benchmark, the fool rate for the GAN-based method PULSE was 24.6% while the fool rate for FSRGAN was 8.9%, which showcases just how large of an improvement this is.

insert image here!!

All in all, SR3 is a state-of-the-art diffusion-based super-resolution method that offers key advantages over GAN-based methods such as a stronger ability to generate sharp and detailed images, absence of GAN instability issues, such as mode collapse, due to probabilistic modeling, as well as efficiency due to its cascaded architecutre that allows for modular and parallel training.

 

   



## Method 2

## Method 3

## Main Content
Your survey starts here. You can refer to the [source code](https://github.com/lilianweng/lil-log/tree/master/_posts) of [lil's blogs](https://lilianweng.github.io/lil-log/) for article structure ideas or Markdown syntax. We've provided a [sample post](https://ucladeepvision.github.io/CS188-Projects-2022Winter/2017/06/21/an-overview-of-deep-learning.html) from Lilian Weng and you can find the source code [here](https://raw.githubusercontent.com/UCLAdeepvision/CS188-Projects-2022Winter/main/_posts/2017-06-21-an-overview-of-deep-learning.md)

## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/UCLAdeepvision/object_detection.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].



Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

---
